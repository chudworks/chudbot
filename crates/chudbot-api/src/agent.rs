//! Provider-neutral agent runtime.
//!
//! This module is the orchestration contract between a configured model,
//! provider-neutral transcripts, and locally executed client tools. It does not
//! know about Discord, storage, HTTP providers, or concrete tool registries:
//! callers build those pieces elsewhere and pass in a [`Model`] plus one
//! [`ClientToolExecutor`].
//!
//! A normal run flows through four stages:
//!
//! 1. [`AgentSpec`] supplies static agent behavior: instructions, loop limits,
//!    and optional tool allowlists.
//! 2. [`Agent::run`] injects the instructions into the incoming [`Transcript`],
//!    normalizes model/agent tool exposure, and calls the [`LlmBackend`] one
//!    step at a time.
//! 3. Provider-side tools and grounding are recorded as trace data, while
//!    client-side tool calls are executed locally and returned to the model as a
//!    user turn.
//! 4. The final [`AgentRun`] returns the completed transcript, provider
//!    continuation data, trace records, and usage records that higher layers can
//!    persist.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use futures::Stream;
use futures::stream::{FuturesOrdered, StreamExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::Instrument;

use crate::ids::{ModelId, ProviderName, ToolName};
use crate::llm::{
    LlmBackend, Model, ModelStep, ModelStepCollector, ModelStepDelta, ModelStepEvent,
    ModelStepOutput, ModelStepReducerError, ServerToolSet,
};
use crate::media::BoxedMediaRef;
use crate::storage::{ModelStepKind, ModelStepTrace};
use crate::tool::{
    ClientToolCall, ClientToolDefinition, ClientToolExecutor, ClientToolExecutorError,
    ClientToolOutput, ClientToolResult, ClientToolResultContent, ClientToolSpec, ClientToolTrace,
    NoClientTools, ToolTrace,
};
use crate::transcript::{ContentBlock, ProviderContinuation, Transcript, TranscriptTurn, TurnRole};
use crate::usage::UsageRecord;

/// Static, provider-neutral agent configuration.
///
/// This is TOML-shaped data for the parts of an agent that are independent of
/// a concrete provider or platform. It intentionally does not carry runtime
/// tool implementations, provider clients, or model routing; callers pair it
/// with a [`Model`] and a [`ClientToolExecutor`] when building an [`Agent`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    /// Agent instructions applied to each run's transcript.
    pub system_prompt: String,
    /// Provider-side/server-side tools this agent allows.
    ///
    /// `None` means every server tool allowed by the model config is available
    /// to the provider. When set, the runner intersects this set with the
    /// model's server-tool set after trimming and lowercasing names.
    #[serde(default)]
    pub server_tools: Option<ServerToolSet>,
    /// Optional static list of client tool names enabled for this agent.
    ///
    /// `None` means every runtime tool supplied by the executor is exposed. A
    /// populated list only filters the executor's definitions; it does not
    /// create tools by name.
    #[serde(default)]
    pub client_tools: Option<Vec<ToolName>>,
    /// Agent loop limits.
    pub limits: AgentLimits,
}

impl AgentSpec {
    /// Create an agent spec with default loop limits and no tool restrictions.
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            server_tools: None,
            client_tools: None,
            limits: AgentLimits::default(),
        }
    }

    /// Restrict the runtime client-tool surface to these tool names.
    ///
    /// The names are matched against the [`ClientToolDefinition`] values
    /// returned by the runtime executor. Unknown names simply expose nothing by
    /// themselves.
    pub fn with_client_tools(mut self, client_tools: Vec<ToolName>) -> Self {
        self.client_tools = Some(client_tools);
        self
    }

    /// Set the model/tool loop limits for this agent.
    pub fn with_limits(mut self, limits: AgentLimits) -> Self {
        self.limits = limits;
        self
    }
}

/// Concrete provider-neutral agent runtime.
///
/// `Agent` is intentionally small: the model contains provider routing and
/// model config, the spec contains agent behavior, and the executor contains
/// local tool implementations. Platform-specific code decides how to build
/// those pieces before calling [`Self::run`].
pub struct Agent<B, T = NoClientTools> {
    /// Callable model.
    model: Model<B>,
    /// Agent instructions.
    spec: AgentSpec,
    /// Runtime client tool executor available to this agent.
    tool_executor: T,
}

impl<B, T> Agent<B, T>
where
    T: ClientToolExecutor,
{
    /// Construct a runnable agent from a model, static spec, and tool executor.
    ///
    /// The executor owns all local/client-side tool implementations for this
    /// agent. Agent-level `client_tools` config is only an allowlist over the
    /// executor's advertised definitions.
    pub fn new(model: Model<B>, spec: AgentSpec, tool_executor: T) -> Self {
        Self {
            model,
            spec,
            tool_executor,
        }
    }
}

impl<B, T> fmt::Debug for Agent<B, T>
where
    B: fmt::Debug,
    T: ClientToolExecutor,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("model", &self.model)
            .field("spec", &self.spec)
            .field("client_tools", &tool_specs(self.tool_executor.tools()))
            .finish()
    }
}

/// Terminal error returned by [`Agent::run`] before an [`AgentRun`] exists.
///
/// Recoverable model-requested tool failures are converted into tool result
/// blocks and sent back to the model; provider/backend failures stay out of the
/// transcript and are returned through this error type.
#[derive(Debug, Error)]
pub enum AgentRunError<BE>
where
    BE: std::error::Error + Send + Sync + 'static,
{
    /// Provider step failed.
    #[error("model error: {0}")]
    Model(#[source] BE),
    /// Provider step events could not be reduced into a valid model step.
    #[error("model step stream error: {0}")]
    ModelStep(#[from] ModelStepReducerError),
    /// Agent event stream ended before a terminal run event.
    #[error("agent run stream ended before a finished event")]
    RunStreamEnded,
    /// Agent event stream emitted more than one terminal run event.
    #[error("agent run stream emitted more than one finished event")]
    DuplicateRunFinished,
}

/// Limits for the provider/tool loop inside [`Agent::run`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AgentLimits {
    /// Maximum model steps before the run returns
    /// [`AgentOutcome::IterationLimit`].
    pub max_iterations: u32,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self { max_iterations: 8 }
    }
}

/// Complete output from a finished agent loop.
///
/// This is the handoff object for bot/runtime layers: it includes the
/// model-facing transcript after the attempt, auditable tool/model traces,
/// provider continuation state for replay, and direct model usage. Tool traces
/// may carry additional usage from nested agents or media providers.
#[derive(Debug, Clone)]
pub struct AgentRun {
    /// Final outcome of the loop.
    pub outcome: AgentOutcome,
    /// Transcript after the run, including final assistant content or any
    /// intermediate assistant/tool-result turns that were produced before the
    /// loop stopped.
    pub transcript: Transcript,
    /// Tool trace records in model-observed order.
    pub trace: Vec<ToolTrace>,
    /// Ordered provider model-step traces for replay and auditing.
    pub model_steps: Vec<ModelStepTrace>,
    /// Last concrete model id reported by a provider during the run.
    pub last_model_id: Option<ModelId>,
    /// Last provider continuation to persist for cross-turn replay.
    pub final_continuation: Option<ProviderContinuation>,
    /// Usage/cost accumulated directly by model steps in this agent run.
    ///
    /// Client and server tool trace entries may also carry their own usage; use
    /// [`Self::all_usage`] when billing/reporting code needs the full total.
    pub usage: Vec<UsageRecord>,
}

impl AgentRun {
    /// Collect model usage plus every traced client/server tool usage record.
    ///
    /// Grounding-only trace records have no usage channel today, so they are
    /// intentionally skipped.
    pub fn all_usage(&self) -> Vec<UsageRecord> {
        let mut usage = self.usage.clone();
        for trace in &self.trace {
            match trace {
                ToolTrace::Client { trace } => usage.extend(trace.usage.iter().cloned()),
                ToolTrace::Server { tool } => usage.extend(tool.usage.iter().cloned()),
                ToolTrace::Grounding { .. } => {}
            }
        }
        usage
    }
}

/// Streaming event emitted by [`Agent::run`].
#[derive(Debug, Clone)]
pub enum AgentRunEvent {
    /// Agent run started.
    RunStarted,
    /// One provider/model step started.
    ModelStepStarted {
        /// Zero-based model-step ordinal.
        ordinal: u32,
    },
    /// Provider/model event forwarded from the active step.
    ModelEvent {
        /// Zero-based model-step ordinal.
        ordinal: u32,
        /// Provider event.
        event: ModelStepEvent,
    },
    /// Chudbot-owned client tool execution started.
    ClientToolStarted {
        /// Tool call requested by the model.
        call: ClientToolCall,
    },
    /// Chudbot-owned client tool execution finished.
    ClientToolFinished {
        /// Model-facing tool result.
        result: ClientToolResult,
        /// Persistable tool trace.
        trace: ClientToolTrace,
        /// Media references appended to the following user turn.
        media: Vec<BoxedMediaRef>,
    },
    /// Agent run finished and produced the stable collected output.
    RunFinished {
        /// Stable output used by existing callers.
        run: AgentRun,
    },
}

impl AgentRunEvent {
    /// Stable kind label for logging and diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::RunStarted => "run_started",
            Self::ModelStepStarted { .. } => "model_step_started",
            Self::ModelEvent { .. } => "model_event",
            Self::ClientToolStarted { .. } => "client_tool_started",
            Self::ClientToolFinished { .. } => "client_tool_finished",
            Self::RunFinished { .. } => "run_finished",
        }
    }
}

/// Collect a streaming agent run into the stable output shape.
pub async fn collect_agent_run<S, BE>(events: S) -> Result<AgentRun, AgentRunError<BE>>
where
    S: Stream<Item = Result<AgentRunEvent, AgentRunError<BE>>> + Send,
    BE: std::error::Error + Send + Sync + 'static,
{
    futures::pin_mut!(events);
    let mut finished = None;
    while let Some(event) = events.next().await {
        let event = event?;
        tracing::trace!(event = event.kind(), "collecting agent run event");
        match event {
            AgentRunEvent::RunFinished { run } => {
                tracing::trace!(
                    event = "run_finished",
                    outcome = run.outcome.kind(),
                    turns = run.transcript.turns.len(),
                    trace_records = run.trace.len(),
                    model_steps = run.model_steps.len(),
                    usage_records = run.usage.len(),
                    "collecting agent run event"
                );
                if finished.replace(run).is_some() {
                    return Err(AgentRunError::DuplicateRunFinished);
                }
            }
            AgentRunEvent::RunStarted
            | AgentRunEvent::ModelStepStarted { .. }
            | AgentRunEvent::ModelEvent { .. }
            | AgentRunEvent::ClientToolStarted { .. }
            | AgentRunEvent::ClientToolFinished { .. } => {}
        }
    }
    finished.ok_or(AgentRunError::RunStreamEnded)
}

/// Outcome recorded for an agent attempt.
///
/// [`Agent::run`] currently returns transport/provider failures as
/// [`AgentRunError`]. The `Failed` variant is still part of the persistable
/// contract for higher layers that may convert non-transport failures into a
/// partial run.
#[derive(Debug, Clone)]
pub enum AgentOutcome {
    /// Completed with a final answer.
    Completed {
        /// Assistant answer.
        answer: AssistantAnswer,
    },
    /// Failed before a final answer.
    Failed {
        /// Error.
        error: AgentError,
        /// Partial assistant answer if any.
        partial: Option<AssistantAnswer>,
    },
    /// Hit the iteration cap before the provider returned a final answer.
    IterationLimit {
        /// Configured maximum.
        max_iterations: u32,
    },
    /// Cancelled by the caller.
    Cancelled {
        /// Cancellation reason.
        reason: String,
    },
}

impl AgentOutcome {
    /// Stable kind label for logging and diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Completed { .. } => "completed",
            Self::Failed { .. } => "failed",
            Self::IterationLimit { .. } => "iteration_limit",
            Self::Cancelled { .. } => "cancelled",
        }
    }
}

/// Final assistant answer returned by a completed run.
#[derive(Debug, Clone)]
pub struct AssistantAnswer {
    /// Plain text answer derived by concatenating final text blocks.
    pub text: String,
    /// Full provider-neutral answer blocks, including non-text media if present.
    pub blocks: Vec<ContentBlock>,
}

/// Persistable agent failure reason.
///
/// This is separate from [`AgentRunError`]: `AgentRunError` is the Rust error
/// channel for a run that could not produce an [`AgentRun`], while `AgentError`
/// is stored inside [`AgentOutcome::Failed`] when higher layers have a partial
/// run to persist.
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentError {
    /// Provider step failed.
    #[error("model error: {message}")]
    Model {
        /// Error message.
        message: String,
    },
    /// Tool protocol failed in a way that could not be returned to the model.
    #[error("tool loop error: {message}")]
    ToolLoop {
        /// Error message.
        message: String,
    },
}

impl<B, T> Agent<B, T>
where
    B: LlmBackend,
    T: ClientToolExecutor,
{
    /// Run this agent against one transcript until final answer or loop limit.
    ///
    /// The incoming transcript supplies the conversation turns and optional
    /// provider routing id. The runner replaces any existing instructions with
    /// this agent's system prompt, then repeats provider calls until the backend
    /// returns final content, asks for client tools, asks for a provider
    /// continuation, or the configured iteration limit is reached.
    #[tracing::instrument(
        name = "agent.run",
        skip_all,
        fields(
            transcript_id = tracing::field::Empty,
            provider = tracing::field::Empty,
            model = tracing::field::Empty,
            initial_turns = tracing::field::Empty,
            max_iterations = tracing::field::Empty,
            client_tools = tracing::field::Empty,
            server_tools = tracing::field::Empty,
        )
    )]
    pub fn run(
        &self,
        transcript: Transcript,
    ) -> impl Stream<Item = Result<AgentRunEvent, AgentRunError<B::Error>>> + Send + '_ {
        async_stream::try_stream! {
            let event = AgentRunEvent::RunStarted;
            trace_agent_run_event(&event);
            yield event;
            // Prepare model input. Agent instructions are owned by the spec, so a
            // caller-provided transcript cannot accidentally carry stale system text.
            let Transcript { id, turns, .. } = transcript;
            let initial_turns = turns.len();
            let mut transcript = Transcript {
                id,
                instructions: Some(self.spec.system_prompt.clone()),
                turns,
            };

            // Resolve the model-visible tool surfaces once for this run. Provider
            // crates receive normalized server-tool names; local client tools remain
            // keyed by the repo's typed `ToolName`.
            let client_tools = enabled_tool_specs(&self.spec, &self.tool_executor);
            let server_tools = enabled_server_tools(
                &self.model.spec.server_tools,
                self.spec.server_tools.as_ref(),
            );
            // Populate structured tracing fields after computing the effective
            // request shape. These fields are intentionally low-cardinality except
            // for transcript/model ids.
            let span = tracing::Span::current();
            if let Some(id) = transcript.id.as_deref() {
                span.record("transcript_id", id);
            }
            span.record(
                "provider",
                tracing::field::display(self.model.backend.backend_name()),
            );
            span.record("model", tracing::field::display(&self.model.spec.id));
            span.record("initial_turns", initial_turns);
            span.record("max_iterations", self.spec.limits.max_iterations);
            span.record("client_tools", client_tools.len());
            span.record("server_tools", server_tools.len());
            tracing::info!("starting agent run");

            // Accumulators returned in `AgentRun`. The transcript is the model's
            // replayable view; trace/model_steps/usage are the audit view.
            let mut trace = Vec::new();
            let mut model_steps = Vec::new();
            let mut usage = Vec::new();
            let mut last_model_id = None;
            let mut final_continuation = None;
            let provider = self.model.backend.backend_name().clone();

            for iteration in 0..self.spec.limits.max_iterations {
                // Step 1: ask the routed backend what should happen next.
                tracing::debug!(
                    iteration = iteration + 1,
                    turns = transcript.turns.len(),
                    trace_records = trace.len(),
                    usage_records = usage.len(),
                    "requesting model step"
                );
                let event = AgentRunEvent::ModelStepStarted { ordinal: iteration };
                trace_agent_run_event(&event);
                yield event;
                let events = self.model.backend.step(crate::llm::ModelStepRequest {
                    model: self.model.spec.id.clone(),
                    transcript: transcript.clone(),
                    client_tools: client_tools.clone(),
                    server_tools: server_tools.clone(),
                    sampling: self.model.spec.sampling,
                    provider_options: self.model.spec.provider_options.clone(),
                });
                futures::pin_mut!(events);
                let mut collector = ModelStepCollector::default();
                while let Some(event) = events.next().await {
                    let event = match event {
                        Ok(event) => event,
                        Err(error) => {
                            tracing::warn!(
                                iteration = iteration + 1,
                                error = %error,
                                "model step failed"
                            );
                            Err(AgentRunError::Model(error))?
                        }
                    };
                    collector.push(event.clone())?;
                    let event = AgentRunEvent::ModelEvent {
                        ordinal: iteration,
                        event,
                    };
                    trace_agent_run_event(&event);
                    yield event;
                }
                let step = collector.finish()?;
                tracing::trace!(
                    iteration = iteration + 1,
                    step = ?step,
                    "collected model step"
                );

                // Step 2: fold the provider's step into transcript, traces, usage,
                // and continuation state. Client tool calls get an extra local
                // dispatch phase before the next provider step.
                let ModelStep { kind, output } = step;
                match kind {
                    ModelStepKind::Final => {
                        // Final content is both returned as `AssistantAnswer` and
                        // appended to the replay transcript as an assistant turn.
                        model_steps.push(model_step_trace(
                            iteration,
                            kind,
                            &provider,
                            &output,
                        ));
                        let answer_blocks = output.answer_blocks();
                        tracing::debug!(
                            iteration = iteration + 1,
                            model = %output.model_id,
                            content_blocks = answer_blocks.len(),
                            server_tools = output.server_tool_uses().count(),
                            grounding = output.grounding().count(),
                            usage_records = output.usage.len(),
                            has_continuation = output.continuation().is_some(),
                            "model returned final answer"
                        );
                        append_step_trace(&mut trace, &output);
                        usage.extend(output.usage.iter().cloned());
                        last_model_id = Some(output.model_id.clone());
                        final_continuation = output.continuation().cloned();
                        let answer = answer_from_content(answer_blocks.clone());
                        transcript.push(TranscriptTurn {
                            role: TurnRole::Assistant,
                            blocks: answer_blocks,
                            metadata: serde_json::Value::Null,
                        });
                        tracing::info!(
                            iterations = iteration + 1,
                            answer_chars = answer.text.chars().count(),
                            trace_records = trace.len(),
                            usage_records = usage.len(),
                            "agent run completed"
                        );
                        let run = AgentRun {
                            outcome: AgentOutcome::Completed { answer },
                            transcript,
                            trace,
                            model_steps,
                            last_model_id,
                            final_continuation,
                            usage,
                        };
                        let event = AgentRunEvent::RunFinished { run };
                        trace_agent_run_event(&event);
                        yield event;
                        return;
                    }
                    ModelStepKind::Continue => {
                        // Continuations are provider-owned state. We record and
                        // replay them as transcript blocks, but only the backend
                        // that created the continuation should interpret them.
                        model_steps.push(model_step_trace(
                            iteration,
                            kind,
                            &provider,
                            &output,
                        ));
                        tracing::debug!(
                            iteration = iteration + 1,
                            model = %output.model_id,
                            content_blocks = output.transcript_blocks().count(),
                            server_tools = output.server_tool_uses().count(),
                            grounding = output.grounding().count(),
                            usage_records = output.usage.len(),
                            has_continuation = output.continuation().is_some(),
                            "model requested continuation"
                        );
                        append_step_trace(&mut trace, &output);
                        usage.extend(output.usage.iter().cloned());
                        last_model_id = Some(output.model_id.clone());
                        final_continuation = output.continuation().cloned();
                        append_assistant_step(&mut transcript, output);
                    }
                    ModelStepKind::ClientTools => {
                        // Preserve the assistant tool-call turn before appending
                        // local results. Providers rely on this call/result order
                        // when converting the neutral transcript to native shapes.
                        model_steps.push(model_step_trace(
                            iteration,
                            kind,
                            &provider,
                            &output,
                        ));
                        let calls: Vec<_> = output.client_tool_calls().cloned().collect();
                        tracing::debug!(
                            iteration = iteration + 1,
                            model = %output.model_id,
                            content_blocks = output.transcript_blocks().count(),
                            client_tool_calls = calls.len(),
                            server_tools = output.server_tool_uses().count(),
                            grounding = output.grounding().count(),
                            usage_records = output.usage.len(),
                            has_continuation = output.continuation().is_some(),
                            "model requested client tools"
                        );
                        append_step_trace(&mut trace, &output);
                        usage.extend(output.usage.iter().cloned());
                        last_model_id = Some(output.model_id.clone());
                        final_continuation = output.continuation().cloned();

                        for call in &calls {
                            let event = AgentRunEvent::ClientToolStarted { call: call.clone() };
                            trace_agent_run_event(&event);
                            yield event;
                        }
                        append_assistant_step(&mut transcript, output);

                        // Tool calls run concurrently, then are yielded back in
                        // provider-requested order before they become the next
                        // user turn.
                        let tool_results =
                            execute_client_tool_calls(&client_tools, &self.tool_executor, calls).await;
                        let mut result_blocks = Vec::with_capacity(tool_results.len());
                        for (result, tool_trace, media) in tool_results {
                            let event = AgentRunEvent::ClientToolFinished {
                                result: result.clone(),
                                trace: tool_trace.clone(),
                                media: media.clone(),
                            };
                            trace_agent_run_event(&event);
                            yield event;
                            result_blocks.push(ContentBlock::ClientToolResult(result));
                            // Media handles are shown to the next model step as
                            // native media blocks, while the trace keeps only the
                            // JSON/text protocol result and audit payload.
                            result_blocks
                                .extend(media.into_iter().map(|media| ContentBlock::Media { media }));
                            trace.push(ToolTrace::Client { trace: tool_trace });
                        }
                        transcript.push(TranscriptTurn {
                            role: TurnRole::User,
                            blocks: result_blocks,
                            metadata: serde_json::Value::Null,
                        });
                        tracing::debug!(
                            iteration = iteration + 1,
                            turns = transcript.turns.len(),
                            trace_records = trace.len(),
                            "client tool results appended"
                        );
                    }
                }
            }

            // The caller still gets the partial transcript and audit data so a UI
            // can show exactly how far the loop got before the limit stopped it.
            tracing::warn!(
                max_iterations = self.spec.limits.max_iterations,
                trace_records = trace.len(),
                usage_records = usage.len(),
                "agent run hit iteration limit"
            );
            let run = AgentRun {
                outcome: AgentOutcome::IterationLimit {
                    max_iterations: self.spec.limits.max_iterations,
                },
                transcript,
                trace,
                model_steps,
                last_model_id,
                final_continuation,
                usage,
            };
            let event = AgentRunEvent::RunFinished { run };
            trace_agent_run_event(&event);
            yield event;
        }
    }
}

fn trace_agent_run_event(event: &AgentRunEvent) {
    match event {
        AgentRunEvent::RunStarted => {
            tracing::trace!(event = "run_started", "yielding agent run event");
        }
        AgentRunEvent::ModelStepStarted { ordinal } => {
            tracing::trace!(
                event = "model_step_started",
                ordinal,
                iteration = ordinal + 1,
                "yielding agent run event"
            );
        }
        AgentRunEvent::ModelEvent { ordinal, event } => {
            trace_model_step_event(*ordinal, event);
        }
        AgentRunEvent::ClientToolStarted { call } => {
            tracing::trace!(
                event = "client_tool_started",
                tool_use_id = %call.id,
                tool = %call.name,
                "yielding agent run event"
            );
        }
        AgentRunEvent::ClientToolFinished {
            result,
            trace,
            media,
        } => {
            tracing::trace!(
                event = "client_tool_finished",
                tool_use_id = %result.tool_use_id,
                is_error = result.is_error,
                content_kind = result.content.kind(),
                content = ?result.content,
                trace_response = ?trace.trace_response,
                usage_records = trace.usage.len(),
                media = media.len(),
                "yielding agent run event"
            );
        }
        AgentRunEvent::RunFinished { run } => {
            tracing::trace!(
                event = "run_finished",
                outcome = run.outcome.kind(),
                turns = run.transcript.turns.len(),
                trace_records = run.trace.len(),
                model_steps = run.model_steps.len(),
                usage_records = run.usage.len(),
                has_last_model = run.last_model_id.is_some(),
                has_final_continuation = run.final_continuation.is_some(),
                "yielding agent run event"
            );
        }
    }
}

fn trace_model_step_event(ordinal: u32, event: &ModelStepEvent) {
    match event {
        ModelStepEvent::Delta(delta) => trace_model_step_delta(ordinal, delta),
        ModelStepEvent::Continuation(continuation) => {
            tracing::trace!(
                event = "model_event",
                model_event = "continuation",
                ordinal,
                iteration = ordinal + 1,
                provider = %continuation.provider,
                data_kind = continuation.data.kind(),
                "yielding agent run event"
            );
        }
        ModelStepEvent::ServerToolUse(tool) => {
            tracing::trace!(
                event = "model_event",
                model_event = "server_tool_use",
                ordinal,
                iteration = ordinal + 1,
                provider = %tool.provider,
                tool = %tool.name,
                id = tool.id.as_deref(),
                status = tool.status.as_deref(),
                raw = ?tool.raw,
                usage_records = tool.usage.len(),
                "yielding agent run event"
            );
        }
        ModelStepEvent::Grounding(metadata) => {
            tracing::trace!(
                event = "model_event",
                model_event = "grounding",
                ordinal,
                iteration = ordinal + 1,
                provider = %metadata.provider,
                raw = ?metadata.raw,
                "yielding agent run event"
            );
        }
        ModelStepEvent::Usage(usage) => {
            tracing::trace!(
                event = "model_event",
                model_event = "usage",
                ordinal,
                iteration = ordinal + 1,
                provider = %usage.provider,
                model = usage.model.as_ref().map(ModelId::as_str),
                input_tokens = usage.input_tokens,
                cached_input_tokens = usage.cached_input_tokens,
                output_tokens = usage.output_tokens,
                reasoning_tokens = usage.reasoning_tokens,
                total_tokens = usage.total_tokens,
                has_cost = usage.cost.is_some(),
                "yielding agent run event"
            );
        }
        ModelStepEvent::Finished { kind, model_id } => {
            tracing::trace!(
                event = "model_event",
                model_event = "finished",
                ordinal,
                iteration = ordinal + 1,
                kind = kind.label(),
                model = %model_id,
                "yielding agent run event"
            );
        }
    }
}

fn trace_model_step_delta(ordinal: u32, delta: &ModelStepDelta) {
    match delta {
        ModelStepDelta::Text { item_id, delta } => {
            tracing::trace!(
                event = "model_event",
                model_event = "delta",
                delta = "text",
                ordinal,
                iteration = ordinal + 1,
                item_id = item_id.as_str(),
                delta_chars = delta.chars().count(),
                text_delta = ?delta,
                "yielding agent run event"
            );
        }
        ModelStepDelta::ReasoningSummary {
            item_id,
            provider,
            kind,
            delta,
        } => {
            tracing::trace!(
                event = "model_event",
                model_event = "delta",
                delta = "reasoning_summary",
                ordinal,
                iteration = ordinal + 1,
                item_id = item_id.as_str(),
                provider = %provider,
                kind = kind.as_deref(),
                delta_chars = delta.chars().count(),
                summary_delta = ?delta,
                "yielding agent run event"
            );
        }
        ModelStepDelta::ClientToolCall {
            item_id,
            id,
            name,
            arguments_delta,
        } => {
            tracing::trace!(
                event = "model_event",
                model_event = "delta",
                delta = "client_tool_call",
                ordinal,
                iteration = ordinal + 1,
                item_id = item_id.as_str(),
                tool_use_id = %id,
                tool = name.as_ref().map(ToolName::as_str),
                arguments_delta_chars = arguments_delta.chars().count(),
                arguments_delta = ?arguments_delta,
                "yielding agent run event"
            );
        }
    }
}

trait JsonValueKind {
    fn kind(&self) -> &'static str;
}

impl JsonValueKind for serde_json::Value {
    fn kind(&self) -> &'static str {
        match self {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "bool",
            serde_json::Value::Number(_) => "number",
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        }
    }
}

/// Build the compact provider-step record stored with a turn.
///
/// `iteration` is zero-based to match `ModelStepTrace::ordinal`; logs display
/// one-based iteration numbers for human reading.
fn model_step_trace(
    iteration: u32,
    kind: ModelStepKind,
    provider: &ProviderName,
    output: &ModelStepOutput,
) -> ModelStepTrace {
    ModelStepTrace {
        ordinal: i32::try_from(iteration).unwrap_or(i32::MAX),
        kind,
        provider: provider.clone(),
        model: output.model_id.clone(),
        continuation: output.continuation().cloned(),
    }
}

/// Append provider-owned trace events from one model step.
///
/// Server tools and grounding are already complete when the provider returns a
/// step. They do not produce local tool results, so they go straight into the
/// trace stream instead of the transcript.
fn append_step_trace(trace: &mut Vec<ToolTrace>, output: &ModelStepOutput) {
    trace.extend(
        output
            .server_tool_uses()
            .cloned()
            .map(|tool| ToolTrace::Server { tool }),
    );
    trace.extend(
        output
            .grounding()
            .cloned()
            .map(|metadata| ToolTrace::Grounding { metadata }),
    );
}

/// Resolve the client tools visible to the model for this agent run.
///
/// The runtime executor advertises the real available tools. `AgentSpec` can
/// only narrow that set by name; this keeps static config from inventing a
/// callable tool that the executor does not own.
fn enabled_tool_specs(
    spec: &AgentSpec,
    tool_executor: &impl ClientToolExecutor,
) -> BTreeMap<ToolName, ClientToolSpec> {
    let tool_specs = tool_specs(tool_executor.tools());
    let Some(enabled) = &spec.client_tools else {
        return tool_specs;
    };

    tool_specs
        .into_iter()
        .filter(|(name, _)| enabled.iter().any(|enabled_name| enabled_name == name))
        .collect()
}

/// Resolve provider-side tools by intersecting model and agent allowlists.
///
/// The model config is the hard upper bound for provider-native capabilities.
/// Agent config can narrow that set, but cannot enable a provider-side tool the
/// model config did not allow.
fn enabled_server_tools(
    model_tools: &ServerToolSet,
    agent_tools: Option<&ServerToolSet>,
) -> ServerToolSet {
    let mut model_tools = normalized_server_tools(model_tools);
    let Some(agent_tools) = agent_tools else {
        return model_tools;
    };

    let agent_tools = normalized_server_tools(agent_tools);
    model_tools.retain(|tool| agent_tools.contains(tool));
    model_tools
}

/// Normalize a set of provider-side tool names for cross-provider matching.
fn normalized_server_tools(tools: &ServerToolSet) -> BTreeSet<String> {
    tools
        .iter()
        .filter_map(|tool| normalize_server_tool(tool))
        .collect()
}

/// Normalize one provider-side tool name.
///
/// Empty entries are ignored so config like `["web_search", " "]` does not
/// produce a phantom provider tool.
fn normalize_server_tool(tool: &str) -> Option<String> {
    let tool = tool.trim();
    (!tool.is_empty()).then(|| tool.to_ascii_lowercase())
}

/// Execute all client tool calls from one provider step.
///
/// Calls are dispatched concurrently to avoid serializing independent tool
/// latency. Results are yielded in the provider's original call order before
/// they are appended to the transcript.
async fn execute_client_tool_calls(
    enabled_tools: &BTreeMap<ToolName, ClientToolSpec>,
    tool_executor: &impl ClientToolExecutor,
    calls: Vec<ClientToolCall>,
) -> Vec<(
    ClientToolResult,
    ClientToolTrace,
    Vec<crate::media::BoxedMediaRef>,
)> {
    tracing::debug!(calls = calls.len(), "executing client tool calls");
    let mut pending = FuturesOrdered::new();
    for call in calls {
        let tool_name = call.name.to_string();
        let tool_use_id = call.id.to_string();
        pending.push_back(
            async move {
                let output = call_client_tool(enabled_tools, tool_executor, call.clone()).await;
                (call, output)
            }
            .instrument(tracing::debug_span!(
                "agent.client_tool",
                tool = %tool_name,
                tool_use_id = %tool_use_id,
            )),
        );
    }

    let mut completed = Vec::new();
    while let Some((call, output)) = pending.next().await {
        let output = match output {
            Ok(output) => {
                tracing::debug!(
                    tool = %call.name,
                    tool_use_id = %call.id,
                    is_error = output.is_error,
                    usage_records = output.usage.len(),
                    "client tool completed"
                );
                output
            }
            Err(error) => {
                // Tool failures are model-visible results, not fatal agent
                // errors. The model gets one more chance to recover or explain.
                tracing::warn!(
                    tool = %call.name,
                    tool_use_id = %call.id,
                    error = %error,
                    "client tool failed"
                );
                ClientToolOutput {
                    result: ClientToolResultContent::Text {
                        text: format!("tool `{}` failed: {error}", call.name),
                    },
                    media: Vec::new(),
                    is_error: true,
                    trace_response: serde_json::json!({
                        "error": error.to_string(),
                    }),
                    usage: Vec::new(),
                }
            }
        };
        let result = ClientToolResult {
            tool_use_id: call.id.clone(),
            content: output.result.clone(),
            is_error: output.is_error,
        };
        let tool_trace = ClientToolTrace {
            call,
            result: result.clone(),
            trace_response: output.trace_response,
            usage: output.usage,
        };
        completed.push((result, tool_trace, output.media));
    }

    tracing::debug!(completed = completed.len(), "client tool calls finished");
    completed
}

/// Convert executor definitions to the map shape expected by providers.
///
/// The first definition for a name wins. Duplicate names are almost certainly a
/// wiring bug, but keeping the first definition preserves deterministic request
/// shape and avoids changing behavior mid-run.
fn tool_specs(definitions: Vec<ClientToolDefinition>) -> BTreeMap<ToolName, ClientToolSpec> {
    let mut specs = BTreeMap::new();
    for definition in definitions {
        if specs.contains_key(&definition.name) {
            tracing::warn!(
                tool = %definition.name,
                "duplicate client tool definition ignored"
            );
            continue;
        }
        specs.insert(definition.name, definition.spec);
    }
    specs
}

/// Dispatch one enabled client tool call through the runtime executor.
async fn call_client_tool<T>(
    enabled_tools: &BTreeMap<ToolName, ClientToolSpec>,
    tool_executor: &T,
    call: ClientToolCall,
) -> Result<ClientToolOutput, ClientToolExecutorError<T::Error>>
where
    T: ClientToolExecutor,
{
    let tool_name = call.name.clone();
    let tool_use_id = call.id.clone();
    tracing::trace!(
        tool = %tool_name,
        tool_use_id = %tool_use_id,
        "dispatching client tool"
    );
    if !enabled_tools.contains_key(&tool_name) {
        // The model asked for a tool outside this agent's allowlist. Return an
        // unknown-tool error so the outer loop can surface it as a tool result.
        tracing::warn!(
            tool = %tool_name,
            tool_use_id = %tool_use_id,
            "disabled client tool requested"
        );
        return Err(ClientToolExecutorError::unknown(call.name));
    }

    let output = tool_executor.execute(call).await;
    if let Err(error) = &output
        && error.is_unknown()
    {
        // The allowlist said the tool should exist, but the executor could not
        // route it. That points to stale config or inconsistent registration.
        tracing::warn!(
            tool = %tool_name,
            tool_use_id = %tool_use_id,
            "unknown client tool requested"
        );
    }
    output
}

/// Append assistant content, client tool calls, and continuation state.
///
/// Empty assistant steps are skipped to avoid adding no-op turns during
/// provider continuation loops.
fn append_assistant_step(transcript: &mut Transcript, output: ModelStepOutput) {
    let blocks: Vec<_> = output.transcript_blocks().collect();
    if !blocks.is_empty() {
        transcript.push(TranscriptTurn {
            role: TurnRole::Assistant,
            blocks,
            metadata: serde_json::Value::Null,
        });
    }
}

/// Split final content into the convenience text field and the full block list.
fn answer_from_content(content: Vec<ContentBlock>) -> AssistantAnswer {
    let text = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::Media { .. }
            | ContentBlock::ClientToolCall(_)
            | ContentBlock::ClientToolResult(_)
            | ContentBlock::Continuation(_) => None,
        })
        .collect::<Vec<_>>()
        .join("");
    AssistantAnswer {
        text,
        blocks: content,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use serde_json::json;
    use tokio::sync::Barrier;
    use tokio::time::{Duration, timeout};

    use crate::ids::{ProviderName, ToolUseId};
    use crate::llm::{
        ModelOutputBlock, ModelSpec, ModelStepEvent, ModelStepItem, ModelStepOutput,
        SamplingOptions,
    };
    use crate::tool::ToolInputSchema;

    use super::*;

    #[derive(Debug)]
    struct TestError;

    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("test error")
        }
    }

    impl std::error::Error for TestError {}

    fn test_model<B>(backend: B) -> Model<B> {
        Model {
            backend,
            spec: ModelSpec {
                id: ModelId::new("test-model"),
                server_tools: ServerToolSet::default(),
                sampling: SamplingOptions::default(),
                provider_options: None,
            },
        }
    }

    fn test_tool_spec(description: impl Into<String>) -> ClientToolSpec {
        ClientToolSpec {
            description: description.into(),
            input_schema: ToolInputSchema::default(),
        }
    }

    fn final_text_step(text: impl Into<String>) -> ModelStep {
        ModelStep::new(
            ModelStepKind::Final,
            ModelStepOutput {
                model_id: ModelId::new("test-model"),
                items: vec![ModelStepItem::OutputBlock(ModelOutputBlock::Text {
                    text: text.into(),
                })],
                usage: Vec::new(),
            },
        )
    }

    fn client_tool_step(calls: Vec<ClientToolCall>) -> ModelStep {
        ModelStep::new(
            ModelStepKind::ClientTools,
            ModelStepOutput {
                model_id: ModelId::new("test-model"),
                items: calls
                    .into_iter()
                    .map(ModelOutputBlock::ClientToolCall)
                    .map(ModelStepItem::OutputBlock)
                    .collect(),
                usage: Vec::new(),
            },
        )
    }

    fn test_step_events<E: Send>(
        step: ModelStep,
    ) -> impl Stream<Item = Result<ModelStepEvent, E>> + Send {
        let kind = step.kind();
        let output = step.into_output();
        let model_id = output.model_id;
        let mut events = Vec::new();
        for (index, item) in output.items.into_iter().enumerate() {
            append_test_item_events(&mut events, index, item);
        }
        events.extend(output.usage.into_iter().map(ModelStepEvent::Usage).map(Ok));
        events.push(Ok(ModelStepEvent::Finished { kind, model_id }));
        futures::stream::iter(events)
    }

    fn append_test_item_events<E: Send>(
        events: &mut Vec<Result<ModelStepEvent, E>>,
        index: usize,
        item: ModelStepItem,
    ) {
        match item {
            ModelStepItem::OutputBlock(ModelOutputBlock::Text { text }) => {
                events.push(Ok(ModelStepEvent::Delta(ModelStepDelta::Text {
                    item_id: format!("test_text:{index}"),
                    delta: text,
                })));
            }
            ModelStepItem::OutputBlock(ModelOutputBlock::ClientToolCall(call)) => {
                events.push(Ok(ModelStepEvent::Delta(ModelStepDelta::ClientToolCall {
                    item_id: format!("test_tool:{index}"),
                    id: call.id,
                    name: Some(call.name),
                    arguments_delta: call.input.to_string(),
                })));
            }
            ModelStepItem::OutputBlock(ModelOutputBlock::Continuation(continuation)) => {
                events.push(Ok(ModelStepEvent::Continuation(continuation)));
            }
            ModelStepItem::Reasoning(reasoning) => {
                let item_id = reasoning
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("test_reasoning:{index}"));
                for summary in reasoning.summary {
                    if summary.text.is_empty() {
                        continue;
                    }
                    events.push(Ok(ModelStepEvent::Delta(
                        ModelStepDelta::ReasoningSummary {
                            item_id: item_id.clone(),
                            provider: reasoning.provider.clone(),
                            kind: summary.kind,
                            delta: summary.text,
                        },
                    )));
                }
            }
            ModelStepItem::ServerToolUse(tool) => {
                events.push(Ok(ModelStepEvent::ServerToolUse(tool)));
            }
            ModelStepItem::Grounding(metadata) => {
                events.push(Ok(ModelStepEvent::Grounding(metadata)));
            }
        }
    }

    #[derive(Debug, Clone)]
    struct EchoExecutor;

    impl ClientToolExecutor for EchoExecutor {
        type Error = TestError;

        fn tools(&self) -> Vec<ClientToolDefinition> {
            vec![ClientToolDefinition::new(
                "echo",
                test_tool_spec("Echo input."),
            )]
        }

        async fn execute(
            &self,
            call: ClientToolCall,
        ) -> Result<ClientToolOutput, ClientToolExecutorError<Self::Error>> {
            if call.name.as_str() != "echo" {
                return Err(ClientToolExecutorError::unknown(call.name));
            }
            Ok(ClientToolOutput {
                result: ClientToolResultContent::Json { value: call.input },
                media: Vec::new(),
                is_error: false,
                trace_response: json!({ "ok": true }),
                usage: Vec::new(),
            })
        }
    }

    #[derive(Debug, Clone)]
    struct NamedToolsExecutor {
        names: Vec<&'static str>,
    }

    impl ClientToolExecutor for NamedToolsExecutor {
        type Error = TestError;

        fn tools(&self) -> Vec<ClientToolDefinition> {
            self.names
                .iter()
                .map(|name| {
                    ClientToolDefinition::new(*name, test_tool_spec(format!("{name} tool.")))
                })
                .collect()
        }

        async fn execute(
            &self,
            call: ClientToolCall,
        ) -> Result<ClientToolOutput, ClientToolExecutorError<Self::Error>> {
            let name = call.name.as_str();
            if !self.names.contains(&name) {
                return Err(ClientToolExecutorError::unknown(call.name));
            }
            Ok(ClientToolOutput {
                result: ClientToolResultContent::Text {
                    text: name.to_string(),
                },
                media: Vec::new(),
                is_error: false,
                trace_response: json!({ "tool": name }),
                usage: Vec::new(),
            })
        }
    }

    #[tokio::test]
    async fn client_tools_dispatches_by_tool_name() {
        let tool_executor = EchoExecutor;
        let enabled = tool_specs(tool_executor.tools());
        assert_eq!(tool_specs(tool_executor.tools()).len(), 1);

        let output = call_client_tool(
            &enabled,
            &tool_executor,
            ClientToolCall {
                id: ToolUseId::new("call-1"),
                name: ToolName::new("echo"),
                input: json!({ "text": "hello" }),
            },
        )
        .await
        .unwrap();

        match output.result {
            ClientToolResultContent::Json { value } => {
                assert_eq!(value, json!({ "text": "hello" }));
            }
            ClientToolResultContent::Text { .. } => panic!("expected json output"),
        }
    }

    #[tokio::test]
    async fn disabled_client_tools_are_not_callable() {
        let tool_executor = EchoExecutor;
        let enabled = BTreeMap::new();

        let output = call_client_tool(
            &enabled,
            &tool_executor,
            ClientToolCall {
                id: ToolUseId::new("call-1"),
                name: ToolName::new("echo"),
                input: json!({ "text": "hello" }),
            },
        )
        .await;

        assert!(matches!(
            output,
            Err(ClientToolExecutorError::Unknown { .. })
        ));
    }

    #[tokio::test]
    async fn agent_accumulates_registered_tool_executor_specs() {
        #[derive(Debug, Clone)]
        struct ToolListingBackend {
            name: ProviderName,
            seen_tools: Arc<Mutex<Vec<String>>>,
        }

        impl LlmBackend for ToolListingBackend {
            type Error = TestError;

            fn backend_name(&self) -> &ProviderName {
                &self.name
            }

            fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_ {
                *self.seen_tools.lock().unwrap() = request
                    .client_tools
                    .keys()
                    .map(|name| name.as_str().to_string())
                    .collect();
                test_step_events(final_text_step("done"))
            }
        }

        let seen_tools = Arc::new(Mutex::new(Vec::new()));
        let backend = ToolListingBackend {
            name: ProviderName::new("test"),
            seen_tools: seen_tools.clone(),
        };
        let spec = AgentSpec::new("system");
        let tools = NamedToolsExecutor {
            names: vec!["alpha", "beta"],
        };
        let agent = Agent::new(test_model(backend), spec, tools);

        let run = collect_agent_run(agent.run(Transcript::from_user_text("list tools")))
            .await
            .unwrap();

        assert!(matches!(run.outcome, AgentOutcome::Completed { .. }));
        assert_eq!(
            *seen_tools.lock().unwrap(),
            vec!["alpha".to_string(), "beta".to_string()]
        );
    }

    #[tokio::test]
    async fn agent_replaces_transcript_instructions_without_adding_system_turn() {
        #[derive(Debug, Clone)]
        struct InstructionBackend {
            name: ProviderName,
            seen: Arc<Mutex<Option<Transcript>>>,
        }

        impl LlmBackend for InstructionBackend {
            type Error = TestError;

            fn backend_name(&self) -> &ProviderName {
                &self.name
            }

            fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_ {
                *self.seen.lock().unwrap() = Some(request.transcript);
                test_step_events(final_text_step("done"))
            }
        }

        let seen = Arc::new(Mutex::new(None));
        let backend = InstructionBackend {
            name: ProviderName::new("test"),
            seen: seen.clone(),
        };
        let mut input = Transcript::from_user_text("hello");
        input.id = Some("conversation-1".to_string());
        input.instructions = Some("old saved system prompt".to_string());

        let agent = Agent::new(
            test_model(backend),
            AgentSpec::new("new system prompt"),
            NoClientTools,
        );
        let run = collect_agent_run(agent.run(input)).await.unwrap();

        assert!(matches!(run.outcome, AgentOutcome::Completed { .. }));
        let seen = seen.lock().unwrap().clone().unwrap();
        assert_eq!(seen.id.as_deref(), Some("conversation-1"));
        assert_eq!(seen.instructions.as_deref(), Some("new system prompt"));
        assert_eq!(seen.turns.len(), 1);
        assert_text_block(&seen.turns[0], TurnRole::User, "hello");
    }

    #[tokio::test]
    async fn agent_intersects_server_tools_case_insensitively() {
        #[derive(Debug, Clone)]
        struct ServerToolBackend {
            name: ProviderName,
            seen_tools: Arc<Mutex<Vec<String>>>,
        }

        impl LlmBackend for ServerToolBackend {
            type Error = TestError;

            fn backend_name(&self) -> &ProviderName {
                &self.name
            }

            fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_ {
                *self.seen_tools.lock().unwrap() = request.server_tools.into_iter().collect();
                test_step_events(final_text_step("done"))
            }
        }

        let seen_tools = Arc::new(Mutex::new(Vec::new()));
        let backend = ServerToolBackend {
            name: ProviderName::new("test"),
            seen_tools: seen_tools.clone(),
        };
        let mut model = test_model(backend);
        model.spec.server_tools = ServerToolSet::from([
            "WEB_SEARCH".to_string(),
            "x_search".to_string(),
            "model_only".to_string(),
        ]);
        let mut spec = AgentSpec::new("system");
        spec.server_tools = Some(ServerToolSet::from([
            " web_search ".to_string(),
            "agent_only".to_string(),
            "X_SEARCH".to_string(),
        ]));

        let agent = Agent::new(model, spec, NoClientTools);
        let run = collect_agent_run(agent.run(Transcript::from_user_text("use server tools")))
            .await
            .unwrap();

        assert!(matches!(run.outcome, AgentOutcome::Completed { .. }));
        assert_eq!(
            *seen_tools.lock().unwrap(),
            vec!["web_search".to_string(), "x_search".to_string()]
        );
    }

    #[tokio::test]
    async fn agent_allows_model_server_tools_by_default() {
        #[derive(Debug, Clone)]
        struct ServerToolBackend {
            name: ProviderName,
            seen_tools: Arc<Mutex<Vec<String>>>,
        }

        impl LlmBackend for ServerToolBackend {
            type Error = TestError;

            fn backend_name(&self) -> &ProviderName {
                &self.name
            }

            fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_ {
                *self.seen_tools.lock().unwrap() = request.server_tools.into_iter().collect();
                test_step_events(final_text_step("done"))
            }
        }

        let seen_tools = Arc::new(Mutex::new(Vec::new()));
        let backend = ServerToolBackend {
            name: ProviderName::new("test"),
            seen_tools: seen_tools.clone(),
        };
        let mut model = test_model(backend);
        model.spec.server_tools =
            ServerToolSet::from(["WEB_SEARCH".to_string(), "x_search".to_string()]);

        let agent = Agent::new(model, AgentSpec::new("system"), NoClientTools);
        let run = collect_agent_run(agent.run(Transcript::from_user_text("use server tools")))
            .await
            .unwrap();

        assert!(matches!(run.outcome, AgentOutcome::Completed { .. }));
        assert_eq!(
            *seen_tools.lock().unwrap(),
            vec!["web_search".to_string(), "x_search".to_string()]
        );
    }

    #[tokio::test]
    async fn agent_runs_client_tool_calls_concurrently() {
        #[derive(Debug, Clone)]
        struct TwoToolCallBackend {
            name: ProviderName,
            calls: Arc<Mutex<usize>>,
            result_order: Arc<Mutex<Vec<String>>>,
        }

        impl LlmBackend for TwoToolCallBackend {
            type Error = TestError;

            fn backend_name(&self) -> &ProviderName {
                &self.name
            }

            fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_ {
                let first_call = {
                    let mut calls = self.calls.lock().unwrap();
                    let first_call = *calls == 0;
                    *calls += 1;
                    first_call
                };
                let step = if first_call {
                    client_tool_step(vec![
                        ClientToolCall {
                            id: ToolUseId::new("call-1"),
                            name: ToolName::new("wait"),
                            input: json!({ "label": "first" }),
                        },
                        ClientToolCall {
                            id: ToolUseId::new("call-2"),
                            name: ToolName::new("wait"),
                            input: json!({ "label": "second" }),
                        },
                    ])
                } else {
                    let observed = request
                        .transcript
                        .turns
                        .last()
                        .map(|message| {
                            message
                                .blocks
                                .iter()
                                .filter_map(|block| match block {
                                    ContentBlock::ClientToolResult(result) => {
                                        Some(result.tool_use_id.as_str().to_string())
                                    }
                                    ContentBlock::Text { .. }
                                    | ContentBlock::Media { .. }
                                    | ContentBlock::ClientToolCall(_)
                                    | ContentBlock::Continuation(_) => None,
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    *self.result_order.lock().unwrap() = observed;

                    final_text_step("done")
                };

                test_step_events(step)
            }
        }

        #[derive(Debug)]
        struct WaitingExecutor {
            barrier: Arc<Barrier>,
        }

        impl ClientToolExecutor for WaitingExecutor {
            type Error = TestError;

            fn tools(&self) -> Vec<ClientToolDefinition> {
                vec![ClientToolDefinition::new(
                    "wait",
                    test_tool_spec("Wait until both calls are running."),
                )]
            }

            async fn execute(
                &self,
                call: ClientToolCall,
            ) -> Result<ClientToolOutput, ClientToolExecutorError<Self::Error>> {
                if call.name.as_str() != "wait" {
                    return Err(ClientToolExecutorError::unknown(call.name));
                }
                self.barrier.wait().await;
                Ok(ClientToolOutput {
                    result: ClientToolResultContent::Text {
                        text: call.id.as_str().to_string(),
                    },
                    media: Vec::new(),
                    is_error: false,
                    trace_response: json!({ "tool_use_id": call.id.as_str() }),
                    usage: Vec::new(),
                })
            }
        }

        let result_order = Arc::new(Mutex::new(Vec::new()));
        let backend = TwoToolCallBackend {
            name: ProviderName::new("test"),
            calls: Arc::new(Mutex::new(0)),
            result_order: result_order.clone(),
        };
        let spec = AgentSpec::new("system").with_limits(AgentLimits { max_iterations: 4 });
        let tools = WaitingExecutor {
            barrier: Arc::new(Barrier::new(2)),
        };
        let agent = Agent::new(test_model(backend), spec, tools);

        let run = timeout(
            Duration::from_secs(1),
            collect_agent_run(agent.run(Transcript::from_user_text("run both tools"))),
        )
        .await
        .expect("tool calls should run concurrently")
        .unwrap();

        assert!(matches!(run.outcome, AgentOutcome::Completed { .. }));
        let observed = result_order.lock().unwrap().clone();
        assert_eq!(observed, vec!["call-1".to_string(), "call-2".to_string()]);
    }

    fn assert_text_block(message: &TranscriptTurn, role: TurnRole, text: &str) {
        assert_eq!(message.role, role);
        match message.blocks.as_slice() {
            [ContentBlock::Text { text: actual }] => assert_eq!(actual, text),
            blocks => panic!("expected one text block, got {blocks:?}"),
        }
    }
}
