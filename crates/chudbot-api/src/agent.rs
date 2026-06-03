//! Agent runtime and sub-agent adapters.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::future::Future;
use std::pin::Pin;

use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::Instrument;

use crate::ids::{ModelId, ToolName};
use crate::llm::{AssistantStep, LlmBackend, Model, ModelStep, ServerToolSet};
use crate::tool::{
    ClientTool, ClientToolCall, ClientToolOutput, ClientToolResult, ClientToolResultContent,
    ClientToolSpec, ClientToolTrace, ToolInputSchema, ToolTrace,
};
use crate::transcript::{ContentBlock, ProviderContinuation, Transcript, TranscriptTurn, TurnRole};
use crate::usage::UsageRecord;

/// Static agent configuration.
///
/// This is TOML-shaped data. It intentionally does not carry runtime tool
/// implementations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSpec {
    /// Agent instructions applied to each run's transcript.
    pub system_prompt: String,
    /// Provider-side/server-side tools this agent allows. `None` means all
    /// server tools allowed by the model config are available.
    #[serde(default)]
    pub server_tools: Option<ServerToolSet>,
    /// Optional static list of client tool names enabled for this agent. `None`
    /// means all runtime tools supplied to the agent are exposed.
    #[serde(default)]
    pub client_tools: Option<Vec<ToolName>>,
    /// Agent loop limits.
    pub limits: AgentLimits,
}

impl AgentSpec {
    /// Create an agent spec.
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            server_tools: None,
            client_tools: None,
            limits: AgentLimits::default(),
        }
    }

    /// Restrict the runtime tool surface to these tool names.
    pub fn with_client_tools(mut self, client_tools: Vec<ToolName>) -> Self {
        self.client_tools = Some(client_tools);
        self
    }

    /// Set loop limits.
    pub fn with_limits(mut self, limits: AgentLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Add one runtime client tool and continue building an agent.
    pub fn with_tool<T>(self, name: impl Into<ToolName>, tool: T) -> AgentBuilder
    where
        T: ClientTool + 'static,
    {
        AgentBuilder::new(self).with_tool(name, tool)
    }

    /// Combine this spec with a model into a runnable agent.
    pub fn into_agent<B>(self, model: Model<B>) -> Agent<B> {
        AgentBuilder::new(self).into_agent(model)
    }
}

/// Builder for attaching runtime client tools before creating an [`Agent`].
pub struct AgentBuilder {
    spec: AgentSpec,
    tools: BTreeMap<ToolName, Box<dyn DynClientTool>>,
}

impl fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("spec", &self.spec)
            .field("client_tools", &tool_specs(&self.tools))
            .finish()
    }
}

impl AgentBuilder {
    fn new(spec: AgentSpec) -> Self {
        Self {
            spec,
            tools: BTreeMap::new(),
        }
    }

    /// Add one runtime client tool.
    pub fn with_tool<T>(mut self, name: impl Into<ToolName>, tool: T) -> Self
    where
        T: ClientTool + 'static,
    {
        self.tools.insert(name.into(), Box::new(tool));
        self
    }

    /// Combine the configured spec/tools with a model into a runnable agent.
    pub fn into_agent<B>(self, model: Model<B>) -> Agent<B> {
        Agent {
            model,
            spec: self.spec,
            tools: self.tools,
        }
    }
}

/// Concrete agent built from a model, an agent spec, and runtime client tools.
pub struct Agent<B> {
    /// Callable model.
    model: Model<B>,
    /// Agent instructions.
    spec: AgentSpec,
    /// Runtime client tools available to this agent.
    tools: BTreeMap<ToolName, Box<dyn DynClientTool>>,
}

impl<B> fmt::Debug for Agent<B>
where
    B: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
            .field("model", &self.model)
            .field("spec", &self.spec)
            .field("client_tools", &tool_specs(&self.tools))
            .finish()
    }
}

impl<B> Agent<B> {
    /// Convert this agent into a client-side tool.
    pub fn into_subagent(self, tool_description: impl Into<String>) -> Subagent<B> {
        Subagent {
            agent: self,
            tool_description: tool_description.into(),
        }
    }
}

type ClientToolFuture<'a> =
    Pin<Box<dyn Future<Output = Result<ClientToolOutput, ToolDispatchError>> + Send + 'a>>;

/// Error produced while dispatching a model-requested tool call.
#[derive(Debug, Error)]
enum ToolDispatchError {
    #[error("unknown tool `{0}`")]
    Unknown(String),
    #[error("execution failed: {0}")]
    Execution(String),
}

trait DynClientTool: Send + Sync {
    fn spec(&self) -> ClientToolSpec;
    fn call_dyn<'a>(&'a self, call: ClientToolCall) -> ClientToolFuture<'a>;
}

impl<T> DynClientTool for T
where
    T: ClientTool,
{
    fn spec(&self) -> ClientToolSpec {
        ClientTool::spec(self)
    }

    fn call_dyn<'a>(&'a self, call: ClientToolCall) -> ClientToolFuture<'a> {
        Box::pin(async move {
            ClientTool::call(self, call)
                .await
                .map_err(|e| ToolDispatchError::Execution(e.to_string()))
        })
    }
}

/// Error produced while running an [`Agent`].
#[derive(Debug, Error)]
pub enum AgentRunError<BE>
where
    BE: std::error::Error + Send + Sync + 'static,
{
    /// Provider step failed.
    #[error("model error: {0}")]
    Model(#[source] BE),
}

/// Agent loop limits.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AgentLimits {
    /// Maximum model/tool iterations.
    pub max_iterations: u32,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self { max_iterations: 8 }
    }
}

/// Agent run output.
#[derive(Debug, Clone)]
pub struct AgentRun {
    /// Final outcome.
    pub outcome: AgentOutcome,
    /// Transcript after the run.
    pub transcript: Transcript,
    /// Tool trace records.
    pub trace: Vec<ToolTrace>,
    /// Last model id reported by a provider.
    pub last_model_id: Option<ModelId>,
    /// Final provider continuation to persist for cross-turn replay.
    pub final_continuation: Option<ProviderContinuation>,
    /// Usage/cost accumulated directly by the agent run. Tool trace entries
    /// may also carry their own usage.
    pub usage: Vec<UsageRecord>,
}

impl AgentRun {
    /// Collect usage from the run and every traced tool that carries usage.
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

/// Agent run outcome.
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
    /// Hit the iteration cap.
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

/// Assistant answer.
#[derive(Debug, Clone)]
pub struct AssistantAnswer {
    /// Plain text answer. Derived from text blocks for convenience.
    pub text: String,
    /// Full answer blocks.
    pub blocks: Vec<ContentBlock>,
}

/// Agent error.
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

impl<B> Agent<B>
where
    B: LlmBackend,
{
    /// Run this agent against one transcript.
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
    pub async fn run(&self, transcript: Transcript) -> Result<AgentRun, AgentRunError<B::Error>> {
        let Transcript { id, turns, .. } = transcript;
        let initial_turns = turns.len();
        let mut transcript = Transcript {
            id,
            instructions: Some(self.spec.system_prompt.clone()),
            turns,
        };

        let client_tools = enabled_tool_specs(&self.spec, &self.tools);
        let server_tools = enabled_server_tools(
            &self.model.spec.server_tools,
            self.spec.server_tools.as_ref(),
        );
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

        let mut trace = Vec::new();
        let mut usage = Vec::new();
        let mut last_model_id = None;
        let mut final_continuation = None;

        for iteration in 0..self.spec.limits.max_iterations {
            tracing::debug!(
                iteration = iteration + 1,
                turns = transcript.turns.len(),
                trace_records = trace.len(),
                usage_records = usage.len(),
                "requesting model step"
            );
            let step = match self
                .model
                .backend
                .step(crate::llm::ModelStepRequest {
                    model: self.model.spec.id.clone(),
                    transcript: transcript.clone(),
                    client_tools: client_tools.clone(),
                    server_tools: server_tools.clone(),
                    sampling: self.model.spec.sampling,
                    provider_options: self.model.spec.provider_options.clone(),
                })
                .await
            {
                Ok(step) => step,
                Err(error) => {
                    tracing::warn!(
                        iteration = iteration + 1,
                        error = %error,
                        "model step failed"
                    );
                    return Err(AgentRunError::Model(error));
                }
            };

            match step {
                ModelStep::Final { step } => {
                    tracing::debug!(
                        iteration = iteration + 1,
                        model = %step.model_id,
                        content_blocks = step.content.len(),
                        server_tools = step.server_tool_uses.len(),
                        grounding = step.grounding.len(),
                        usage_records = step.usage.len(),
                        has_continuation = step.continuation.is_some(),
                        "model returned final answer"
                    );
                    append_step_trace(&mut trace, &step);
                    usage.extend(step.usage.iter().cloned());
                    last_model_id = Some(step.model_id.clone());
                    final_continuation = step.continuation.clone();
                    let answer = answer_from_content(step.content.clone());
                    transcript.push(TranscriptTurn {
                        role: TurnRole::Assistant,
                        blocks: step.content,
                        metadata: serde_json::Value::Null,
                    });
                    tracing::info!(
                        iterations = iteration + 1,
                        answer_chars = answer.text.chars().count(),
                        trace_records = trace.len(),
                        usage_records = usage.len(),
                        "agent run completed"
                    );
                    return Ok(AgentRun {
                        outcome: AgentOutcome::Completed { answer },
                        transcript,
                        trace,
                        last_model_id,
                        final_continuation,
                        usage,
                    });
                }
                ModelStep::Continue { step } => {
                    tracing::debug!(
                        iteration = iteration + 1,
                        model = %step.model_id,
                        content_blocks = step.content.len(),
                        server_tools = step.server_tool_uses.len(),
                        grounding = step.grounding.len(),
                        usage_records = step.usage.len(),
                        has_continuation = step.continuation.is_some(),
                        "model requested continuation"
                    );
                    append_step_trace(&mut trace, &step);
                    usage.extend(step.usage.iter().cloned());
                    last_model_id = Some(step.model_id.clone());
                    final_continuation = step.continuation.clone();
                    append_assistant_step(&mut transcript, step);
                }
                ModelStep::UseClientTools { step } => {
                    tracing::debug!(
                        iteration = iteration + 1,
                        model = %step.model_id,
                        content_blocks = step.content.len(),
                        client_tool_calls = step.client_tool_calls.len(),
                        server_tools = step.server_tool_uses.len(),
                        grounding = step.grounding.len(),
                        usage_records = step.usage.len(),
                        has_continuation = step.continuation.is_some(),
                        "model requested client tools"
                    );
                    append_step_trace(&mut trace, &step);
                    usage.extend(step.usage.iter().cloned());
                    last_model_id = Some(step.model_id.clone());
                    final_continuation = step.continuation.clone();

                    let calls = step.client_tool_calls.clone();
                    append_assistant_step(&mut transcript, step);

                    let tool_results =
                        execute_client_tool_calls(&client_tools, &self.tools, calls).await;
                    let mut result_blocks = Vec::with_capacity(tool_results.len());
                    for (result, tool_trace) in tool_results {
                        result_blocks.push(ContentBlock::ClientToolResult(result));
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

        tracing::warn!(
            max_iterations = self.spec.limits.max_iterations,
            trace_records = trace.len(),
            usage_records = usage.len(),
            "agent run hit iteration limit"
        );
        Ok(AgentRun {
            outcome: AgentOutcome::IterationLimit {
                max_iterations: self.spec.limits.max_iterations,
            },
            transcript,
            trace,
            last_model_id,
            final_continuation,
            usage,
        })
    }
}

fn append_step_trace(trace: &mut Vec<ToolTrace>, step: &AssistantStep) {
    trace.extend(
        step.server_tool_uses
            .iter()
            .cloned()
            .map(|tool| ToolTrace::Server { tool }),
    );
    trace.extend(
        step.grounding
            .iter()
            .cloned()
            .map(|metadata| ToolTrace::Grounding { metadata }),
    );
}

fn enabled_tool_specs(
    spec: &AgentSpec,
    tools: &BTreeMap<ToolName, Box<dyn DynClientTool>>,
) -> BTreeMap<ToolName, ClientToolSpec> {
    let tool_specs = tool_specs(tools);
    let Some(enabled) = &spec.client_tools else {
        return tool_specs;
    };

    tool_specs
        .into_iter()
        .filter(|(name, _)| enabled.iter().any(|enabled_name| enabled_name == name))
        .collect()
}

fn enabled_server_tools(
    model_tools: &ServerToolSet,
    agent_tools: Option<&ServerToolSet>,
) -> ServerToolSet {
    let model_tools = normalized_server_tools(model_tools);
    let Some(agent_tools) = agent_tools else {
        return model_tools;
    };

    let agent_tools = normalized_server_tools(agent_tools);
    model_tools.intersection(&agent_tools).cloned().collect()
}

fn normalized_server_tools(tools: &ServerToolSet) -> BTreeSet<String> {
    tools
        .iter()
        .filter_map(|tool| normalize_server_tool(tool))
        .collect()
}

fn normalize_server_tool(tool: &str) -> Option<String> {
    let tool = tool.trim();
    (!tool.is_empty()).then(|| tool.to_ascii_lowercase())
}

async fn execute_client_tool_calls(
    enabled_tools: &BTreeMap<ToolName, ClientToolSpec>,
    tools: &BTreeMap<ToolName, Box<dyn DynClientTool>>,
    calls: Vec<ClientToolCall>,
) -> Vec<(ClientToolResult, ClientToolTrace)> {
    tracing::debug!(calls = calls.len(), "executing client tool calls");
    let mut pending = FuturesUnordered::new();
    for (index, call) in calls.into_iter().enumerate() {
        let tool_name = call.name.to_string();
        let tool_use_id = call.id.to_string();
        pending.push(
            async move {
                let output = call_client_tool(enabled_tools, tools, call.clone()).await;
                (index, call, output)
            }
            .instrument(tracing::debug_span!(
                "agent.client_tool",
                tool = %tool_name,
                tool_use_id = %tool_use_id,
            )),
        );
    }

    let mut completed = Vec::new();
    while let Some((index, call, output)) = pending.next().await {
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
        completed.push((index, result, tool_trace));
    }

    completed.sort_by_key(|(index, _, _)| *index);
    let completed = completed
        .into_iter()
        .map(|(_, result, tool_trace)| (result, tool_trace))
        .collect::<Vec<_>>();
    tracing::debug!(completed = completed.len(), "client tool calls finished");
    completed
}

fn tool_specs(
    tools: &BTreeMap<ToolName, Box<dyn DynClientTool>>,
) -> BTreeMap<ToolName, ClientToolSpec> {
    tools
        .iter()
        .map(|(name, tool)| (name.clone(), tool.spec()))
        .collect()
}

fn call_client_tool<'a>(
    enabled_tools: &'a BTreeMap<ToolName, ClientToolSpec>,
    tools: &'a BTreeMap<ToolName, Box<dyn DynClientTool>>,
    call: ClientToolCall,
) -> ClientToolFuture<'a> {
    Box::pin(async move {
        tracing::trace!(
            tool = %call.name,
            tool_use_id = %call.id,
            "dispatching client tool"
        );
        if !enabled_tools.contains_key(&call.name) {
            tracing::warn!(
                tool = %call.name,
                tool_use_id = %call.id,
                "disabled client tool requested"
            );
            return Err(ToolDispatchError::Unknown(call.name.to_string()));
        }
        let Some(tool) = tools.get(&call.name) else {
            tracing::warn!(
                tool = %call.name,
                tool_use_id = %call.id,
                "unknown client tool requested"
            );
            return Err(ToolDispatchError::Unknown(call.name.to_string()));
        };

        tool.call_dyn(call).await
    })
}

fn append_assistant_step(transcript: &mut Transcript, step: AssistantStep) {
    let mut blocks = step.content;
    blocks.extend(
        step.client_tool_calls
            .into_iter()
            .map(ContentBlock::ClientToolCall),
    );
    if let Some(continuation) = step.continuation {
        blocks.push(ContentBlock::Continuation(continuation));
    }
    if !blocks.is_empty() {
        transcript.push(TranscriptTurn {
            role: TurnRole::Assistant,
            blocks,
            metadata: serde_json::Value::Null,
        });
    }
}

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

/// An agent exposed as one client-side tool.
///
/// `Subagent` owns an agent and turns a parent model's tool call into
/// a nested [`Transcript`].
#[derive(Debug)]
pub struct Subagent<B> {
    /// Nested agent runtime.
    agent: Agent<B>,
    /// Tool description exposed to the parent model.
    tool_description: String,
}

impl<B> Subagent<B>
where
    B: LlmBackend,
{
    fn transcript_for_input(&self, input: serde_json::Value) -> Transcript {
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn::text(TurnRole::User, tool_input_text(input)));
        transcript
    }

    fn output_from_run(&self, run: &AgentRun) -> ClientToolOutput {
        let (result, is_error) = match &run.outcome {
            AgentOutcome::Completed { answer } => (
                ClientToolResultContent::Text {
                    text: answer.text.clone(),
                },
                false,
            ),
            AgentOutcome::Failed { error, .. } => (
                ClientToolResultContent::Text {
                    text: format!("sub-agent failed: {error}"),
                },
                true,
            ),
            AgentOutcome::IterationLimit { max_iterations } => (
                ClientToolResultContent::Text {
                    text: format!("sub-agent hit iteration limit ({max_iterations})"),
                },
                true,
            ),
            AgentOutcome::Cancelled { reason } => (
                ClientToolResultContent::Text {
                    text: format!("sub-agent was cancelled: {reason}"),
                },
                true,
            ),
        };
        ClientToolOutput {
            result,
            is_error,
            trace_response: subagent_trace_response(run),
            usage: run.all_usage(),
        }
    }
}

impl<B> ClientTool for Subagent<B>
where
    B: LlmBackend,
{
    type Error = AgentRunError<B::Error>;

    fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.tool_description.clone(),
            input_schema: prompt_input_schema(),
        }
    }

    #[tracing::instrument(
        name = "subagent.call",
        skip_all,
        fields(tool = %call.name, tool_use_id = %call.id)
    )]
    async fn call(&self, call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
        tracing::debug!("starting subagent tool call");
        let run = self
            .agent
            .run(self.transcript_for_input(call.input))
            .await?;
        tracing::debug!(
            outcome = agent_outcome_kind(&run.outcome),
            usage_records = run.all_usage().len(),
            trace_records = run.trace.len(),
            "subagent tool call finished"
        );
        Ok(self.output_from_run(&run))
    }
}

fn prompt_input_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The task or question for the sub-agent."
            }
        },
        "additionalProperties": false
    }))
}

fn tool_input_text(input: serde_json::Value) -> String {
    input
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| input.to_string())
}

fn subagent_trace_response(run: &AgentRun) -> serde_json::Value {
    match &run.outcome {
        AgentOutcome::Completed { answer } => serde_json::json!({
            "outcome": "completed",
            "text": answer.text,
            "usage": run.all_usage(),
        }),
        AgentOutcome::Failed { error, .. } => serde_json::json!({
            "outcome": "failed",
            "error": error.to_string(),
            "usage": run.all_usage(),
        }),
        AgentOutcome::IterationLimit { max_iterations } => serde_json::json!({
            "outcome": "iteration_limit",
            "max_iterations": max_iterations,
            "usage": run.all_usage(),
        }),
        AgentOutcome::Cancelled { reason } => serde_json::json!({
            "outcome": "cancelled",
            "reason": reason,
            "usage": run.all_usage(),
        }),
    }
}

fn agent_outcome_kind(outcome: &AgentOutcome) -> &'static str {
    match outcome {
        AgentOutcome::Completed { .. } => "completed",
        AgentOutcome::Failed { .. } => "failed",
        AgentOutcome::IterationLimit { .. } => "iteration_limit",
        AgentOutcome::Cancelled { .. } => "cancelled",
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
    use crate::llm::{ModelSpec, SamplingOptions};
    use crate::usage::UsageSubject;

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

    #[derive(Debug, Clone)]
    struct RecordingBackend {
        name: ProviderName,
        requests: Arc<Mutex<Vec<crate::llm::ModelStepRequest>>>,
        step: AssistantStep,
    }

    impl LlmBackend for RecordingBackend {
        type Error = TestError;

        fn backend_name(&self) -> &ProviderName {
            &self.name
        }

        async fn step(
            &self,
            request: crate::llm::ModelStepRequest,
        ) -> Result<ModelStep, Self::Error> {
            self.requests.lock().unwrap().push(request);
            Ok(ModelStep::Final {
                step: self.step.clone(),
            })
        }
    }

    #[tokio::test]
    async fn subagent_exposes_spec_and_executes_nested_agent() {
        let usage = UsageRecord {
            provider: ProviderName::new("openai"),
            model: Some(ModelId::new("gpt-5")),
            subject: UsageSubject::ModelStep,
            input_tokens: Some(100),
            cached_input_tokens: Some(25),
            output_tokens: Some(40),
            reasoning_tokens: Some(10),
            total_tokens: Some(140),
            cost: None,
            raw: None,
        };
        let requests = Arc::new(Mutex::new(Vec::new()));
        let backend = RecordingBackend {
            name: ProviderName::new("openai"),
            requests: requests.clone(),
            step: AssistantStep {
                content: vec![ContentBlock::Text {
                    text: "use VTI".to_string(),
                }],
                client_tool_calls: Vec::new(),
                server_tool_uses: Vec::new(),
                grounding: Vec::new(),
                model_id: ModelId::new("gpt-5"),
                continuation: None,
                usage: vec![usage],
            },
        };
        let agent = AgentSpec::new("expert").into_agent(test_model(backend));

        let subagent = agent.into_subagent("Ask the OpenAI expert for a second opinion.");
        let spec = ClientTool::spec(&subagent);
        assert_eq!(
            spec.description,
            "Ask the OpenAI expert for a second opinion."
        );
        assert!(
            spec.input_schema
                .as_json_schema()
                .get("properties")
                .is_some()
        );

        let output = subagent
            .call(ClientToolCall {
                id: ToolUseId::new("call-1"),
                name: ToolName::new("ask_openai_expert"),
                input: json!({ "prompt": "Which total-market ETF is best?" }),
            })
            .await
            .unwrap();

        assert!(!output.is_error);
        assert_eq!(output.usage.len(), 1);
        match output.result {
            ClientToolResultContent::Text { text } => assert_eq!(text, "use VTI"),
            ClientToolResultContent::Json { .. } => panic!("expected text output"),
        }

        let inputs = requests.lock().unwrap();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].transcript.instructions.as_deref(), Some("expert"));
        assert_eq!(inputs[0].transcript.turns.len(), 1);
        assert_text_block(
            &inputs[0].transcript.turns[0],
            TurnRole::User,
            "Which total-market ETF is best?",
        );
    }

    #[tokio::test]
    async fn client_tools_dispatches_by_tool_name() {
        #[derive(Debug)]
        struct EchoTool;

        impl ClientTool for EchoTool {
            type Error = TestError;

            fn spec(&self) -> ClientToolSpec {
                ClientToolSpec {
                    description: "Echo input.".to_string(),
                    input_schema: ToolInputSchema::new(json!({ "type": "object" })),
                }
            }

            async fn call(&self, call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
                Ok(ClientToolOutput {
                    result: ClientToolResultContent::Json { value: call.input },
                    is_error: false,
                    trace_response: json!({ "ok": true }),
                    usage: Vec::new(),
                })
            }
        }

        let tools: BTreeMap<ToolName, Box<dyn DynClientTool>> = BTreeMap::from([(
            ToolName::new("echo"),
            Box::new(EchoTool) as Box<dyn DynClientTool>,
        )]);
        let enabled = tool_specs(&tools);
        assert_eq!(tool_specs(&tools).len(), 1);

        let output = call_client_tool(
            &enabled,
            &tools,
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
        #[derive(Debug)]
        struct EchoTool;

        impl ClientTool for EchoTool {
            type Error = TestError;

            fn spec(&self) -> ClientToolSpec {
                ClientToolSpec {
                    description: "Echo input.".to_string(),
                    input_schema: ToolInputSchema::new(json!({ "type": "object" })),
                }
            }

            async fn call(&self, call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
                Ok(ClientToolOutput {
                    result: ClientToolResultContent::Json { value: call.input },
                    is_error: false,
                    trace_response: json!({ "ok": true }),
                    usage: Vec::new(),
                })
            }
        }

        let tools: BTreeMap<ToolName, Box<dyn DynClientTool>> = BTreeMap::from([(
            ToolName::new("echo"),
            Box::new(EchoTool) as Box<dyn DynClientTool>,
        )]);
        let enabled = BTreeMap::new();

        let output = call_client_tool(
            &enabled,
            &tools,
            ClientToolCall {
                id: ToolUseId::new("call-1"),
                name: ToolName::new("echo"),
                input: json!({ "text": "hello" }),
            },
        )
        .await;

        assert!(matches!(output, Err(ToolDispatchError::Unknown(_))));
    }

    #[tokio::test]
    async fn agent_with_tool_accumulates_registered_tools() {
        #[derive(Debug)]
        struct NoopTool;

        impl ClientTool for NoopTool {
            type Error = TestError;

            fn spec(&self) -> ClientToolSpec {
                ClientToolSpec {
                    description: "No-op tool.".to_string(),
                    input_schema: ToolInputSchema::new(json!({ "type": "object" })),
                }
            }

            async fn call(&self, _call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
                Ok(ClientToolOutput {
                    result: ClientToolResultContent::Text {
                        text: String::new(),
                    },
                    is_error: false,
                    trace_response: json!({ "ok": true }),
                    usage: Vec::new(),
                })
            }
        }

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

            async fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> Result<ModelStep, Self::Error> {
                *self.seen_tools.lock().unwrap() = request
                    .client_tools
                    .keys()
                    .map(|name| name.as_str().to_string())
                    .collect();
                Ok(ModelStep::Final {
                    step: AssistantStep {
                        content: vec![ContentBlock::Text {
                            text: "done".to_string(),
                        }],
                        client_tool_calls: Vec::new(),
                        server_tool_uses: Vec::new(),
                        grounding: Vec::new(),
                        model_id: ModelId::new("test-model"),
                        continuation: None,
                        usage: Vec::new(),
                    },
                })
            }
        }

        let seen_tools = Arc::new(Mutex::new(Vec::new()));
        let backend = ToolListingBackend {
            name: ProviderName::new("test"),
            seen_tools: seen_tools.clone(),
        };
        let agent = AgentSpec::new("system")
            .with_tool("alpha", NoopTool)
            .with_tool("beta", NoopTool)
            .into_agent(test_model(backend));

        let run = agent
            .run(Transcript::from_user_text("list tools"))
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

            async fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> Result<ModelStep, Self::Error> {
                *self.seen.lock().unwrap() = Some(request.transcript);
                Ok(ModelStep::Final {
                    step: AssistantStep {
                        content: vec![ContentBlock::Text {
                            text: "done".to_string(),
                        }],
                        client_tool_calls: Vec::new(),
                        server_tool_uses: Vec::new(),
                        grounding: Vec::new(),
                        model_id: ModelId::new("test-model"),
                        continuation: None,
                        usage: Vec::new(),
                    },
                })
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

        let run = AgentSpec::new("new system prompt")
            .into_agent(test_model(backend))
            .run(input)
            .await
            .unwrap();

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

            async fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> Result<ModelStep, Self::Error> {
                *self.seen_tools.lock().unwrap() = request.server_tools.into_iter().collect();
                Ok(ModelStep::Final {
                    step: AssistantStep {
                        content: vec![ContentBlock::Text {
                            text: "done".to_string(),
                        }],
                        client_tool_calls: Vec::new(),
                        server_tool_uses: Vec::new(),
                        grounding: Vec::new(),
                        model_id: ModelId::new("test-model"),
                        continuation: None,
                        usage: Vec::new(),
                    },
                })
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

        let run = spec
            .into_agent(model)
            .run(Transcript::from_user_text("use server tools"))
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

            async fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> Result<ModelStep, Self::Error> {
                *self.seen_tools.lock().unwrap() = request.server_tools.into_iter().collect();
                Ok(ModelStep::Final {
                    step: AssistantStep {
                        content: vec![ContentBlock::Text {
                            text: "done".to_string(),
                        }],
                        client_tool_calls: Vec::new(),
                        server_tool_uses: Vec::new(),
                        grounding: Vec::new(),
                        model_id: ModelId::new("test-model"),
                        continuation: None,
                        usage: Vec::new(),
                    },
                })
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

        let run = AgentSpec::new("system")
            .into_agent(model)
            .run(Transcript::from_user_text("use server tools"))
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

            async fn step(
                &self,
                request: crate::llm::ModelStepRequest,
            ) -> Result<ModelStep, Self::Error> {
                let mut calls = self.calls.lock().unwrap();
                if *calls == 0 {
                    *calls += 1;
                    return Ok(ModelStep::UseClientTools {
                        step: AssistantStep {
                            content: Vec::new(),
                            client_tool_calls: vec![
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
                            ],
                            server_tool_uses: Vec::new(),
                            grounding: Vec::new(),
                            model_id: ModelId::new("test-model"),
                            continuation: None,
                            usage: Vec::new(),
                        },
                    });
                }
                *calls += 1;
                drop(calls);

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

                Ok(ModelStep::Final {
                    step: AssistantStep {
                        content: vec![ContentBlock::Text {
                            text: "done".to_string(),
                        }],
                        client_tool_calls: Vec::new(),
                        server_tool_uses: Vec::new(),
                        grounding: Vec::new(),
                        model_id: ModelId::new("test-model"),
                        continuation: None,
                        usage: Vec::new(),
                    },
                })
            }
        }

        #[derive(Debug)]
        struct WaitingTool {
            barrier: Arc<Barrier>,
        }

        impl ClientTool for WaitingTool {
            type Error = TestError;

            fn spec(&self) -> ClientToolSpec {
                ClientToolSpec {
                    description: "Wait until both calls are running.".to_string(),
                    input_schema: ToolInputSchema::new(json!({ "type": "object" })),
                }
            }

            async fn call(&self, call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
                self.barrier.wait().await;
                Ok(ClientToolOutput {
                    result: ClientToolResultContent::Text {
                        text: call.id.as_str().to_string(),
                    },
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
        let agent = AgentSpec::new("system")
            .with_limits(AgentLimits { max_iterations: 4 })
            .with_tool(
                "wait",
                WaitingTool {
                    barrier: Arc::new(Barrier::new(2)),
                },
            )
            .into_agent(test_model(backend));

        let run = timeout(
            Duration::from_secs(1),
            agent.run(Transcript::from_user_text("run both tools")),
        )
        .await
        .expect("tool calls should run concurrently")
        .unwrap();

        assert!(matches!(run.outcome, AgentOutcome::Completed { .. }));
        let mut observed = result_order.lock().unwrap().clone();
        observed.sort();
        assert_eq!(observed, vec!["call-1".to_string(), "call-2".to_string()]);
    }

    #[tokio::test]
    async fn subagent_ignores_registration_name() {
        let backend = RecordingBackend {
            name: ProviderName::new("openai"),
            requests: Arc::new(Mutex::new(Vec::new())),
            step: AssistantStep {
                content: vec![ContentBlock::Text {
                    text: "ok".to_string(),
                }],
                client_tool_calls: Vec::new(),
                server_tool_uses: Vec::new(),
                grounding: Vec::new(),
                model_id: ModelId::new("gpt-5"),
                continuation: None,
                usage: Vec::new(),
            },
        };
        let agent = AgentSpec::new("expert").into_agent(test_model(backend));
        let subagent = agent.into_subagent("Ask the OpenAI expert.");

        let output = subagent
            .call(ClientToolCall {
                id: ToolUseId::new("call-1"),
                name: ToolName::new("registered_elsewhere"),
                input: json!({ "prompt": "hello" }),
            })
            .await
            .unwrap();

        assert!(!output.is_error);
    }

    fn assert_text_block(message: &TranscriptTurn, role: TurnRole, text: &str) {
        assert_eq!(message.role, role);
        match message.blocks.as_slice() {
            [ContentBlock::Text { text: actual }] => assert_eq!(actual, text),
            blocks => panic!("expected one text block, got {blocks:?}"),
        }
    }
}
