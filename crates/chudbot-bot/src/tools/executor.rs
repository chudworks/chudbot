//! Runtime implementation of Chudbot's provider-neutral client tool executor.
//!
//! The agent loop asks this executor for model-visible tool specifications and
//! later calls it for each model-requested client tool call. This module owns
//! single-call dispatch only: the surrounding agent runtime handles launching
//! multiple calls concurrently, restoring the model-emitted call order, turning
//! results into transcript blocks, and recording client tool traces for storage.

use super::*;
use tracing::Instrument;

/// Runtime tool execution error after tool-specific errors are stringified.
///
/// Individual tools use different concrete error types. The runtime executor
/// erases those failures into one displayable error so the agent loop can apply
/// uniform error-result and trace serialization behavior.
#[derive(Debug)]
pub(crate) struct RuntimeToolError(String);

impl std::fmt::Display for RuntimeToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RuntimeToolError {}

/// One configured agent exposed as a named client tool by its parent executor.
pub(crate) struct Subagent<B, T = NoClientTools> {
    /// Model-facing tool description.
    description: String,
    /// Nested agent runtime.
    agent: Agent<B, T>,
}

impl<B, T> std::fmt::Debug for Subagent<B, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subagent")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl<B, T> Subagent<B, T> {
    /// Build a configured subagent tool from a nested agent.
    pub(crate) fn new(description: impl Into<String>, agent: Agent<B, T>) -> Self {
        Self {
            description: description.into(),
            agent,
        }
    }
}

impl<B, T> Subagent<B, T>
where
    B: chudbot_api::LlmBackend,
    T: ClientToolExecutor,
{
    /// Build the tool specification shown to a parent model.
    pub(crate) fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.description.clone(),
            input_schema: subagent_input_schema(),
        }
    }

    /// Execute a parent-model tool call by running the nested agent.
    pub(crate) fn call(
        &self,
        call: ClientToolCall,
    ) -> impl Future<Output = Result<ClientToolOutput, AgentRunError<B::Error>>> + Send + '_ {
        async move {
            let span = tracing::debug_span!(
                "subagent.call",
                tool = %call.name,
                tool_use_id = %call.id
            );
            async move {
                tracing::debug!("starting subagent tool call");
                let run = self
                    .agent
                    .run(subagent_transcript_for_input(call.input))
                    .await?;
                tracing::debug!(
                    outcome = subagent_outcome_kind(&run.outcome),
                    usage_records = run.all_usage().len(),
                    trace_records = run.trace.len(),
                    "subagent tool call finished"
                );
                Ok(subagent_output_from_run(&run))
            }
            .instrument(span)
            .await
        }
    }
}

/// Shared runtime services available to all tool calls for one turn.
///
/// These are cloneable handles into the platform, storage, media, and provider
/// registries. Tool wrappers are built from them lazily for each advertised or
/// executed tool rather than being stored as separate executor fields.
pub(crate) struct RuntimeToolDeps<R: BotRuntimeTypes> {
    /// Platform registry used by tools that fetch messages or modify replies.
    pub(crate) platforms: R::Platforms,
    /// Storage implementation used for conversation, usage, memory, and jobs.
    pub(crate) storage: R::Storage,
    /// Media store used by generation, transcription, and stored-asset tools.
    pub(crate) media_store: R::Media,
    /// Image provider registry used by the configured image-generation binding.
    pub(crate) images: R::Images,
    /// Video provider registry used by the configured video-generation binding.
    pub(crate) videos: R::Videos,
    /// Audio provider registry used by the configured transcription binding.
    pub(crate) audio: R::Audio,
    /// Per-scope locks shared by persistent video generation tools.
    pub(crate) video_rate_limit_locks: VideoRateLimitLocks,
}

/// Per-turn context captured by tools that interact with the current conversation.
pub(crate) struct RuntimeToolContext {
    /// Channel used when a tool needs the current conversation's default target.
    pub(crate) default_channel: ChannelRef,
    /// Bot message the turn is replying through.
    pub(crate) reply_to: MessageRef,
    /// Conversation whose trace and user-facing link this turn belongs to.
    pub(crate) conversation_id: ConversationId,
    /// Storage turn id used by status and persistent media job tools.
    pub(crate) turn_id: TurnId,
    /// User who triggered the turn, used for scoped memory and rate limits.
    pub(crate) turn_user: UserRef,
    /// Privacy mode that gates history-fetch behavior for this turn.
    pub(crate) privacy: PrivacyMode,
}

/// Dynamic tool exposure toggles for one agent run.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RuntimeToolFlags {
    /// Allow the model to fetch platform history according to privacy rules.
    pub(crate) fetch_messages: bool,
    /// Allow the model to post progress/status updates in the reply channel.
    pub(crate) post_status: bool,
    /// Allow the model to react to the message being handled.
    pub(crate) add_reaction: bool,
    /// Allow the model to inspect usage/cost for the current channel.
    pub(crate) usage_report: bool,
    /// Stored-media access tools exposed for this run.
    pub(crate) media_access: RuntimeMediaAccessFlags,
    /// User-memory tools exposed for this run.
    pub(crate) memory: RuntimeMemoryFlags,
}

/// Enabled stored-media access operations.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RuntimeMediaAccessFlags {
    /// Allow reading stored media into the next model step.
    pub(crate) read: bool,
    /// Allow inspecting stored media metadata.
    pub(crate) stat: bool,
    /// Allow resolving a stored asset to a public URL when supported.
    pub(crate) public_url: bool,
    /// Allow queueing a stored asset for the final platform reply.
    pub(crate) attach: bool,
}

/// Enabled user-memory operations.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RuntimeMemoryFlags {
    /// Allow lookup of relevant memories for the current user.
    pub(crate) lookup: bool,
    /// Allow writing a new memory for the current user.
    pub(crate) remember: bool,
    /// Allow deleting memories for the current user.
    pub(crate) forget: bool,
}

impl RuntimeMemoryFlags {
    /// Enable the full read/write/delete memory surface for a configured agent.
    pub(crate) fn all() -> Self {
        Self {
            lookup: true,
            remember: true,
            forget: true,
        }
    }
}

/// Tool executor for an agent run.
///
/// Tool specs and tool execution are both derived from the same dependencies,
/// context, and feature flags, so enabled/disabled behavior stays in one place.
/// Each successful call returns a `ClientToolOutput`: its `result` is sent back
/// to the model, its `trace_response` is serialized into the client tool trace,
/// and any media handles are carried only to the next model step.
///
/// The executor handles one call at a time. The agent runtime may call it for
/// several model-emitted tool calls in parallel, then sort those completed
/// results back into original call order before appending transcript blocks and
/// later persisting trace rows.
pub(crate) struct RuntimeToolExecutor<R: BotRuntimeTypes> {
    /// Shared service handles used to construct concrete tools.
    pub(crate) deps: RuntimeToolDeps<R>,
    /// Conversation and user context captured for this turn.
    pub(crate) context: RuntimeToolContext,
    /// Runtime flags controlling which built-in tools are advertised and accepted.
    pub(crate) enabled: RuntimeToolFlags,
    /// Memory context, present only when memory tools are enabled.
    pub(crate) memory: Option<MemoryToolContext>,
    /// Configured image-generation binding exposed as `generate_image`.
    pub(crate) image_generation: Option<GenerationBinding>,
    /// Configured video-generation binding exposed as `generate_video`.
    pub(crate) video_generation: Option<GenerationBinding>,
    /// Configured audio-transcription binding exposed as `transcribe_audio`.
    pub(crate) audio_transcription: Option<TranscriptionBinding>,
    /// Configured subagents exposed as additional named client tools.
    pub(crate) subagents:
        BTreeMap<ToolName, Subagent<RoutedLlmBackend<<R as BotRuntimeTypes>::Llms>, Self>>,
}

impl<R> std::fmt::Debug for RuntimeToolExecutor<R>
where
    R: BotRuntimeTypes,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeToolExecutor")
            .field("enabled", &self.enabled)
            .field("memory", &self.memory.is_some())
            .field("image_generation", &self.image_generation.is_some())
            .field("video_generation", &self.video_generation.is_some())
            .field("audio_transcription", &self.audio_transcription.is_some())
            .field("subagents", &self.subagents)
            .finish()
    }
}

impl<R> ClientToolExecutor for RuntimeToolExecutor<R>
where
    R: BotRuntimeTypes,
{
    type Error = RuntimeToolError;

    async fn execute(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<Self::Error>> {
        let name = call.name.clone();
        // This is the single-call dispatch point invoked by the concurrent
        // agent loop. Unknown names fall through to subagents, then to the
        // executor's sentinel unknown-tool error.
        match name.as_str() {
            FETCH_MESSAGES_TOOL if self.enabled.fetch_messages => self.fetch_messages(call).await,
            POST_STATUS_TOOL if self.enabled.post_status => self.post_status(call).await,
            ADD_REACTION_TOOL if self.enabled.add_reaction => self.add_reaction(call).await,
            USAGE_REPORT_TOOL if self.enabled.usage_report => self.usage_report(call).await,
            GENERATE_IMAGE_TOOL if self.image_generation.is_some() => {
                self.generate_image(call).await
            }
            GENERATE_VIDEO_TOOL if self.video_generation.is_some() => {
                self.generate_video(call).await
            }
            TRANSCRIBE_AUDIO_TOOL if self.audio_transcription.is_some() => {
                self.transcribe_audio(call).await
            }
            READ_ASSET_TOOL if self.enabled.media_access.read => self.read_asset(call).await,
            STAT_ASSET_TOOL if self.enabled.media_access.stat => self.stat_asset(call).await,
            PUBLIC_URL_ASSET_TOOL if self.enabled.media_access.public_url => {
                self.public_url_asset(call).await
            }
            ATTACH_ASSET_TOOL if self.enabled.media_access.attach => self.attach_asset(call).await,
            LOOKUP_USER_MEMORY_TOOL if self.enabled.memory.lookup => {
                self.lookup_user_memory(call).await
            }
            REMEMBER_USER_MEMORY_TOOL if self.enabled.memory.remember => {
                self.remember_user_memory(call).await
            }
            FORGET_USER_MEMORY_TOOL if self.enabled.memory.forget => {
                self.forget_user_memory(call).await
            }
            _ => self.execute_subagent_or_unknown(call).await,
        }
    }

    fn tools(&self) -> Vec<ClientToolDefinition> {
        // Keep this list in sync with `execute`: a tool should be advertised
        // only when the matching dispatch arm is enabled.
        let mut definitions = Vec::new();
        if self.enabled.fetch_messages {
            definitions.push(ClientToolDefinition::new(
                FETCH_MESSAGES_TOOL,
                self.fetch_messages_tool().spec(),
            ));
        }
        if self.enabled.post_status {
            definitions.push(ClientToolDefinition::new(
                POST_STATUS_TOOL,
                self.post_status_tool().spec(),
            ));
        }
        if self.enabled.add_reaction {
            definitions.push(ClientToolDefinition::new(
                ADD_REACTION_TOOL,
                self.add_reaction_tool().spec(),
            ));
        }
        if self.enabled.usage_report {
            definitions.push(ClientToolDefinition::new(
                USAGE_REPORT_TOOL,
                self.usage_report_tool().spec(),
            ));
        }
        if let Some(tool) = self.image_generation_tool() {
            definitions.push(ClientToolDefinition::new(GENERATE_IMAGE_TOOL, tool.spec()));
        }
        if let Some(tool) = self.video_generation_tool() {
            definitions.push(ClientToolDefinition::new(GENERATE_VIDEO_TOOL, tool.spec()));
        }
        if let Some(tool) = self.audio_transcription_tool() {
            definitions.push(ClientToolDefinition::new(
                TRANSCRIBE_AUDIO_TOOL,
                tool.spec(),
            ));
        }
        if self.enabled.media_access.read {
            definitions.push(ClientToolDefinition::new(
                READ_ASSET_TOOL,
                read_asset_spec(),
            ));
        }
        if self.enabled.media_access.stat {
            definitions.push(ClientToolDefinition::new(
                STAT_ASSET_TOOL,
                stat_asset_spec(),
            ));
        }
        if self.enabled.media_access.public_url {
            definitions.push(ClientToolDefinition::new(
                PUBLIC_URL_ASSET_TOOL,
                public_url_asset_spec(),
            ));
        }
        if self.enabled.media_access.attach {
            definitions.push(ClientToolDefinition::new(
                ATTACH_ASSET_TOOL,
                attach_asset_spec(),
            ));
        }
        if self.enabled.memory.lookup {
            definitions.push(ClientToolDefinition::new(
                LOOKUP_USER_MEMORY_TOOL,
                lookup_user_memory_spec(),
            ));
        }
        if self.enabled.memory.remember {
            definitions.push(ClientToolDefinition::new(
                REMEMBER_USER_MEMORY_TOOL,
                remember_user_memory_spec(),
            ));
        }
        if self.enabled.memory.forget {
            definitions.push(ClientToolDefinition::new(
                FORGET_USER_MEMORY_TOOL,
                forget_user_memory_spec(),
            ));
        }
        for (name, subagent) in &self.subagents {
            definitions.push(ClientToolDefinition::new(name.clone(), subagent.spec()));
        }
        definitions
    }
}

impl<R> RuntimeToolExecutor<R>
where
    R: BotRuntimeTypes,
{
    /// Build an executor with no optional tools enabled.
    pub(crate) fn new(deps: RuntimeToolDeps<R>, context: RuntimeToolContext) -> Self {
        Self {
            deps,
            context,
            enabled: RuntimeToolFlags::default(),
            memory: None,
            image_generation: None,
            video_generation: None,
            audio_transcription: None,
            subagents: BTreeMap::new(),
        }
    }

    /// Attach scoped memory context and expose the complete memory tool set.
    pub(crate) fn enable_memory(&mut self, context: MemoryToolContext) {
        self.memory = Some(context);
        self.enabled.memory = RuntimeMemoryFlags::all();
    }

    // Tool wrappers are built lazily so the executor stores services and flags,
    // not one field per possible tool.
    fn fetch_messages_tool(&self) -> FetchMessagesTool<R::Platforms, R::Storage> {
        FetchMessagesTool {
            platforms: self.deps.platforms.clone(),
            storage: self.deps.storage.clone(),
            default_channel: self.context.default_channel.clone(),
            privacy: self.context.privacy.clone(),
        }
    }

    fn post_status_tool(&self) -> PostStatusTool<R::Platforms, R::Storage> {
        PostStatusTool {
            platforms: self.deps.platforms.clone(),
            storage: self.deps.storage.clone(),
            channel: self.context.default_channel.clone(),
            reply_to: self.context.reply_to.clone(),
            conversation_id: self.context.conversation_id,
            turn_id: self.context.turn_id,
        }
    }

    fn add_reaction_tool(&self) -> AddReactionTool<R::Platforms> {
        AddReactionTool {
            platforms: self.deps.platforms.clone(),
            message: self.context.reply_to.clone(),
        }
    }

    fn usage_report_tool(&self) -> UsageReportTool<R::Storage> {
        UsageReportTool {
            storage: self.deps.storage.clone(),
            channel: self.context.default_channel.clone(),
        }
    }

    fn image_generation_tool(
        &self,
    ) -> Option<ImageGeneratorTool<RoutedImageGenerator<R::Images>, R::Media>> {
        self.image_generation.as_ref().map(|binding| {
            ImageGeneratorTool::new(
                RoutedImageGenerator::new(
                    self.deps.images.clone(),
                    binding.provider.clone(),
                    binding.model.clone(),
                ),
                self.deps.media_store.clone(),
            )
            .with_description(image_generation_tool_description(
                &binding.provider,
                &binding.model,
            ))
        })
    }

    fn video_generation_tool(
        &self,
    ) -> Option<PersistentVideoGeneratorTool<RoutedVideoGenerator<R::Videos>, R::Media, R::Storage>>
    {
        self.video_generation.as_ref().map(|binding| {
            PersistentVideoGeneratorTool::new(
                RoutedVideoGenerator::new(
                    self.deps.videos.clone(),
                    binding.provider.clone(),
                    binding.model.clone(),
                ),
                self.deps.media_store.clone(),
                self.deps.storage.clone(),
                self.deps.video_rate_limit_locks.clone(),
                self.context.turn_id,
                self.context.turn_user.clone(),
                binding.provider.clone(),
                binding.rate_limit.clone(),
            )
            .with_description(video_generation_tool_description(binding))
        })
    }

    fn audio_transcription_tool(
        &self,
    ) -> Option<AudioTranscriptionTool<RoutedAudioTranscriber<R::Audio>, R::Media>> {
        self.audio_transcription.as_ref().map(|binding| {
            AudioTranscriptionTool::new(
                RoutedAudioTranscriber::new(
                    self.deps.audio.clone(),
                    binding.provider.clone(),
                    binding.model.clone(),
                ),
                self.deps.media_store.clone(),
            )
            .with_default_keyterms(audio_transcription_default_keyterms(binding))
            .with_description(format!(
                "Transcribe a stored audio attachment with the configured `{}` audio provider{} and return the speech as text.",
                binding.provider,
                binding
                    .model
                    .as_ref()
                    .map(|model| format!(" and `{model}` model"))
                    .unwrap_or_default()
            ))
        })
    }

    // The wrapper methods below preserve each tool's `ClientToolOutput` on
    // success. On failure, they normalize tool-specific errors into
    // `RuntimeToolError`; the agent loop converts that into an `is_error` tool
    // result plus an error JSON object in the client tool trace.

    async fn fetch_messages(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        self.fetch_messages_tool()
            .call(call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn post_status(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        self.post_status_tool()
            .call(call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn add_reaction(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        self.add_reaction_tool()
            .call(call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn usage_report(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        self.usage_report_tool()
            .call(call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn generate_image(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        // Guard against stale model-visible calls if bindings changed between
        // spec generation and dispatch.
        let Some(tool) = self.image_generation_tool() else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        tool.call(call).await.map_err(runtime_tool_execution_error)
    }

    async fn generate_video(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        // Guard against stale model-visible calls if bindings changed between
        // spec generation and dispatch.
        let Some(tool) = self.video_generation_tool() else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        tool.call(call).await.map_err(runtime_tool_execution_error)
    }

    async fn transcribe_audio(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        // Guard against stale model-visible calls if bindings changed between
        // spec generation and dispatch.
        let Some(tool) = self.audio_transcription_tool() else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        tool.call(call).await.map_err(runtime_tool_execution_error)
    }

    async fn read_asset(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        read_asset(&self.deps.media_store, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn stat_asset(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        stat_asset(&self.deps.media_store, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn public_url_asset(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        public_url_asset(&self.deps.media_store, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn attach_asset(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        attach_asset(&self.deps.media_store, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn lookup_user_memory(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        // Memory tools require both the enable flag and scoped memory context.
        let Some(context) = &self.memory else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        lookup_user_memory(&self.deps.storage, context, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn remember_user_memory(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        // Memory tools require both the enable flag and scoped memory context.
        let Some(context) = &self.memory else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        remember_user_memory(&self.deps.storage, context, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn forget_user_memory(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        // Memory tools require both the enable flag and scoped memory context.
        let Some(context) = &self.memory else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        forget_user_memory(&self.deps.storage, context, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn execute_subagent_or_unknown(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        if let Some(subagent) = self.subagents.get(&call.name) {
            // Subagents use the same client-tool output contract. Their nested
            // trace is packed into `trace_response` by `Subagent::call`.
            return subagent
                .call(call)
                .await
                .map_err(runtime_tool_execution_error);
        }
        Err(ClientToolExecutorError::unknown(call.name))
    }
}

/// JSON Schema for all subagent tool inputs.
fn subagent_input_schema() -> ToolInputSchema {
    ToolInputSchema::object([ToolInputField::required(
        "prompt",
        ToolInputValueSchema::string().description("The task or question for the sub-agent."),
    )])
}

/// Convert the parent tool input into the nested agent's user message.
fn subagent_transcript_for_input(input: serde_json::Value) -> Transcript {
    let mut transcript = Transcript::new();
    transcript.push(TranscriptTurn::text(
        TurnRole::User,
        subagent_input_text(input),
    ));
    transcript
}

/// Extract the prompt string from a subagent call input.
///
/// Falling back to the whole JSON value keeps malformed inputs observable to
/// the nested model instead of dropping context on the floor.
fn subagent_input_text(input: serde_json::Value) -> String {
    input
        .get("prompt")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| input.to_string())
}

/// Convert the nested run into the generic client-tool output contract.
fn subagent_output_from_run(run: &AgentRun) -> ClientToolOutput {
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
        media: Vec::new(),
        is_error,
        trace_response: subagent_trace_response(run),
        usage: run.all_usage(),
    }
}

/// Serialize the nested run summary into one trace payload.
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

/// Low-cardinality nested-agent outcome label for structured logs.
fn subagent_outcome_kind(outcome: &AgentOutcome) -> &'static str {
    match outcome {
        AgentOutcome::Completed { .. } => "completed",
        AgentOutcome::Failed { .. } => "failed",
        AgentOutcome::IterationLimit { .. } => "iteration_limit",
        AgentOutcome::Cancelled { .. } => "cancelled",
    }
}

/// Convert a concrete tool failure into the executor's single error type.
///
/// The caller records this as an execution failure, sends an error result back
/// to the model, and stores the display string in trace JSON.
pub(crate) fn runtime_tool_execution_error(
    error: impl std::fmt::Display,
) -> ClientToolExecutorError<RuntimeToolError> {
    ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
}
