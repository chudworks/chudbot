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

type RuntimeVideoGenerationTool<R> = PersistentVideoGeneratorTool<
    RoutedVideoGenerator<<R as BotRuntimeTypes>::Videos>,
    <R as BotRuntimeTypes>::Media,
    <R as BotRuntimeTypes>::Storage,
>;

const GUILD_ICON_URI_PREFIX: &str = "guild_icon://";
const USER_AVATAR_URI_PREFIX: &str = "user_avatar://";

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
    #[allow(
        clippy::manual_async_fn,
        reason = "native async triggers a compiler cycle through recursive subagent executor types"
    )]
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
                let run =
                    collect_agent_run(self.agent.run(subagent_transcript_for_input(call.input)))
                        .await?;
                tracing::debug!(
                    outcome = run.outcome.kind(),
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
#[derive(Debug, Clone)]
pub(crate) struct RuntimeToolContext {
    /// Channel used when a tool needs the current conversation's default target.
    pub(in crate::tools) default_channel: ChannelRef,
    /// Bot message the turn is replying through.
    pub(in crate::tools) reply_to: MessageRef,
    /// Conversation whose trace and user-facing link this turn belongs to.
    pub(in crate::tools) conversation_id: ConversationId,
    /// Storage turn id used by status and persistent media job tools.
    pub(in crate::tools) turn_id: TurnId,
    /// User who triggered the turn, used for scoped memory and rate limits.
    pub(in crate::tools) turn_user: UserRef,
    /// Privacy mode that gates history-fetch behavior for this turn.
    pub(in crate::tools) privacy: PrivacyMode,
}

impl RuntimeToolContext {
    /// Capture the turn identity used by runtime tools.
    ///
    /// The default channel is derived from the reply message so callers cannot
    /// accidentally construct a context where `reply_to` and `default_channel`
    /// point at different platform surfaces.
    pub(crate) fn new(
        reply_to: MessageRef,
        conversation_id: ConversationId,
        turn_id: TurnId,
        turn_user: UserRef,
        privacy: PrivacyMode,
    ) -> Self {
        let default_channel = channel_from_message(&reply_to);
        Self {
            default_channel,
            reply_to,
            conversation_id,
            turn_id,
            turn_user,
            privacy,
        }
    }
}

bitflags::bitflags! {
    /// Dynamic tool exposure toggles for one agent run.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub(crate) struct RuntimeToolFlags: u16 {
        /// Allow the model to fetch platform history according to privacy rules.
        const FETCH_MESSAGES = 1 << 0;
        /// Allow the model to post progress/status updates in the reply channel.
        const POST_STATUS = 1 << 1;
        /// Allow the model to react to the message being handled.
        const ADD_REACTION = 1 << 2;
        /// Allow the model to inspect usage/cost for the current channel.
        const USAGE_REPORT = 1 << 3;
        /// Allow reading stored media into the next model step.
        const MEDIA_READ = 1 << 4;
        /// Allow inspecting stored media metadata.
        const MEDIA_STAT = 1 << 5;
        /// Allow resolving a stored asset to a public URL when supported.
        const MEDIA_PUBLIC_URL = 1 << 6;
        /// Allow queueing a stored asset for the final platform reply.
        const MEDIA_ATTACH = 1 << 7;

        /// Conversation helper tools exposed to every conversation agent.
        const CONVERSATION_HELPERS =
            Self::POST_STATUS.bits()
            | Self::ADD_REACTION.bits()
            | Self::USAGE_REPORT.bits();
        /// Stored-media inspection tools exposed to every conversation agent.
        const MEDIA_INSPECT =
            Self::MEDIA_READ.bits()
            | Self::MEDIA_STAT.bits()
            | Self::MEDIA_PUBLIC_URL.bits();
    }
}

/// Enabled user-memory surface plus the context every memory call requires.
#[derive(Debug, Clone)]
enum RuntimeMemoryTools {
    /// Read-only memory lookup, used by subagents.
    Lookup { context: MemoryToolContext },
    /// Full memory lookup/write/delete surface, used by top-level agents.
    Full { context: MemoryToolContext },
}

impl RuntimeMemoryTools {
    fn mode(&self) -> &'static str {
        match self {
            Self::Lookup { .. } => "lookup",
            Self::Full { .. } => "full",
        }
    }

    fn lookup_context(&self) -> &MemoryToolContext {
        match self {
            Self::Lookup { context } | Self::Full { context } => context,
        }
    }

    fn write_context(&self) -> Option<&MemoryToolContext> {
        match self {
            Self::Lookup { .. } => None,
            Self::Full { context } => Some(context),
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
    deps: RuntimeToolDeps<R>,
    /// Conversation and user context captured for this turn.
    context: RuntimeToolContext,
    /// Runtime flags controlling which built-in tools are advertised and accepted.
    enabled: RuntimeToolFlags,
    /// Memory tools, present only when memory is enabled for this run.
    memory: Option<RuntimeMemoryTools>,
    /// Configured image-generation binding exposed as `generate_image`.
    image_generation: Option<GenerationBinding>,
    /// Configured video-generation binding exposed as `generate_video`.
    video_generation: Option<GenerationBinding>,
    /// Configured audio-transcription binding exposed as `transcribe_audio`.
    audio_transcription: Option<TranscriptionBinding>,
    /// Configured subagents exposed as additional named client tools.
    subagents: BTreeMap<ToolName, Subagent<RoutedLlmBackend<<R as BotRuntimeTypes>::Llms>, Self>>,
}

impl<R> std::fmt::Debug for RuntimeToolExecutor<R>
where
    R: BotRuntimeTypes,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeToolExecutor")
            .field("enabled", &self.enabled)
            .field(
                "memory",
                &self.memory.as_ref().map(RuntimeMemoryTools::mode),
            )
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
        // This is the single-call dispatch point invoked by the concurrent
        // agent loop. Unknown names fall through to subagents, then to the
        // executor's sentinel unknown-tool error.
        match call.name.as_str() {
            FETCH_MESSAGES_TOOL if self.enabled.contains(RuntimeToolFlags::FETCH_MESSAGES) => {
                self.fetch_messages(call).await
            }
            POST_STATUS_TOOL if self.enabled.contains(RuntimeToolFlags::POST_STATUS) => {
                self.post_status(call).await
            }
            ADD_REACTION_TOOL if self.enabled.contains(RuntimeToolFlags::ADD_REACTION) => {
                self.add_reaction(call).await
            }
            USAGE_REPORT_TOOL if self.enabled.contains(RuntimeToolFlags::USAGE_REPORT) => {
                self.usage_report(call).await
            }
            GENERATE_IMAGE_TOOL if self.image_generation.is_some() => {
                self.generate_image(call).await
            }
            GENERATE_VIDEO_TOOL if self.video_generation.is_some() => {
                self.generate_video(call).await
            }
            TRANSCRIBE_AUDIO_TOOL if self.audio_transcription.is_some() => {
                self.transcribe_audio(call).await
            }
            READ_ASSET_TOOL if self.enabled.contains(RuntimeToolFlags::MEDIA_READ) => {
                self.read_asset(call).await
            }
            STAT_ASSET_TOOL if self.enabled.contains(RuntimeToolFlags::MEDIA_STAT) => {
                self.stat_asset(call).await
            }
            PUBLIC_URL_ASSET_TOOL if self.enabled.contains(RuntimeToolFlags::MEDIA_PUBLIC_URL) => {
                self.public_url_asset(call).await
            }
            ATTACH_ASSET_TOOL if self.enabled.contains(RuntimeToolFlags::MEDIA_ATTACH) => {
                self.attach_asset(call).await
            }
            LOOKUP_USER_MEMORY_TOOL if self.memory_lookup_enabled() => {
                self.lookup_user_memory(call).await
            }
            REMEMBER_USER_MEMORY_TOOL if self.memory_writes_enabled() => {
                self.remember_user_memory(call).await
            }
            FORGET_USER_MEMORY_TOOL if self.memory_writes_enabled() => {
                self.forget_user_memory(call).await
            }
            _ => self.execute_subagent_or_unknown(call).await,
        }
    }

    fn tools(&self) -> Vec<ClientToolDefinition> {
        // Keep this list in sync with `execute`: a tool should be advertised
        // only when the matching dispatch arm is enabled.
        let mut definitions = Vec::new();
        if self.enabled.contains(RuntimeToolFlags::FETCH_MESSAGES) {
            definitions.push(ClientToolDefinition::new(
                FETCH_MESSAGES_TOOL,
                self.fetch_messages_tool().spec(),
            ));
        }
        if self.enabled.contains(RuntimeToolFlags::POST_STATUS) {
            definitions.push(ClientToolDefinition::new(
                POST_STATUS_TOOL,
                self.post_status_tool().spec(),
            ));
        }
        if self.enabled.contains(RuntimeToolFlags::ADD_REACTION) {
            definitions.push(ClientToolDefinition::new(
                ADD_REACTION_TOOL,
                self.add_reaction_tool().spec(),
            ));
        }
        if self.enabled.contains(RuntimeToolFlags::USAGE_REPORT) {
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
        if self.enabled.contains(RuntimeToolFlags::MEDIA_READ) {
            definitions.push(ClientToolDefinition::new(
                READ_ASSET_TOOL,
                read_asset_spec(),
            ));
        }
        if self.enabled.contains(RuntimeToolFlags::MEDIA_STAT) {
            definitions.push(ClientToolDefinition::new(
                STAT_ASSET_TOOL,
                stat_asset_spec(),
            ));
        }
        if self.enabled.contains(RuntimeToolFlags::MEDIA_PUBLIC_URL) {
            definitions.push(ClientToolDefinition::new(
                PUBLIC_URL_ASSET_TOOL,
                public_url_asset_spec(),
            ));
        }
        if self.enabled.contains(RuntimeToolFlags::MEDIA_ATTACH) {
            definitions.push(ClientToolDefinition::new(
                ATTACH_ASSET_TOOL,
                attach_asset_spec(),
            ));
        }
        if self.memory_lookup_enabled() {
            definitions.push(ClientToolDefinition::new(
                LOOKUP_USER_MEMORY_TOOL,
                lookup_user_memory_spec(),
            ));
        }
        if self.memory_writes_enabled() {
            definitions.push(ClientToolDefinition::new(
                REMEMBER_USER_MEMORY_TOOL,
                remember_user_memory_spec(),
            ));
        }
        if self.memory_writes_enabled() {
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
        self.memory = Some(RuntimeMemoryTools::Full { context });
    }

    /// Attach scoped memory context and expose read-only memory lookup.
    pub(crate) fn enable_memory_lookup(&mut self, context: MemoryToolContext) {
        self.memory = Some(RuntimeMemoryTools::Lookup { context });
    }

    pub(crate) fn enable_tools(&mut self, flags: RuntimeToolFlags) {
        self.enabled.insert(flags);
    }

    pub(crate) fn enable_image_generation(&mut self, binding: GenerationBinding) {
        self.image_generation = Some(binding);
    }

    pub(crate) fn enable_video_generation(&mut self, binding: GenerationBinding) {
        self.video_generation = Some(binding);
    }

    pub(crate) fn enable_audio_transcription(&mut self, binding: TranscriptionBinding) {
        self.audio_transcription = Some(binding);
    }

    pub(crate) fn add_subagent(
        &mut self,
        name: ToolName,
        description: impl Into<String>,
        agent: Agent<RoutedLlmBackend<<R as BotRuntimeTypes>::Llms>, Self>,
    ) {
        self.subagents
            .insert(name, Subagent::new(description, agent));
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

    fn video_generation_tool(&self) -> Option<RuntimeVideoGenerationTool<R>> {
        self.video_generation
            .as_ref()
            .map(|binding| PersistentVideoGeneratorTool {
                generator: RoutedVideoGenerator::new(
                    self.deps.videos.clone(),
                    binding.provider.clone(),
                    binding.model.clone(),
                ),
                media_store: self.deps.media_store.clone(),
                storage: self.deps.storage.clone(),
                rate_limit_locks: self.deps.video_rate_limit_locks.clone(),
                context: self.context.clone(),
                binding: binding.clone(),
                poll_interval: DEFAULT_VIDEO_POLL_INTERVAL,
                max_polls: DEFAULT_VIDEO_MAX_POLLS,
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

    fn memory_lookup_enabled(&self) -> bool {
        self.memory.is_some()
    }

    fn memory_writes_enabled(&self) -> bool {
        self.memory
            .as_ref()
            .and_then(RuntimeMemoryTools::write_context)
            .is_some()
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

    async fn resolve_media_access_call(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolCall, ClientToolExecutorError<RuntimeToolError>> {
        resolve_media_access_tool_call(&self.deps.storage, &self.context, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn read_asset(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let call = self.resolve_media_access_call(call).await?;
        read_asset(&self.deps.media_store, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn stat_asset(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let call = self.resolve_media_access_call(call).await?;
        stat_asset(&self.deps.media_store, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn public_url_asset(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let call = self.resolve_media_access_call(call).await?;
        public_url_asset(&self.deps.media_store, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn attach_asset(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let call = self.resolve_media_access_call(call).await?;
        attach_asset(&self.deps.media_store, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn lookup_user_memory(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let Some(context) = self.memory.as_ref().map(RuntimeMemoryTools::lookup_context) else {
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
        let Some(context) = self
            .memory
            .as_ref()
            .and_then(RuntimeMemoryTools::write_context)
        else {
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
        let Some(context) = self
            .memory
            .as_ref()
            .and_then(RuntimeMemoryTools::write_context)
        else {
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

async fn resolve_media_access_tool_call<S>(
    storage: &S,
    context: &RuntimeToolContext,
    mut call: ClientToolCall,
) -> Result<ClientToolCall, BotToolError>
where
    S: BotStorage,
{
    let uri = media_access_uri_from_context(storage, context, &call.input).await?;
    if let Some(input) = call.input.as_object_mut() {
        input.insert(
            "uri".to_string(),
            serde_json::Value::String(uri.as_str().to_string()),
        );
    } else {
        call.input = serde_json::json!({ "uri": uri.as_str() });
    }
    Ok(call)
}

async fn media_access_uri_from_context<S>(
    storage: &S,
    context: &RuntimeToolContext,
    input: &serde_json::Value,
) -> Result<MediaUri, BotToolError>
where
    S: BotStorage,
{
    let uri = tool_required_string(input, "uri")?;
    if let Some(target) = uri.strip_prefix(GUILD_ICON_URI_PREFIX) {
        return canonical_media_access_uri(current_guild_icon_uri(storage, context, target).await?);
    }
    if let Some(target) = uri.strip_prefix(USER_AVATAR_URI_PREFIX) {
        return canonical_media_access_uri(
            current_user_avatar_uri(storage, context, target).await?,
        );
    }
    media_uri_from_tool_input(input)
}

fn canonical_media_access_uri(uri: MediaUri) -> Result<MediaUri, BotToolError> {
    canonical_stored_media_uri(&uri).map_err(|_| {
        BotToolError::InvalidInput("resolved media handle is not a stored media:// URI".to_string())
    })
}

async fn current_guild_icon_uri<S>(
    storage: &S,
    context: &RuntimeToolContext,
    target: &str,
) -> Result<MediaUri, BotToolError>
where
    S: BotStorage,
{
    let Some(current_guild) = context.default_channel.guild_id.as_ref() else {
        return Err(BotToolError::InvalidInput(
            "`guild_icon://...` is only available inside a guild channel".to_string(),
        ));
    };

    if target.is_empty() {
        return Err(BotToolError::InvalidInput(
            "`guild_icon://...` must name `current` or the current guild id".to_string(),
        ));
    }

    if target != "current" && target != current_guild.as_str() {
        return Err(BotToolError::InvalidInput(
            "`guild_icon://...` may only reference the current guild".to_string(),
        ));
    }

    storage
        .load_guild_icon(
            context.default_channel.platform.clone(),
            current_guild.clone(),
        )
        .await
        .map_err(|error| BotToolError::Storage(error.to_string()))?
        .ok_or_else(|| {
            BotToolError::InvalidInput(
                "no cached guild icon is available for the current guild".to_string(),
            )
        })
}

async fn current_user_avatar_uri<S>(
    storage: &S,
    context: &RuntimeToolContext,
    target: &str,
) -> Result<MediaUri, BotToolError>
where
    S: BotStorage,
{
    let user_id = if target == "current" {
        context.turn_user.user_id.clone()
    } else if target.is_empty() {
        return Err(BotToolError::InvalidInput(
            "`user_avatar://...` must name `current` or a user id".to_string(),
        ));
    } else {
        ExternalId::new(target)
    };

    storage
        .load_user_avatar(UserRef {
            platform: context.default_channel.platform.clone(),
            guild_id: context.default_channel.guild_id.clone(),
            user_id,
        })
        .await
        .map_err(|error| BotToolError::Storage(error.to_string()))?
        .ok_or_else(|| {
            BotToolError::InvalidInput("no cached avatar is available for that user id".to_string())
        })
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
