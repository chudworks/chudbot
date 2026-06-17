//! Single runtime tool executor that owns tool discovery and name-based dispatch.

use super::*;

/// Runtime tool execution error after tool-specific errors are stringified.
#[derive(Debug)]
pub(crate) struct RuntimeToolError(String);

impl std::fmt::Display for RuntimeToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RuntimeToolError {}

pub(crate) type RuntimeAgent<R> =
    Agent<RoutedLlmBackend<<R as BotRuntimeTypes>::Llms>, RuntimeToolExecutor<R>>;

/// Boxed subagent tool type used when config creates recursive agent graphs.
pub(crate) type RuntimeSubagentTool<R> =
    Subagent<RoutedLlmBackend<<R as BotRuntimeTypes>::Llms>, RuntimeToolExecutor<R>>;

/// One configured subagent exposed as a named client tool.
pub(crate) struct RuntimeSubagent<R: BotRuntimeTypes> {
    pub(crate) name: ToolName,
    pub(crate) tool: Box<RuntimeSubagentTool<R>>,
}

/// Shared runtime services available to all tool calls for one turn.
pub(crate) struct RuntimeToolDeps<R: BotRuntimeTypes> {
    pub(crate) platforms: R::Platforms,
    pub(crate) storage: R::Storage,
    pub(crate) media_store: R::Media,
    pub(crate) images: R::Images,
    pub(crate) videos: R::Videos,
    pub(crate) audio: R::Audio,
    pub(crate) video_rate_limit_locks: VideoRateLimitLocks,
}

/// Per-turn context captured by tools that interact with the current conversation.
pub(crate) struct RuntimeToolContext {
    pub(crate) default_channel: ChannelRef,
    pub(crate) reply_to: MessageRef,
    pub(crate) conversation_id: ConversationId,
    pub(crate) turn_id: TurnId,
    pub(crate) turn_user: UserRef,
    pub(crate) privacy: PrivacyMode,
}

/// Dynamic tool exposure toggles for one agent run.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RuntimeToolFlags {
    pub(crate) fetch_messages: bool,
    pub(crate) post_status: bool,
    pub(crate) add_reaction: bool,
    pub(crate) usage_report: bool,
    pub(crate) media_access: RuntimeMediaAccessFlags,
    pub(crate) memory: RuntimeMemoryFlags,
}

/// Enabled stored-media access operations.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RuntimeMediaAccessFlags {
    pub(crate) read: bool,
    pub(crate) stat: bool,
    pub(crate) public_url: bool,
    pub(crate) attach: bool,
}

/// Enabled user-memory operations.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RuntimeMemoryFlags {
    pub(crate) lookup: bool,
    pub(crate) remember: bool,
    pub(crate) forget: bool,
}

impl RuntimeMemoryFlags {
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
pub(crate) struct RuntimeToolExecutor<R: BotRuntimeTypes> {
    pub(crate) deps: RuntimeToolDeps<R>,
    pub(crate) context: RuntimeToolContext,
    pub(crate) enabled: RuntimeToolFlags,
    pub(crate) memory: Option<memory::MemoryToolContext>,
    pub(crate) image_generation: Option<GenerationBinding>,
    pub(crate) video_generation: Option<GenerationBinding>,
    pub(crate) audio_transcription: Option<TranscriptionBinding>,
    pub(crate) subagents: Vec<RuntimeSubagent<R>>,
}

impl<R> std::fmt::Debug for RuntimeSubagent<R>
where
    R: BotRuntimeTypes,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeSubagent")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
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

impl<R> RuntimeToolExecutor<R>
where
    R: BotRuntimeTypes,
{
    pub(crate) fn new(deps: RuntimeToolDeps<R>, context: RuntimeToolContext) -> Self {
        Self {
            deps,
            context,
            enabled: RuntimeToolFlags::default(),
            memory: None,
            image_generation: None,
            video_generation: None,
            audio_transcription: None,
            subagents: Vec::new(),
        }
    }

    pub(crate) fn enable_memory(&mut self, context: memory::MemoryToolContext) {
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
        let Some(tool) = self.image_generation_tool() else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        tool.call(call).await.map_err(runtime_tool_execution_error)
    }

    async fn generate_video(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let Some(tool) = self.video_generation_tool() else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        tool.call(call).await.map_err(runtime_tool_execution_error)
    }

    async fn transcribe_audio(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
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
        let Some(context) = &self.memory else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        memory::lookup_user_memory(&self.deps.storage, context, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn remember_user_memory(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let Some(context) = &self.memory else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        memory::remember_user_memory(&self.deps.storage, context, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn forget_user_memory(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let Some(context) = &self.memory else {
            return Err(ClientToolExecutorError::unknown(call.name));
        };
        memory::forget_user_memory(&self.deps.storage, context, call)
            .await
            .map_err(runtime_tool_execution_error)
    }

    async fn execute_subagent_or_unknown(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<RuntimeToolError>> {
        let name = call.name.clone();
        let Some(subagent) = self
            .subagents
            .iter()
            .find(|subagent| subagent.name.as_str() == name.as_str())
        else {
            return Err(ClientToolExecutorError::unknown(name));
        };
        subagent
            .tool
            .call(call)
            .await
            .map_err(runtime_tool_execution_error)
    }
}

impl<R> ClientToolExecutor for RuntimeToolExecutor<R>
where
    R: BotRuntimeTypes,
{
    type Error = RuntimeToolError;

    fn tools(&self) -> Vec<ClientToolDefinition> {
        // Keep this list in sync with `execute`: a tool should be advertised only
        // when the matching dispatch arm is enabled.
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
                memory::LOOKUP_USER_MEMORY_TOOL,
                memory::lookup_user_memory_spec(),
            ));
        }
        if self.enabled.memory.remember {
            definitions.push(ClientToolDefinition::new(
                memory::REMEMBER_USER_MEMORY_TOOL,
                memory::remember_user_memory_spec(),
            ));
        }
        if self.enabled.memory.forget {
            definitions.push(ClientToolDefinition::new(
                memory::FORGET_USER_MEMORY_TOOL,
                memory::forget_user_memory_spec(),
            ));
        }
        for subagent in &self.subagents {
            definitions.push(ClientToolDefinition::new(
                subagent.name.clone(),
                subagent.tool.spec(),
            ));
        }
        definitions
    }

    async fn execute(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<Self::Error>> {
        let name = call.name.clone();
        // Dispatch by stable tool name. Unknown names fall through to subagents,
        // then to the executor's sentinel unknown-tool error.
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
            memory::LOOKUP_USER_MEMORY_TOOL if self.enabled.memory.lookup => {
                self.lookup_user_memory(call).await
            }
            memory::REMEMBER_USER_MEMORY_TOOL if self.enabled.memory.remember => {
                self.remember_user_memory(call).await
            }
            memory::FORGET_USER_MEMORY_TOOL if self.enabled.memory.forget => {
                self.forget_user_memory(call).await
            }
            _ => self.execute_subagent_or_unknown(call).await,
        }
    }
}

pub(crate) fn runtime_tool_execution_error(
    error: impl std::fmt::Display,
) -> ClientToolExecutorError<RuntimeToolError> {
    ClientToolExecutorError::execution(RuntimeToolError(error.to_string()))
}
