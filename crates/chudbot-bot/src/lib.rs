//! Platform-neutral bot orchestration.
//!
//! This crate owns the chudbot turn lifecycle without knowing about Discord,
//! Postgres, Axum, or concrete model-provider HTTP clients. It consumes the
//! contracts from `chudbot-api` and routes work through named service
//! registries supplied by the binary crate.

#![allow(async_fn_in_trait)]

pub mod memory;

pub use memory::MemoryConfig;

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chudbot_api::{
    Agent, AgentBuilder, AgentLimits, AgentOutcome, AgentSelection, AgentSpec, AttachmentRef,
    AudioTranscriber, AudioTranscription, AudioTranscriptionRequest, BeginTurn, BotStorage,
    ChannelLink, ChannelRef, ClientTool, ClientToolCall, ClientToolOutput, ClientToolResultContent,
    ClientToolSpec, ContentBlock, Conversation, ConversationEventKind, ConversationId,
    ConversationLookup, ConversationSnapshot, ConversationStop, CreateMedia, CreateVideoJob,
    EventSink, ExternalId, FetchMessages, FinishTurn, GeneratedImage, ImageGenerator,
    ImageGeneratorTool, ImageRequest, LiveEvent, LlmBackend, MediaCategory, MediaStore, MediaUri,
    MessageLink, MessagePlatform, MessageRef, Model, ModelId, ModelSpec, ModelStep,
    ModelStepRequest, OpenConversation, OutgoingAttachment, PlatformCommand,
    PlatformCommandDefinition, PlatformCommandInput, PlatformCommandOption,
    PlatformCommandOptionChoice, PlatformCommandOptionKind, PlatformCommandResponse,
    PlatformCommandValue, PlatformEvent, PlatformMessage, PlatformMessageReference,
    PlatformMessageRelationship, PlatformName, PlatformReaction, PostedMessage, PrivacyMode,
    ProviderName, ReactionKind, ResolveAgent, RuntimeSettings, SamplingOptions, SaveTurnInput,
    SendMessage, Subagent, ThreadRequest, ToolInputSchema, ToolName, ToolTrace, Transcript,
    TranscriptTurn, Turn, TurnAsset, TurnId, TurnRole, TurnSnapshot, UpdateVideoJob, UrlMediaRef,
    UserProfile, UserRef, VideoGenerator, VideoJobId, VideoJobStatus, VideoRequest,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::task::{JoinError, JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

const FETCH_MESSAGES_TOOL: &str = "fetch_messages";
const GENERATE_IMAGE_TOOL: &str = "generate_image";
const GENERATE_VIDEO_TOOL: &str = "generate_video";
const TRANSCRIBE_AUDIO_TOOL: &str = "transcribe_audio";
const POST_STATUS_TOOL: &str = "post_status_message";
const WORKING_REACTION: &str = "👀";
const SUCCESS_REACTION: &str = "✅";
const ERROR_REACTION: &str = "❌";
const RETRY_REACTION: &str = "🔄";
const STOP_REACTION: &str = "🛑";
const REFUSED_REACTION: &str = "❓";
const DEFAULT_SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
const TYPING_REFRESH_INTERVAL: Duration = Duration::from_secs(8);
const MAX_OUTGOING_ATTACHMENT_BYTES: usize = 25 * 1024 * 1024;
const HISTORY_SIZE_MIN: i64 = 1;
const HISTORY_SIZE_MAX: i64 = 100;
const TITLE_MAX_CHARS: usize = 80;
const TITLE_MAX_TOKENS: u32 = 96;
const DEFAULT_THREAD_THRESHOLD_CHARS: usize = 1500;
const DEFAULT_THREAD_THRESHOLD_LINES: usize = 20;
const THREAD_REPLY_WRAP_WIDTH: usize = 80;
const MODEL_TRANSCRIPT_IMAGE_MIME_TYPES: &[&str] = &[
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/x-icon",
    "image/vnd.microsoft.icon",
];

const TITLE_SYSTEM_PROMPT: &str = "You write very short conversation titles. \
Output ONLY a title for the conversation below: five words or fewer, no quotes, \
no period, no leading 'Re:' or 'Conversation about'. Just the title text.";

const MODERATION_PROMPT: &str = "You are a TOS compliance classifier for a \
private friends-only Discord server. Each message you classify is prefixed \
with the sender's display name as `[name]: `. The DEFAULT IS ALLOW. Only \
REFUSE the narrowly listed categories below.

REFUSE these:
- CSAM or any sexualization of minors
- Doxxing: sharing someone's non-public personal info with apparent intent to harm
- Credible, specific threats of violence against a real, identifiable person
- Coordinated incitement to suicide or self-harm directed at a specific person
- Illegal arrangements: drug or weapon sales, human trafficking, exploitation rings
- Malware, phishing payloads, or coordinated large-scale spam campaigns

ALLOW EVERYTHING ELSE, including profanity, insults, dark humor, political \
opinions, criticism of public figures, news/current-events questions, and \
edgy art requests that do not involve minors.

When in any doubt, ALLOW.

Respond with EXACTLY one token: ALLOW or REFUSE. No punctuation. No explanation.";

/// Registry of named LLM provider services.
pub trait LlmProviderRegistry: Clone + Send + Sync {
    /// Registry error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Whether a provider is configured.
    fn contains_provider(&self, provider: &ProviderName) -> bool;

    /// Execute one model step against a named provider.
    fn step(
        &self,
        provider: &ProviderName,
        request: ModelStepRequest,
    ) -> impl Future<Output = Result<ModelStep, Self::Error>> + Send;
}

/// Registry of named image generation services.
pub trait ImageGeneratorRegistry: Clone + Send + Sync {
    /// Registry error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Whether a generator is configured.
    fn contains_generator(&self, provider: &ProviderName) -> bool;

    /// Generate one image through a named provider.
    fn generate_image(
        &self,
        provider: &ProviderName,
        request: ImageRequest,
    ) -> impl Future<Output = Result<GeneratedImage, Self::Error>> + Send;
}

/// Registry of named video generation services.
pub trait VideoGeneratorRegistry: Clone + Send + Sync {
    /// Registry error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Whether a generator is configured.
    fn contains_generator(&self, provider: &ProviderName) -> bool;

    /// Submit a video generation job through a named provider.
    fn submit_video(
        &self,
        provider: &ProviderName,
        request: VideoRequest,
    ) -> impl Future<Output = Result<VideoJobId, Self::Error>> + Send;

    /// Poll a video generation job once through a named provider.
    fn check_video(
        &self,
        provider: &ProviderName,
        job: VideoJobId,
    ) -> impl Future<Output = Result<VideoJobStatus, Self::Error>> + Send;

    /// Download a completed video through a named provider.
    fn download_video(
        &self,
        provider: &ProviderName,
        url: String,
    ) -> impl Future<Output = Result<Vec<u8>, Self::Error>> + Send;
}

/// Registry of named audio transcription services.
pub trait AudioTranscriberRegistry: Clone + Send + Sync {
    /// Registry error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Whether a transcriber is configured.
    fn contains_transcriber(&self, provider: &ProviderName) -> bool;

    /// Transcribe one audio file through a named provider.
    fn transcribe_audio(
        &self,
        provider: &ProviderName,
        request: AudioTranscriptionRequest,
    ) -> impl Future<Output = Result<AudioTranscription, Self::Error>> + Send;
}

/// Registry of named message platform services.
pub trait MessagePlatformRegistry: Clone + Send + Sync {
    /// Registry error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Fetch the bot user for a platform.
    fn bot_user(
        &self,
        platform: &PlatformName,
    ) -> impl Future<Output = Result<UserProfile, Self::Error>> + Send;

    /// Register bot commands across configured platforms.
    fn register_commands(
        &self,
        commands: Vec<PlatformCommandDefinition>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Read the next event from any configured platform.
    fn next_event(&self) -> impl Future<Output = Result<PlatformEvent, Self::Error>> + Send;

    /// Gracefully stop platform services owned by this registry.
    fn shutdown(&self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        async { Ok(()) }
    }

    /// Respond to a command invocation.
    fn respond_to_command(
        &self,
        response: PlatformCommandResponse,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Send a message through the platform named by `request.channel.platform`.
    fn send_message(
        &self,
        request: SendMessage,
    ) -> impl Future<Output = Result<PostedMessage, Self::Error>> + Send;

    /// Delete a platform message.
    fn delete_message(
        &self,
        message: MessageRef,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Add a reaction to a platform message.
    fn add_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Remove the bot's own reaction from a platform message.
    fn remove_own_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Trigger a platform typing indicator.
    fn typing(&self, channel: ChannelRef) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Fetch messages through the platform named by `request.channel.platform`.
    fn fetch_messages(
        &self,
        request: FetchMessages,
    ) -> impl Future<Output = Result<Vec<PlatformMessage>, Self::Error>> + Send;

    /// Render a platform message for model context.
    fn message_context(
        &self,
        message: &PlatformMessage,
        relationship: PlatformMessageRelationship,
    ) -> impl Future<Output = Result<serde_json::Value, Self::Error>> + Send;

    /// Resolve a platform channel's parent channel.
    fn parent_channel(
        &self,
        channel: ChannelRef,
    ) -> impl Future<Output = Result<ChannelRef, Self::Error>> + Send;
}

impl<T> MessagePlatformRegistry for T
where
    T: MessagePlatform + Clone,
{
    type Error = T::Error;

    async fn bot_user(&self, _platform: &PlatformName) -> Result<UserProfile, Self::Error> {
        MessagePlatform::bot_user(self).await
    }

    async fn register_commands(
        &self,
        commands: Vec<PlatformCommandDefinition>,
    ) -> Result<(), Self::Error> {
        MessagePlatform::register_commands(self, commands, None).await
    }

    async fn next_event(&self) -> Result<PlatformEvent, Self::Error> {
        MessagePlatform::next_event(self).await
    }

    async fn respond_to_command(
        &self,
        response: PlatformCommandResponse,
    ) -> Result<(), Self::Error> {
        MessagePlatform::respond_to_command(self, response).await
    }

    async fn send_message(&self, request: SendMessage) -> Result<PostedMessage, Self::Error> {
        MessagePlatform::send_message(self, request).await
    }

    async fn delete_message(&self, message: MessageRef) -> Result<(), Self::Error> {
        MessagePlatform::delete_message(self, message).await
    }

    async fn add_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> Result<(), Self::Error> {
        MessagePlatform::add_reaction(self, message, reaction).await
    }

    async fn remove_own_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> Result<(), Self::Error> {
        MessagePlatform::remove_own_reaction(self, message, reaction).await
    }

    async fn typing(&self, channel: ChannelRef) -> Result<(), Self::Error> {
        MessagePlatform::typing(self, channel).await
    }

    async fn fetch_messages(
        &self,
        request: FetchMessages,
    ) -> Result<Vec<PlatformMessage>, Self::Error> {
        MessagePlatform::fetch_messages(self, request).await
    }

    async fn message_context(
        &self,
        message: &PlatformMessage,
        relationship: PlatformMessageRelationship,
    ) -> Result<serde_json::Value, Self::Error> {
        MessagePlatform::message_context(self, message, relationship).await
    }

    async fn parent_channel(&self, channel: ChannelRef) -> Result<ChannelRef, Self::Error> {
        MessagePlatform::parent_channel(self, channel).await
    }
}

/// `LlmBackend` adapter for one configured provider inside a registry.
#[derive(Debug, Clone)]
pub struct RoutedLlmBackend<R> {
    registry: R,
    provider: ProviderName,
}

impl<R> RoutedLlmBackend<R> {
    /// Build a routed backend.
    pub fn new(registry: R, provider: ProviderName) -> Self {
        Self { registry, provider }
    }
}

impl<R> LlmBackend for RoutedLlmBackend<R>
where
    R: LlmProviderRegistry,
{
    type Error = R::Error;

    fn backend_name(&self) -> &ProviderName {
        &self.provider
    }

    #[tracing::instrument(
        name = "llm.routed_step",
        skip_all,
        fields(
            provider = %self.provider,
            model = %request.model,
            transcript_id = ?request.transcript.id,
            turns = request.transcript.turns.len(),
            client_tools = request.client_tools.len(),
            server_tools = request.server_tools.len(),
        )
    )]
    async fn step(&self, request: ModelStepRequest) -> Result<ModelStep, Self::Error> {
        tracing::debug!("dispatching model step through provider registry");
        let result = self.registry.step(&self.provider, request).await;
        match &result {
            Ok(step) => tracing::debug!(outcome = model_step_kind(step), "model step completed"),
            Err(error) => tracing::warn!(error = %error, "model step failed"),
        }
        result
    }
}

/// `ImageGenerator` adapter for one configured provider inside a registry.
#[derive(Debug, Clone)]
pub struct RoutedImageGenerator<R> {
    registry: R,
    provider: ProviderName,
    model: ModelId,
}

impl<R> RoutedImageGenerator<R> {
    /// Build a routed image generator.
    pub fn new(registry: R, provider: ProviderName, model: ModelId) -> Self {
        Self {
            registry,
            provider,
            model,
        }
    }
}

impl<R> ImageGenerator for RoutedImageGenerator<R>
where
    R: ImageGeneratorRegistry,
{
    type Error = R::Error;

    fn backend_name(&self) -> &ProviderName {
        &self.provider
    }

    #[tracing::instrument(
        name = "image.routed_generate",
        skip_all,
        fields(provider = %self.provider, configured_model = %self.model)
    )]
    async fn generate_image(
        &self,
        mut request: ImageRequest,
    ) -> Result<GeneratedImage, Self::Error> {
        if request.model.is_none() {
            request.model = Some(self.model.clone());
        }
        tracing::debug!(
            request_model = ?request.model.as_ref(),
            "dispatching image generation through registry"
        );
        let result = self.registry.generate_image(&self.provider, request).await;
        match &result {
            Ok(image) => tracing::info!(
                model = %image.model,
                mime_type = %image.mime_type,
                bytes = image.bytes.len(),
                "image generation completed"
            ),
            Err(error) => tracing::warn!(error = %error, "image generation failed"),
        }
        result
    }
}

/// `VideoGenerator` adapter for one configured provider inside a registry.
#[derive(Debug, Clone)]
pub struct RoutedVideoGenerator<R> {
    registry: R,
    provider: ProviderName,
    model: ModelId,
}

impl<R> RoutedVideoGenerator<R> {
    /// Build a routed video generator.
    pub fn new(registry: R, provider: ProviderName, model: ModelId) -> Self {
        Self {
            registry,
            provider,
            model,
        }
    }
}

impl<R> VideoGenerator for RoutedVideoGenerator<R>
where
    R: VideoGeneratorRegistry,
{
    type Error = R::Error;

    fn backend_name(&self) -> &ProviderName {
        &self.provider
    }

    #[tracing::instrument(
        name = "video.routed_submit",
        skip_all,
        fields(provider = %self.provider, configured_model = %self.model)
    )]
    async fn submit_video(&self, mut request: VideoRequest) -> Result<VideoJobId, Self::Error> {
        if request.model.is_none() {
            request.model = Some(self.model.clone());
        }
        tracing::debug!(
            request_model = ?request.model.as_ref(),
            "submitting video generation through registry"
        );
        let result = self.registry.submit_video(&self.provider, request).await;
        match &result {
            Ok(job) => tracing::info!(job = %job, "video generation submitted"),
            Err(error) => tracing::warn!(error = %error, "video generation submit failed"),
        }
        result
    }

    #[tracing::instrument(
        name = "video.routed_check",
        skip_all,
        fields(provider = %self.provider, job = %job)
    )]
    async fn check_video(&self, job: VideoJobId) -> Result<VideoJobStatus, Self::Error> {
        self.registry.check_video(&self.provider, job).await
    }

    #[tracing::instrument(name = "video.routed_download", skip_all, fields(provider = %self.provider))]
    async fn download_video(&self, url: String) -> Result<Vec<u8>, Self::Error> {
        self.registry.download_video(&self.provider, url).await
    }
}

/// `AudioTranscriber` adapter for one configured provider inside a registry.
#[derive(Debug, Clone)]
pub struct RoutedAudioTranscriber<R> {
    registry: R,
    provider: ProviderName,
    model: Option<ModelId>,
}

impl<R> RoutedAudioTranscriber<R> {
    /// Build a routed audio transcriber.
    pub fn new(registry: R, provider: ProviderName, model: Option<ModelId>) -> Self {
        Self {
            registry,
            provider,
            model,
        }
    }
}

impl<R> AudioTranscriber for RoutedAudioTranscriber<R>
where
    R: AudioTranscriberRegistry,
{
    type Error = R::Error;

    fn backend_name(&self) -> &ProviderName {
        &self.provider
    }

    #[tracing::instrument(
        name = "audio.routed_transcribe",
        skip_all,
        fields(provider = %self.provider, configured_model = ?self.model.as_ref())
    )]
    async fn transcribe_audio(
        &self,
        mut request: AudioTranscriptionRequest,
    ) -> Result<AudioTranscription, Self::Error> {
        if request.model.is_none() {
            request.model = self.model.clone();
        }
        tracing::debug!(
            request_model = ?request.model.as_ref(),
            "dispatching audio transcription through registry"
        );
        let result = self
            .registry
            .transcribe_audio(&self.provider, request)
            .await;
        match &result {
            Ok(transcription) => tracing::info!(
                duration_seconds = transcription.duration_seconds,
                text_chars = transcription.text.chars().count(),
                usage_records = transcription.usage.len(),
                "audio transcription completed"
            ),
            Err(error) => tracing::warn!(error = %error, "audio transcription failed"),
        }
        result
    }
}

/// Runtime configuration owned by the bot crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotConfig {
    /// Public viewer base URL used in the first reply for a conversation.
    pub web_base_url: String,
    /// Default top-level agent when a platform has no explicit binding.
    pub default_agent: String,
    /// Named agents. An agent may be top-level, subagent-only, or both.
    pub agents: BTreeMap<String, AgentConfig>,
    /// Operator users allowed to stop/resume conversations with the stop
    /// reaction. A missing `guild_id` applies across the platform.
    #[serde(default)]
    pub admins: Vec<chudbot_api::UserRef>,
    /// Platform default bindings, e.g. `discord -> chudbot`.
    #[serde(default)]
    pub platforms: BTreeMap<PlatformName, PlatformBinding>,
    /// Optional operator-wide policy text.
    #[serde(default)]
    pub extra_system_prompt: Option<String>,
    /// Build/version label included in the operational system prompt.
    #[serde(default)]
    pub version: String,
    /// Default model/tool loop limits for agents that do not override them.
    #[serde(default)]
    pub limits: AgentLimits,
    /// Reply length above which a new conversation asks the platform to open a
    /// thread when supported.
    #[serde(default = "default_thread_threshold_chars")]
    pub thread_threshold_chars: usize,
    /// Approximate visible reply rows above which a new conversation asks the
    /// platform to open a thread when supported.
    #[serde(default = "default_thread_threshold_lines")]
    pub thread_threshold_lines: usize,
}

impl BotConfig {
    /// Validate static agent references.
    #[tracing::instrument(
        name = "bot.config.validate",
        skip_all,
        fields(
            agents = self.agents.len(),
            admins = self.admins.len(),
            platforms = self.platforms.len(),
            default_agent = %self.default_agent,
        )
    )]
    pub fn validate(&self) -> Result<(), BotError> {
        tracing::debug!("validating bot config");
        if !self.agents.contains_key(&self.default_agent) {
            tracing::warn!(
                missing_agent = %self.default_agent,
                "default agent is not configured"
            );
            return Err(BotError::MissingAgent {
                name: self.default_agent.clone(),
            });
        }
        for binding in self.platforms.values() {
            if !self.agents.contains_key(&binding.agent) {
                tracing::warn!(
                    missing_agent = %binding.agent,
                    "platform binding references missing agent"
                );
                return Err(BotError::MissingAgent {
                    name: binding.agent.clone(),
                });
            }
        }
        for (agent_name, agent) in &self.agents {
            if let Some(binding) = &agent.image_generation {
                validate_generation_binding(agent_name, "image_generation", binding)?;
            }
            if let Some(binding) = &agent.video_generation {
                validate_generation_binding(agent_name, "video_generation", binding)?;
            }
            if let Some(binding) = &agent.audio_transcription {
                validate_transcription_binding(agent_name, "audio_transcription", binding)?;
            }
            for binding in agent.subagents.values() {
                if !self.agents.contains_key(&binding.agent) {
                    tracing::warn!(
                        agent = %agent_name,
                        missing_subagent = %binding.agent,
                        "subagent binding references missing agent"
                    );
                    return Err(BotError::MissingSubagent {
                        agent: agent_name.clone(),
                        subagent: binding.agent.clone(),
                    });
                }
            }
        }
        tracing::info!("bot config validated");
        Ok(())
    }

    /// Resolve an agent name with fallback to the platform binding and default
    /// agent.
    pub fn agent_or_platform_default(
        &self,
        requested: Option<&str>,
        platform: &PlatformName,
    ) -> Result<(String, &AgentConfig), BotError> {
        if let Some(name) = requested
            && let Some(agent) = self.agents.get(name)
        {
            tracing::debug!(
                requested_agent = %name,
                platform = %platform,
                provider = %agent.provider,
                model = %agent.model.id,
                "resolved requested agent"
            );
            return Ok((name.to_string(), agent));
        }

        let platform_default = self
            .platforms
            .get(platform)
            .map(|binding| binding.agent.as_str())
            .unwrap_or(self.default_agent.as_str());
        let resolved = self
            .agents
            .get(platform_default)
            .map(|agent| (platform_default.to_string(), agent))
            .ok_or_else(|| BotError::MissingAgent {
                name: platform_default.to_string(),
            })?;
        tracing::debug!(
            requested_agent = ?requested,
            platform = %platform,
            resolved_agent = %resolved.0,
            provider = %resolved.1.provider,
            model = %resolved.1.model.id,
            "resolved platform/default agent"
        );
        Ok(resolved)
    }
}

fn default_thread_threshold_chars() -> usize {
    DEFAULT_THREAD_THRESHOLD_CHARS
}

fn default_thread_threshold_lines() -> usize {
    DEFAULT_THREAD_THRESHOLD_LINES
}

fn image_generation_tool_description(provider: &ProviderName, model: &ModelId) -> String {
    format!(
        concat!(
            "Generate an image with the configured `{}` image provider and `{}` model, ",
            "save it to media storage, and return its media URI.\n\n",
            "Use this whenever the user asks for an image, picture, drawing, illustration, ",
            "infographic, or other visual.\n\n",
            "To edit, restyle, transform, make a variation of, or combine images already ",
            "visible in the conversation, pass their exact `file://images/...` URI(s) in ",
            "`reference_images`. This is the expected path for requests like \"turn this ",
            "image into...\", \"make the image...\", \"use the previous image\", or ",
            "\"here's a different version\". User-uploaded images are listed in image ",
            "attachment reference notes; generated images are listed in prior tool ",
            "results and generated-media reference notes. Never invent or guess paths. ",
            "For two or three references, refer to them in the prompt as <IMAGE_0>, ",
            "<IMAGE_1>, etc. in the same order. If no real URI applies, omit ",
            "`reference_images` and generate from text alone.\n\n",
            "Generated media is attached to the final platform reply automatically. ",
            "Do not paste media URIs, filenames, public URLs, or markdown media links ",
            "in user-facing text."
        ),
        provider, model
    )
}

fn validate_generation_binding(
    agent_name: &str,
    field: &'static str,
    binding: &GenerationBinding,
) -> Result<(), BotError> {
    if binding.provider.as_str().trim().is_empty() {
        tracing::warn!(agent = %agent_name, field, "media generation provider is empty");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "provider is empty".to_string(),
        });
    }
    if binding.model.as_str().trim().is_empty() {
        tracing::warn!(agent = %agent_name, field, "media generation model is empty");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "model is empty".to_string(),
        });
    }
    Ok(())
}

fn validate_transcription_binding(
    agent_name: &str,
    field: &'static str,
    binding: &TranscriptionBinding,
) -> Result<(), BotError> {
    if binding.provider.as_str().trim().is_empty() {
        tracing::warn!(agent = %agent_name, field, "audio transcription provider is empty");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "provider is empty".to_string(),
        });
    }
    if let Some(model) = &binding.model
        && model.as_str().trim().is_empty()
    {
        tracing::warn!(agent = %agent_name, field, "audio transcription model is empty");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "model is empty".to_string(),
        });
    }
    Ok(())
}

/// One named agent: prompt, provider/model, tool exposure, and subagents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// LLM provider registry key.
    pub provider: ProviderName,
    /// System prompt / agent instructions.
    pub system_prompt: String,
    /// Model config used for this agent.
    pub model: ModelSpec,
    /// Optional server-tool restriction for this agent. `None` means all
    /// server tools allowed by the model config are exposed.
    #[serde(default)]
    pub server_tools: Option<chudbot_api::ServerToolSet>,
    /// Optional client-tool allowlist. `None` means all runtime tools assembled
    /// for this agent are exposed.
    #[serde(default)]
    pub client_tools: Option<Vec<ToolName>>,
    /// Optional per-agent loop limits.
    #[serde(default)]
    pub limits: Option<AgentLimits>,
    /// Optional image generation binding exposed through `generate_image`.
    #[serde(default)]
    pub image_generation: Option<GenerationBinding>,
    /// Optional video generation binding exposed through `generate_video`.
    #[serde(default)]
    pub video_generation: Option<GenerationBinding>,
    /// Optional audio transcription binding exposed through `transcribe_audio`.
    #[serde(default)]
    pub audio_transcription: Option<TranscriptionBinding>,
    /// Whether top-level runs for this agent receive user-memory tools.
    #[serde(default)]
    pub memory: bool,
    /// Subagents exposed as named client-side tools.
    #[serde(default)]
    pub subagents: BTreeMap<ToolName, SubagentBinding>,
}

/// Binding from an agent to a media-generation provider and default model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationBinding {
    /// Media-generation provider registry key.
    pub provider: ProviderName,
    /// Provider-specific image/video model id or tier.
    pub model: ModelId,
}

/// Binding from an agent to an audio transcription provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionBinding {
    /// Audio transcription provider registry key.
    pub provider: ProviderName,
    /// Provider-specific transcription model id when applicable.
    #[serde(default)]
    pub model: Option<ModelId>,
}

/// Platform default binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformBinding {
    /// Agent name used for this platform by default.
    pub agent: String,
}

/// Runtime controls for the bot event loop.
#[derive(Debug, Clone, Copy)]
pub struct BotRunOptions {
    /// How long graceful shutdown waits for in-flight event tasks.
    pub drain_timeout: Duration,
}

impl Default for BotRunOptions {
    fn default() -> Self {
        Self {
            drain_timeout: DEFAULT_SHUTDOWN_DRAIN_TIMEOUT,
        }
    }
}

/// A tool binding from one agent to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentBinding {
    /// Target agent name.
    pub agent: String,
    /// Tool description shown to the parent model.
    pub description: String,
}

/// Platform-neutral bot runtime.
#[derive(Debug, Clone)]
pub struct BotRuntime<P, S, M, L, I, V, A, E> {
    platforms: P,
    storage: S,
    media_store: M,
    llms: L,
    images: I,
    videos: V,
    audio: A,
    events: E,
    background: TaskTracker,
    turn_cancellations: TurnCancellations,
    download_http: reqwest::Client,
    config: BotConfig,
    memory_config: memory::MemoryConfig,
}

/// Runtime service implementations supplied by the binary crate.
#[derive(Debug, Clone)]
pub struct BotRuntimeParts<P, S, M, L, I, V, A, E> {
    /// Message platform registry.
    pub platforms: P,
    /// Durable bot storage.
    pub storage: S,
    /// Media storage.
    pub media_store: M,
    /// LLM provider registry.
    pub llms: L,
    /// Image generation registry.
    pub images: I,
    /// Video generation registry.
    pub videos: V,
    /// Audio transcription registry.
    pub audio: A,
    /// Live event sink.
    pub events: E,
    /// User-memory runtime configuration.
    pub memory: memory::MemoryConfig,
}

#[derive(Debug, Clone, Default)]
struct TurnCancellations {
    inner: Arc<Mutex<BTreeMap<ConversationId, BTreeMap<TurnId, CancellationToken>>>>,
}

impl TurnCancellations {
    fn register(&self, conversation_id: ConversationId, turn_id: TurnId) -> TurnCancellationGuard {
        let token = CancellationToken::new();
        self.inner
            .lock()
            .expect("turn cancellation mutex poisoned")
            .entry(conversation_id)
            .or_default()
            .insert(turn_id, token.clone());
        TurnCancellationGuard {
            registry: self.clone(),
            conversation_id,
            turn_id,
            token,
        }
    }

    fn unregister(&self, conversation_id: ConversationId, turn_id: TurnId) {
        let mut inner = self.inner.lock().expect("turn cancellation mutex poisoned");
        if let Some(turns) = inner.get_mut(&conversation_id) {
            turns.remove(&turn_id);
            if turns.is_empty() {
                inner.remove(&conversation_id);
            }
        }
    }

    fn cancel_conversation(&self, conversation_id: ConversationId) -> usize {
        let tokens = self
            .inner
            .lock()
            .expect("turn cancellation mutex poisoned")
            .get(&conversation_id)
            .map(|turns| turns.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let count = tokens.len();
        for token in tokens {
            token.cancel();
        }
        count
    }
}

#[derive(Debug)]
struct TurnCancellationGuard {
    registry: TurnCancellations,
    conversation_id: ConversationId,
    turn_id: TurnId,
    token: CancellationToken,
}

impl TurnCancellationGuard {
    fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

impl Drop for TurnCancellationGuard {
    fn drop(&mut self) {
        self.registry.unregister(self.conversation_id, self.turn_id);
    }
}

#[derive(Debug)]
struct TypingIndicator {
    stop: CancellationToken,
    task: JoinHandle<()>,
}

impl TypingIndicator {
    async fn stop(self) {
        self.stop.cancel();
        if let Err(error) = self.task.await {
            log_task_join_error("typing indicator", &error);
        }
    }
}

impl<P, S, M, L, I, V, A, E> BotRuntime<P, S, M, L, I, V, A, E> {
    /// Construct a bot runtime.
    pub fn new(parts: BotRuntimeParts<P, S, M, L, I, V, A, E>, config: BotConfig) -> Self {
        tracing::debug!(
            agents = config.agents.len(),
            platforms = config.platforms.len(),
            default_agent = %config.default_agent,
            "constructing bot runtime"
        );
        Self {
            platforms: parts.platforms,
            storage: parts.storage,
            media_store: parts.media_store,
            llms: parts.llms,
            images: parts.images,
            videos: parts.videos,
            audio: parts.audio,
            events: parts.events,
            background: TaskTracker::new(),
            turn_cancellations: TurnCancellations::default(),
            download_http: reqwest::Client::new(),
            config,
            memory_config: parts.memory,
        }
    }

    /// Borrow the bot config.
    pub fn config(&self) -> &BotConfig {
        &self.config
    }
}

impl<P, S, M, L, I, V, A, E> BotRuntime<P, S, M, L, I, V, A, E>
where
    P: MessagePlatformRegistry + Clone + 'static,
    S: BotStorage + Clone + 'static,
    M: MediaStore + Clone + 'static,
    L: LlmProviderRegistry + Clone + 'static,
    I: ImageGeneratorRegistry + Clone + 'static,
    V: VideoGeneratorRegistry + Clone + 'static,
    A: AudioTranscriberRegistry + Clone + 'static,
    E: EventSink + Clone + 'static,
{
    /// Run the platform event loop until the registry emits shutdown.
    #[tracing::instrument(
        name = "bot.run",
        skip_all,
        fields(
            agents = self.config.agents.len(),
            platforms = self.config.platforms.len(),
            default_agent = %self.config.default_agent,
        )
    )]
    pub async fn run(&self) -> Result<(), BotError> {
        self.run_until_shutdown(CancellationToken::new()).await
    }

    /// Run the platform event loop until the registry emits shutdown or the
    /// supplied shutdown token is cancelled.
    pub async fn run_until_shutdown(&self, shutdown: CancellationToken) -> Result<(), BotError> {
        self.run_with_options(shutdown, BotRunOptions::default())
            .await
    }

    /// Run the platform event loop with explicit shutdown behavior.
    #[tracing::instrument(
        name = "bot.run_with_options",
        skip_all,
        fields(
            agents = self.config.agents.len(),
            platforms = self.config.platforms.len(),
            default_agent = %self.config.default_agent,
            drain_timeout_ms = options.drain_timeout.as_millis(),
        )
    )]
    pub async fn run_with_options(
        &self,
        shutdown: CancellationToken,
        options: BotRunOptions,
    ) -> Result<(), BotError> {
        self.platforms
            .register_commands(command_definitions())
            .await
            .map_err(platform_error)?;
        let memory_shutdown = shutdown.child_token();
        self.spawn_memory_runtime(memory_shutdown.clone());
        tracing::info!("bot event loop starting");
        let mut tasks = JoinSet::new();
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    tracing::info!("bot shutdown requested; stopping platform event intake");
                    break;
                }
                Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                    log_event_task_result(result);
                }
                event = self.platforms.next_event() => {
                    let event = match event {
                        Ok(event) => event,
                        Err(error) => {
                            memory_shutdown.cancel();
                            return Err(platform_error(error));
                        }
                    };
                    tracing::trace!(
                        event = platform_event_kind(&event),
                        "received platform event"
                    );
                    if matches!(event, PlatformEvent::Shutdown) {
                        tracing::info!("platform event stream requested shutdown");
                        break;
                    }
                    let runtime = (*self).clone();
                    tasks.spawn(async move {
                        let event_name = platform_event_kind(&event);
                        let result = runtime.handle_event(event).await;
                        (event_name, result)
                    });
                }
            }
        }

        memory_shutdown.cancel();
        drain_event_tasks(&mut tasks, options.drain_timeout).await;
        drain_background_tasks(&self.background, options.drain_timeout).await;
        self.platforms.shutdown().await.map_err(platform_error)?;
        tracing::info!("bot event loop stopped");
        Ok(())
    }

    /// Handle one platform event.
    #[tracing::instrument(
        name = "bot.handle_event",
        skip_all,
        fields(event = platform_event_kind(&event))
    )]
    pub async fn handle_event(&self, event: PlatformEvent) -> Result<BotAction, BotError> {
        let action = match event {
            PlatformEvent::Ready { .. } => Ok(BotAction::Ignored),
            PlatformEvent::MessageCreated { message } => self.handle_message(message).await,
            PlatformEvent::ReactionAdded { reaction } => {
                self.handle_reaction(reaction, false).await
            }
            PlatformEvent::ReactionRemoved { reaction } => {
                self.handle_reaction(reaction, true).await
            }
            PlatformEvent::Command { command } => self.handle_command(command).await,
            PlatformEvent::Shutdown => Ok(BotAction::Shutdown),
        };
        match &action {
            Ok(action) => {
                tracing::debug!(action = bot_action_kind(action), "platform event handled")
            }
            Err(error) => tracing::warn!(error = %error, "platform event handling failed"),
        }
        action
    }

    /// Handle one incoming message.
    #[tracing::instrument(
        name = "bot.handle_message",
        skip_all,
        fields(
            platform = %message.id.platform,
            guild = ?message.id.guild_id,
            channel = %message.id.channel_id,
            message = %message.id.message_id,
            author = %message.author.id.user_id,
            mentions = message.mentions.len(),
            attachments = message.attachments.len(),
            content_chars = message.content.chars().count(),
            conversation = tracing::field::Empty,
            turn = tracing::field::Empty,
            agent = tracing::field::Empty,
            provider = tracing::field::Empty,
            model = tracing::field::Empty,
            is_new = tracing::field::Empty,
        )
    )]
    pub async fn handle_message(
        &self,
        mut message: PlatformMessage,
    ) -> Result<BotAction, BotError> {
        let referenced = message.referenced_message_id();
        tracing::debug!(
            author_is_bot = message.author.is_bot,
            mentions = message.mentions.len(),
            mention_profiles = message.mention_profiles.len(),
            attachments = message.attachments.len(),
            content_chars = message.content.chars().count(),
            reference_kind = platform_message_reference_kind(&message.reference),
            reference_platform = ?referenced.map(|message| message.platform.as_str()),
            reference_guild = ?referenced
                .and_then(|message| message.guild_id.as_ref().map(ExternalId::as_str)),
            reference_channel = ?referenced.map(|message| message.channel_id.as_str()),
            reference_message = ?referenced.map(|message| message.message_id.as_str()),
            has_hydrated_reference = message.referenced_message().is_some(),
            "received platform message"
        );
        let bot_user = self
            .platforms
            .bot_user(&message.id.platform)
            .await
            .map_err(platform_error)?;
        if message.author.is_bot
            || !message
                .mentions
                .iter()
                .any(|user| same_platform_user(user, &bot_user.id))
        {
            tracing::debug!(
                author_is_bot = message.author.is_bot,
                mentioned_bot = message
                    .mentions
                    .iter()
                    .any(|user| same_platform_user(user, &bot_user.id)),
                "ignoring message"
            );
            return Ok(BotAction::Ignored);
        }
        message.content = normalize_mention_content(
            &message.content,
            &bot_user.id,
            &message.mentions,
            &message.mention_profiles,
        );

        self.storage
            .upsert_user(message.author.clone())
            .await
            .map_err(storage_error)?;
        self.spawn_avatar_download(message.author.clone());
        self.publish_user(message.author.id.clone());

        let settings = self.runtime_settings(&message).await?;
        tracing::debug!(
            privacy = privacy_mode_kind(&settings.privacy),
            user_opted_in = settings.user_opted_in,
            "loaded runtime settings"
        );

        let existing = self.lookup_existing_conversation(&message).await?;
        if !self
            .privacy_allows_message_channel(&settings.privacy, &message.id, existing.as_ref())
            .await?
        {
            tracing::debug!(
                privacy = privacy_mode_kind(&settings.privacy),
                "privacy mode rejected message channel"
            );
            return Ok(BotAction::Ignored);
        }
        if let Some(snapshot) = &existing
            && snapshot.conversation.stopped_at.is_some()
        {
            tracing::debug!(
                conversation = %snapshot.conversation.id,
                stopped_at = ?snapshot.conversation.stopped_at,
                "ignoring message because conversation is stopped"
            );
            return Ok(BotAction::Ignored);
        }

        let user_message = message.id.clone();
        self.add_unicode_reaction(&user_message, WORKING_REACTION, "turn_working")
            .await;
        let action = self
            .process_mentioned_message(message, existing, settings)
            .await;
        self.remove_own_unicode_reaction(&user_message, WORKING_REACTION, "turn_working")
            .await;
        self.react_for_action(&user_message, &action).await;
        action
    }

    async fn process_mentioned_message(
        &self,
        message: PlatformMessage,
        existing: Option<ConversationSnapshot>,
        settings: RuntimeSettings,
    ) -> Result<BotAction, BotError> {
        let user_display_name = display_name(&message);
        if !self.moderation_allows(&message, &user_display_name).await? {
            tracing::info!("message refused by moderation preflight");
            return Ok(BotAction::RefusedMessage);
        }

        let resolved_agent = self
            .storage
            .resolve_agent(ResolveAgent {
                message_provider: message.id.platform.clone(),
                conversation_id: existing.as_ref().map(|s| s.conversation.id),
                guild_key: guild_key(&message.id),
                channel_key: self
                    .agent_scope_channel(&message.id)
                    .await
                    .channel_id
                    .as_str()
                    .to_string(),
                user_key: message.author.id.user_id.as_str().to_string(),
            })
            .await
            .map_err(storage_error)?;
        let (agent_name, agent_config) = self
            .config
            .agent_or_platform_default(resolved_agent.as_deref(), &message.id.platform)?;
        let agent_config = agent_config.clone();
        self.ensure_agent_services_exist(&agent_name, &agent_config)?;
        tracing::Span::current().record("agent", tracing::field::display(&agent_name));
        tracing::Span::current()
            .record("provider", tracing::field::display(&agent_config.provider));
        tracing::Span::current().record("model", tracing::field::display(&agent_config.model.id));
        tracing::debug!(
            resolved_agent = %agent_name,
            storage_agent = ?resolved_agent,
            provider = %agent_config.provider,
            model = %agent_config.model.id,
            "resolved agent for turn"
        );

        let system_instructions = self.compose_system_prompt(&agent_config, &settings.privacy);

        let (snapshot, is_new) = match existing {
            Some(snapshot) => (snapshot, false),
            None => {
                let snapshot = self
                    .storage
                    .open_conversation(OpenConversation {
                        channel: channel_from_message(&message.id),
                        created_by: message.author.id.clone(),
                        root_message: message.id.clone(),
                        initial_model: agent_config.model.id.clone(),
                        agent_name: agent_name.clone(),
                        provider: agent_config.provider.clone(),
                        system_instructions: system_instructions.clone(),
                        title: None,
                    })
                    .await
                    .map_err(storage_error)?;
                self.publish_conversation(snapshot.conversation.id, ConversationEventKind::Created);
                tracing::info!(
                    conversation = %snapshot.conversation.id,
                    "opened new conversation"
                );
                (snapshot, true)
            }
        };
        tracing::Span::current().record(
            "conversation",
            tracing::field::display(snapshot.conversation.id),
        );
        tracing::Span::current().record("is_new", is_new);

        let turn = self
            .storage
            .begin_turn(BeginTurn {
                conversation_id: snapshot.conversation.id,
                user_message: message.id.clone(),
                user_message_created_at: message.created_at,
                user: message.author.id.clone(),
                user_display_name: display_name(&message),
                user_content: message.content.clone(),
            })
            .await
            .map_err(storage_error)?;
        tracing::Span::current().record("turn", tracing::field::display(turn.id));
        self.publish_conversation(snapshot.conversation.id, ConversationEventKind::TurnStarted);
        tracing::info!(
            conversation = %snapshot.conversation.id,
            turn = %turn.id,
            turn_ordinal = turn.ordinal,
            history_cutoff = ?turn.history_cutoff,
            is_new,
            "started turn"
        );

        self.storage
            .link_message(MessageLink {
                message: message.id.clone(),
                conversation_id: snapshot.conversation.id,
                turn_id: turn.id,
                role: "user".to_string(),
            })
            .await
            .map_err(storage_error)?;
        tracing::debug!("linked user message to turn");

        let turn_context = self
            .prepare_turn_context(&message, &settings, &snapshot.conversation)
            .await?;
        let prompt_snapshot = self
            .storage
            .load_conversation(ConversationLookup::Id {
                id: snapshot.conversation.id,
            })
            .await
            .map_err(storage_error)?
            .ok_or(BotError::MissingConversation {
                conversation_id: snapshot.conversation.id,
            })?;
        let transcript = self
            .transcript_for_turn(&prompt_snapshot, &turn, &turn_context.items)
            .await?;
        tracing::debug!(
            transcript_turns = transcript.turns.len(),
            system_instructions_chars = system_instructions.chars().count(),
            "assembled model transcript"
        );
        self.storage
            .save_turn_input(SaveTurnInput {
                turn_id: turn.id,
                agent_name: agent_name.clone(),
                provider: agent_config.provider.clone(),
                model: agent_config.model.id.clone(),
                system_instructions: system_instructions.clone(),
                context: turn_context.items,
                transcript: Some(transcript.clone()),
            })
            .await
            .map_err(storage_error)?;
        self.publish_conversation(
            snapshot.conversation.id,
            ConversationEventKind::ContextRecorded,
        );
        tracing::debug!("saved turn input");

        self.execute_turn(TurnExecution {
            conversation: prompt_snapshot.conversation,
            turn,
            agent_name,
            agent_config,
            system_prompt: system_instructions,
            transcript,
            settings,
            reply_to: message.id,
            is_new,
        })
        .await
    }

    #[tracing::instrument(
        name = "bot.handle_reaction",
        skip_all,
        fields(
            platform = %reaction.message.platform,
            guild = ?reaction.message.guild_id,
            channel = %reaction.message.channel_id,
            message = %reaction.message.message_id,
            user = %reaction.user.user_id,
            removed,
        )
    )]
    async fn handle_reaction(
        &self,
        reaction: PlatformReaction,
        removed: bool,
    ) -> Result<BotAction, BotError> {
        let bot_user = self
            .platforms
            .bot_user(&reaction.message.platform)
            .await
            .map_err(platform_error)?;
        if same_platform_user(&reaction.user, &bot_user.id) {
            tracing::debug!("ignoring bot's own reaction");
            return Ok(BotAction::Ignored);
        }
        let ReactionKind::Unicode { name } = &reaction.reaction else {
            tracing::debug!("ignoring non-unicode reaction");
            return Ok(BotAction::Ignored);
        };
        tracing::debug!(reaction = %name, "handling unicode reaction");

        match (name.as_str(), removed) {
            (RETRY_REACTION, false) => self.retry_from_message(reaction.message).await,
            (STOP_REACTION, _) => {
                if !self.is_admin(&reaction.user) {
                    tracing::debug!("stop reaction ignored because user is not configured admin");
                    return Ok(BotAction::Ignored);
                }
                self.set_stop(reaction.message, reaction.user, !removed)
                    .await
            }
            _ => {
                tracing::debug!(reaction = %name, "ignoring reaction");
                Ok(BotAction::Ignored)
            }
        }
    }

    #[tracing::instrument(
        name = "bot.retry_from_message",
        skip_all,
        fields(
            platform = %message.platform,
            guild = ?message.guild_id,
            channel = %message.channel_id,
            message = %message.message_id,
            conversation = tracing::field::Empty,
            turn = tracing::field::Empty,
            agent = tracing::field::Empty,
        )
    )]
    async fn retry_from_message(&self, message: MessageRef) -> Result<BotAction, BotError> {
        let Some(link) = self
            .storage
            .load_message_link(message)
            .await
            .map_err(storage_error)?
        else {
            tracing::debug!("retry ignored because message has no link");
            return Ok(BotAction::Ignored);
        };
        tracing::Span::current().record(
            "conversation",
            tracing::field::display(link.conversation_id),
        );
        tracing::Span::current().record("turn", tracing::field::display(link.turn_id));
        let prior_links = self
            .storage
            .load_message_links_for_turn(link.turn_id)
            .await
            .map_err(storage_error)?;
        if self
            .storage
            .load_conversation(ConversationLookup::Id {
                id: link.conversation_id,
            })
            .await
            .map_err(storage_error)?
            .as_ref()
            .is_some_and(|snapshot| snapshot.conversation.stopped_at.is_some())
        {
            tracing::info!("retry ignored because conversation is stopped");
            return Ok(BotAction::Ignored);
        }
        let Some(retry) = self
            .storage
            .prepare_retry(link.turn_id)
            .await
            .map_err(storage_error)?
        else {
            tracing::debug!("retry ignored because turn is not eligible");
            return Ok(BotAction::Ignored);
        };
        let Some(turn_snapshot) = retry
            .conversation
            .turns
            .iter()
            .find(|snapshot| snapshot.turn.id == retry.turn_id)
        else {
            return Err(BotError::MissingRetryTurn {
                turn_id: retry.turn_id,
            });
        };
        let turn = turn_snapshot.turn.clone();
        if retry.conversation.conversation.stopped_at.is_some() {
            tracing::info!("retry ignored because conversation is stopped");
            return Ok(BotAction::Ignored);
        }
        let (agent_name, agent_config) = self
            .config
            .agent_or_platform_default(turn.agent_name.as_deref(), &turn.user_message.platform)?;
        let agent_config = agent_config.clone();
        self.ensure_agent_services_exist(&agent_name, &agent_config)?;
        tracing::Span::current().record("agent", tracing::field::display(&agent_name));
        tracing::debug!(
            provider = %agent_config.provider,
            model = %agent_config.model.id,
            "prepared turn retry"
        );
        let settings = RuntimeSettings {
            privacy: PrivacyMode::ConversationOnly,
            user_opted_in: true,
        };
        let system_instructions = turn_snapshot
            .system_instructions
            .clone()
            .unwrap_or_else(|| self.compose_system_prompt(&agent_config, &settings.privacy));
        let stored_context = replayable_context_items(&turn_snapshot.context);
        let has_stored_context = !stored_context.is_empty();
        let transcript = self
            .transcript_for_retry(
                &retry.conversation,
                turn_snapshot,
                &stored_context,
                has_stored_context,
            )
            .await?;
        self.storage
            .save_turn_input(SaveTurnInput {
                turn_id: turn.id,
                agent_name: agent_name.clone(),
                provider: agent_config.provider.clone(),
                model: agent_config.model.id.clone(),
                system_instructions: system_instructions.clone(),
                context: stored_context,
                transcript: Some(transcript.clone()),
            })
            .await
            .map_err(storage_error)?;
        self.publish_conversation(
            retry.conversation.conversation.id,
            ConversationEventKind::ContextRecorded,
        );

        for link in prior_links
            .iter()
            .filter(|link| link.role.starts_with("assistant"))
        {
            if let Err(error) = self.platforms.delete_message(link.message.clone()).await {
                tracing::warn!(
                    error = %error,
                    message = %link.message.message_id,
                    "failed to delete prior failed reply during retry"
                );
            }
        }

        let retry_user_message = turn.user_message.clone();
        self.remove_own_unicode_reaction(&retry_user_message, ERROR_REACTION, "retry_error_clear")
            .await;
        self.add_unicode_reaction(&retry_user_message, WORKING_REACTION, "retry_working")
            .await;
        let action = self
            .execute_turn(TurnExecution {
                conversation: retry.conversation.conversation,
                turn,
                agent_name,
                agent_config,
                system_prompt: system_instructions,
                transcript,
                settings,
                reply_to: retry_user_message.clone(),
                is_new: false,
            })
            .await;
        self.remove_own_unicode_reaction(&retry_user_message, WORKING_REACTION, "retry_working")
            .await;
        self.react_for_action(&retry_user_message, &action).await;
        action
    }

    #[tracing::instrument(
        name = "bot.set_stop",
        skip_all,
        fields(
            platform = %message.platform,
            guild = ?message.guild_id,
            channel = %message.channel_id,
            message = %message.message_id,
            user = %user.user_id,
            stop,
            conversation = tracing::field::Empty,
        )
    )]
    async fn set_stop(
        &self,
        message: MessageRef,
        user: chudbot_api::UserRef,
        stop: bool,
    ) -> Result<BotAction, BotError> {
        let snapshot = self
            .storage
            .load_conversation(ConversationLookup::Message {
                message: message.clone(),
            })
            .await
            .map_err(storage_error)?;
        let snapshot = match snapshot {
            Some(snapshot) => Some(snapshot),
            None => self
                .storage
                .load_conversation(ConversationLookup::Channel {
                    channel: channel_from_message(&message),
                })
                .await
                .map_err(storage_error)?,
        };
        let Some(snapshot) = snapshot else {
            tracing::debug!("stop/resume ignored because message maps to no conversation");
            return Ok(BotAction::Ignored);
        };
        let conversation_id = snapshot.conversation.id;
        tracing::Span::current().record("conversation", tracing::field::display(conversation_id));
        let changed = self
            .storage
            .set_conversation_stop(if stop {
                ConversationStop::Stop {
                    conversation_id,
                    stopped_by: user,
                }
            } else {
                ConversationStop::Resume { conversation_id }
            })
            .await
            .map_err(storage_error)?;
        if changed {
            self.publish_conversation(conversation_id, ConversationEventKind::ConversationUpdated);
            if stop {
                let cancelled = self.turn_cancellations.cancel_conversation(conversation_id);
                if cancelled > 0 {
                    tracing::info!(
                        cancelled,
                        "cancelled in-flight turn(s) for stopped conversation"
                    );
                }
            }
            tracing::info!(changed, "conversation stop state updated");
        } else {
            tracing::debug!("conversation stop state was unchanged");
        }
        Ok(if stop {
            BotAction::StoppedConversation { conversation_id }
        } else {
            BotAction::ResumedConversation { conversation_id }
        })
    }

    #[tracing::instrument(
        name = "bot.execute_turn",
        skip_all,
        fields(
            conversation = %execution.conversation.id,
            turn = %execution.turn.id,
            turn_ordinal = execution.turn.ordinal,
            history_cutoff = ?execution.turn.history_cutoff,
            response_ordinal = ?execution.turn.response_ordinal,
            agent = %execution.agent_name,
            provider = %execution.agent_config.provider,
            model = %execution.agent_config.model.id,
            transcript_turns = execution.transcript.turns.len(),
            is_new = execution.is_new,
        )
    )]
    async fn execute_turn(&self, mut execution: TurnExecution) -> Result<BotAction, BotError> {
        if self
            .storage
            .load_conversation(ConversationLookup::Id {
                id: execution.conversation.id,
            })
            .await
            .map_err(storage_error)?
            .as_ref()
            .is_some_and(|snapshot| snapshot.conversation.stopped_at.is_some())
        {
            tracing::info!("turn cancelled because conversation is stopped before execution");
            self.storage
                .finish_turn(FinishTurn::Cancelled {
                    turn_id: execution.turn.id,
                    reason: "cancelled by admin stop reaction".to_string(),
                    usage: Vec::new(),
                })
                .await
                .map_err(storage_error)?;
            self.publish_conversation(
                execution.conversation.id,
                ConversationEventKind::TurnUpdated,
            );
            return Ok(BotAction::CancelledTurn {
                conversation_id: execution.conversation.id,
                turn_id: execution.turn.id,
            });
        }
        tracing::debug!("building agent for turn");
        let agent = self.build_agent(
            &execution.agent_name,
            &execution.agent_config,
            execution.system_prompt.clone(),
            &execution.settings,
            &execution.reply_to,
            &execution.turn.user,
            &execution.turn.user_display_name,
            execution.conversation.id,
            execution.turn.id,
            true,
            &mut Vec::new(),
        )?;
        let transcript = std::mem::take(&mut execution.transcript);
        let replayed_media_refs = media_reply_refs_from_transcript(&transcript).await;
        tracing::info!("running agent");
        let cancel_guard = self
            .turn_cancellations
            .register(execution.conversation.id, execution.turn.id);
        let cancel_token = cancel_guard.token();
        let typing = self.spawn_typing_indicator(channel_from_message(&execution.reply_to));
        let run = tokio::select! {
            biased;
            () = cancel_token.cancelled() => {
                tracing::info!("turn cancelled before agent completed");
                None
            }
            run = agent.run(transcript) => Some(run),
        };
        typing.stop().await;
        let Some(run) = run else {
            self.storage
                .finish_turn(FinishTurn::Cancelled {
                    turn_id: execution.turn.id,
                    reason: "cancelled by admin stop reaction".to_string(),
                    usage: Vec::new(),
                })
                .await
                .map_err(storage_error)?;
            self.publish_conversation(
                execution.conversation.id,
                ConversationEventKind::TurnUpdated,
            );
            return Ok(BotAction::CancelledTurn {
                conversation_id: execution.conversation.id,
                turn_id: execution.turn.id,
            });
        };
        let run = match run {
            Ok(run) => run,
            Err(error) => {
                tracing::warn!(error = %error, "agent failed before producing run output");
                let message = error.to_string();
                if error_indicates_safety_refusal(&message) {
                    return self
                        .refuse_turn(&execution, "refused by upstream safety")
                        .await;
                }
                return self
                    .fail_turn(&execution, format!("model failed: {message}"))
                    .await;
            }
        };
        drop(cancel_guard);
        tracing::debug!(
            outcome = agent_outcome_kind(&run.outcome),
            trace_events = run.trace.len(),
            usage_records = run.all_usage().len(),
            last_model = ?run.last_model_id,
            has_continuation = run.final_continuation.is_some(),
            "agent run completed"
        );

        for (ordinal, trace) in run.trace.iter().cloned().enumerate() {
            let trace_kind = tool_trace_kind(&trace);
            self.storage
                .append_tool_trace(
                    execution.turn.id,
                    i32::try_from(ordinal).unwrap_or(i32::MAX),
                    trace,
                )
                .await
                .map_err(storage_error)?;
            self.publish_conversation(
                execution.conversation.id,
                ConversationEventKind::ToolTraceRecorded,
            );
            tracing::trace!(ordinal, trace_kind, "recorded tool trace");
        }

        if safety_refusal_in_tool_trace(&run.trace) {
            tracing::info!("turn refused by upstream safety in a client tool");
            return self
                .refuse_turn(&execution, "refused by upstream safety")
                .await;
        }

        let mut generated_media_refs = generated_media_reply_refs(&run.trace);
        for reference in replayed_media_refs {
            if !generated_media_refs.iter().any(|seen| seen == &reference) {
                generated_media_refs.push(reference);
            }
        }
        let generated_attachments = self.generated_attachments(&run.trace).await;

        match &run.outcome {
            AgentOutcome::Completed { answer } => {
                let text = strip_generated_media_refs(&answer.text, &generated_media_refs);
                let text = if text.trim().is_empty() {
                    "Done.".to_string()
                } else {
                    text
                };
                let content = self.format_reply(&text, execution.is_new, execution.conversation.id);
                let rendered_lines = rendered_line_count(&content);
                let open_thread = should_thread(
                    execution.is_new,
                    &content,
                    self.config.thread_threshold_chars,
                    self.config.thread_threshold_lines,
                )
                .then(|| ThreadRequest {
                    title: thread_title(&execution),
                });
                let posted = self
                    .platforms
                    .send_message(SendMessage {
                        channel: channel_from_message(&execution.reply_to),
                        reply_to: Some(execution.reply_to.clone()),
                        content: content.clone(),
                        attachments: generated_attachments,
                        suppress_embeds: true,
                        open_thread,
                    })
                    .await
                    .map_err(platform_error)?;
                tracing::info!(
                    reply_message = %posted.id.message_id,
                    reply_channel = %posted.channel.channel_id,
                    answer_chars = content.chars().count(),
                    rendered_lines,
                    thread_threshold_chars = self.config.thread_threshold_chars,
                    thread_threshold_lines = self.config.thread_threshold_lines,
                    opened_thread = posted.channel != channel_from_message(&execution.reply_to),
                    "posted assistant reply"
                );
                self.storage
                    .link_message(MessageLink {
                        message: posted.id.clone(),
                        conversation_id: execution.conversation.id,
                        turn_id: execution.turn.id,
                        role: "assistant".to_string(),
                    })
                    .await
                    .map_err(storage_error)?;
                for message in &posted.extra_messages {
                    self.storage
                        .link_message(MessageLink {
                            message: message.clone(),
                            conversation_id: execution.conversation.id,
                            turn_id: execution.turn.id,
                            role: "assistant".to_string(),
                        })
                        .await
                        .map_err(storage_error)?;
                }
                if posted.channel != channel_from_message(&execution.reply_to) {
                    self.storage
                        .link_channel(ChannelLink {
                            channel: posted.channel.clone(),
                            conversation_id: execution.conversation.id,
                            turn_id: execution.turn.id,
                            role: "thread".to_string(),
                        })
                        .await
                        .map_err(storage_error)?;
                    tracing::debug!(
                        thread_channel = %posted.channel.channel_id,
                        "linked thread channel to conversation"
                    );
                }
                self.storage
                    .finish_turn(FinishTurn::Completed {
                        turn_id: execution.turn.id,
                        assistant_content: content,
                        assistant_message: posted.id,
                        continuation: run.final_continuation.clone(),
                        usage: run.all_usage(),
                    })
                    .await
                    .map_err(storage_error)?;
                self.publish_conversation(
                    execution.conversation.id,
                    ConversationEventKind::TurnUpdated,
                );
                if execution.turn.ordinal == 0 && execution.conversation.title.is_none() {
                    self.spawn_title_generation(
                        execution.conversation.id,
                        execution.agent_name.clone(),
                    );
                }
                tracing::info!("turn completed");
                Ok(BotAction::CompletedTurn {
                    conversation_id: execution.conversation.id,
                    turn_id: execution.turn.id,
                })
            }
            AgentOutcome::IterationLimit { max_iterations } => {
                tracing::warn!(max_iterations, "turn hit agent iteration limit");
                self.fail_turn(
                    &execution,
                    format!("model hit iteration limit ({max_iterations})"),
                )
                .await
            }
            AgentOutcome::Failed { error, partial } => {
                tracing::warn!(
                    error = %error,
                    has_partial = partial.is_some(),
                    "agent reported failed outcome"
                );
                let mut message = error.to_string();
                if let Some(partial) = partial
                    && !partial.text.trim().is_empty()
                {
                    message.push_str("\n\nPartial answer:\n");
                    message.push_str(&partial.text);
                }
                self.fail_turn(&execution, message).await
            }
            AgentOutcome::Cancelled { reason } => {
                tracing::info!(reason = %reason, "turn cancelled");
                self.storage
                    .finish_turn(FinishTurn::Cancelled {
                        turn_id: execution.turn.id,
                        reason: reason.clone(),
                        usage: run.all_usage(),
                    })
                    .await
                    .map_err(storage_error)?;
                self.publish_conversation(
                    execution.conversation.id,
                    ConversationEventKind::TurnUpdated,
                );
                Ok(BotAction::CancelledTurn {
                    conversation_id: execution.conversation.id,
                    turn_id: execution.turn.id,
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        name = "bot.build_agent",
        skip_all,
        fields(
            agent = %agent_name,
            provider = %agent_config.provider,
            model = %agent_config.model.id,
            top_level,
            stack_depth = stack.len(),
            subagents = agent_config.subagents.len(),
        )
    )]
    fn build_agent(
        &self,
        agent_name: &str,
        agent_config: &AgentConfig,
        system_prompt: String,
        settings: &RuntimeSettings,
        reply_to: &MessageRef,
        turn_user: &UserRef,
        turn_user_display_name: &str,
        conversation_id: ConversationId,
        turn_id: TurnId,
        top_level: bool,
        stack: &mut Vec<String>,
    ) -> Result<Agent<RoutedLlmBackend<L>>, BotError> {
        self.ensure_agent_services_exist(agent_name, agent_config)?;
        if stack.iter().any(|name| name == agent_name) {
            tracing::warn!("recursive agent reference detected");
            return Err(BotError::RecursiveAgent {
                name: agent_name.to_string(),
            });
        }
        stack.push(agent_name.to_string());

        let mut spec = AgentSpec::new(system_prompt)
            .with_limits(agent_config.limits.unwrap_or(self.config.limits));
        spec.server_tools = agent_config.server_tools.clone();
        spec.client_tools = agent_config.client_tools.clone();
        if top_level && self.agent_memory_enabled(agent_config) {
            ensure_client_tool_enabled(&mut spec.client_tools, memory::LOOKUP_USER_MEMORY_TOOL);
            ensure_client_tool_enabled(&mut spec.client_tools, memory::REMEMBER_USER_MEMORY_TOOL);
            ensure_client_tool_enabled(&mut spec.client_tools, memory::FORGET_USER_MEMORY_TOOL);
        }

        let mut tools: Vec<RuntimeTool<P, S, M, L, I, V, A>> = Vec::new();
        if top_level {
            if !matches!(settings.privacy, PrivacyMode::ConversationOnly) {
                tracing::debug!(tool = FETCH_MESSAGES_TOOL, "attaching runtime tool");
                tools.push(RuntimeTool::Fetch {
                    name: ToolName::new(FETCH_MESSAGES_TOOL),
                    tool: FetchMessagesTool {
                        platforms: self.platforms.clone(),
                        storage: self.storage.clone(),
                        default_channel: channel_from_message(reply_to),
                        privacy: settings.privacy.clone(),
                    },
                });
            }
            if self.agent_memory_enabled(agent_config) {
                let base_key = memory::key_from_user_ref(turn_user);
                tracing::debug!("attaching user memory tools");
                tools.push(RuntimeTool::Memory {
                    name: ToolName::new(memory::LOOKUP_USER_MEMORY_TOOL),
                    tool: memory::MemoryClientTool::new(
                        self.storage.clone(),
                        memory::MemoryToolKind::Lookup,
                        base_key.clone(),
                        turn_user_display_name.to_string(),
                        conversation_id,
                        turn_id,
                    ),
                });
                tools.push(RuntimeTool::Memory {
                    name: ToolName::new(memory::REMEMBER_USER_MEMORY_TOOL),
                    tool: memory::MemoryClientTool::new(
                        self.storage.clone(),
                        memory::MemoryToolKind::Remember,
                        base_key.clone(),
                        turn_user_display_name.to_string(),
                        conversation_id,
                        turn_id,
                    ),
                });
                tools.push(RuntimeTool::Memory {
                    name: ToolName::new(memory::FORGET_USER_MEMORY_TOOL),
                    tool: memory::MemoryClientTool::new(
                        self.storage.clone(),
                        memory::MemoryToolKind::Forget,
                        base_key,
                        turn_user_display_name.to_string(),
                        conversation_id,
                        turn_id,
                    ),
                });
            }
            tracing::debug!(tool = POST_STATUS_TOOL, "attaching runtime tool");
            tools.push(RuntimeTool::Status {
                name: ToolName::new(POST_STATUS_TOOL),
                tool: PostStatusTool {
                    platforms: self.platforms.clone(),
                    storage: self.storage.clone(),
                    channel: channel_from_message(reply_to),
                    reply_to: reply_to.clone(),
                    conversation_id,
                    turn_id,
                },
            });
        }

        if let Some(binding) = &agent_config.image_generation {
            tracing::debug!(
                tool = GENERATE_IMAGE_TOOL,
                provider = %binding.provider,
                model = %binding.model,
                "attaching image generation tool"
            );
            tools.push(RuntimeTool::Image {
                name: ToolName::new(GENERATE_IMAGE_TOOL),
                tool: ImageGeneratorTool::new(
                    RoutedImageGenerator::new(
                        self.images.clone(),
                        binding.provider.clone(),
                        binding.model.clone(),
                    ),
                    self.media_store.clone(),
                )
                .with_description(image_generation_tool_description(
                    &binding.provider,
                    &binding.model,
                )),
            });
        }

        if let Some(binding) = &agent_config.video_generation {
            tracing::debug!(
                tool = GENERATE_VIDEO_TOOL,
                provider = %binding.provider,
                model = %binding.model,
                "attaching video generation tool"
            );
            tools.push(RuntimeTool::Video {
                name: ToolName::new(GENERATE_VIDEO_TOOL),
                tool: PersistentVideoGeneratorTool::new(
                    RoutedVideoGenerator::new(
                        self.videos.clone(),
                        binding.provider.clone(),
                        binding.model.clone(),
                    ),
                    self.media_store.clone(),
                    self.storage.clone(),
                    turn_id,
                    binding.provider.clone(),
                )
                .with_description(format!(
                    "Generate a video with the configured `{}` video provider and `{}` model, save it to media storage, and return its media URI.",
                    binding.provider, binding.model
                )),
            });
        }

        if let Some(binding) = &agent_config.audio_transcription {
            tracing::debug!(
                tool = TRANSCRIBE_AUDIO_TOOL,
                provider = %binding.provider,
                model = ?binding.model.as_ref(),
                "attaching audio transcription tool"
            );
            tools.push(RuntimeTool::Audio {
                name: ToolName::new(TRANSCRIBE_AUDIO_TOOL),
                tool: AudioTranscriptionTool::new(
                    RoutedAudioTranscriber::new(
                        self.audio.clone(),
                        binding.provider.clone(),
                        binding.model.clone(),
                    ),
                    self.media_store.clone(),
                )
                .with_description(format!(
                    "Transcribe a stored audio attachment with the configured `{}` audio provider{} and return the speech as text.",
                    binding.provider,
                    binding
                        .model
                        .as_ref()
                        .map(|model| format!(" and `{model}` model"))
                        .unwrap_or_default()
                )),
            });
        }

        for (tool_name, binding) in &agent_config.subagents {
            let (subagent_name, subagent_config) = self
                .config
                .agent_or_platform_default(Some(&binding.agent), &reply_to.platform)?;
            tracing::debug!(
                tool = %tool_name,
                subagent = %subagent_name,
                provider = %subagent_config.provider,
                model = %subagent_config.model.id,
                "attaching subagent tool"
            );
            let prompt = self
                .compose_subagent_system_prompt(subagent_config, &PrivacyMode::ConversationOnly);
            let nested = self.build_agent(
                &subagent_name,
                subagent_config,
                prompt,
                settings,
                reply_to,
                turn_user,
                turn_user_display_name,
                conversation_id,
                turn_id,
                false,
                stack,
            )?;
            tools.push(RuntimeTool::Subagent {
                name: tool_name.clone(),
                tool: nested.into_subagent(binding.description.clone()),
            });
        }

        stack.pop();
        let model = Model {
            backend: RoutedLlmBackend::new(self.llms.clone(), agent_config.provider.clone()),
            spec: agent_config.model.clone(),
        };
        let mut iter = tools.into_iter();
        let Some(first) = iter.next() else {
            tracing::debug!("built agent without client tools");
            return Ok(spec.into_agent(model));
        };
        let mut builder = first.start(spec);
        let mut tool_count = 1usize;
        for tool in iter {
            builder = tool.append(builder);
            tool_count += 1;
        }
        tracing::debug!(client_tools = tool_count, "built agent with client tools");
        Ok(builder.into_agent(model))
    }

    fn utility_agent(
        &self,
        provider: ProviderName,
        model: ModelSpec,
        system_prompt: &'static str,
        limits: AgentLimits,
    ) -> Agent<RoutedLlmBackend<L>> {
        AgentSpec::new(system_prompt)
            .with_limits(limits)
            .into_agent(Model {
                backend: RoutedLlmBackend::new(self.llms.clone(), provider),
                spec: model,
            })
    }

    #[tracing::instrument(
        name = "bot.fail_turn",
        skip_all,
        fields(
            conversation = %execution.conversation.id,
            turn = %execution.turn.id,
            agent = %execution.agent_name,
            error_chars = error.chars().count(),
        )
    )]
    async fn fail_turn(
        &self,
        execution: &TurnExecution,
        error: String,
    ) -> Result<BotAction, BotError> {
        let content = format!("Warning: {error}");
        let posted = self
            .platforms
            .send_message(SendMessage {
                channel: channel_from_message(&execution.reply_to),
                reply_to: Some(execution.reply_to.clone()),
                content,
                attachments: Vec::new(),
                suppress_embeds: true,
                open_thread: None,
            })
            .await
            .map_err(platform_error)?;
        tracing::info!(
            error_message = %posted.id.message_id,
            channel = %posted.channel.channel_id,
            "posted turn failure reply"
        );
        self.storage
            .finish_turn(FinishTurn::Failed {
                turn_id: execution.turn.id,
                error,
                assistant_content: None,
                assistant_message: Some(posted.id.clone()),
                usage: Vec::new(),
            })
            .await
            .map_err(storage_error)?;
        self.storage
            .link_message(MessageLink {
                message: posted.id.clone(),
                conversation_id: execution.conversation.id,
                turn_id: execution.turn.id,
                role: "assistant_error".to_string(),
            })
            .await
            .map_err(storage_error)?;
        for message in &posted.extra_messages {
            self.storage
                .link_message(MessageLink {
                    message: message.clone(),
                    conversation_id: execution.conversation.id,
                    turn_id: execution.turn.id,
                    role: "assistant_error".to_string(),
                })
                .await
                .map_err(storage_error)?;
        }
        if let Err(error) = self
            .platforms
            .add_reaction(
                posted.id,
                ReactionKind::Unicode {
                    name: RETRY_REACTION.to_string(),
                },
            )
            .await
        {
            tracing::warn!(error = %error, "failed to add retry reaction to failed reply");
        }
        self.publish_conversation(
            execution.conversation.id,
            ConversationEventKind::TurnUpdated,
        );
        tracing::warn!("turn marked failed");
        Ok(BotAction::FailedTurn {
            conversation_id: execution.conversation.id,
            turn_id: execution.turn.id,
        })
    }

    #[tracing::instrument(
        name = "bot.refuse_turn",
        skip_all,
        fields(
            conversation = %execution.conversation.id,
            turn = %execution.turn.id,
            agent = %execution.agent_name,
        )
    )]
    async fn refuse_turn(
        &self,
        execution: &TurnExecution,
        reason: &str,
    ) -> Result<BotAction, BotError> {
        self.storage
            .finish_turn(FinishTurn::Failed {
                turn_id: execution.turn.id,
                error: reason.to_string(),
                assistant_content: None,
                assistant_message: None,
                usage: Vec::new(),
            })
            .await
            .map_err(storage_error)?;
        self.publish_conversation(
            execution.conversation.id,
            ConversationEventKind::TurnUpdated,
        );
        Ok(BotAction::RefusedMessage)
    }

    async fn generated_attachments(&self, trace: &[ToolTrace]) -> Vec<OutgoingAttachment> {
        let uris = media_uris_from_tool_traces(trace);
        let mut attachments = Vec::with_capacity(uris.len());
        for uri in uris {
            let media = match self.media_store.media_from_uri(&uri).await {
                Ok(media) => media,
                Err(error) => {
                    tracing::warn!(error = %error, uri = %uri, "generated media was not found");
                    continue;
                }
            };
            let loaded = match media.load().await {
                Ok(loaded) => loaded,
                Err(error) => {
                    tracing::warn!(error = %error, uri = %uri, "failed to load generated media");
                    continue;
                }
            };
            if loaded.bytes.len() > MAX_OUTGOING_ATTACHMENT_BYTES {
                tracing::warn!(
                    uri = %uri,
                    bytes = loaded.bytes.len(),
                    limit = MAX_OUTGOING_ATTACHMENT_BYTES,
                    "generated media exceeds outgoing attachment size limit; skipping"
                );
                continue;
            }
            tracing::debug!(
                uri = %uri,
                filename = loaded.media.name(),
                mime_type = loaded.media.mime_type(),
                bytes = loaded.bytes.len(),
                "prepared generated media attachment"
            );
            attachments.push(OutgoingAttachment {
                filename: loaded.media.name().to_string(),
                content_type: loaded.media.mime_type().to_string(),
                bytes: loaded.bytes,
            });
        }
        attachments
    }

    fn spawn_typing_indicator(&self, channel: ChannelRef) -> TypingIndicator {
        let platforms = self.platforms.clone();
        let stop = CancellationToken::new();
        let stopped = stop.clone();
        let task = tokio::spawn(async move {
            loop {
                if let Err(error) = platforms.typing(channel.clone()).await {
                    tracing::warn!(
                        error = %error,
                        channel = %channel.channel_id,
                        "failed to send typing indicator"
                    );
                }
                tokio::select! {
                    biased;
                    () = stopped.cancelled() => break,
                    () = tokio::time::sleep(TYPING_REFRESH_INTERVAL) => {}
                }
            }
        });
        TypingIndicator { stop, task }
    }

    #[tracing::instrument(
        name = "bot.handle_command",
        skip_all,
        fields(
            command = %command.name,
            platform = %command.channel.platform,
            guild = ?command.channel.guild_id,
            channel = %command.channel.channel_id,
            user = %command.user.user_id,
            is_admin = command.is_admin,
        )
    )]
    async fn handle_command(&self, command: PlatformCommand) -> Result<BotAction, BotError> {
        let handled = match command.name.as_str() {
            "chudbot-privacy" => self.handle_privacy_command(&command).await,
            "chudbot-mode" => self.handle_mode_command(&command).await,
            "chudbot-agent" => self.handle_agent_command(&command).await,
            other => {
                tracing::warn!(name = other, "unknown command");
                Ok("Unknown command. Try `/chudbot-privacy`, `/chudbot-mode`, or `/chudbot-agent`."
                    .to_string())
            }
        };
        let content = match handled {
            Ok(content) => content,
            Err(BotError::CommandInput(message)) => message,
            Err(error) => return Err(error),
        };
        self.platforms
            .respond_to_command(PlatformCommandResponse {
                target: command.response_target,
                content,
                ephemeral: true,
            })
            .await
            .map_err(platform_error)?;
        Ok(BotAction::HandledCommand)
    }

    async fn handle_privacy_command(&self, command: &PlatformCommand) -> Result<String, BotError> {
        let Some(guild) = command.channel.guild_id.as_ref() else {
            return Ok(
                "Privacy preferences are per-server. Run this from inside a server channel."
                    .to_string(),
            );
        };
        let Some(sub) = command_subcommand(command) else {
            return Ok("Missing subcommand.".to_string());
        };
        match sub.name.as_str() {
            "in" => {
                self.storage
                    .set_user_privacy(
                        command.channel.platform.clone(),
                        guild.as_str().to_string(),
                        command.user.user_id.as_str().to_string(),
                        true,
                    )
                    .await
                    .map_err(storage_error)?;
                Ok("Opted in. Chudbot may use your quoted messages as context here.".to_string())
            }
            "out" => {
                self.storage
                    .set_user_privacy(
                        command.channel.platform.clone(),
                        guild.as_str().to_string(),
                        command.user.user_id.as_str().to_string(),
                        false,
                    )
                    .await
                    .map_err(storage_error)?;
                Ok(
                    "Opted out. Your direct mentions and messages inside a Chudbot thread still remain visible so the bot can answer."
                        .to_string(),
                )
            }
            "status" => {
                let opted_in = self
                    .storage
                    .user_privacy(
                        command.channel.platform.clone(),
                        guild.as_str().to_string(),
                        command.user.user_id.as_str().to_string(),
                    )
                    .await
                    .map_err(storage_error)?
                    .unwrap_or(false);
                if opted_in {
                    Ok("You are opted in here.".to_string())
                } else {
                    Ok("You are opted out here. Use `/chudbot-privacy in` to opt in.".to_string())
                }
            }
            other => Ok(format!("Unknown subcommand `{other}`.")),
        }
    }

    async fn handle_mode_command(&self, command: &PlatformCommand) -> Result<String, BotError> {
        let Some(guild) = command.channel.guild_id.as_ref() else {
            return Ok(
                "Privacy mode is per-server. Run this from inside a server channel.".to_string(),
            );
        };
        if !command.is_admin {
            return Ok(
                "Changing server privacy mode requires administrator privileges.".to_string(),
            );
        }
        let Some(sub) = command_subcommand(command) else {
            return Ok("Missing subcommand.".to_string());
        };
        match sub.name.as_str() {
            "show" => {
                let settings = self
                    .storage
                    .runtime_settings(
                        command.channel.platform.clone(),
                        Some(guild.as_str().to_string()),
                        command.user.user_id.as_str().to_string(),
                    )
                    .await
                    .map_err(storage_error)?;
                Ok(format!(
                    "Current mode: `{}`\n\n```json\n{}\n```",
                    privacy_mode_kind(&settings.privacy),
                    pretty_json(&settings.privacy),
                ))
            }
            "set" => {
                let mode = sub_option_string(&sub, "mode")
                    .ok_or_else(|| BotError::CommandInput("missing `mode`".to_string()))?;
                let channel = sub_option_string(&sub, "channel");
                let history_size = sub_option_integer(&sub, "history_size").map(|value| {
                    u32::try_from(value.clamp(HISTORY_SIZE_MIN, HISTORY_SIZE_MAX)).unwrap_or(20)
                });
                let privacy = command_privacy_mode(
                    command.channel.platform.clone(),
                    guild.as_str().to_string(),
                    mode,
                    channel,
                    history_size,
                )?;
                self.storage
                    .set_privacy_mode(
                        command.channel.platform.clone(),
                        guild.as_str().to_string(),
                        privacy.clone(),
                    )
                    .await
                    .map_err(storage_error)?;
                Ok(format!(
                    "Mode set to `{}`.\n```json\n{}\n```",
                    privacy_mode_kind(&privacy),
                    pretty_json(&privacy),
                ))
            }
            other => Ok(format!("Unknown subcommand `{other}`.")),
        }
    }

    async fn handle_agent_command(&self, command: &PlatformCommand) -> Result<String, BotError> {
        let Some(sub) = command_subcommand(command) else {
            return Ok("Missing subcommand.".to_string());
        };
        match sub.name.as_str() {
            "list" => Ok(agent_list_response(&self.config)),
            "show" => self.handle_agent_show(command).await,
            "set" => self.handle_agent_set(command, &sub).await,
            "clear" => self.handle_agent_clear(command, &sub).await,
            other => Ok(format!("Unknown subcommand `{other}`.")),
        }
    }

    async fn handle_agent_show(&self, command: &PlatformCommand) -> Result<String, BotError> {
        let conversation = self.command_conversation(command).await?;
        let channel = self.command_scope_channel(command).await;
        let guild = command
            .channel
            .guild_id
            .as_ref()
            .map(|id| id.as_str().to_string());

        let conv_pick = match conversation {
            Some(conversation_id) => self
                .storage
                .load_agent_selection(AgentSelection::Conversation { conversation_id })
                .await
                .map_err(storage_error)?,
            None => None,
        };
        let user_pick = match guild.as_deref() {
            Some(guild) => self
                .storage
                .load_agent_selection(AgentSelection::User {
                    message_provider: command.channel.platform.clone(),
                    guild_key: guild.to_string(),
                    user_key: command.user.user_id.as_str().to_string(),
                })
                .await
                .map_err(storage_error)?,
            None => None,
        };
        let channel_pick = self
            .storage
            .load_agent_selection(AgentSelection::Channel {
                message_provider: command.channel.platform.clone(),
                guild_key: guild.clone(),
                channel_key: channel.channel_id.as_str().to_string(),
            })
            .await
            .map_err(storage_error)?;
        let guild_pick = match guild.as_deref() {
            Some(guild) => self
                .storage
                .load_agent_selection(AgentSelection::Guild {
                    message_provider: command.channel.platform.clone(),
                    guild_key: guild.to_string(),
                })
                .await
                .map_err(storage_error)?,
            None => None,
        };
        let platform_pick = self
            .storage
            .load_agent_selection(AgentSelection::Platform {
                message_provider: command.channel.platform.clone(),
            })
            .await
            .map_err(storage_error)?;

        let active_name = conv_pick
            .clone()
            .or_else(|| user_pick.clone())
            .or_else(|| channel_pick.clone())
            .or_else(|| guild_pick.clone())
            .or_else(|| platform_pick.clone())
            .unwrap_or_else(|| self.config.default_agent.clone());
        let active = self.config.agents.get(&active_name);
        let mut out = String::from("Agent resolution here\n");
        out.push_str(&format!(
            "conversation: {}\n",
            option_tick(conv_pick.as_deref())
        ));
        out.push_str(&format!("user: {}\n", option_tick(user_pick.as_deref())));
        out.push_str(&format!(
            "channel: {}\n",
            option_tick(channel_pick.as_deref())
        ));
        out.push_str(&format!("guild: {}\n", option_tick(guild_pick.as_deref())));
        out.push_str(&format!(
            "platform: {}\n",
            option_tick(platform_pick.as_deref())
        ));
        out.push_str(&format!("fallback: `{}`\n", self.config.default_agent));
        match active {
            Some(agent) => out.push_str(&format!(
                "\nActive: `{active_name}`: `{}` / `{}`",
                agent.provider, agent.model.id
            )),
            None => out.push_str(&format!(
                "\nActive: `{active_name}` is no longer configured; falling back to `{}`",
                self.config.default_agent
            )),
        }
        Ok(out)
    }

    async fn handle_agent_set(
        &self,
        command: &PlatformCommand,
        sub: &PlatformCommandInput,
    ) -> Result<String, BotError> {
        let Some(name) = sub_option_string(sub, "name") else {
            return Ok("Missing `name`.".to_string());
        };
        if !self.config.agents.contains_key(name) {
            return Ok(format!(
                "Unknown agent `{name}`. {}",
                available_agents(&self.config)
            ));
        }
        let Some(scope) = sub_option_string(sub, "scope") else {
            return Ok("Missing `scope`.".to_string());
        };
        let selection = self.command_agent_selection(command, scope, true).await?;
        self.storage
            .set_agent_selection(selection, name.to_string())
            .await
            .map_err(storage_error)?;
        Ok(format!(
            "Set agent for {} to `{name}`.",
            scope_description(scope)
        ))
    }

    async fn handle_agent_clear(
        &self,
        command: &PlatformCommand,
        sub: &PlatformCommandInput,
    ) -> Result<String, BotError> {
        let Some(scope) = sub_option_string(sub, "scope") else {
            return Ok("Missing `scope`.".to_string());
        };
        let selection = self.command_agent_selection(command, scope, true).await?;
        let cleared = self
            .storage
            .clear_agent_selection(selection)
            .await
            .map_err(storage_error)?;
        if cleared {
            Ok(format!(
                "Cleared agent override for {}.",
                scope_description(scope)
            ))
        } else {
            Ok(format!(
                "No agent override was set for {}.",
                scope_description(scope)
            ))
        }
    }

    async fn command_agent_selection(
        &self,
        command: &PlatformCommand,
        scope: &str,
        enforce_admin: bool,
    ) -> Result<AgentSelection, BotError> {
        match scope {
            "conversation" => {
                let Some(conversation_id) = self.command_conversation(command).await? else {
                    return Err(BotError::CommandInput(
                        "No conversation is bound to this channel. Run this inside a thread the bot opened for an answer."
                            .to_string(),
                    ));
                };
                Ok(AgentSelection::Conversation { conversation_id })
            }
            "user" => {
                let Some(guild) = command.channel.guild_id.as_ref() else {
                    return Err(BotError::CommandInput(
                        "User-scoped agent selection only makes sense in a server.".to_string(),
                    ));
                };
                Ok(AgentSelection::User {
                    message_provider: command.channel.platform.clone(),
                    guild_key: guild.as_str().to_string(),
                    user_key: command.user.user_id.as_str().to_string(),
                })
            }
            "channel" => {
                if enforce_admin && !command.is_admin {
                    return Err(BotError::CommandInput(
                        "Channel-scoped agent selection requires administrator privileges."
                            .to_string(),
                    ));
                }
                let channel = self.command_scope_channel(command).await;
                Ok(AgentSelection::Channel {
                    message_provider: command.channel.platform.clone(),
                    guild_key: command
                        .channel
                        .guild_id
                        .as_ref()
                        .map(|id| id.as_str().to_string()),
                    channel_key: channel.channel_id.as_str().to_string(),
                })
            }
            "guild" => {
                if enforce_admin && !command.is_admin {
                    return Err(BotError::CommandInput(
                        "Guild-scoped agent selection requires administrator privileges."
                            .to_string(),
                    ));
                }
                let Some(guild) = command.channel.guild_id.as_ref() else {
                    return Err(BotError::CommandInput(
                        "Guild-scoped agent selection only makes sense in a server.".to_string(),
                    ));
                };
                Ok(AgentSelection::Guild {
                    message_provider: command.channel.platform.clone(),
                    guild_key: guild.as_str().to_string(),
                })
            }
            other => Err(BotError::CommandInput(format!("Unknown scope `{other}`."))),
        }
    }

    async fn command_conversation(
        &self,
        command: &PlatformCommand,
    ) -> Result<Option<ConversationId>, BotError> {
        let snapshot = self
            .storage
            .load_conversation(ConversationLookup::Channel {
                channel: command.channel.clone(),
            })
            .await
            .map_err(storage_error)?;
        Ok(snapshot.map(|snapshot| snapshot.conversation.id))
    }

    async fn command_scope_channel(&self, command: &PlatformCommand) -> ChannelRef {
        match self.platforms.parent_channel(command.channel.clone()).await {
            Ok(parent) => parent,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    channel = %command.channel.channel_id,
                    "failed to resolve command parent channel; using interaction channel"
                );
                command.channel.clone()
            }
        }
    }

    async fn moderation_allows(
        &self,
        message: &PlatformMessage,
        display_name: &str,
    ) -> Result<bool, BotError> {
        let (agent_name, agent_config) = self
            .config
            .agent_or_platform_default(None, &message.id.platform)?;
        if !self.llms.contains_provider(&agent_config.provider) {
            tracing::warn!(
                agent = %agent_name,
                provider = %agent_config.provider,
                "moderation provider is missing; failing open"
            );
            return Ok(true);
        }

        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn::text(
            TurnRole::User,
            format!(
                "Message to classify:\n<<<\n[{display_name}]: {}\n>>>",
                message.content
            ),
        ));
        let agent = self.utility_agent(
            agent_config.provider.clone(),
            ModelSpec {
                id: agent_config.model.id.clone(),
                server_tools: Default::default(),
                sampling: SamplingOptions {
                    max_output_tokens: Some(8),
                    temperature: Some(0.0),
                    top_p: None,
                },
                provider_options: None,
            },
            MODERATION_PROMPT,
            agent_config.limits.unwrap_or(self.config.limits),
        );
        let run = match agent.run(transcript).await {
            Ok(run) => run,
            Err(error) => {
                let message = error.to_string();
                if error_indicates_safety_refusal(&message) {
                    tracing::info!(
                        error = %error,
                        "moderation provider refusal detected; treating as refused"
                    );
                    return Ok(false);
                }
                tracing::warn!(error = %error, "moderation errored; failing open");
                return Ok(true);
            }
        };
        match run.outcome {
            AgentOutcome::Completed { answer } => {
                let verdict = answer.text.trim().to_ascii_uppercase();
                let allowed = !verdict.starts_with("REFUSE")
                    && !verdict.contains(" REFUSE")
                    && verdict != "REFUSE";
                tracing::info!(verdict = %verdict, allowed, "moderation classified message");
                Ok(allowed)
            }
            AgentOutcome::IterationLimit { max_iterations } => {
                tracing::warn!(
                    max_iterations,
                    "moderation hit iteration limit; failing open"
                );
                Ok(true)
            }
            AgentOutcome::Failed { error, .. } => {
                if error_indicates_safety_refusal(&error.to_string()) {
                    tracing::info!(
                        error = %error,
                        "moderation provider refusal detected; treating as refused"
                    );
                    return Ok(false);
                }
                tracing::warn!(error = %error, "moderation failed; failing open");
                Ok(true)
            }
            AgentOutcome::Cancelled { reason } => {
                tracing::warn!(reason = %reason, "moderation was cancelled; failing open");
                Ok(true)
            }
        }
    }

    fn spawn_title_generation(&self, conversation_id: ConversationId, agent_name: String) {
        let runtime = (*self).clone();
        spawn_background_task(&self.background, "title generation", async move {
            if let Err(error) = runtime.generate_title(conversation_id, &agent_name).await {
                tracing::warn!(
                    conversation = %conversation_id,
                    agent = %agent_name,
                    error = %error,
                    "title generation failed"
                );
            }
        });
    }

    fn spawn_memory_runtime(&self, shutdown: CancellationToken) {
        if !self.memory_config.enabled {
            return;
        }
        let runtime = memory::MemoryRuntime::new(
            self.storage.clone(),
            self.llms.clone(),
            self.memory_config.clone(),
        );
        spawn_background_task(&self.background, "memory runtime", async move {
            if let Err(error) = runtime.run_until_shutdown(shutdown).await {
                tracing::warn!(error = %error, "memory runtime stopped with error");
            }
        });
    }

    async fn generate_title(
        &self,
        conversation_id: ConversationId,
        agent_name: &str,
    ) -> Result<(), BotError> {
        let Some(snapshot) = self
            .storage
            .load_conversation(ConversationLookup::Id {
                id: conversation_id,
            })
            .await
            .map_err(storage_error)?
        else {
            return Err(BotError::MissingConversation { conversation_id });
        };
        if snapshot.conversation.title.is_some() {
            tracing::debug!("conversation title already exists; skipping");
            return Ok(());
        }
        let Some(first) = snapshot
            .turns
            .iter()
            .find(|turn| matches!(turn.turn.status, chudbot_api::TurnStatus::Completed))
        else {
            tracing::debug!("no completed turns available for title generation");
            return Ok(());
        };
        let (_, agent) = self
            .config
            .agent_or_platform_default(Some(agent_name), &snapshot.conversation.channel.platform)?;
        let user_text = format!(
            "User said:\n{}\n\nAssistant replied:\n{}",
            first.turn.user_content,
            first.turn.assistant_content.as_deref().unwrap_or("")
        );
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn::text(TurnRole::User, user_text));
        let agent_runtime = self.utility_agent(
            agent.provider.clone(),
            ModelSpec {
                id: agent.model.id.clone(),
                server_tools: Default::default(),
                sampling: SamplingOptions {
                    max_output_tokens: Some(TITLE_MAX_TOKENS),
                    temperature: Some(0.3),
                    top_p: None,
                },
                provider_options: agent.model.provider_options.clone(),
            },
            TITLE_SYSTEM_PROMPT,
            agent.limits.unwrap_or(self.config.limits),
        );
        let run = agent_runtime
            .run(transcript)
            .await
            .map_err(|error| BotError::Model {
                message: error.to_string(),
            })?;
        let raw = match run.outcome {
            AgentOutcome::Completed { answer } => answer.text,
            AgentOutcome::IterationLimit { max_iterations } => {
                return Err(BotError::Model {
                    message: format!("title generation hit iteration limit ({max_iterations})"),
                });
            }
            AgentOutcome::Failed { error, partial } => {
                let mut message = error.to_string();
                if let Some(partial) = partial
                    && !partial.text.trim().is_empty()
                {
                    message.push_str("\n\nPartial answer:\n");
                    message.push_str(&partial.text);
                }
                return Err(BotError::Model { message });
            }
            AgentOutcome::Cancelled { reason } => {
                return Err(BotError::Model {
                    message: format!("title generation cancelled: {reason}"),
                });
            }
        };
        let title = clean_title(&raw);
        if title.is_empty() {
            tracing::warn!(raw = %raw, "title generation returned empty title");
            return Ok(());
        }
        self.storage
            .set_conversation_title(conversation_id, title.clone())
            .await
            .map_err(storage_error)?;
        self.publish_conversation(conversation_id, ConversationEventKind::TitleUpdated);
        tracing::info!(title = %title, "conversation title set");
        Ok(())
    }

    fn spawn_avatar_download(&self, user: UserProfile) {
        let Some(url) = user
            .avatar_url
            .as_deref()
            .filter(|url| !url.trim().is_empty())
            .map(str::to_string)
        else {
            return;
        };
        let runtime = (*self).clone();
        spawn_background_task(&self.background, "avatar download", async move {
            if let Err(error) = runtime.download_avatar(user, url).await {
                tracing::warn!(error = %error, "avatar download failed");
            }
        });
    }

    async fn download_avatar(&self, user: UserProfile, url: String) -> Result<(), BotError> {
        let name = avatar_media_name(&user, &url);
        let expected_uri = MediaUri::new(format!("file://avatars/{name}"));
        if self
            .storage
            .load_user_avatar(user.id.clone())
            .await
            .map_err(storage_error)?
            .as_ref()
            .is_some_and(|uri| uri == &expected_uri)
        {
            tracing::trace!(uri = %expected_uri, "avatar already cached");
            return Ok(());
        }

        let response = reqwest::Client::new()
            .get(&url)
            .send()
            .await
            .map_err(|error| BotError::AvatarDownload(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(BotError::AvatarDownload(format!("http {status}")));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| BotError::AvatarDownload(error.to_string()))?
            .to_vec();
        let media = self
            .media_store
            .create_media(CreateMedia {
                category: MediaCategory::Avatar,
                bytes,
                mime_type: Some("image/png".to_string()),
                name: Some(name),
                extension: Some("png".to_string()),
            })
            .await
            .map_err(|error| BotError::AvatarDownload(error.to_string()))?;
        self.storage
            .set_user_avatar(user.id.clone(), media.uri().clone())
            .await
            .map_err(storage_error)?;
        self.publish_user(user.id);
        tracing::info!(uri = %media.uri(), "avatar cached");
        Ok(())
    }

    fn publish_user(&self, user: chudbot_api::UserRef) {
        self.events.publish(LiveEvent::UserProfileUpdated { user });
    }

    fn is_admin(&self, user: &chudbot_api::UserRef) -> bool {
        self.config.admins.iter().any(|admin| {
            admin.platform == user.platform
                && admin.user_id == user.user_id
                && admin
                    .guild_id
                    .as_ref()
                    .is_none_or(|guild| user.guild_id.as_ref() == Some(guild))
        })
    }

    async fn add_unicode_reaction(&self, message: &MessageRef, name: &str, label: &str) {
        if let Err(error) = self
            .platforms
            .add_reaction(
                message.clone(),
                ReactionKind::Unicode {
                    name: name.to_string(),
                },
            )
            .await
        {
            tracing::warn!(error = %error, reaction = name, label, "failed to add reaction");
        }
    }

    async fn remove_own_unicode_reaction(&self, message: &MessageRef, name: &str, label: &str) {
        if let Err(error) = self
            .platforms
            .remove_own_reaction(
                message.clone(),
                ReactionKind::Unicode {
                    name: name.to_string(),
                },
            )
            .await
        {
            tracing::warn!(error = %error, reaction = name, label, "failed to remove reaction");
        }
    }

    async fn react_for_action(&self, message: &MessageRef, action: &Result<BotAction, BotError>) {
        match action {
            Ok(BotAction::CompletedTurn { .. }) => {
                self.add_unicode_reaction(message, SUCCESS_REACTION, "turn_success")
                    .await;
            }
            Ok(BotAction::FailedTurn { .. }) | Err(_) => {
                self.add_unicode_reaction(message, ERROR_REACTION, "turn_error")
                    .await;
            }
            Ok(BotAction::RefusedMessage) => {
                self.add_unicode_reaction(message, REFUSED_REACTION, "turn_refused")
                    .await;
            }
            Ok(BotAction::CancelledTurn { .. }) => {
                tracing::info!("turn cancelled; leaving only the stop reaction as status");
            }
            Ok(_) => {}
        }
    }

    async fn runtime_settings(
        &self,
        message: &PlatformMessage,
    ) -> Result<RuntimeSettings, BotError> {
        let settings = self
            .storage
            .runtime_settings(
                message.id.platform.clone(),
                guild_key(&message.id),
                message.author.id.user_id.as_str().to_string(),
            )
            .await
            .map_err(storage_error)?;
        tracing::trace!(
            platform = %message.id.platform,
            guild = ?message.id.guild_id,
            user = %message.author.id.user_id,
            privacy = privacy_mode_kind(&settings.privacy),
            opted_in = settings.user_opted_in,
            "runtime settings loaded"
        );
        Ok(settings)
    }

    async fn agent_scope_channel(&self, message: &MessageRef) -> ChannelRef {
        let channel = channel_from_message(message);
        match self.platforms.parent_channel(channel.clone()).await {
            Ok(parent) => parent,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    platform = %message.platform,
                    channel = %message.channel_id,
                    "failed to resolve parent channel for agent scope; using event channel"
                );
                channel
            }
        }
    }

    async fn privacy_allows_message_channel(
        &self,
        mode: &PrivacyMode,
        message: &MessageRef,
        existing: Option<&ConversationSnapshot>,
    ) -> Result<bool, BotError> {
        let PrivacyMode::ChannelOnly {
            channel: allowed, ..
        } = mode
        else {
            return Ok(true);
        };
        let actual = channel_from_message(message);
        if &actual == allowed {
            return Ok(true);
        }
        if existing.is_some() {
            tracing::debug!(
                actual_channel = %actual.channel_id,
                allowed_channel = %allowed.channel_id,
                "allowing channel_only message because it continues an existing conversation"
            );
            return Ok(true);
        }
        let parent = self
            .platforms
            .parent_channel(actual)
            .await
            .map_err(platform_error)?;
        Ok(&parent == allowed)
    }

    #[tracing::instrument(
        name = "bot.lookup_conversation",
        skip_all,
        fields(
            platform = %message.id.platform,
            guild = ?message.id.guild_id,
            channel = %message.id.channel_id,
            message = %message.id.message_id,
            has_reference = message.referenced_message_id().is_some(),
        )
    )]
    async fn lookup_existing_conversation(
        &self,
        message: &PlatformMessage,
    ) -> Result<Option<ConversationSnapshot>, BotError> {
        let channel = channel_from_message(&message.id);
        tracing::debug!(
            lookup = "channel",
            lookup_platform = %channel.platform,
            lookup_guild = ?channel.guild_id.as_ref().map(ExternalId::as_str),
            lookup_channel = %channel.channel_id,
            "looking up existing conversation by channel"
        );
        if let Some(snapshot) = self
            .storage
            .load_conversation(ConversationLookup::Channel {
                channel: channel.clone(),
            })
            .await
            .map_err(storage_error)?
        {
            tracing::debug!(
                conversation = %snapshot.conversation.id,
                source = "channel",
                "found existing conversation"
            );
            return Ok(Some(snapshot));
        }
        tracing::debug!(
            lookup = "channel",
            lookup_platform = %channel.platform,
            lookup_guild = ?channel.guild_id.as_ref().map(ExternalId::as_str),
            lookup_channel = %channel.channel_id,
            "no existing conversation found by channel"
        );

        if let Some(referenced) = message.referenced_message_id().cloned() {
            tracing::debug!(
                lookup = "referenced_message",
                reference_kind = platform_message_reference_kind(&message.reference),
                lookup_platform = %referenced.platform,
                lookup_guild = ?referenced.guild_id.as_ref().map(ExternalId::as_str),
                lookup_channel = %referenced.channel_id,
                lookup_message = %referenced.message_id,
                "looking up existing conversation by referenced message"
            );
            if let Some(snapshot) = self
                .storage
                .load_conversation(ConversationLookup::Message {
                    message: referenced.clone(),
                })
                .await
                .map_err(storage_error)?
            {
                tracing::debug!(
                    conversation = %snapshot.conversation.id,
                    source = "referenced_message",
                    lookup_platform = %referenced.platform,
                    lookup_guild = ?referenced.guild_id.as_ref().map(ExternalId::as_str),
                    lookup_channel = %referenced.channel_id,
                    lookup_message = %referenced.message_id,
                    "found existing conversation"
                );
                return Ok(Some(snapshot));
            }
            tracing::debug!(
                lookup = "referenced_message",
                reference_kind = platform_message_reference_kind(&message.reference),
                lookup_platform = %referenced.platform,
                lookup_guild = ?referenced.guild_id.as_ref().map(ExternalId::as_str),
                lookup_channel = %referenced.channel_id,
                lookup_message = %referenced.message_id,
                "no existing conversation found by referenced message"
            );
        } else {
            tracing::debug!(
                reference_kind = platform_message_reference_kind(&message.reference),
                "skipping referenced-message lookup because no referenced message id was available"
            );
        }

        tracing::debug!(
            lookup = "message",
            lookup_platform = %message.id.platform,
            lookup_guild = ?message.id.guild_id.as_ref().map(ExternalId::as_str),
            lookup_channel = %message.id.channel_id,
            lookup_message = %message.id.message_id,
            "looking up existing conversation by current message"
        );
        let snapshot = self
            .storage
            .load_conversation(ConversationLookup::Message {
                message: message.id.clone(),
            })
            .await
            .map_err(storage_error)?;
        if let Some(snapshot) = &snapshot {
            tracing::debug!(
                conversation = %snapshot.conversation.id,
                source = "message",
                lookup_platform = %message.id.platform,
                lookup_guild = ?message.id.guild_id.as_ref().map(ExternalId::as_str),
                lookup_channel = %message.id.channel_id,
                lookup_message = %message.id.message_id,
                "found existing conversation"
            );
        } else {
            tracing::debug!(
                lookup = "message",
                lookup_platform = %message.id.platform,
                lookup_guild = ?message.id.guild_id.as_ref().map(ExternalId::as_str),
                lookup_channel = %message.id.channel_id,
                lookup_message = %message.id.message_id,
                "no existing conversation found by current message"
            );
        }
        Ok(snapshot)
    }

    #[tracing::instrument(
        name = "bot.transcript_for_turn",
        skip_all,
        fields(
            conversation = %snapshot.conversation.id,
            turn = %turn.id,
            turn_ordinal = turn.ordinal,
            history_cutoff = ?turn.history_cutoff,
        )
    )]
    async fn transcript_for_turn(
        &self,
        snapshot: &ConversationSnapshot,
        turn: &Turn,
        context: &[chudbot_api::ContextItem],
    ) -> Result<Transcript, BotError> {
        let mut transcript = self
            .transcript_from_snapshot(snapshot, turn.history_cutoff)
            .await?;
        transcript.push(self.transcript_turn_from_context(turn.id, context).await);
        tracing::debug!(
            turns = transcript.turns.len(),
            "assembled transcript for live turn"
        );
        Ok(transcript)
    }

    #[tracing::instrument(
        name = "bot.transcript_for_retry",
        skip_all,
        fields(
            conversation = %snapshot.conversation.id,
            retry_turn = %retry_turn.turn.id,
            retry_turn_ordinal = retry_turn.turn.ordinal,
            history_cutoff = ?retry_turn.turn.history_cutoff,
        )
    )]
    async fn transcript_for_retry(
        &self,
        snapshot: &ConversationSnapshot,
        retry_turn: &TurnSnapshot,
        context: &[chudbot_api::ContextItem],
        has_stored_context: bool,
    ) -> Result<Transcript, BotError> {
        let mut transcript = self
            .transcript_from_snapshot(snapshot, retry_turn.turn.history_cutoff)
            .await?;
        if has_stored_context {
            transcript.push(
                self.transcript_turn_from_context(retry_turn.turn.id, context)
                    .await,
            );
        } else {
            let mut turn = TranscriptTurn::text(
                TurnRole::User,
                format!(
                    "[{}]: {}",
                    retry_turn.turn.user_display_name, retry_turn.turn.user_content
                ),
            );
            turn.metadata =
                transcript_message_metadata(turn_transcript_message_id(retry_turn.turn.id, "user"));
            let mut extra_blocks = self.context_blocks_from_items(context).await;
            turn.blocks.append(&mut extra_blocks);
            transcript.push(turn);
        }
        tracing::debug!(
            turns = transcript.turns.len(),
            "assembled transcript for retry"
        );
        Ok(transcript)
    }

    async fn transcript_turn_from_context(
        &self,
        turn_id: TurnId,
        context: &[chudbot_api::ContextItem],
    ) -> TranscriptTurn {
        let mut blocks = self.context_blocks_from_items(context).await;
        if blocks.is_empty() {
            blocks.push(ContentBlock::Text {
                text: "(no message content)".to_string(),
            });
        }
        TranscriptTurn {
            role: TurnRole::User,
            blocks,
            metadata: transcript_message_metadata(turn_transcript_message_id(turn_id, "user")),
        }
    }

    async fn context_blocks_from_items(
        &self,
        context: &[chudbot_api::ContextItem],
    ) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();
        for item in context {
            if item.content.starts_with("file://") {
                match self
                    .media_store
                    .media_from_uri(&MediaUri::new(item.content.clone()))
                    .await
                {
                    Ok(media) if model_transcript_supports_media(media.as_ref()) => {
                        blocks.push(ContentBlock::Media { media })
                    }
                    Ok(media) => tracing::debug!(
                        source = %item.source,
                        uri = %media.uri(),
                        category = ?media.category(),
                        mime_type = %media.mime_type(),
                        "skipping unsupported context media while assembling transcript"
                    ),
                    Err(error) => tracing::warn!(
                        error = %error,
                        source = %item.source,
                        uri = %item.content,
                        "skipping context media while assembling transcript"
                    ),
                }
                continue;
            }
            blocks.push(ContentBlock::Text {
                text: item.content.clone(),
            });
        }
        blocks
    }

    async fn prepare_turn_context(
        &self,
        message: &PlatformMessage,
        settings: &RuntimeSettings,
        conversation: &Conversation,
    ) -> Result<PreparedTurnContext, BotError> {
        let mut items = Vec::new();
        let mut position = 0;

        if let Some(referenced) = message.referenced_message()
            && self
                .quoted_message_allowed(referenced, settings, conversation)
                .await?
            && !self
                .quoted_assistant_message_already_replays(referenced, conversation)
                .await?
        {
            self.push_message_context(
                &mut items,
                &mut position,
                "quoted",
                referenced,
                PlatformMessageRelationship::Referenced,
            )
            .await?;
        }

        self.push_message_context(
            &mut items,
            &mut position,
            "message",
            message,
            PlatformMessageRelationship::Current,
        )
        .await?;

        Ok(PreparedTurnContext { items })
    }

    async fn push_message_context(
        &self,
        items: &mut Vec<chudbot_api::ContextItem>,
        position: &mut i32,
        kind: &str,
        message: &PlatformMessage,
        relationship: PlatformMessageRelationship,
    ) -> Result<(), BotError> {
        let image_media = self
            .save_matching_attachments(message, MediaCategory::Image, "image", looks_like_image_ref)
            .await;
        let audio_media = self
            .save_matching_attachments(message, MediaCategory::Audio, "audio", looks_like_audio_ref)
            .await;

        let mut value = self
            .platforms
            .message_context(message, relationship)
            .await
            .map_err(platform_error)?;
        inject_audio_attachment_refs(&mut value, &audio_media);
        let content = serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
        items.push(chudbot_api::ContextItem {
            position: *position,
            source: format!("platform:{kind}:{}", message.id.message_id.as_str()),
            role: "user".to_string(),
            content,
            message: Some(message.id.clone()),
        });
        *position += 1;

        let mut image_refs = Vec::new();
        for saved in image_media {
            let uri = saved.media.uri().to_string();
            image_refs.push(uri.clone());
            items.push(chudbot_api::ContextItem {
                position: *position,
                source: format!(
                    "platform:{kind}:{}:image:{}",
                    message.id.message_id.as_str(),
                    saved.attachment_index
                ),
                role: "user".to_string(),
                content: uri,
                message: Some(message.id.clone()),
            });
            *position += 1;
        }
        for saved in &audio_media {
            items.push(chudbot_api::ContextItem {
                position: *position,
                source: format!(
                    "platform:{kind}:{}:audio:{}",
                    message.id.message_id.as_str(),
                    saved.attachment_index
                ),
                role: "user".to_string(),
                content: saved.media.uri().to_string(),
                message: Some(message.id.clone()),
            });
            *position += 1;
        }
        if !image_refs.is_empty() {
            items.push(chudbot_api::ContextItem {
                position: *position,
                source: format!(
                    "platform:{kind}:{}:image_refs",
                    message.id.message_id.as_str()
                ),
                role: "user".to_string(),
                content: format!(
                    "Image attachment reference IDs available for tool calls: {}",
                    image_refs.join(", ")
                ),
                message: Some(message.id.clone()),
            });
            *position += 1;
        }
        Ok(())
    }

    async fn quoted_message_allowed(
        &self,
        referenced: &PlatformMessage,
        settings: &RuntimeSettings,
        conversation: &Conversation,
    ) -> Result<bool, BotError> {
        match &settings.privacy {
            PrivacyMode::Open { .. } | PrivacyMode::ChannelOnly { .. } => Ok(true),
            PrivacyMode::ConversationOnly => Ok(false),
            PrivacyMode::OptIn => {
                if referenced.author.is_bot {
                    return Ok(true);
                }
                if self
                    .storage
                    .load_conversation(ConversationLookup::Channel {
                        channel: channel_from_message(&referenced.id),
                    })
                    .await
                    .map_err(storage_error)?
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.conversation.id == conversation.id)
                {
                    return Ok(true);
                }
                let Some(guild) = referenced.id.guild_id.as_ref() else {
                    return Ok(false);
                };
                self.storage
                    .user_privacy(
                        referenced.id.platform.clone(),
                        guild.as_str().to_string(),
                        referenced.author.id.user_id.as_str().to_string(),
                    )
                    .await
                    .map_err(storage_error)
                    .map(|opted_in| opted_in.unwrap_or(false))
            }
        }
    }

    async fn quoted_assistant_message_already_replays(
        &self,
        referenced: &PlatformMessage,
        conversation: &Conversation,
    ) -> Result<bool, BotError> {
        let Some(link) = self
            .storage
            .load_message_link(referenced.id.clone())
            .await
            .map_err(storage_error)?
        else {
            return Ok(false);
        };
        let already_replays = message_link_replays_as_assistant(&link, conversation.id);
        if already_replays {
            tracing::trace!(
                conversation = %conversation.id,
                message = %referenced.id.message_id,
                "skipping quoted assistant message already present in transcript"
            );
        }
        Ok(already_replays)
    }

    async fn save_matching_attachments(
        &self,
        message: &PlatformMessage,
        category: MediaCategory,
        label: &'static str,
        predicate: fn(&AttachmentRef) -> bool,
    ) -> Vec<StoredAttachmentMedia> {
        let mut out = Vec::new();
        for (attachment_index, attachment) in message.attachments.iter().enumerate() {
            if !predicate(attachment) {
                continue;
            }
            let response = match self.download_http.get(&attachment.url).send().await {
                Ok(response) => response,
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        filename = %attachment.filename,
                        media_type = label,
                        "failed to download media attachment"
                    );
                    continue;
                }
            };
            let status = response.status();
            if !status.is_success() {
                tracing::warn!(
                    status = status.as_u16(),
                    filename = %attachment.filename,
                    media_type = label,
                    "media attachment download returned non-success status"
                );
                continue;
            }
            let bytes = match response.bytes().await {
                Ok(bytes) => bytes.to_vec(),
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        filename = %attachment.filename,
                        media_type = label,
                        "failed to read media attachment bytes"
                    );
                    continue;
                }
            };
            match self
                .media_store
                .create_media(CreateMedia {
                    category: category.clone(),
                    bytes,
                    mime_type: attachment.content_type.clone(),
                    name: None,
                    extension: extension_from_filename(&attachment.filename),
                })
                .await
            {
                Ok(media) => {
                    tracing::info!(
                        uri = %media.uri(),
                        filename = %attachment.filename,
                        media_type = label,
                        "saved media attachment"
                    );
                    out.push(StoredAttachmentMedia {
                        attachment_index,
                        media,
                    });
                }
                Err(error) => tracing::warn!(
                    error = %error,
                    filename = %attachment.filename,
                    media_type = label,
                    "failed to store media attachment"
                ),
            }
        }
        out
    }

    #[tracing::instrument(
        name = "bot.transcript_from_snapshot",
        skip_all,
        fields(
            conversation = %snapshot.conversation.id,
            stored_turns = snapshot.turns.len(),
            history_cutoff = ?history_cutoff,
        )
    )]
    async fn transcript_from_snapshot(
        &self,
        snapshot: &ConversationSnapshot,
        history_cutoff: Option<i64>,
    ) -> Result<Transcript, BotError> {
        let mut transcript = Transcript::new();
        transcript.id = Some(snapshot.conversation.id.to_string());

        let mut replay_turns = snapshot
            .turns
            .iter()
            .filter(|turn| matches!(turn.turn.status, chudbot_api::TurnStatus::Completed))
            .filter(|turn| {
                let Some(history_cutoff) = history_cutoff else {
                    return false;
                };
                turn.turn
                    .response_ordinal
                    .is_some_and(|ordinal| ordinal <= history_cutoff)
            })
            .collect::<Vec<_>>();
        replay_turns.sort_by_key(|turn| {
            (
                turn.turn.response_ordinal.unwrap_or(i64::MAX),
                turn.turn.ordinal,
            )
        });

        for turn in replay_turns {
            let replay_context = replayable_context_items(&turn.context);
            let mut user_turn = if replay_context.is_empty() {
                TranscriptTurn {
                    role: TurnRole::User,
                    blocks: vec![ContentBlock::Text {
                        text: format!(
                            "[{}]: {}",
                            turn.turn.user_display_name, turn.turn.user_content
                        ),
                    }],
                    metadata: transcript_message_metadata(turn_transcript_message_id(
                        turn.turn.id,
                        "user",
                    )),
                }
            } else {
                self.transcript_turn_from_context(turn.turn.id, &replay_context)
                    .await
            };
            let mut replayed_media = replay_context
                .iter()
                .filter_map(|item| {
                    item.content
                        .starts_with("file://")
                        .then(|| item.content.clone())
                })
                .collect::<Vec<_>>();
            let mut generated_media_refs = Vec::new();
            let mut generated_media_blocks = Vec::new();
            for asset in &turn.replay_assets {
                if replayed_media
                    .iter()
                    .any(|uri| uri.as_str() == asset.uri.as_str())
                {
                    continue;
                }
                match self.media_store.media_from_uri(&asset.uri).await {
                    Ok(media) => {
                        if !model_transcript_supports_media(media.as_ref()) {
                            tracing::debug!(
                                source = %asset.source,
                                uri = %media.uri(),
                                category = ?media.category(),
                                mime_type = %media.mime_type(),
                                "skipping unsupported replay media while rebuilding transcript"
                            );
                            continue;
                        }
                        replayed_media.push(asset.uri.as_str().to_string());
                        if replay_asset_belongs_to_user_turn(asset) {
                            user_turn.blocks.push(ContentBlock::Media { media });
                        } else {
                            generated_media_refs.push(asset.uri.as_str().to_string());
                            generated_media_blocks.push(ContentBlock::Media { media });
                        }
                    }
                    Err(error) => tracing::warn!(
                        error = %error,
                        uri = %asset.uri,
                        "skipping replay media while rebuilding transcript"
                    ),
                }
            }
            tracing::trace!(
                turn = %turn.turn.id,
                replay_assets = turn.replay_assets.len(),
                user_blocks = user_turn.blocks.len(),
                "added prior user turn to transcript"
            );
            transcript.push(user_turn);
            append_client_tool_replay(&mut transcript, &turn.tool_trace);

            if let Some(answer) = &turn.turn.assistant_content {
                let mut blocks = Vec::new();
                if let Some(continuation) = &turn.turn.continuation {
                    blocks.push(ContentBlock::Continuation(continuation.clone()));
                }
                blocks.push(ContentBlock::Text {
                    text: answer.clone(),
                });
                transcript.push(TranscriptTurn {
                    role: TurnRole::Assistant,
                    blocks,
                    metadata: transcript_message_metadata(turn_transcript_message_id(
                        turn.turn.id,
                        "assistant",
                    )),
                });
                tracing::trace!(
                    turn = %turn.turn.id,
                    has_continuation = turn.turn.continuation.is_some(),
                    "added prior assistant turn to transcript"
                );
            }
            append_generated_media_replay(
                &mut transcript,
                turn.turn.id,
                generated_media_refs,
                generated_media_blocks,
            );
        }

        tracing::debug!(
            transcript_turns = transcript.turns.len(),
            "rebuilt transcript from snapshot"
        );
        Ok(transcript)
    }

    fn compose_system_prompt(&self, agent: &AgentConfig, privacy: &PrivacyMode) -> String {
        self.compose_system_prompt_inner(agent, privacy, self.agent_memory_enabled(agent))
    }

    fn compose_subagent_system_prompt(&self, agent: &AgentConfig, privacy: &PrivacyMode) -> String {
        self.compose_system_prompt_inner(agent, privacy, false)
    }

    fn compose_system_prompt_inner(
        &self,
        agent: &AgentConfig,
        privacy: &PrivacyMode,
        include_memory: bool,
    ) -> String {
        let mut out = String::new();
        if let Some(extra) = self
            .config
            .extra_system_prompt
            .as_deref()
            .map(str::trim)
            .filter(|extra| !extra.is_empty())
        {
            out.push_str("Operator policy:\n");
            out.push_str(extra);
            out.push_str("\n\n");
        }
        out.push_str("Operational context:\n");
        out.push_str(&format!(
            "Bot build: {}. You are answering as model `{}` via `{}`.\n",
            self.config.version, agent.model.id, agent.provider
        ));
        out.push_str("Capabilities this turn:\n");
        if !agent.model.server_tools.is_empty() {
            out.push_str("- Provider-side tools configured on this model.\n");
        }
        if !matches!(privacy, PrivacyMode::ConversationOnly) {
            out.push_str("- Recent platform messages are available through fetch_messages.\n");
        }
        if let Some(binding) = &agent.image_generation {
            out.push_str(&format!(
                concat!(
                    "- Image generation and image editing are available through generate_image ",
                    "using provider `{}` and model `{}`. When the user asks to edit, restyle, ",
                    "transform, or make a variation of an existing image, pass the exact ",
                    "available URI in reference_images.\n"
                ),
                binding.provider, binding.model
            ));
        }
        if let Some(binding) = &agent.video_generation {
            out.push_str(&format!(
                "- Video generation is available through generate_video using provider `{}` and model `{}`.\n",
                binding.provider, binding.model
            ));
        }
        if let Some(binding) = &agent.audio_transcription {
            out.push_str(&format!(
                "- Audio transcription is available through transcribe_audio using provider `{}`{}.\n",
                binding.provider,
                binding
                    .model
                    .as_ref()
                    .map(|model| format!(" and model `{model}`"))
                    .unwrap_or_default()
            ));
            out.push_str("- Platform message JSON may include `audio_attachments` or attachment `audio_uri` fields. Use transcribe_audio with those file://audio/... URIs when the user's audio is relevant.\n");
        }
        if !agent.subagents.is_empty() {
            out.push_str("- Specialist subagents are available as tools.\n");
        }
        if include_memory {
            out.push_str("- User memory is available through lookup_user_memory, remember_user_memory, and forget_user_memory.\n");
        }
        out.push_str("- Generated image and video media are attached to the final platform reply automatically; do not paste media URLs, file:// URIs, filenames, or markdown media links in user-facing text.\n");
        out.push_str("- Slow work (video generation, subagent calls, research) SHOULD be narrated with calls to the post_status_message tool.\n");
        if include_memory {
            out.push_str(memory::prompt_guidance());
        }
        out.push_str("Agent Persona Prompt:\n");
        out.push_str(agent.system_prompt.trim());
        out
    }

    fn agent_memory_enabled(&self, agent: &AgentConfig) -> bool {
        self.memory_config.enabled && agent.memory
    }

    fn publish_conversation(&self, conversation_id: ConversationId, kind: ConversationEventKind) {
        tracing::trace!(
            conversation = %conversation_id,
            event = conversation_event_kind(kind),
            "publishing conversation event"
        );
        self.events.publish(LiveEvent::Conversation {
            conversation_id,
            kind,
        });
    }

    fn format_reply(&self, text: &str, is_new: bool, conversation_id: ConversationId) -> String {
        format_reply_content(text, is_new, conversation_id, &self.config.web_base_url)
    }

    fn ensure_provider_exists(
        &self,
        agent_name: &str,
        agent: &AgentConfig,
    ) -> Result<(), BotError> {
        if self.llms.contains_provider(&agent.provider) {
            tracing::trace!(
                agent = %agent_name,
                provider = %agent.provider,
                "provider is available"
            );
            return Ok(());
        }
        tracing::warn!(
            agent = %agent_name,
            provider = %agent.provider,
            "agent provider is not configured"
        );
        Err(BotError::MissingProvider {
            agent: agent_name.to_string(),
            provider: agent.provider.clone(),
        })
    }

    fn ensure_agent_services_exist(
        &self,
        agent_name: &str,
        agent: &AgentConfig,
    ) -> Result<(), BotError> {
        self.ensure_provider_exists(agent_name, agent)?;
        if let Some(binding) = &agent.image_generation
            && !self.images.contains_generator(&binding.provider)
        {
            tracing::warn!(
                agent = %agent_name,
                provider = %binding.provider,
                model = %binding.model,
                "agent image generation provider is not configured"
            );
            return Err(BotError::MissingImageGenerator {
                agent: agent_name.to_string(),
                provider: binding.provider.clone(),
            });
        }
        if let Some(binding) = &agent.video_generation
            && !self.videos.contains_generator(&binding.provider)
        {
            tracing::warn!(
                agent = %agent_name,
                provider = %binding.provider,
                model = %binding.model,
                "agent video generation provider is not configured"
            );
            return Err(BotError::MissingVideoGenerator {
                agent: agent_name.to_string(),
                provider: binding.provider.clone(),
            });
        }
        if let Some(binding) = &agent.audio_transcription
            && !self.audio.contains_transcriber(&binding.provider)
        {
            tracing::warn!(
                agent = %agent_name,
                provider = %binding.provider,
                model = ?binding.model.as_ref(),
                "agent audio transcription provider is not configured"
            );
            return Err(BotError::MissingAudioTranscriber {
                agent: agent_name.to_string(),
                provider: binding.provider.clone(),
            });
        }
        Ok(())
    }
}

/// Result of handling one event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BotAction {
    /// Event did not require work.
    Ignored,
    /// Event stream asked the bot to stop.
    Shutdown,
    /// Turn completed.
    CompletedTurn {
        /// Conversation id.
        conversation_id: ConversationId,
        /// Turn id.
        turn_id: TurnId,
    },
    /// Turn failed.
    FailedTurn {
        /// Conversation id.
        conversation_id: ConversationId,
        /// Turn id.
        turn_id: TurnId,
    },
    /// Turn was cancelled.
    CancelledTurn {
        /// Conversation id.
        conversation_id: ConversationId,
        /// Turn id.
        turn_id: TurnId,
    },
    /// Conversation was stopped.
    StoppedConversation {
        /// Conversation id.
        conversation_id: ConversationId,
    },
    /// Conversation was resumed.
    ResumedConversation {
        /// Conversation id.
        conversation_id: ConversationId,
    },
    /// Message was refused before turn creation.
    RefusedMessage,
    /// Platform command was handled.
    HandledCommand,
}

fn transcript_message_metadata(id: String) -> serde_json::Value {
    serde_json::json!({ "id": id })
}

fn turn_transcript_message_id(turn_id: TurnId, role: &str) -> String {
    format!("chudbot_turn_{turn_id}_{role}")
}

/// Errors from platform-neutral bot orchestration.
#[derive(Debug, Error)]
pub enum BotError {
    /// Messaging platform failed.
    #[error("platform error: {message}")]
    Platform {
        /// Error message.
        message: String,
    },
    /// Storage failed.
    #[error("storage error: {message}")]
    Storage {
        /// Error message.
        message: String,
    },
    /// Configured agent is missing.
    #[error("agent `{name}` is not configured")]
    MissingAgent {
        /// Agent name.
        name: String,
    },
    /// A subagent binding points at an unknown agent.
    #[error("agent `{agent}` references missing subagent `{subagent}`")]
    MissingSubagent {
        /// Parent agent name.
        agent: String,
        /// Missing subagent name.
        subagent: String,
    },
    /// Retry storage result did not include the requested turn.
    #[error("retry turn `{turn_id}` was not present in the loaded conversation")]
    MissingRetryTurn {
        /// Turn id.
        turn_id: TurnId,
    },
    /// Storage could not reload a conversation that was just referenced.
    #[error("conversation `{conversation_id}` was not found")]
    MissingConversation {
        /// Conversation id.
        conversation_id: ConversationId,
    },
    /// Agent references an unavailable provider.
    #[error("agent `{agent}` uses provider `{provider}` but that provider is not configured")]
    MissingProvider {
        /// Agent name.
        agent: String,
        /// Missing provider.
        provider: ProviderName,
    },
    /// Agent references an unavailable image generator.
    #[error(
        "agent `{agent}` uses image provider `{provider}` but that generator is not configured"
    )]
    MissingImageGenerator {
        /// Agent name.
        agent: String,
        /// Missing image provider.
        provider: ProviderName,
    },
    /// Agent references an unavailable video generator.
    #[error(
        "agent `{agent}` uses video provider `{provider}` but that generator is not configured"
    )]
    MissingVideoGenerator {
        /// Agent name.
        agent: String,
        /// Missing video provider.
        provider: ProviderName,
    },
    /// Agent references an unavailable audio transcriber.
    #[error(
        "agent `{agent}` uses audio provider `{provider}` but that transcriber is not configured"
    )]
    MissingAudioTranscriber {
        /// Agent name.
        agent: String,
        /// Missing audio provider.
        provider: ProviderName,
    },
    /// Agent media-generation binding is malformed.
    #[error("agent `{agent}` has invalid `{field}` binding: {message}")]
    InvalidGenerationBinding {
        /// Agent name.
        agent: String,
        /// Config field.
        field: &'static str,
        /// Error detail.
        message: String,
    },
    /// Agent graph is recursive.
    #[error("agent `{name}` recursively references itself through subagents")]
    RecursiveAgent {
        /// Agent name.
        name: String,
    },
    /// Command input could not be resolved.
    #[error("command input: {0}")]
    CommandInput(String),
    /// One-shot model operation failed.
    #[error("model operation failed: {message}")]
    Model {
        /// Error message.
        message: String,
    },
    /// Avatar download failed.
    #[error("avatar download failed: {0}")]
    AvatarDownload(String),
}

#[derive(Debug, Clone)]
struct TurnExecution {
    conversation: Conversation,
    turn: Turn,
    agent_name: String,
    agent_config: AgentConfig,
    system_prompt: String,
    transcript: Transcript,
    settings: RuntimeSettings,
    reply_to: MessageRef,
    is_new: bool,
}

#[derive(Debug, Clone)]
struct PreparedTurnContext {
    items: Vec<chudbot_api::ContextItem>,
}

#[derive(Debug)]
struct StoredAttachmentMedia {
    attachment_index: usize,
    media: chudbot_api::BoxedMediaRef,
}

#[derive(Debug)]
enum RuntimeTool<P, S, M, L, I, V, A>
where
    L: LlmProviderRegistry,
    I: ImageGeneratorRegistry,
    V: VideoGeneratorRegistry,
    A: AudioTranscriberRegistry,
{
    Fetch {
        name: ToolName,
        tool: FetchMessagesTool<P, S>,
    },
    Status {
        name: ToolName,
        tool: PostStatusTool<P, S>,
    },
    Image {
        name: ToolName,
        tool: ImageGeneratorTool<RoutedImageGenerator<I>, M>,
    },
    Video {
        name: ToolName,
        tool: PersistentVideoGeneratorTool<RoutedVideoGenerator<V>, M, S>,
    },
    Audio {
        name: ToolName,
        tool: AudioTranscriptionTool<RoutedAudioTranscriber<A>, M>,
    },
    Memory {
        name: ToolName,
        tool: memory::MemoryClientTool<S>,
    },
    Subagent {
        name: ToolName,
        tool: Subagent<RoutedLlmBackend<L>>,
    },
}

impl<P, S, M, L, I, V, A> RuntimeTool<P, S, M, L, I, V, A>
where
    P: MessagePlatformRegistry + Clone + 'static,
    S: BotStorage + Clone + 'static,
    M: MediaStore + Clone + 'static,
    L: LlmProviderRegistry + Clone + 'static,
    I: ImageGeneratorRegistry + Clone + 'static,
    V: VideoGeneratorRegistry + Clone + 'static,
    A: AudioTranscriberRegistry + Clone + 'static,
{
    fn start(self, spec: AgentSpec) -> AgentBuilder {
        match self {
            Self::Fetch { name, tool } => spec.with_tool(name, tool),
            Self::Status { name, tool } => spec.with_tool(name, tool),
            Self::Image { name, tool } => spec.with_tool(name, tool),
            Self::Video { name, tool } => spec.with_tool(name, tool),
            Self::Audio { name, tool } => spec.with_tool(name, tool),
            Self::Memory { name, tool } => spec.with_tool(name, tool),
            Self::Subagent { name, tool } => spec.with_tool(name, tool),
        }
    }

    fn append(self, builder: AgentBuilder) -> AgentBuilder {
        match self {
            Self::Fetch { name, tool } => builder.with_tool(name, tool),
            Self::Status { name, tool } => builder.with_tool(name, tool),
            Self::Image { name, tool } => builder.with_tool(name, tool),
            Self::Video { name, tool } => builder.with_tool(name, tool),
            Self::Audio { name, tool } => builder.with_tool(name, tool),
            Self::Memory { name, tool } => builder.with_tool(name, tool),
            Self::Subagent { name, tool } => builder.with_tool(name, tool),
        }
    }
}

#[derive(Debug, Clone)]
struct AudioTranscriptionTool<T, M> {
    transcriber: T,
    media_store: M,
    description: String,
}

impl<T, M> AudioTranscriptionTool<T, M> {
    fn new(transcriber: T, media_store: M) -> Self {
        Self {
            transcriber,
            media_store,
            description: "Transcribe a stored audio attachment and return its speech as text."
                .to_string(),
        }
    }

    fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

impl<T, M> ClientTool for AudioTranscriptionTool<T, M>
where
    T: AudioTranscriber,
    M: MediaStore,
{
    type Error = BotToolError;

    fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.description.clone(),
            input_schema: audio_transcription_tool_schema(),
        }
    }

    #[tracing::instrument(
        name = "tool.transcribe_audio",
        skip_all,
        fields(tool_call = %call.id)
    )]
    async fn call(&self, call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
        let request =
            audio_transcription_request_from_tool_input(&self.media_store, call.input).await?;
        let audio_uri = request.audio.uri().to_string();
        let audio_mime_type = request.audio.mime_type().to_string();
        let audio_size_bytes = request.audio.size_bytes();
        tracing::debug!(
            audio_uri = %audio_uri,
            audio_mime_type = %audio_mime_type,
            audio_size_bytes,
            language = ?request.language.as_deref(),
            keyterms = request.keyterms.len(),
            model = ?request.model.as_ref().map(ModelId::as_str),
            "parsed audio transcription request"
        );
        let transcription = self
            .transcriber
            .transcribe_audio(request)
            .await
            .map_err(|error| {
                tracing::warn!(error = %error, "audio transcription failed");
                BotToolError::Generator(error.to_string())
            })?;
        let result = audio_transcription_model_result_json(&transcription);
        let trace_response = serde_json::json!({
            "audio": {
                "uri": audio_uri,
                "mime_type": audio_mime_type,
                "size_bytes": audio_size_bytes,
            },
            "transcription": result,
        });
        tracing::info!(
            duration_seconds = transcription.duration_seconds,
            text_chars = transcription.text.chars().count(),
            usage_records = transcription.usage.len(),
            "audio transcription tool completed"
        );

        Ok(ClientToolOutput {
            result: ClientToolResultContent::Json {
                value: result.clone(),
            },
            is_error: false,
            trace_response,
            usage: transcription.usage,
        })
    }
}

#[derive(Debug, Clone)]
struct PersistentVideoGeneratorTool<G, M, S> {
    generator: G,
    media_store: M,
    storage: S,
    turn_id: TurnId,
    provider: ProviderName,
    description: String,
    poll_interval: Duration,
    max_polls: u32,
}

impl<G, M, S> PersistentVideoGeneratorTool<G, M, S> {
    fn new(
        generator: G,
        media_store: M,
        storage: S,
        turn_id: TurnId,
        provider: ProviderName,
    ) -> Self {
        Self {
            generator,
            media_store,
            storage,
            turn_id,
            provider,
            description: "Generate a video, save it to media storage, and return its media URI."
                .to_string(),
            poll_interval: Duration::from_secs(2),
            max_polls: 600,
        }
    }

    fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

impl<G, M, S> ClientTool for PersistentVideoGeneratorTool<G, M, S>
where
    G: VideoGenerator,
    M: MediaStore,
    S: BotStorage,
{
    type Error = BotToolError;

    fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: self.description.clone(),
            input_schema: video_tool_schema(),
        }
    }

    #[tracing::instrument(
        name = "tool.generate_video",
        skip_all,
        fields(turn = %self.turn_id, provider = %self.provider, tool_call = %call.id)
    )]
    async fn call(&self, call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
        let request = video_request_from_tool_input(&self.media_store, call.input).await?;
        let prompt = request.prompt.clone();
        let job_id = self
            .generator
            .submit_video(request)
            .await
            .map_err(|error| BotToolError::Generator(error.to_string()))?;
        self.storage
            .create_video_job(CreateVideoJob {
                turn_id: self.turn_id,
                provider: self.provider.clone(),
                provider_job_id: job_id.as_str().to_string(),
                prompt,
            })
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?;
        tracing::info!(job = %job_id, "video job submitted and persisted");

        for poll in 0..self.max_polls {
            match self
                .generator
                .check_video(job_id.clone())
                .await
                .map_err(|error| BotToolError::Generator(error.to_string()))?
            {
                VideoJobStatus::Pending => {
                    if poll + 1 < self.max_polls {
                        tokio::time::sleep(self.poll_interval).await;
                    }
                }
                VideoJobStatus::Done { meta } => {
                    let bytes = self
                        .generator
                        .download_video(meta.url.clone())
                        .await
                        .map_err(|error| BotToolError::Generator(error.to_string()))?;
                    let media = self
                        .media_store
                        .create_media(CreateMedia {
                            category: MediaCategory::Video,
                            bytes,
                            mime_type: None,
                            name: None,
                            extension: None,
                        })
                        .await
                        .map_err(|error| BotToolError::Media(error.to_string()))?;
                    self.storage
                        .update_video_job(UpdateVideoJob {
                            provider: self.provider.clone(),
                            provider_job_id: job_id.as_str().to_string(),
                            status: "done".to_string(),
                            output_uri: Some(media.uri().clone()),
                            error: None,
                        })
                        .await
                        .map_err(|error| BotToolError::Storage(error.to_string()))?;
                    let public_url = media.public_url().await.ok();
                    let trace_response = media_tool_trace_json(
                        media.as_ref(),
                        public_url.as_ref().map(|url| url.as_str()),
                        serde_json::json!({
                            "provider_job_id": job_id.as_str(),
                            "download_url": meta.url,
                            "duration_seconds": meta.duration_seconds,
                        }),
                    );
                    let result = media_tool_model_result_json(
                        media.as_ref(),
                        serde_json::json!({
                            "provider_job_id": job_id.as_str(),
                            "duration_seconds": meta.duration_seconds,
                        }),
                    );
                    tracing::info!(job = %job_id, uri = %media.uri(), "video job completed");
                    return Ok(ClientToolOutput {
                        result: ClientToolResultContent::Json {
                            value: result.clone(),
                        },
                        is_error: false,
                        trace_response,
                        usage: meta.usage,
                    });
                }
                VideoJobStatus::Failed { message } => {
                    self.storage
                        .update_video_job(UpdateVideoJob {
                            provider: self.provider.clone(),
                            provider_job_id: job_id.as_str().to_string(),
                            status: "failed".to_string(),
                            output_uri: None,
                            error: Some(message.clone()),
                        })
                        .await
                        .map_err(|error| BotToolError::Storage(error.to_string()))?;
                    return Err(BotToolError::Generator(format!(
                        "video generation failed: {message}"
                    )));
                }
                VideoJobStatus::Expired => {
                    self.storage
                        .update_video_job(UpdateVideoJob {
                            provider: self.provider.clone(),
                            provider_job_id: job_id.as_str().to_string(),
                            status: "expired".to_string(),
                            output_uri: None,
                            error: Some("expired".to_string()),
                        })
                        .await
                        .map_err(|error| BotToolError::Storage(error.to_string()))?;
                    return Err(BotToolError::Generator(
                        "video generation job expired".to_string(),
                    ));
                }
            }
        }

        let message = format!(
            "video generation still pending after {} polls: {}",
            self.max_polls, job_id
        );
        self.storage
            .update_video_job(UpdateVideoJob {
                provider: self.provider.clone(),
                provider_job_id: job_id.as_str().to_string(),
                status: "pending".to_string(),
                output_uri: None,
                error: Some(message.clone()),
            })
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?;
        Err(BotToolError::Generator(message))
    }
}

#[derive(Debug, Clone)]
struct FetchMessagesTool<P, S> {
    platforms: P,
    storage: S,
    default_channel: ChannelRef,
    privacy: PrivacyMode,
}

impl<P, S> ClientTool for FetchMessagesTool<P, S>
where
    P: MessagePlatformRegistry + Clone,
    S: BotStorage + Clone,
{
    type Error = BotToolError;

    fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: "Fetch recent messages from the current channel for context.".to_string(),
            input_schema: ToolInputSchema::new(serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Optional platform channel id. Defaults to the current channel."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 20
                    },
                    "before_message_id": {
                        "type": "string",
                        "description": "Optional platform message id to page before."
                    }
                },
                "additionalProperties": false
            })),
        }
    }

    #[tracing::instrument(
        name = "tool.fetch_messages",
        skip_all,
        fields(
            tool_call = %call.id,
            default_platform = %self.default_channel.platform,
            default_channel = %self.default_channel.channel_id,
            privacy = privacy_mode_kind(&self.privacy),
        )
    )]
    async fn call(&self, call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
        let channel = requested_channel(&self.default_channel, &call.input)?;
        if let PrivacyMode::ChannelOnly {
            channel: allowed, ..
        } = &self.privacy
            && &channel != allowed
        {
            tracing::warn!(
                requested_channel = %channel.channel_id,
                allowed_channel = %allowed.channel_id,
                "fetch_messages rejected by channel_only privacy mode"
            );
            return Err(BotToolError::InvalidInput(
                "fetch_messages is limited to the configured channel".to_string(),
            ));
        }
        let limit = call
            .input
            .get("limit")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u16::try_from(value).ok())
            .unwrap_or(20)
            .clamp(1, 100);
        let before = call
            .input
            .get("before_message_id")
            .and_then(serde_json::Value::as_str)
            .map(|message_id| MessageRef {
                platform: channel.platform.clone(),
                guild_id: channel.guild_id.clone(),
                channel_id: channel.channel_id.clone(),
                message_id: message_id.into(),
            });
        let messages = self
            .platforms
            .fetch_messages(FetchMessages {
                channel: channel.clone(),
                limit,
                before,
            })
            .await
            .map_err(|error| BotToolError::Platform(error.to_string()))?;
        let messages =
            redact_messages_for_privacy(&self.storage, &self.privacy, &channel, messages).await?;
        tracing::info!(
            messages = messages.len(),
            limit,
            "fetched platform messages"
        );
        let mut rendered = Vec::with_capacity(messages.len());
        for message in &messages {
            rendered.push(
                self.platforms
                    .message_context(message, PlatformMessageRelationship::Fetched)
                    .await
                    .map_err(|error| BotToolError::Platform(error.to_string()))?,
            );
        }
        let value = serde_json::Value::Array(rendered);
        Ok(ClientToolOutput {
            result: ClientToolResultContent::Json {
                value: value.clone(),
            },
            is_error: false,
            trace_response: value,
            usage: Vec::new(),
        })
    }
}

async fn redact_messages_for_privacy<S>(
    storage: &S,
    privacy: &PrivacyMode,
    channel: &ChannelRef,
    messages: Vec<PlatformMessage>,
) -> Result<Vec<PlatformMessage>, BotToolError>
where
    S: BotStorage,
{
    if !matches!(privacy, PrivacyMode::OptIn) {
        return Ok(messages);
    }
    let Some(guild_id) = channel.guild_id.as_ref() else {
        return Ok(messages);
    };
    let mut redacted = Vec::with_capacity(messages.len());
    for mut message in messages {
        let opted_in = storage
            .user_privacy(
                channel.platform.clone(),
                guild_id.as_str().to_string(),
                message.author.id.user_id.as_str().to_string(),
            )
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?
            .unwrap_or(false);
        if !opted_in {
            message.content = "[redacted: user has not opted in]".to_string();
            message.mentions.clear();
            message.mention_profiles.clear();
            message.attachments.clear();
            message.reference = PlatformMessageReference::None;
        }
        redacted.push(message);
    }
    Ok(redacted)
}

#[derive(Debug, Clone)]
struct PostStatusTool<P, S> {
    platforms: P,
    storage: S,
    channel: ChannelRef,
    reply_to: MessageRef,
    conversation_id: ConversationId,
    turn_id: TurnId,
}

impl<P, S> ClientTool for PostStatusTool<P, S>
where
    P: MessagePlatformRegistry + Clone,
    S: BotStorage + Clone,
{
    type Error = BotToolError;

    fn spec(&self) -> ClientToolSpec {
        ClientToolSpec {
            description: "Post a short interim status reply before slow work.".to_string(),
            input_schema: ToolInputSchema::new(serde_json::json!({
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Short status message to send to the user."
                    }
                },
                "additionalProperties": false
            })),
        }
    }

    #[tracing::instrument(
        name = "tool.post_status_message",
        skip_all,
        fields(
            tool_call = %call.id,
            conversation = %self.conversation_id,
            turn = %self.turn_id,
            platform = %self.channel.platform,
            channel = %self.channel.channel_id,
            reply_to = %self.reply_to.message_id,
        )
    )]
    async fn call(&self, call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
        let text = call
            .input
            .get("text")
            .and_then(serde_json::Value::as_str)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| BotToolError::InvalidInput("`text` is required".to_string()))?;
        tracing::debug!(text_chars = text.chars().count(), "posting status message");
        let posted = self
            .platforms
            .send_message(SendMessage {
                channel: self.channel.clone(),
                reply_to: Some(self.reply_to.clone()),
                content: text.to_string(),
                attachments: Vec::new(),
                suppress_embeds: true,
                open_thread: None,
            })
            .await
            .map_err(|error| BotToolError::Platform(error.to_string()))?;
        tracing::info!(
            message = %posted.id.message_id,
            channel = %posted.channel.channel_id,
            "posted status message"
        );
        self.storage
            .link_message(MessageLink {
                message: posted.id.clone(),
                conversation_id: self.conversation_id,
                turn_id: self.turn_id,
                role: "assistant_status".to_string(),
            })
            .await
            .map_err(|error| BotToolError::Storage(error.to_string()))?;
        for message in &posted.extra_messages {
            self.storage
                .link_message(MessageLink {
                    message: message.clone(),
                    conversation_id: self.conversation_id,
                    turn_id: self.turn_id,
                    role: "assistant_status".to_string(),
                })
                .await
                .map_err(|error| BotToolError::Storage(error.to_string()))?;
        }
        let value = serde_json::json!({
            "message": posted.id,
            "channel": posted.channel,
            "extra_messages": posted.extra_messages,
        });
        Ok(ClientToolOutput {
            result: ClientToolResultContent::Json {
                value: value.clone(),
            },
            is_error: false,
            trace_response: value,
            usage: Vec::new(),
        })
    }
}

#[derive(Debug, Error)]
enum BotToolError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("platform error: {0}")]
    Platform(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("generator error: {0}")]
    Generator(String),
    #[error("media error: {0}")]
    Media(String),
}

async fn video_request_from_tool_input<M>(
    media_store: &M,
    input: serde_json::Value,
) -> Result<VideoRequest, BotToolError>
where
    M: MediaStore,
{
    let prompt = tool_required_string(&input, "prompt")?;
    let image = match input.get("image").or_else(|| input.get("image_url")) {
        Some(value) => {
            Some(resolve_tool_media_arg(media_store, MediaCategory::Image, value).await?)
        }
        None => None,
    };
    Ok(VideoRequest {
        prompt,
        image,
        duration_seconds: tool_optional_u8_bounded(&input, "duration_seconds", 15)?,
        aspect_ratio: tool_optional_string(&input, "aspect_ratio")?,
        resolution: tool_optional_string(&input, "resolution")?,
        model: tool_optional_string(&input, "model")?.map(ModelId::new),
    })
}

async fn audio_transcription_request_from_tool_input<M>(
    media_store: &M,
    input: serde_json::Value,
) -> Result<AudioTranscriptionRequest, BotToolError>
where
    M: MediaStore,
{
    let audio_value = input
        .get("audio_uri")
        .or_else(|| input.get("audio"))
        .ok_or_else(|| BotToolError::InvalidInput("`audio_uri` is required".to_string()))?;
    let audio = resolve_tool_media_arg(media_store, MediaCategory::Audio, audio_value).await?;
    let keyterms = match tool_optional_string_list(&input, "keyterm")? {
        Some(keyterms) => keyterms,
        None => tool_optional_string_list(&input, "keyterms")?.unwrap_or_default(),
    };
    Ok(AudioTranscriptionRequest {
        audio,
        language: tool_optional_string(&input, "language")?,
        keyterms,
        model: tool_optional_string(&input, "model")?.map(ModelId::new),
    })
}

async fn resolve_tool_media_arg<M>(
    media_store: &M,
    category: MediaCategory,
    value: &serde_json::Value,
) -> Result<chudbot_api::BoxedMediaRef, BotToolError>
where
    M: MediaStore,
{
    let text = value.as_str().ok_or_else(|| {
        BotToolError::InvalidInput("media references must be strings".to_string())
    })?;
    if text.starts_with("http://") || text.starts_with("https://") {
        return Ok(UrlMediaRef::new(category, text, "application/octet-stream").boxed());
    }
    media_store
        .media_from_uri(&MediaUri::new(text))
        .await
        .map_err(|error| BotToolError::Media(error.to_string()))
}

fn tool_required_string(input: &serde_json::Value, field: &str) -> Result<String, BotToolError> {
    input
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| BotToolError::InvalidInput(format!("`{field}` is required")))
}

fn tool_optional_string(
    input: &serde_json::Value,
    field: &str,
) -> Result<Option<String>, BotToolError> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    value
        .as_str()
        .map(str::to_string)
        .map(Some)
        .ok_or_else(|| BotToolError::InvalidInput(format!("`{field}` must be a string")))
}

fn tool_optional_string_list(
    input: &serde_json::Value,
    field: &str,
) -> Result<Option<Vec<String>>, BotToolError> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    if let Some(text) = value.as_str() {
        return Ok(Some(vec![text.to_string()]));
    }
    let Some(values) = value.as_array() else {
        return Err(BotToolError::InvalidInput(format!(
            "`{field}` must be a string or array of strings"
        )));
    };
    values
        .iter()
        .map(|value| {
            value.as_str().map(str::to_string).ok_or_else(|| {
                BotToolError::InvalidInput(format!("`{field}` must only contain strings"))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn tool_optional_u8(input: &serde_json::Value, field: &str) -> Result<Option<u8>, BotToolError> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    let Some(value) = value.as_u64() else {
        return Err(BotToolError::InvalidInput(format!(
            "`{field}` must be an integer"
        )));
    };
    u8::try_from(value)
        .map(Some)
        .map_err(|_| BotToolError::InvalidInput(format!("`{field}` is too large")))
}

fn tool_optional_u8_bounded(
    input: &serde_json::Value,
    field: &str,
    max: u8,
) -> Result<Option<u8>, BotToolError> {
    let value = tool_optional_u8(input, field)?;
    if let Some(value) = value
        && value > max
    {
        return Err(BotToolError::InvalidInput(format!(
            "`{field}` must be at most {max}"
        )));
    }
    Ok(value)
}

fn media_tool_trace_json(
    media: &dyn chudbot_api::MediaRef,
    public_url: Option<&str>,
    extra: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "uri": media.uri().as_str(),
        "category": media.category(),
        "name": media.name(),
        "mime_type": media.mime_type(),
        "size_bytes": media.size_bytes(),
        "public_url": public_url,
        "extra": extra,
    })
}

fn media_tool_model_result_json(
    media: &dyn chudbot_api::MediaRef,
    extra: serde_json::Value,
) -> serde_json::Value {
    serde_json::json!({
        "uri": media.uri().as_str(),
        "category": media.category(),
        "mime_type": media.mime_type(),
        "size_bytes": media.size_bytes(),
        "delivery": {
            "platform_reply": "The generated media will be attached to the final platform reply automatically. Do not paste media URIs, filenames, public URLs, or markdown image/video links in user-facing text."
        },
        "extra": extra,
    })
}

fn audio_transcription_model_result_json(transcription: &AudioTranscription) -> serde_json::Value {
    serde_json::json!({
        "text": transcription.text,
        "language": transcription.language,
        "duration_seconds": transcription.duration_seconds,
        "words": transcription.words,
        "channels": transcription.channels,
        "model": transcription.model.as_ref().map(ModelId::as_str),
    })
}

fn audio_transcription_tool_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "required": ["audio_uri"],
        "properties": {
            "audio_uri": {
                "type": "string",
                "description": "A file://audio/... URI from the message JSON audio_attachments or attachment audio_uri field."
            },
            "audio": {
                "type": "string",
                "description": "Alias for audio_uri."
            },
            "language": {
                "type": "string",
                "description": "Optional language code such as en, fr, de, or ja for text formatting."
            },
            "keyterm": {
                "oneOf": [
                    { "type": "string" },
                    { "type": "array", "items": { "type": "string" } }
                ],
                "description": "Optional key term or terms to bias transcription toward."
            },
            "keyterms": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Alias for keyterm when passing multiple terms."
            },
            "model": {
                "type": "string",
                "description": "Optional provider-specific transcription model id."
            }
        },
        "additionalProperties": false
    }))
}

fn video_tool_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The video prompt."
            },
            "image": {
                "type": "string",
                "description": "Optional media URI or public URL for an image to animate. Use file:// media URIs from prior tool results; do not invent local filesystem paths."
            },
            "image_url": {
                "type": "string",
                "description": "Alias for image."
            },
            "duration_seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 15
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Optional provider-specific aspect ratio."
            },
            "resolution": {
                "type": "string",
                "description": "Optional provider-specific resolution or quality tier."
            },
            "model": {
                "type": "string",
                "description": "Optional provider-specific model id."
            }
        },
        "additionalProperties": false
    }))
}

fn media_uris_from_tool_traces(trace: &[ToolTrace]) -> Vec<MediaUri> {
    let mut seen = Vec::<String>::new();
    let mut out = Vec::new();
    for trace in trace {
        let ToolTrace::Client { trace } = trace else {
            continue;
        };
        if trace.result.is_error {
            continue;
        }
        let Some(uri) = trace
            .trace_response
            .get("uri")
            .or_else(|| trace.trace_response.get("image_uri"))
            .or_else(|| trace.trace_response.get("video_uri"))
            .and_then(serde_json::Value::as_str)
            .filter(|uri| uri.starts_with("file://"))
        else {
            continue;
        };
        if seen.iter().any(|seen| seen == uri) {
            continue;
        }
        seen.push(uri.to_string());
        out.push(MediaUri::new(uri));
    }
    out
}

fn generated_media_reply_refs(trace: &[ToolTrace]) -> Vec<String> {
    let mut out = Vec::new();
    for trace in trace {
        let ToolTrace::Client { trace } = trace else {
            continue;
        };
        if trace.result.is_error {
            continue;
        }
        collect_generated_media_reply_refs(&trace.trace_response, &mut out);
        if let ClientToolResultContent::Json { value } = &trace.result.content {
            collect_generated_media_reply_refs(value, &mut out);
        }
    }
    out
}

fn collect_generated_media_reply_refs(value: &serde_json::Value, out: &mut Vec<String>) {
    for key in ["public_url", "uri", "image_uri", "video_uri"] {
        let Some(reference) = value.get(key).and_then(serde_json::Value::as_str) else {
            continue;
        };
        if reference.is_empty() || out.iter().any(|seen| seen == reference) {
            continue;
        }
        out.push(reference.to_string());
    }
}

fn replay_asset_belongs_to_user_turn(asset: &TurnAsset) -> bool {
    asset.source.starts_with("platform:")
}

fn append_generated_media_replay(
    transcript: &mut Transcript,
    turn_id: TurnId,
    media_refs: Vec<String>,
    mut media_blocks: Vec<ContentBlock>,
) {
    if media_blocks.is_empty() {
        return;
    }

    let mut text = "Generated media attached to the previous assistant reply.".to_string();
    if !media_refs.is_empty() {
        text.push_str(" Image reference IDs available for tool calls: ");
        text.push_str(&media_refs.join(", "));
        text.push_str(concat!(
            ". Use these exact IDs in generate_image.reference_images when the user asks to ",
            "edit, restyle, transform, or make a variation of the images."
        ));
    }

    let mut blocks = Vec::with_capacity(media_blocks.len() + 1);
    blocks.push(ContentBlock::Text { text });
    blocks.append(&mut media_blocks);
    transcript.push(TranscriptTurn {
        role: TurnRole::User,
        blocks,
        metadata: transcript_message_metadata(turn_transcript_message_id(
            turn_id,
            "assistant_media",
        )),
    });
}

fn append_client_tool_replay(transcript: &mut Transcript, traces: &[ToolTrace]) {
    let mut call_blocks = Vec::new();
    let mut result_blocks = Vec::new();
    for trace in traces {
        let ToolTrace::Client { trace } = trace else {
            continue;
        };
        call_blocks.push(ContentBlock::ClientToolCall(trace.call.clone()));
        result_blocks.push(ContentBlock::ClientToolResult(trace.result.clone()));
    }
    if call_blocks.is_empty() {
        return;
    }
    transcript.push(TranscriptTurn {
        role: TurnRole::Assistant,
        blocks: call_blocks,
        metadata: serde_json::Value::Null,
    });
    transcript.push(TranscriptTurn {
        role: TurnRole::User,
        blocks: result_blocks,
        metadata: serde_json::Value::Null,
    });
}

async fn media_reply_refs_from_transcript(transcript: &Transcript) -> Vec<String> {
    let mut out = Vec::new();
    for turn in &transcript.turns {
        for block in &turn.blocks {
            let ContentBlock::Media { media } = block else {
                if let ContentBlock::ClientToolResult(result) = block
                    && let ClientToolResultContent::Json { value } = &result.content
                {
                    collect_generated_media_reply_refs(value, &mut out);
                }
                continue;
            };
            push_unique_string(&mut out, media.uri().as_str());
            if let Ok(public_url) = media.public_url().await {
                push_unique_string(&mut out, public_url.as_str());
            }
        }
    }
    out
}

fn push_unique_string(out: &mut Vec<String>, value: &str) {
    if value.is_empty() || out.iter().any(|seen| seen == value) {
        return;
    }
    out.push(value.to_string());
}

fn strip_generated_media_refs(text: &str, refs: &[String]) -> String {
    if refs.is_empty() {
        return text.to_string();
    }

    let mut out = text.to_string();
    for reference in refs {
        if reference.is_empty() {
            continue;
        }
        while let Some(index) = out.find(reference) {
            let (start, end) = markdown_link_bounds(&out, index, reference.len())
                .unwrap_or((index, index + reference.len()));
            out.replace_range(start..end, "");
        }
    }
    normalize_stripped_reply(&out)
}

fn markdown_link_bounds(
    text: &str,
    reference_start: usize,
    reference_len: usize,
) -> Option<(usize, usize)> {
    let open_paren = reference_start.checked_sub(1)?;
    if text.as_bytes().get(open_paren) != Some(&b'(') {
        return None;
    }
    let close_paren = reference_start.checked_add(reference_len)?;
    if text.as_bytes().get(close_paren) != Some(&b')') {
        return None;
    }
    let before_open = &text[..open_paren];
    if !before_open.ends_with(']') {
        return None;
    }
    let close_bracket = before_open.len().checked_sub(1)?;
    let open_bracket = before_open[..close_bracket].rfind('[')?;
    let start = if open_bracket > 0 && text.as_bytes().get(open_bracket - 1) == Some(&b'!') {
        open_bracket - 1
    } else {
        open_bracket
    };
    Some((start, close_paren + 1))
}

fn normalize_stripped_reply(text: &str) -> String {
    let mut lines = Vec::new();
    let mut previous_blank = true;
    for line in text.lines() {
        let line = line.trim_end();
        let blank = line.trim().is_empty();
        if blank {
            if !previous_blank {
                lines.push(String::new());
            }
        } else {
            lines.push(line.to_string());
        }
        previous_blank = blank;
    }
    while lines.last().is_some_and(|line| line.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

fn platform_error(error: impl std::fmt::Display) -> BotError {
    BotError::Platform {
        message: error.to_string(),
    }
}

fn storage_error(error: impl std::fmt::Display) -> BotError {
    BotError::Storage {
        message: error.to_string(),
    }
}

fn log_task_join_error(task: &'static str, error: &JoinError) {
    if error.is_cancelled() {
        tracing::debug!(task, error = %error, "task was cancelled");
    } else if error.is_panic() {
        tracing::error!(task, error = %error, "task panicked");
    } else {
        tracing::warn!(task, error = %error, "task join failed");
    }
}

fn spawn_background_task<F>(tracker: &TaskTracker, task: &'static str, future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    tracker.spawn(async move {
        if let Err(error) = tokio::spawn(future).await {
            log_task_join_error(task, &error);
        }
    });
}

fn log_event_task_result(result: Result<(&'static str, Result<BotAction, BotError>), JoinError>) {
    match result {
        Ok((event, Ok(action))) => {
            tracing::debug!(
                event,
                action = bot_action_kind(&action),
                "event task completed"
            )
        }
        Ok((event, Err(error))) => {
            tracing::warn!(event, error = %error, "event task failed")
        }
        Err(error) if error.is_cancelled() => {
            tracing::debug!("event task was cancelled during shutdown")
        }
        Err(error) if error.is_panic() => tracing::error!(error = %error, "event task panicked"),
        Err(error) => tracing::warn!(error = %error, "event task join failed"),
    }
}

async fn drain_event_tasks(
    tasks: &mut JoinSet<(&'static str, Result<BotAction, BotError>)>,
    timeout: Duration,
) {
    if tasks.is_empty() {
        tracing::debug!("no in-flight event tasks to drain");
        return;
    }

    tracing::info!(
        in_flight = tasks.len(),
        timeout_ms = timeout.as_millis(),
        "draining in-flight event tasks"
    );
    let drained = tokio::time::timeout(timeout, async {
        while let Some(result) = tasks.join_next().await {
            log_event_task_result(result);
        }
    })
    .await;
    if drained.is_ok() {
        tracing::info!("in-flight event tasks drained");
        return;
    }

    let remaining = tasks.len();
    tracing::warn!(
        remaining,
        timeout_ms = timeout.as_millis(),
        "event task drain timed out; aborting remaining tasks"
    );
    tasks.abort_all();
    while let Some(result) = tasks.join_next().await {
        log_event_task_result(result);
    }
}

async fn drain_background_tasks(tracker: &TaskTracker, timeout: Duration) {
    if tracker.is_empty() {
        tracing::debug!("no background tasks to drain");
        tracker.close();
        return;
    }

    tracing::info!(
        in_flight = tracker.len(),
        timeout_ms = timeout.as_millis(),
        "draining background tasks"
    );
    tracker.close();

    if tokio::time::timeout(timeout, tracker.wait()).await.is_ok() {
        tracing::info!("background tasks drained");
        return;
    }

    tracing::warn!(
        remaining = tracker.len(),
        timeout_ms = timeout.as_millis(),
        "background task drain timed out"
    );
}

fn guild_key(message: &MessageRef) -> Option<String> {
    message.guild_id.as_ref().map(|id| id.as_str().to_string())
}

fn channel_from_message(message: &MessageRef) -> ChannelRef {
    ChannelRef {
        platform: message.platform.clone(),
        guild_id: message.guild_id.clone(),
        channel_id: message.channel_id.clone(),
    }
}

fn display_name(message: &PlatformMessage) -> String {
    message
        .author
        .display_name
        .clone()
        .or_else(|| message.author.name.clone())
        .unwrap_or_else(|| message.author.username.clone())
}

fn same_platform_user(left: &chudbot_api::UserRef, right: &chudbot_api::UserRef) -> bool {
    left.platform == right.platform && left.user_id == right.user_id
}

fn normalize_mention_content(
    content: &str,
    bot_user: &chudbot_api::UserRef,
    mentions: &[chudbot_api::UserRef],
    profiles: &[UserProfile],
) -> String {
    let mut out = strip_user_mention(content, bot_user).trim().to_string();
    for mention in mentions {
        if same_platform_user(mention, bot_user) {
            continue;
        }
        let label = profiles
            .iter()
            .find(|profile| same_platform_user(&profile.id, mention))
            .map(display_name_for_profile)
            .unwrap_or_else(|| mention.user_id.as_str().to_string());
        let replacement = format!("{label} (<@{}>)", mention.user_id.as_str());
        out = out
            .replace(&format!("<@{}>", mention.user_id.as_str()), &replacement)
            .replace(&format!("<@!{}>", mention.user_id.as_str()), &replacement);
    }
    out
}

fn strip_user_mention(content: &str, user: &chudbot_api::UserRef) -> String {
    content
        .replace(&format!("<@{}>", user.user_id.as_str()), "")
        .replace(&format!("<@!{}>", user.user_id.as_str()), "")
}

fn display_name_for_profile(profile: &UserProfile) -> String {
    profile
        .display_name
        .clone()
        .or_else(|| profile.name.clone())
        .unwrap_or_else(|| profile.username.clone())
}

fn message_link_replays_as_assistant(link: &MessageLink, conversation_id: ConversationId) -> bool {
    link.conversation_id == conversation_id && link.role == "assistant"
}

fn model_transcript_supports_media(media: &dyn chudbot_api::MediaRef) -> bool {
    matches!(media.category(), MediaCategory::Image)
        && model_transcript_supports_image_mime_type(media.mime_type())
}

fn model_transcript_supports_image_mime_type(mime_type: &str) -> bool {
    let mime_type = mime_type.split(';').next().unwrap_or("").trim();
    MODEL_TRANSCRIPT_IMAGE_MIME_TYPES
        .iter()
        .any(|supported| mime_type.eq_ignore_ascii_case(supported))
}

fn inject_audio_attachment_refs(
    value: &mut serde_json::Value,
    audio_media: &[StoredAttachmentMedia],
) {
    if audio_media.is_empty() {
        return;
    }
    let Some(object) = value.as_object_mut() else {
        return;
    };
    let audio_attachments = audio_media
        .iter()
        .map(|saved| serde_json::Value::String(saved.media.uri().to_string()))
        .collect::<Vec<_>>();
    if let Some(attachments) = object
        .get_mut("attachments")
        .and_then(serde_json::Value::as_array_mut)
    {
        for saved in audio_media {
            let Some(attachment) = attachments
                .get_mut(saved.attachment_index)
                .and_then(serde_json::Value::as_object_mut)
            else {
                continue;
            };
            attachment.insert(
                "audio_uri".to_string(),
                serde_json::Value::String(saved.media.uri().to_string()),
            );
        }
    }
    object.insert(
        "audio_attachments".to_string(),
        serde_json::Value::Array(audio_attachments),
    );
}

fn looks_like_image_ref(attachment: &AttachmentRef) -> bool {
    looks_like_image(attachment.content_type.as_deref(), &attachment.filename)
}

fn looks_like_audio_ref(attachment: &AttachmentRef) -> bool {
    looks_like_audio(
        attachment.content_type.as_deref(),
        &attachment.filename,
        attachment.is_voice_message,
    )
}

fn looks_like_image(content_type: Option<&str>, filename: &str) -> bool {
    if content_type
        .map(|content_type| content_type.starts_with("image/"))
        .unwrap_or(false)
    {
        return true;
    }
    matches!(
        extension_from_filename(filename).as_deref(),
        Some("png" | "jpg" | "jpeg" | "gif" | "webp" | "heic" | "heif")
    )
}

fn looks_like_audio(content_type: Option<&str>, filename: &str, is_voice_message: bool) -> bool {
    if is_voice_message {
        return true;
    }
    if content_type
        .map(|content_type| content_type.starts_with("audio/"))
        .unwrap_or(false)
    {
        return true;
    }
    matches!(
        extension_from_filename(filename).as_deref(),
        Some("mp3" | "wav" | "ogg" | "opus" | "m4a" | "aac" | "flac" | "webm")
    )
}

fn extension_from_filename(filename: &str) -> Option<String> {
    filename
        .rsplit_once('.')
        .map(|(_, extension)| {
            extension
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase()
        })
        .filter(|extension| !extension.is_empty())
}

fn command_definitions() -> Vec<PlatformCommandDefinition> {
    vec![
        PlatformCommandDefinition {
            name: "chudbot-privacy".to_string(),
            description: "Manage your personal Chudbot privacy preference in this server"
                .to_string(),
            admin_only: false,
            options: vec![
                subcommand(
                    "in",
                    "Allow Chudbot to use your messages as quoted-message context",
                    Vec::new(),
                ),
                subcommand(
                    "out",
                    "Stop letting Chudbot use your messages as quoted-message context",
                    Vec::new(),
                ),
                subcommand(
                    "status",
                    "Show your current privacy preference here",
                    Vec::new(),
                ),
            ],
        },
        PlatformCommandDefinition {
            name: "chudbot-mode".to_string(),
            description: "Configure how Chudbot gathers context in this server".to_string(),
            admin_only: true,
            options: vec![
                subcommand(
                    "show",
                    "Show the active privacy mode for this server",
                    Vec::new(),
                ),
                subcommand(
                    "set",
                    "Change the privacy mode for this server",
                    vec![
                        string_option(
                            "mode",
                            "Which context-gathering mode to use",
                            true,
                            vec![
                                choice("Open: see recent channel history", "open"),
                                choice("Channel only", "channel_only"),
                                choice("Opt-in", "opt_in"),
                                choice("Conversation only", "conversation_only"),
                            ],
                        ),
                        option(
                            "channel",
                            "Channel for channel_only mode",
                            PlatformCommandOptionKind::Channel,
                            false,
                        ),
                        integer_option(
                            "history_size",
                            "How many recent channel messages to include",
                            false,
                            Some(HISTORY_SIZE_MIN),
                            Some(HISTORY_SIZE_MAX),
                        ),
                    ],
                ),
            ],
        },
        PlatformCommandDefinition {
            name: "chudbot-agent".to_string(),
            description: "Pick which configured agent Chudbot uses".to_string(),
            admin_only: false,
            options: vec![
                subcommand(
                    "set",
                    "Pick an agent for a scope",
                    vec![
                        option(
                            "name",
                            "Agent name from config",
                            PlatformCommandOptionKind::String,
                            true,
                        ),
                        string_option(
                            "scope",
                            "Which scope this override applies to",
                            true,
                            vec![
                                choice("This conversation", "conversation"),
                                choice("Me in this server", "user"),
                                choice("This channel", "channel"),
                                choice("This server", "guild"),
                            ],
                        ),
                    ],
                ),
                subcommand(
                    "show",
                    "Show which agent is active here and why",
                    Vec::new(),
                ),
                subcommand("list", "List available configured agents", Vec::new()),
                subcommand(
                    "clear",
                    "Remove an agent override",
                    vec![string_option(
                        "scope",
                        "Scope whose override to clear",
                        true,
                        vec![
                            choice("This conversation", "conversation"),
                            choice("Me in this server", "user"),
                            choice("This channel", "channel"),
                            choice("This server", "guild"),
                        ],
                    )],
                ),
            ],
        },
    ]
}

fn option(
    name: &str,
    description: &str,
    kind: PlatformCommandOptionKind,
    required: bool,
) -> PlatformCommandOption {
    PlatformCommandOption {
        name: name.to_string(),
        description: description.to_string(),
        kind,
        required,
        choices: Vec::new(),
        options: Vec::new(),
        min_integer: None,
        max_integer: None,
    }
}

fn subcommand(
    name: &str,
    description: &str,
    options: Vec<PlatformCommandOption>,
) -> PlatformCommandOption {
    PlatformCommandOption {
        options,
        ..option(
            name,
            description,
            PlatformCommandOptionKind::SubCommand,
            false,
        )
    }
}

fn string_option(
    name: &str,
    description: &str,
    required: bool,
    choices: Vec<PlatformCommandOptionChoice>,
) -> PlatformCommandOption {
    PlatformCommandOption {
        choices,
        ..option(
            name,
            description,
            PlatformCommandOptionKind::String,
            required,
        )
    }
}

fn integer_option(
    name: &str,
    description: &str,
    required: bool,
    min_integer: Option<i64>,
    max_integer: Option<i64>,
) -> PlatformCommandOption {
    PlatformCommandOption {
        min_integer,
        max_integer,
        ..option(
            name,
            description,
            PlatformCommandOptionKind::Integer,
            required,
        )
    }
}

fn choice(name: &str, value: &str) -> PlatformCommandOptionChoice {
    PlatformCommandOptionChoice {
        name: name.to_string(),
        value: value.to_string(),
    }
}

fn command_subcommand(command: &PlatformCommand) -> Option<PlatformCommandInput> {
    command.options.first().cloned()
}

fn sub_option_string<'a>(option: &'a PlatformCommandInput, name: &str) -> Option<&'a str> {
    option
        .options
        .iter()
        .find(|option| option.name == name)
        .and_then(|option| option.value.as_ref())
        .and_then(|value| match value {
            PlatformCommandValue::String(value) => Some(value.as_str()),
            PlatformCommandValue::Channel(channel) => Some(channel.channel_id.as_str()),
            _ => None,
        })
}

fn sub_option_integer(option: &PlatformCommandInput, name: &str) -> Option<i64> {
    option
        .options
        .iter()
        .find(|option| option.name == name)
        .and_then(|option| option.value.as_ref())
        .and_then(|value| match value {
            PlatformCommandValue::Integer(value) => Some(*value),
            _ => None,
        })
}

fn command_privacy_mode(
    platform: PlatformName,
    guild: String,
    mode: &str,
    channel: Option<&str>,
    history_size: Option<u32>,
) -> Result<PrivacyMode, BotError> {
    match mode {
        "open" => Ok(PrivacyMode::Open {
            history_size: history_size.unwrap_or(20),
        }),
        "channel_only" => {
            let Some(channel) = channel else {
                return Err(BotError::CommandInput(
                    "`channel_only` requires the `channel` option.".to_string(),
                ));
            };
            Ok(PrivacyMode::ChannelOnly {
                channel: ChannelRef {
                    platform,
                    guild_id: Some(guild.into()),
                    channel_id: channel.into(),
                },
                history_size: history_size.unwrap_or(20),
            })
        }
        "opt_in" => Ok(PrivacyMode::OptIn),
        "conversation_only" => Ok(PrivacyMode::ConversationOnly),
        other => Err(BotError::CommandInput(format!("Unknown mode `{other}`."))),
    }
}

fn agent_list_response(config: &BotConfig) -> String {
    let mut out = String::from("Available agents\n");
    for (name, agent) in &config.agents {
        let marker = if name == &config.default_agent {
            " (default)"
        } else {
            ""
        };
        out.push_str(&format!(
            "`{name}`{marker}: `{}` / `{}`\n",
            agent.provider, agent.model.id
        ));
    }
    out
}

fn available_agents(config: &BotConfig) -> String {
    let names = config
        .agents
        .keys()
        .map(|name| format!("`{name}`"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("Available agents: {names}")
}

fn ensure_client_tool_enabled(tools: &mut Option<Vec<ToolName>>, name: &str) {
    let Some(tools) = tools else {
        return;
    };
    if tools.iter().any(|tool| tool.as_str() == name) {
        return;
    }
    tools.push(ToolName::new(name));
}

fn option_tick(value: Option<&str>) -> String {
    value
        .map(|value| format!("`{value}`"))
        .unwrap_or_else(|| "-".to_string())
}

fn scope_description(scope: &str) -> &'static str {
    match scope {
        "conversation" => "this conversation",
        "user" => "you in this server",
        "channel" => "this channel",
        "guild" => "this server",
        _ => "this scope",
    }
}

fn pretty_json<T>(value: &T) -> String
where
    T: Serialize,
{
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "<unprintable>".to_string())
}

fn clean_title(raw: &str) -> String {
    let trimmed = raw.trim();
    let trimmed = trimmed
        .strip_prefix("Title:")
        .or_else(|| trimmed.strip_prefix("title:"))
        .or_else(|| trimmed.strip_prefix("Conversation:"))
        .unwrap_or(trimmed)
        .trim();
    let trimmed = trimmed
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(trimmed)
        .trim();
    if trimmed.chars().count() <= TITLE_MAX_CHARS {
        return trimmed.to_string();
    }
    trimmed.chars().take(TITLE_MAX_CHARS).collect::<String>()
}

fn avatar_media_name(user: &UserProfile, url: &str) -> String {
    let tail = url
        .split('?')
        .next()
        .and_then(|url| url.rsplit('/').next())
        .unwrap_or("avatar.png");
    let stem = tail.strip_suffix(".png").unwrap_or(tail);
    let stem = if url.contains("/embed/avatars/") {
        format!("default{stem}")
    } else {
        stem.to_string()
    };
    format!(
        "{}_{}.png",
        user.id.user_id.as_str(),
        safe_media_name_part(&stem)
    )
}

fn safe_media_name_part(input: &str) -> String {
    let out = input
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
        .collect::<String>();
    if out.is_empty() {
        "avatar".to_string()
    } else {
        out
    }
}

fn error_indicates_safety_refusal(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("safety_check") || lower.contains("violates usage guidelines")
}

fn safety_refusal_in_tool_trace(trace: &[ToolTrace]) -> bool {
    trace.iter().any(|trace| {
        let ToolTrace::Client { trace } = trace else {
            return false;
        };
        if !trace.result.is_error {
            return false;
        }
        match &trace.result.content {
            ClientToolResultContent::Text { text } => error_indicates_safety_refusal(text),
            ClientToolResultContent::Json { value } => error_indicates_safety_refusal(
                &value
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_else(|| value.as_str().unwrap_or("")),
            ),
        }
    })
}

fn fix_bare_mentions(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '@' {
            out.push(ch);
            continue;
        }
        let mut digits = String::new();
        while let Some(&next) = chars.peek() {
            if next.is_ascii_digit() {
                digits.push(next);
                chars.next();
            } else {
                break;
            }
        }
        let is_snowflake = (17..=20).contains(&digits.len());
        let already_wrapped = out.ends_with('<');
        if is_snowflake && !already_wrapped {
            out.push_str("<@");
            out.push_str(&digits);
            out.push('>');
        } else {
            out.push('@');
            out.push_str(&digits);
        }
    }
    out
}

fn requested_channel(
    default_channel: &ChannelRef,
    input: &serde_json::Value,
) -> Result<ChannelRef, BotToolError> {
    let Some(channel_id) = input.get("channel_id").and_then(serde_json::Value::as_str) else {
        return Ok(default_channel.clone());
    };
    if channel_id.trim().is_empty() {
        return Err(BotToolError::InvalidInput(
            "`channel_id` cannot be empty".to_string(),
        ));
    }
    Ok(ChannelRef {
        platform: default_channel.platform.clone(),
        guild_id: default_channel.guild_id.clone(),
        channel_id: channel_id.into(),
    })
}

fn should_thread(
    is_new: bool,
    content: &str,
    char_threshold: usize,
    line_threshold: usize,
) -> bool {
    if !is_new {
        return false;
    }
    if content.chars().count() > char_threshold {
        return true;
    }
    rendered_line_count(content) > line_threshold
}

fn rendered_line_count(content: &str) -> usize {
    content
        .split('\n')
        .map(|line| {
            let chars = line.chars().count();
            if chars == 0 {
                1
            } else {
                chars.div_ceil(THREAD_REPLY_WRAP_WIDTH)
            }
        })
        .sum()
}

fn format_reply_content(
    text: &str,
    is_new: bool,
    conversation_id: ConversationId,
    web_base_url: &str,
) -> String {
    let text = fix_bare_mentions(text);
    if !is_new {
        return text;
    }
    let base = web_base_url.trim_end_matches('/');
    format!("{text}\n\n-# 🔎 [full trace]({base}/c/{conversation_id})")
}

fn thread_title(execution: &TurnExecution) -> String {
    let mut title = execution
        .turn
        .user_content
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ");
    if title.is_empty() {
        title = execution.agent_name.clone();
    }
    title.chars().take(80).collect()
}

fn platform_event_kind(event: &PlatformEvent) -> &'static str {
    match event {
        PlatformEvent::Ready { .. } => "ready",
        PlatformEvent::MessageCreated { .. } => "message_created",
        PlatformEvent::ReactionAdded { .. } => "reaction_added",
        PlatformEvent::ReactionRemoved { .. } => "reaction_removed",
        PlatformEvent::Command { .. } => "command",
        PlatformEvent::Shutdown => "shutdown",
    }
}

fn bot_action_kind(action: &BotAction) -> &'static str {
    match action {
        BotAction::Ignored => "ignored",
        BotAction::Shutdown => "shutdown",
        BotAction::CompletedTurn { .. } => "completed_turn",
        BotAction::FailedTurn { .. } => "failed_turn",
        BotAction::CancelledTurn { .. } => "cancelled_turn",
        BotAction::StoppedConversation { .. } => "stopped_conversation",
        BotAction::ResumedConversation { .. } => "resumed_conversation",
        BotAction::RefusedMessage => "refused_message",
        BotAction::HandledCommand => "handled_command",
    }
}

fn model_step_kind(step: &ModelStep) -> &'static str {
    match step {
        ModelStep::Final { .. } => "final",
        ModelStep::UseClientTools { .. } => "use_client_tools",
        ModelStep::Continue { .. } => "continue",
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

fn tool_trace_kind(trace: &chudbot_api::ToolTrace) -> &'static str {
    match trace {
        chudbot_api::ToolTrace::Client { .. } => "client",
        chudbot_api::ToolTrace::Server { .. } => "server",
        chudbot_api::ToolTrace::Grounding { .. } => "grounding",
    }
}

fn conversation_event_kind(kind: ConversationEventKind) -> &'static str {
    match kind {
        ConversationEventKind::Created => "created",
        ConversationEventKind::TurnStarted => "turn_started",
        ConversationEventKind::TurnUpdated => "turn_updated",
        ConversationEventKind::ToolTraceRecorded => "tool_trace_recorded",
        ConversationEventKind::ContextRecorded => "context_recorded",
        ConversationEventKind::TitleUpdated => "title_updated",
        ConversationEventKind::ConversationUpdated => "conversation_updated",
    }
}

fn privacy_mode_kind(mode: &PrivacyMode) -> &'static str {
    match mode {
        PrivacyMode::Open { .. } => "open",
        PrivacyMode::ChannelOnly { .. } => "channel_only",
        PrivacyMode::OptIn => "opt_in",
        PrivacyMode::ConversationOnly => "conversation_only",
    }
}

fn platform_message_reference_kind(reference: &PlatformMessageReference) -> &'static str {
    match reference {
        PlatformMessageReference::None => "none",
        PlatformMessageReference::Id(_) => "id",
        PlatformMessageReference::Hydrated(_) => "hydrated",
    }
}

fn replayable_context_items(context: &[chudbot_api::ContextItem]) -> Vec<chudbot_api::ContextItem> {
    context
        .iter()
        .filter(|item| !is_memory_context_item(item))
        .cloned()
        .collect()
}

fn is_memory_context_item(item: &chudbot_api::ContextItem) -> bool {
    item.source.starts_with("memory:")
}

#[cfg(test)]
mod tests {
    use super::*;
    use chudbot_api::{
        ExternalId, MediaError, MediaFuture, MediaMetadata, MediaRef, MediaUri, PlatformName,
        PublicMediaUrl,
    };
    use serde_json::json;
    use test_case::test_case;

    fn user(platform: &str, guild: Option<&str>, id: &str) -> chudbot_api::UserRef {
        chudbot_api::UserRef {
            platform: PlatformName::new(platform),
            guild_id: guild.map(ExternalId::new),
            user_id: ExternalId::new(id),
        }
    }

    fn generated_image_trace(uri: &str, public_url: &str) -> ToolTrace {
        let tool_use_id = chudbot_api::ToolUseId::new("call-1");
        ToolTrace::Client {
            trace: chudbot_api::ClientToolTrace {
                call: ClientToolCall {
                    id: tool_use_id.clone(),
                    name: ToolName::new("generate_image"),
                    input: json!({ "prompt": "a worm" }),
                },
                result: chudbot_api::ClientToolResult {
                    tool_use_id,
                    content: ClientToolResultContent::Json {
                        value: json!({
                            "uri": uri,
                            "category": "image",
                            "name": "generated.jpg",
                            "mime_type": "image/jpeg",
                            "size_bytes": 42,
                            "delivery": {
                                "platform_reply": "attached automatically"
                            },
                            "extra": {}
                        }),
                    },
                    is_error: false,
                },
                trace_response: json!({
                    "uri": uri,
                    "category": "image",
                    "name": "generated.jpg",
                    "mime_type": "image/jpeg",
                    "size_bytes": 42,
                    "public_url": public_url,
                    "extra": {}
                }),
                usage: Vec::new(),
            },
        }
    }

    #[test]
    fn platform_user_match_ignores_guild_scope() {
        let global_bot = user("discord", None, "123456789012345678");
        let guild_mention = user("discord", Some("guild-1"), "123456789012345678");
        let other_platform = user("slack", None, "123456789012345678");

        assert!(same_platform_user(&guild_mention, &global_bot));
        assert!(!same_platform_user(&other_platform, &global_bot));
    }

    #[test]
    fn normalizes_bot_and_member_mentions() {
        let bot = user("discord", None, "111111111111111111");
        let mentioned = user("discord", Some("guild-1"), "222222222222222222");
        let profiles = [UserProfile {
            id: mentioned.clone(),
            username: "alice".to_string(),
            name: Some("Alice Global".to_string()),
            display_name: Some("Alice".to_string()),
            avatar_url: None,
            is_bot: false,
        }];

        let normalized = normalize_mention_content(
            "<@111111111111111111> hi <@!222222222222222222>",
            &bot,
            &[bot.clone(), mentioned],
            &profiles,
        );

        assert_eq!(normalized, "hi Alice (<@222222222222222222>)");
    }

    #[test]
    fn profile_label_falls_back_to_platform_name_before_username() {
        let profile = UserProfile {
            id: user("discord", Some("guild-1"), "222222222222222222"),
            username: "alice".to_string(),
            name: Some("Alice Global".to_string()),
            display_name: None,
            avatar_url: None,
            is_bot: false,
        };

        assert_eq!(display_name_for_profile(&profile), "Alice Global");
    }

    #[test]
    fn linked_assistant_message_replays_only_for_same_conversation() {
        let conversation_id = ConversationId::new();
        let other_conversation_id = ConversationId::new();
        let link = MessageLink {
            message: MessageRef {
                platform: PlatformName::new("discord"),
                guild_id: Some(ExternalId::new("guild-1")),
                channel_id: ExternalId::new("channel-1"),
                message_id: ExternalId::new("assistant-message-1"),
            },
            conversation_id,
            turn_id: TurnId::new(),
            role: "assistant".to_string(),
        };
        let user_link = MessageLink {
            role: "user".to_string(),
            ..link.clone()
        };

        assert!(message_link_replays_as_assistant(&link, conversation_id));
        assert!(!message_link_replays_as_assistant(
            &link,
            other_conversation_id
        ));
        assert!(!message_link_replays_as_assistant(
            &user_link,
            conversation_id
        ));
    }

    #[test]
    fn replayable_context_items_drop_memory_context() {
        let platform_item = chudbot_api::ContextItem {
            position: 0,
            source: "platform:message:message-1".to_string(),
            role: "user".to_string(),
            content: "{\"content\":\"hi\"}".to_string(),
            message: None,
        };
        let memory_item = chudbot_api::ContextItem {
            position: 1,
            source: "memory:user:user-1".to_string(),
            role: "user".to_string(),
            content: "Background memory for the current user.".to_string(),
            message: None,
        };

        let replayable = replayable_context_items(&[platform_item.clone(), memory_item]);

        assert_eq!(replayable.len(), 1);
        assert_eq!(replayable[0].source, platform_item.source);
        assert_eq!(replayable[0].content, platform_item.content);
    }

    #[test]
    fn repairs_bare_snowflake_mentions() {
        assert_eq!(
            fix_bare_mentions("talk to @123456789012345678"),
            "talk to <@123456789012345678>"
        );
        assert_eq!(
            fix_bare_mentions("already <@123456789012345678>"),
            "already <@123456789012345678>"
        );
        assert_eq!(fix_bare_mentions("short @123"), "short @123");
    }

    #[test]
    fn strips_generated_media_markdown_from_reply_text() {
        let trace = generated_image_trace(
            "file://images/generated.jpg",
            "https://chud.example/images/generated.jpg",
        );
        let refs = generated_media_reply_refs(&[trace]);

        let reply = strip_generated_media_refs(
            "Worm generated.\n\n![image](https://chud.example/images/generated.jpg)\n\nfile://images/generated.jpg",
            &refs,
        );

        assert_eq!(reply, "Worm generated.");
    }

    #[test]
    fn generated_media_strip_preserves_unrelated_links() {
        let trace = generated_image_trace(
            "file://images/generated.jpg",
            "https://chud.example/images/generated.jpg",
        );
        let refs = generated_media_reply_refs(&[trace]);

        let reply = strip_generated_media_refs(
            "Done.\n\n-# [full trace](https://chud.example/c/abc)",
            &refs,
        );

        assert_eq!(
            reply,
            "Done.\n\n-# [full trace](https://chud.example/c/abc)"
        );
    }

    #[derive(Debug, Clone)]
    struct PromptMediaRef {
        metadata: MediaMetadata,
        public_url: PublicMediaUrl,
    }

    impl PromptMediaRef {
        fn boxed(uri: &str, public_url: &str) -> chudbot_api::BoxedMediaRef {
            Box::new(Self {
                metadata: MediaMetadata {
                    category: MediaCategory::Image,
                    name: "generated.jpg".to_string(),
                    uri: MediaUri::new(uri),
                    mime_type: "image/jpeg".to_string(),
                    size_bytes: 42,
                },
                public_url: PublicMediaUrl::new(public_url),
            })
        }

        fn boxed_audio(uri: &str) -> chudbot_api::BoxedMediaRef {
            Box::new(Self {
                metadata: MediaMetadata {
                    category: MediaCategory::Audio,
                    name: "voice.ogg".to_string(),
                    uri: MediaUri::new(uri),
                    mime_type: "audio/ogg".to_string(),
                    size_bytes: 42,
                },
                public_url: PublicMediaUrl::new("https://chud.example/audio/voice.ogg"),
            })
        }
    }

    impl MediaRef for PromptMediaRef {
        fn metadata(&self) -> &MediaMetadata {
            &self.metadata
        }

        fn clone_box(&self) -> chudbot_api::BoxedMediaRef {
            Box::new(self.clone())
        }

        fn public_url(&self) -> MediaFuture<'_, PublicMediaUrl> {
            Box::pin(async move { Ok(self.public_url.clone()) })
        }

        fn load(&self) -> MediaFuture<'_, chudbot_api::LoadedMedia> {
            Box::pin(async move {
                Err(MediaError::BytesUnavailable {
                    uri: self.metadata.uri.clone(),
                })
            })
        }
    }

    #[test_case(MediaCategory::Image, "image/png", true ; "png image")]
    #[test_case(MediaCategory::Image, "image/jpeg; charset=binary", true ; "jpeg image with params")]
    #[test_case(MediaCategory::Image, "IMAGE/WEBP", true ; "case insensitive webp")]
    #[test_case(MediaCategory::Image, "image/gif", false ; "unsupported image mime")]
    #[test_case(MediaCategory::Image, "video/mp4", false ; "image category with video mime")]
    #[test_case(MediaCategory::Video, "video/mp4", false ; "video category")]
    #[test_case(MediaCategory::Audio, "audio/ogg", false ; "audio category")]
    fn model_transcript_media_support_matches_llm_image_inputs(
        category: MediaCategory,
        mime_type: &str,
        expected: bool,
    ) {
        let media = PromptMediaRef {
            metadata: MediaMetadata {
                category,
                name: "media.bin".to_string(),
                uri: MediaUri::new("file://media/generated.bin"),
                mime_type: mime_type.to_string(),
                size_bytes: 42,
            },
            public_url: PublicMediaUrl::new("https://chud.example/media/generated.bin"),
        };

        assert_eq!(model_transcript_supports_media(&media), expected);
    }

    #[test_case(None, "voice.dat", true, true ; "voice flag")]
    #[test_case(Some("audio/ogg"), "voice.dat", false, true ; "audio content type")]
    #[test_case(None, "voice.m4a", false, true ; "audio extension")]
    #[test_case(Some("video/mp4"), "clip.mp4", false, false ; "video content type")]
    #[test_case(None, "clip.mp4", false, false ; "ambiguous mp4 extension")]
    fn detects_audio_attachments(
        content_type: Option<&str>,
        filename: &str,
        is_voice_message: bool,
        expected: bool,
    ) {
        assert_eq!(
            looks_like_audio(content_type, filename, is_voice_message),
            expected
        );
    }

    #[test]
    fn injects_audio_refs_into_message_json() {
        let mut value = json!({
            "content": "<@111> voice note",
            "attachments": [
                { "filename": "image.png" },
                { "filename": "voice.ogg" }
            ]
        });
        let audio = PromptMediaRef::boxed_audio("file://audio/voice.ogg");
        let saved = StoredAttachmentMedia {
            attachment_index: 1,
            media: audio,
        };

        inject_audio_attachment_refs(&mut value, &[saved]);

        assert_eq!(value["audio_attachments"][0], "file://audio/voice.ogg");
        assert_eq!(
            value["attachments"][1]["audio_uri"],
            "file://audio/voice.ogg"
        );
        assert!(value["attachments"][0].get("audio_uri").is_none());
    }

    #[tokio::test]
    async fn transcript_media_refs_are_sanitized_from_replies() {
        let media = PromptMediaRef::boxed(
            "file://images/generated.jpg",
            "https://chud.example/images/generated.jpg",
        );
        let transcript = Transcript {
            id: None,
            instructions: None,
            turns: vec![TranscriptTurn {
                role: TurnRole::User,
                blocks: vec![
                    ContentBlock::Media {
                        media: media.clone(),
                    },
                    ContentBlock::Media { media },
                ],
                metadata: serde_json::Value::Null,
            }],
        };

        let refs = media_reply_refs_from_transcript(&transcript).await;
        assert_eq!(
            refs,
            vec![
                "file://images/generated.jpg".to_string(),
                "https://chud.example/images/generated.jpg".to_string()
            ]
        );
        let reply = strip_generated_media_refs(
            "Done.\n\nhttps://chud.example/images/generated.jpg\nfile://images/generated.jpg",
            &refs,
        );

        assert_eq!(reply, "Done.");
    }

    #[tokio::test]
    async fn transcript_reply_refs_include_replayed_tool_result_media() {
        let trace = generated_image_trace(
            "file://videos/generated.mp4",
            "https://chud.example/videos/generated.mp4",
        );
        let mut transcript = Transcript::new();
        append_client_tool_replay(&mut transcript, &[trace]);

        let refs = media_reply_refs_from_transcript(&transcript).await;

        assert_eq!(refs, vec!["file://videos/generated.mp4".to_string()]);
    }

    #[test]
    fn client_tool_trace_replays_as_call_then_result() {
        let trace = generated_image_trace(
            "file://images/generated.jpg",
            "https://chud.example/images/generated.jpg",
        );
        let mut transcript = Transcript::new();

        append_client_tool_replay(&mut transcript, &[trace]);

        assert_eq!(transcript.turns.len(), 2);
        assert_eq!(transcript.turns[0].role, TurnRole::Assistant);
        assert_eq!(transcript.turns[1].role, TurnRole::User);
        let [ContentBlock::ClientToolCall(call)] = transcript.turns[0].blocks.as_slice() else {
            panic!("expected client tool call replay");
        };
        assert_eq!(call.name.as_str(), "generate_image");
        let [ContentBlock::ClientToolResult(result)] = transcript.turns[1].blocks.as_slice() else {
            panic!("expected client tool result replay");
        };
        assert_eq!(result.tool_use_id, call.id);
        let ClientToolResultContent::Json { value } = &result.content else {
            panic!("expected json result");
        };
        assert_eq!(value["uri"], "file://images/generated.jpg");
    }

    #[test]
    fn generated_media_replays_after_assistant_message() {
        let turn_id = TurnId::new();
        let trace = generated_image_trace(
            "file://images/generated.jpg",
            "https://chud.example/images/generated.jpg",
        );
        let media = PromptMediaRef::boxed(
            "file://images/generated.jpg",
            "https://chud.example/images/generated.jpg",
        );
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn::text(TurnRole::User, "draw an image"));
        append_client_tool_replay(&mut transcript, &[trace]);
        transcript.push(TranscriptTurn::text(
            TurnRole::Assistant,
            "Done. Image generated and attached.",
        ));

        append_generated_media_replay(
            &mut transcript,
            turn_id,
            vec!["file://images/generated.jpg".to_string()],
            vec![ContentBlock::Media { media }],
        );

        assert_eq!(transcript.turns.len(), 5);
        assert_eq!(transcript.turns[0].role, TurnRole::User);
        assert_eq!(transcript.turns[1].role, TurnRole::Assistant);
        assert_eq!(transcript.turns[2].role, TurnRole::User);
        assert_eq!(transcript.turns[3].role, TurnRole::Assistant);
        assert_eq!(transcript.turns[4].role, TurnRole::User);
        let expected_id = turn_transcript_message_id(turn_id, "assistant_media");
        assert_eq!(
            transcript.turns[4].metadata["id"].as_str(),
            Some(expected_id.as_str())
        );
        let [ContentBlock::Text { text }, ContentBlock::Media { .. }] =
            transcript.turns[4].blocks.as_slice()
        else {
            panic!("expected generated media replay note and media");
        };
        assert!(text.contains("file://images/generated.jpg"));
        assert!(text.contains("reference_images"));
    }

    #[test_case("platform:message:message-1:image:0", true ; "platform attachment")]
    #[test_case("platform:quoted:message-1:image:0", true ; "quoted platform attachment")]
    #[test_case("generate_image", false ; "generated image")]
    fn replay_asset_user_turn_ownership(source: &str, expected: bool) {
        let asset = TurnAsset {
            uri: MediaUri::new("file://images/image.jpg"),
            turn_id: TurnId::new(),
            source: source.to_string(),
            mime_type: Some("image/jpeg".to_string()),
        };

        assert_eq!(replay_asset_belongs_to_user_turn(&asset), expected);
    }

    #[test]
    fn client_tool_replay_skips_non_client_traces() {
        let mut transcript = Transcript::new();
        append_client_tool_replay(&mut transcript, &[]);

        assert!(transcript.turns.is_empty());
    }

    #[test]
    fn formats_new_conversation_reply_with_legacy_trace_link() {
        let conversation_id = ConversationId::new();

        let reply = format_reply_content("answer", true, conversation_id, "https://chud.example/");

        assert_eq!(
            reply,
            format!("answer\n\n-# 🔎 [full trace](https://chud.example/c/{conversation_id})")
        );
    }

    #[test]
    fn formats_continuation_reply_without_trace_link() {
        let reply = format_reply_content(
            "talk to @123456789012345678",
            false,
            ConversationId::new(),
            "https://chud.example",
        );

        assert_eq!(reply, "talk to <@123456789012345678>");
    }

    #[test]
    fn rendered_lines_count_short_blank_and_wrapped_rows() {
        assert_eq!(rendered_line_count("a\nb\nc"), 3);
        assert_eq!(rendered_line_count("a\n\nb"), 3);

        let wrapped = "x".repeat(THREAD_REPLY_WRAP_WIDTH * 3);
        assert_eq!(rendered_line_count(&wrapped), 3);

        let mixed = format!("hi\n{}", "y".repeat(THREAD_REPLY_WRAP_WIDTH + 1));
        assert_eq!(rendered_line_count(&mixed), 3);
    }

    #[test]
    fn should_thread_respects_char_and_visible_line_thresholds() {
        let big = "x".repeat(DEFAULT_THREAD_THRESHOLD_CHARS + 1);
        assert!(!should_thread(
            false,
            &big,
            DEFAULT_THREAD_THRESHOLD_CHARS,
            DEFAULT_THREAD_THRESHOLD_LINES,
        ));
        assert!(should_thread(
            true,
            &big,
            DEFAULT_THREAD_THRESHOLD_CHARS,
            DEFAULT_THREAD_THRESHOLD_LINES,
        ));

        let tall = (1..=24)
            .map(|index| format!("{index}. short line"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(tall.chars().count() < DEFAULT_THREAD_THRESHOLD_CHARS);
        assert!(rendered_line_count(&tall) > DEFAULT_THREAD_THRESHOLD_LINES);
        assert!(should_thread(
            true,
            &tall,
            DEFAULT_THREAD_THRESHOLD_CHARS,
            DEFAULT_THREAD_THRESHOLD_LINES,
        ));
        assert!(!should_thread(
            true,
            "hi",
            DEFAULT_THREAD_THRESHOLD_CHARS,
            DEFAULT_THREAD_THRESHOLD_LINES,
        ));
    }

    #[test]
    fn rejects_video_duration_over_provider_limit() {
        let error =
            tool_optional_u8_bounded(&json!({ "duration_seconds": 16 }), "duration_seconds", 15)
                .expect_err("duration should be capped");

        assert!(error.to_string().contains("at most 15"));
    }
}
