//! Named runtime service registry contracts.
//!
//! This module defines the name-based routing layer above the provider-neutral
//! contracts in [`crate::llm`], [`crate::media`], and [`crate::platform`].
//! Agent and platform configuration refers to providers by [`ProviderName`] or
//! [`PlatformName`]; registry implementations turn those names into concrete
//! services without exposing Discord, SQLx, Reqwest, Axum, or provider-specific
//! client types to `chudbot-api`.
//!
//! The high-level flow is:
//!
//! 1. Configuration chooses a named runtime service, such as an LLM provider,
//!    image generator, video generator, audio transcriber, or message platform.
//! 2. Bot, tool, memory, and web code pass the selected name plus a neutral
//!    request type to the appropriate registry.
//! 3. The registry implementation owns lookup, fan-out, shutdown, and error
//!    mapping for the concrete service.
//!
//! `contains_*` methods are intentionally part of the contracts so config
//! validation can report missing references before a request reaches runtime.

use std::future::Future;

use crate::ids::{ChannelRef, MessageRef, PlatformName, ProviderName, VideoJobId};
use crate::llm::{ModelInfo, ModelInfoRequest, ModelStep, ModelStepRequest};
use crate::media::{
    AudioTranscription, AudioTranscriptionRequest, GeneratedImage, ImageRequest, VideoJobStatus,
    VideoRequest,
};
use crate::platform::{
    FetchMessages, PlatformCommandDefinition, PlatformCommandResponse, PlatformEvent,
    PlatformMessage, PlatformMessageRelationship, PostedMessage, ReactionKind, SendMessage,
    UserProfile,
};

// Provider registries route agent-selected provider names to concrete runtime
// services while keeping the request/response payloads provider-neutral.

/// Name-based registry for language-model providers.
///
/// This sits one level above [`crate::llm::LlmBackend`]. A backend can execute a
/// request once it has already been selected; the registry is responsible for
/// resolving the [`ProviderName`] selected by agent configuration and returning a
/// registry-level error when that name is absent or the provider fails.
pub trait LlmProviderRegistry: Clone + Send + Sync {
    /// Error type used for lookup failures and concrete provider failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Return whether `provider` is available in this registry.
    ///
    /// Callers use this for validation and capability checks. Runtime request
    /// paths should still handle an error from [`Self::step`] because
    /// configuration or service availability can change between validation and
    /// execution.
    fn contains_provider(&self, provider: &ProviderName) -> bool;

    /// Execute one model step against a named provider.
    ///
    /// The request is already provider-neutral and contains the model id,
    /// transcript, client/server tool allowances, sampling options, and opaque
    /// provider options for the selected backend.
    fn step(
        &self,
        provider: &ProviderName,
        request: ModelStepRequest,
    ) -> impl Future<Output = Result<ModelStep, Self::Error>> + Send;

    /// Fetch model metadata from a named provider, when supported.
    ///
    /// A successful `Ok(None)` means the resolved provider does not expose
    /// metadata for this request. Missing-provider and upstream failures should
    /// be returned through [`Self::Error`].
    fn fetch_model_info(
        &self,
        provider: &ProviderName,
        request: ModelInfoRequest,
    ) -> impl Future<Output = Result<Option<ModelInfo>, Self::Error>> + Send;
}

/// Name-based registry for image generation providers.
///
/// Image generation is modeled as one request/response operation. The registry
/// selects the concrete provider, while [`ImageRequest`] and [`GeneratedImage`]
/// keep the public API independent of provider SDKs.
pub trait ImageGeneratorRegistry: Clone + Send + Sync {
    /// Error type used for lookup failures and concrete provider failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Return whether `provider` names a configured image generator.
    fn contains_generator(&self, provider: &ProviderName) -> bool;

    /// Generate one image through a named provider.
    fn generate_image(
        &self,
        provider: &ProviderName,
        request: ImageRequest,
    ) -> impl Future<Output = Result<GeneratedImage, Self::Error>> + Send;
}

/// Name-based registry for video generation providers.
///
/// Video providers use a three-step lifecycle because upstream generation is
/// typically asynchronous: submit a job, poll the job, then download the
/// finished bytes from the provider-reported URL.
pub trait VideoGeneratorRegistry: Clone + Send + Sync {
    /// Error type used for lookup failures and concrete provider failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Return whether `provider` names a configured video generator.
    fn contains_generator(&self, provider: &ProviderName) -> bool;

    /// Step 1: submit a video generation job through a named provider.
    ///
    /// The returned [`VideoJobId`] is provider-owned. Storage/runtime layers
    /// decide how to persist and later poll it.
    fn submit_video(
        &self,
        provider: &ProviderName,
        request: VideoRequest,
    ) -> impl Future<Output = Result<VideoJobId, Self::Error>> + Send;

    /// Step 2: poll a video generation job once through a named provider.
    ///
    /// This method performs one status check only. Retry, backoff, quota, and
    /// persistence policy belong to runtime/storage code outside this contract.
    fn check_video(
        &self,
        provider: &ProviderName,
        job: VideoJobId,
    ) -> impl Future<Output = Result<VideoJobStatus, Self::Error>> + Send;

    /// Step 3: download a completed video through a named provider.
    ///
    /// The `url` is the provider-reported render location from
    /// [`VideoJobStatus::Done`]. Implementations may need the provider name for
    /// authentication, URL normalization, or provider-local storage.
    fn download_video(
        &self,
        provider: &ProviderName,
        url: String,
    ) -> impl Future<Output = Result<Vec<u8>, Self::Error>> + Send;
}

/// Name-based registry for audio transcription providers.
///
/// Audio transcription is modeled as one request/response operation. The
/// concrete provider owns speech-model details, while the shared
/// [`AudioTranscription`] result carries normalized text, timing, channel, and
/// usage metadata.
pub trait AudioTranscriberRegistry: Clone + Send + Sync {
    /// Error type used for lookup failures and concrete provider failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Return whether `provider` names a configured audio transcriber.
    fn contains_transcriber(&self, provider: &ProviderName) -> bool;

    /// Transcribe one audio file through a named provider.
    fn transcribe_audio(
        &self,
        provider: &ProviderName,
        request: AudioTranscriptionRequest,
    ) -> impl Future<Output = Result<AudioTranscription, Self::Error>> + Send;
}

// Message platform registries are different from provider registries: they
// multiplex event streams and route outgoing operations by platform-scoped ids.

/// Name-based registry for messaging platform services.
///
/// This sits one level above [`crate::platform::MessagePlatform`]. A concrete
/// platform implementation owns the transport, command API, and platform
/// vocabulary; the registry lets bot code work with one merged event stream and
/// route outgoing operations by the [`PlatformName`] embedded in references such
/// as [`ChannelRef`] and [`MessageRef`].
pub trait MessagePlatformRegistry: Clone + Send + Sync {
    /// Error type used for lookup failures and concrete platform failures.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Fetch the bot user for one platform.
    fn bot_user(
        &self,
        platform: &PlatformName,
    ) -> impl Future<Output = Result<UserProfile, Self::Error>> + Send;

    /// Register bot commands across configured platforms.
    ///
    /// Commands are passed once at the registry boundary because each platform
    /// implementation knows how broadly to install them, such as globally or
    /// per workspace/guild.
    fn register_commands(
        &self,
        commands: Vec<PlatformCommandDefinition>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Read the next event from any configured platform.
    ///
    /// Implementations merge one or more platform event streams into the neutral
    /// [`PlatformEvent`] enum. A registry with multiple platforms decides its
    /// own fairness and shutdown behavior.
    fn next_event(&self) -> impl Future<Output = Result<PlatformEvent, Self::Error>> + Send;

    /// Gracefully stop platform services owned by this registry.
    ///
    /// Registries that own gateway pumps or background workers should override
    /// this. Stateless or externally-owned implementations can use the default
    /// no-op.
    fn shutdown(&self) -> impl Future<Output = Result<(), Self::Error>> + Send {
        // Default to a no-op so simple registries do not need to allocate or
        // coordinate shutdown machinery they do not own.
        async { Ok(()) }
    }

    /// Respond to a command invocation.
    ///
    /// The response target contains the platform name and interaction-specific
    /// identifiers needed to route the response.
    fn respond_to_command(
        &self,
        response: PlatformCommandResponse,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Send a message through the platform named by `request.channel.platform`.
    ///
    /// The returned [`PostedMessage`] identifies where the platform actually
    /// posted the logical response, including any extra messages used to split a
    /// long reply.
    fn send_message(
        &self,
        request: SendMessage,
    ) -> impl Future<Output = Result<PostedMessage, Self::Error>> + Send;

    /// Delete a platform message.
    ///
    /// The platform is selected from [`MessageRef::platform`].
    fn delete_message(
        &self,
        message: MessageRef,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Add a reaction to a platform message.
    ///
    /// The platform is selected from [`MessageRef::platform`].
    fn add_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Remove the bot's own reaction from a platform message.
    ///
    /// The platform is selected from [`MessageRef::platform`].
    fn remove_own_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Trigger a platform typing indicator.
    ///
    /// The platform is selected from [`ChannelRef::platform`].
    fn typing(&self, channel: ChannelRef) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Fetch messages through the platform named by `request.channel.platform`.
    fn fetch_messages(
        &self,
        request: FetchMessages,
    ) -> impl Future<Output = Result<Vec<PlatformMessage>, Self::Error>> + Send;

    /// Render a platform message for model context.
    ///
    /// This is a deliberate platform boundary: Discord, Telegram, Slack, and
    /// other adapters may use different words for workspace, guild, server,
    /// channel, thread, or room. The registry forwards this to the owning
    /// platform so model-facing context stays faithful to the source platform.
    fn message_context(
        &self,
        message: &PlatformMessage,
        relationship: PlatformMessageRelationship,
    ) -> impl Future<Output = Result<serde_json::Value, Self::Error>> + Send;

    /// Resolve a platform channel's parent channel.
    ///
    /// Thread-like channels can return their parent; non-thread channels can
    /// return themselves. Callers use this to normalize conversation scope
    /// without knowing platform-specific channel hierarchy rules.
    fn parent_channel(
        &self,
        channel: ChannelRef,
    ) -> impl Future<Output = Result<ChannelRef, Self::Error>> + Send;
}
