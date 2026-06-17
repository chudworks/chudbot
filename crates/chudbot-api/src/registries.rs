//! Named runtime service registry contracts.

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

    /// Fetch model metadata from a named provider, when supported.
    fn fetch_model_info(
        &self,
        provider: &ProviderName,
        request: ModelInfoRequest,
    ) -> impl Future<Output = Result<Option<ModelInfo>, Self::Error>> + Send;
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
