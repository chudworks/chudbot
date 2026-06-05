use std::future::Future;

use chudbot_api::{
    AudioTranscriber, AudioTranscription, AudioTranscriptionRequest, ChannelRef, FetchMessages,
    GeneratedImage, ImageGenerator, ImageRequest, LlmBackend, MessagePlatform, MessageRef, ModelId,
    ModelStep, ModelStepRequest, PlatformCommandDefinition, PlatformCommandResponse, PlatformEvent,
    PlatformMessage, PlatformMessageRelationship, PlatformName, PostedMessage, ProviderName,
    ReactionKind, SendMessage, UserProfile, VideoGenerator, VideoJobId, VideoJobStatus,
    VideoRequest,
};

use crate::model_step_kind;

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
