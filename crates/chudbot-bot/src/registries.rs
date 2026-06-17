use chudbot_api::{
    AudioTranscriber, AudioTranscriberRegistry, AudioTranscription, AudioTranscriptionRequest,
    GeneratedImage, ImageGenerator, ImageGeneratorRegistry, ImageRequest, LlmBackend,
    LlmProviderRegistry, ModelId, ModelInfo, ModelInfoRequest, ModelStep, ModelStepRequest,
    ProviderName, VideoGenerator, VideoGeneratorRegistry, VideoJobId, VideoJobStatus, VideoRequest,
};

use crate::model_step_kind;

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

    #[tracing::instrument(
        name = "llm.routed_model_info",
        skip_all,
        fields(provider = %self.provider, model = %request.model)
    )]
    async fn fetch_model_info(
        &self,
        request: ModelInfoRequest,
    ) -> Result<Option<ModelInfo>, Self::Error> {
        tracing::debug!("dispatching model metadata lookup through provider registry");
        self.registry
            .fetch_model_info(&self.provider, request)
            .await
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
