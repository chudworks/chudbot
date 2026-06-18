//! Runtime registry adapters for agent and tool code.
//!
//! The bot runtime owns named provider registries so configuration can bind
//! agents and tools to services such as `llm.openai`, `image.nano_banana`, or a
//! local transcription provider. Most agent and tool implementations, however,
//! only need a single concrete provider-shaped value. The routed wrappers in
//! this module bridge those two shapes: each wrapper carries a registry clone,
//! the selected provider name, and any configured default model needed by that
//! tool surface.
//!
//! These adapters intentionally do not validate provider existence at
//! construction time. Config validation should catch bad bindings before
//! runtime, and the underlying registry remains the source of truth for any
//! missing-provider error if one slips through.

use chudbot_api::{
    AudioTranscriber, AudioTranscriberRegistry, AudioTranscription, AudioTranscriptionRequest,
    GeneratedImage, ImageGenerator, ImageGeneratorRegistry, ImageRequest, LlmBackend,
    LlmProviderRegistry, ModelId, ModelInfo, ModelInfoRequest, ModelStepEvent, ModelStepRequest,
    ProviderName, VideoGenerator, VideoGeneratorRegistry, VideoJobId, VideoJobStatus, VideoRequest,
};
use futures::{Stream, TryStreamExt};

use crate::model_step_kind_from_event;

/// `LlmBackend` adapter for one configured provider inside a registry.
///
/// Agent runners want a backend that already represents the selected provider,
/// while the process-level dependency is a registry of all configured LLM
/// providers. This wrapper keeps the provider name next to the registry and
/// forwards every LLM request through that route without changing the request.
#[derive(Debug, Clone)]
pub struct RoutedLlmBackend<R> {
    /// Runtime registry that owns the concrete provider implementations.
    registry: R,
    /// Provider selected by the agent configuration that created this backend.
    provider: ProviderName,
}

impl<R> RoutedLlmBackend<R> {
    /// Build a backend view over one provider in a registry.
    ///
    /// The provider is stored verbatim and returned from `backend_name`; no
    /// lookup is performed until a request is dispatched.
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
    fn step(
        &self,
        request: ModelStepRequest,
    ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_ {
        // Step 1: keep LLM request shaping owned by the caller's model config.
        tracing::debug!("dispatching model step through provider registry");

        // Step 2: route the request to the configured provider.
        self.registry
            .step(&self.provider, request)
            .inspect_ok(|event| {
                if let Some(kind) = model_step_kind_from_event(event) {
                    tracing::debug!(outcome = kind, "model step completed");
                }
            })
            .inspect_err(|error| tracing::warn!(error = %error, "model step failed"))
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
        // Metadata lookup follows the same provider route as model steps.
        tracing::debug!("dispatching model metadata lookup through provider registry");
        self.registry
            .fetch_model_info(&self.provider, request)
            .await
    }
}

/// `ImageGenerator` adapter for one configured provider inside a registry.
///
/// Image generation tools are configured with both a provider and a default
/// model. The adapter applies that model only when the request has not already
/// supplied one, then delegates to the named registry entry.
#[derive(Debug, Clone)]
pub struct RoutedImageGenerator<R> {
    /// Runtime registry that owns the configured image providers.
    registry: R,
    /// Provider selected by the agent's image-generation binding.
    provider: ProviderName,
    /// Binding-level fallback model for requests that omit a model.
    model: ModelId,
}

impl<R> RoutedImageGenerator<R> {
    /// Build an image generator view over one provider and default model.
    ///
    /// The default model is applied lazily per request so callers can still
    /// provide an explicit request model when the tool surface allows it.
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
        // Step 1: fill the configured model only when the request is silent.
        if request.model.is_none() {
            request.model = Some(self.model.clone());
        }

        // Step 2: dispatch through the registry entry selected by config.
        tracing::debug!(
            request_model = ?request.model.as_ref(),
            "dispatching image generation through registry"
        );
        let result = self.registry.generate_image(&self.provider, request).await;

        // Step 3: log the compact result shape while preserving the return.
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
///
/// Video generation has a submit/poll/download lifecycle, but only submission
/// needs request shaping. The configured model is injected on submit when the
/// request omits one; later calls route by provider because the job id or
/// download URL already identifies the upstream artifact.
#[derive(Debug, Clone)]
pub struct RoutedVideoGenerator<R> {
    /// Runtime registry that owns the configured video providers.
    registry: R,
    /// Provider selected by the agent's video-generation binding.
    provider: ProviderName,
    /// Binding-level fallback model for new video jobs.
    model: ModelId,
}

impl<R> RoutedVideoGenerator<R> {
    /// Build a video generator view over one provider and default model.
    ///
    /// As with images, request-level models win over the configured fallback.
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
        // Step 1: apply the binding's model only for model-less submissions.
        if request.model.is_none() {
            request.model = Some(self.model.clone());
        }

        // Step 2: submit the shaped request to the configured provider.
        tracing::debug!(
            request_model = ?request.model.as_ref(),
            "submitting video generation through registry"
        );
        let result = self.registry.submit_video(&self.provider, request).await;

        // Step 3: log the queueing outcome without changing the job id/error.
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
        // Polling must use the same provider that accepted the job.
        self.registry.check_video(&self.provider, job).await
    }

    #[tracing::instrument(name = "video.routed_download", skip_all, fields(provider = %self.provider))]
    async fn download_video(&self, url: String) -> Result<Vec<u8>, Self::Error> {
        // Downloads stay provider-routed because URL handling is provider-specific.
        self.registry.download_video(&self.provider, url).await
    }
}

/// `AudioTranscriber` adapter for one configured provider inside a registry.
///
/// Audio transcription bindings may specify a model, but some providers have a
/// useful server-side default. The adapter therefore carries an optional model
/// and only writes it into the request when both the binding and request provide
/// enough information to do so.
#[derive(Debug, Clone)]
pub struct RoutedAudioTranscriber<R> {
    /// Runtime registry that owns the configured transcription providers.
    registry: R,
    /// Provider selected by the agent's audio-transcription binding.
    provider: ProviderName,
    /// Optional binding-level fallback model for transcription requests.
    model: Option<ModelId>,
}

impl<R> RoutedAudioTranscriber<R> {
    /// Build an audio transcriber view over one provider and optional model.
    ///
    /// Passing `None` for the model preserves the provider's own default model
    /// behavior for requests that also omit a model.
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
        // Step 1: prefer an explicit request model over the optional binding default.
        if request.model.is_none() {
            request.model = self.model.clone();
        }

        // Step 2: forward the request to the named transcription provider.
        tracing::debug!(
            request_model = ?request.model.as_ref(),
            "dispatching audio transcription through registry"
        );
        let result = self
            .registry
            .transcribe_audio(&self.provider, request)
            .await;

        // Step 3: summarize the transcript size and usage counts for operations.
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
