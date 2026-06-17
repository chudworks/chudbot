use std::collections::BTreeMap;
use std::sync::Arc;

use chudbot_api::{
    AudioTranscriber, AudioTranscriberRegistry, AudioTranscription, AudioTranscriptionRequest,
    GeneratedImage, ImageGenerator, ImageGeneratorRegistry, ImageRequest, LlmBackend,
    LlmProviderRegistry, ModelInfo, ModelInfoRequest, ModelStep, ModelStepRequest, ProviderName,
    VideoGenerator, VideoGeneratorRegistry, VideoJobId, VideoJobStatus, VideoRequest,
};
use chudbot_asset_local::LocalMediaStore;
use chudbot_web::{EventBus, WebConfig};

use crate::config::{
    AudioProviderConfig, ImageProviderConfig, LlmProviderConfig, RuntimeConfig, VideoProviderConfig,
};
use crate::errors::{
    BinError, ConfiguredAudioError, ConfiguredImageError, ConfiguredLlmError, ConfiguredVideoError,
};

/// Services that can be built before storage/platform implementations exist.
#[derive(Debug)]
pub struct ServicePlan {
    /// LLM provider registry.
    pub llms: ConfiguredLlmProviders,
    /// Image generation registry.
    pub images: ConfiguredImageGenerators,
    /// Video generation registry.
    pub videos: ConfiguredVideoGenerators,
    /// Audio transcription registry.
    pub audio: ConfiguredAudioTranscribers,
    /// Local media store.
    pub media_store: LocalMediaStore,
    /// Web event bus.
    pub events: EventBus,
    /// Web config.
    pub web: WebConfig,
}

impl ServicePlan {
    #[tracing::instrument(
        name = "services.build",
        skip_all,
        fields(
            images_dir = %config.storage.images_dir.display(),
            videos_dir = %config.storage.videos_dir.display(),
            audio_dir = %config.storage.audio_dir.display(),
            avatars_dir = %config.storage.avatars_dir.display(),
            frontend_dir = %config.web.frontend_dir.display(),
        )
    )]
    pub(crate) fn build(config: &RuntimeConfig) -> Result<Self, BinError> {
        let media_store = LocalMediaStore::new(
            config.storage.images_dir.clone(),
            config.storage.videos_dir.clone(),
            config.storage.audio_dir.clone(),
            config.storage.avatars_dir.clone(),
            config
                .storage
                .public_base_url
                .clone()
                .or_else(|| Some(config.bot.web_base_url.clone())),
        );
        let llms = ConfiguredLlmProviders::from_config(&config.llm);
        let images = ConfiguredImageGenerators::from_config(&config.image);
        let videos = ConfiguredVideoGenerators::from_config(&config.video);
        let audio = ConfiguredAudioTranscribers::from_config(&config.audio);
        tracing::info!(
            llm_providers = llms.configured_count(),
            image_providers = images.configured_count(),
            video_providers = videos.configured_count(),
            audio_providers = audio.configured_count(),
            event_capacity = 256,
            "built service plan"
        );
        Ok(Self {
            llms,
            images,
            videos,
            audio,
            media_store,
            events: EventBus::new(256),
            web: config.web.viewer_config(&config.bot.web_base_url),
        })
    }
}

/// Concrete named LLM provider registry for implemented providers.
#[derive(Debug, Clone)]
pub struct ConfiguredLlmProviders {
    inner: Arc<ConfiguredLlmProvidersInner>,
}

#[derive(Debug, Default)]
struct ConfiguredLlmProvidersInner {
    anthropic: BTreeMap<ProviderName, chudbot_anthropic::AnthropicClient>,
    gemini: BTreeMap<ProviderName, chudbot_gemini::GeminiClient>,
    openai: BTreeMap<ProviderName, chudbot_openai::OpenAiClient>,
    openai_compat: BTreeMap<ProviderName, chudbot_openai_compat::OpenAiCompatClient>,
    xai: BTreeMap<ProviderName, chudbot_xai::XaiClient>,
}

impl Default for ConfiguredLlmProviders {
    fn default() -> Self {
        Self {
            inner: Arc::new(ConfiguredLlmProvidersInner::default()),
        }
    }
}

impl ConfiguredLlmProviders {
    #[tracing::instrument(
        name = "llm_registry.from_config",
        skip_all,
        fields(providers = config.len())
    )]
    fn from_config(config: &BTreeMap<ProviderName, LlmProviderConfig>) -> Self {
        let mut providers = ConfiguredLlmProvidersInner::default();
        for (name, provider) in config {
            match provider {
                LlmProviderConfig::Anthropic {
                    api_key,
                    base_url,
                    pricing,
                } => {
                    let mut client = chudbot_anthropic::AnthropicClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    if !pricing.is_empty() {
                        client = client.with_token_pricing(pricing.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "anthropic",
                        base_url_override = base_url.is_some(),
                        pricing_overrides = pricing.len(),
                        "registered LLM provider"
                    );
                    providers.anthropic.insert(name.clone(), client);
                }
                LlmProviderConfig::OpenAi {
                    api_key,
                    base_url,
                    pricing,
                } => {
                    let mut client = chudbot_openai::OpenAiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    if !pricing.is_empty() {
                        client = client.with_token_pricing(pricing.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "openai",
                        base_url_override = base_url.is_some(),
                        pricing_overrides = pricing.len(),
                        "registered LLM provider"
                    );
                    providers.openai.insert(name.clone(), client);
                }
                LlmProviderConfig::OpenAiCompat { base_url, api_key } => {
                    let mut client =
                        chudbot_openai_compat::OpenAiCompatClient::new(base_url.clone());
                    if let Some(api_key) = api_key {
                        client = client.with_api_key(api_key.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "openai_compat",
                        auth_configured = api_key.is_some(),
                        "registered LLM provider"
                    );
                    providers.openai_compat.insert(name.clone(), client);
                }
                LlmProviderConfig::Gemini { api_key, base_url } => {
                    let mut client = chudbot_gemini::GeminiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "gemini",
                        base_url_override = base_url.is_some(),
                        "registered LLM provider"
                    );
                    providers.gemini.insert(name.clone(), client);
                }
                LlmProviderConfig::Xai { api_key, base_url } => {
                    let mut client = chudbot_xai::XaiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "xai",
                        base_url_override = base_url.is_some(),
                        "registered LLM provider"
                    );
                    providers.xai.insert(name.clone(), client);
                }
            }
        }
        Self {
            inner: Arc::new(providers),
        }
    }

    pub(crate) fn configured_count(&self) -> usize {
        self.inner.anthropic.len()
            + self.inner.gemini.len()
            + self.inner.openai.len()
            + self.inner.openai_compat.len()
            + self.inner.xai.len()
    }
}

impl LlmProviderRegistry for ConfiguredLlmProviders {
    type Error = ConfiguredLlmError;

    fn contains_provider(&self, provider: &ProviderName) -> bool {
        let contains = self.inner.anthropic.contains_key(provider)
            || self.inner.gemini.contains_key(provider)
            || self.inner.openai.contains_key(provider)
            || self.inner.openai_compat.contains_key(provider)
            || self.inner.xai.contains_key(provider);
        tracing::trace!(provider = %provider, contains, "checking LLM provider registry");
        contains
    }

    #[tracing::instrument(
        name = "llm_registry.step",
        skip_all,
        fields(provider = %provider, model = %request.model)
    )]
    async fn step(
        &self,
        provider: &ProviderName,
        request: ModelStepRequest,
    ) -> Result<ModelStep, Self::Error> {
        if let Some(client) = self.inner.anthropic.get(provider) {
            tracing::debug!(kind = "anthropic", "dispatching model step");
            return LlmBackend::step(client, request)
                .await
                .map_err(ConfiguredLlmError::Anthropic);
        }
        if let Some(client) = self.inner.openai.get(provider) {
            tracing::debug!(kind = "openai", "dispatching model step");
            return LlmBackend::step(client, request)
                .await
                .map_err(ConfiguredLlmError::OpenAi);
        }
        if let Some(client) = self.inner.openai_compat.get(provider) {
            tracing::debug!(kind = "openai_compat", "dispatching model step");
            return LlmBackend::step(client, request)
                .await
                .map_err(ConfiguredLlmError::OpenAiCompat);
        }
        if let Some(client) = self.inner.gemini.get(provider) {
            tracing::debug!(kind = "gemini", "dispatching model step");
            return LlmBackend::step(client, request)
                .await
                .map_err(ConfiguredLlmError::Gemini);
        }
        if let Some(client) = self.inner.xai.get(provider) {
            tracing::debug!(kind = "xai", "dispatching model step");
            return LlmBackend::step(client, request)
                .await
                .map_err(ConfiguredLlmError::Xai);
        }
        tracing::warn!("requested provider is missing from registry");
        Err(ConfiguredLlmError::Missing(provider.clone()))
    }

    #[tracing::instrument(
        name = "llm_registry.model_info",
        skip_all,
        fields(provider = %provider, model = %request.model)
    )]
    async fn fetch_model_info(
        &self,
        provider: &ProviderName,
        request: ModelInfoRequest,
    ) -> Result<Option<ModelInfo>, Self::Error> {
        if let Some(client) = self.inner.anthropic.get(provider) {
            tracing::debug!(kind = "anthropic", "fetching model metadata");
            return LlmBackend::fetch_model_info(client, request)
                .await
                .map_err(ConfiguredLlmError::Anthropic);
        }
        if let Some(client) = self.inner.openai.get(provider) {
            tracing::debug!(kind = "openai", "fetching model metadata");
            return LlmBackend::fetch_model_info(client, request)
                .await
                .map_err(ConfiguredLlmError::OpenAi);
        }
        if let Some(client) = self.inner.openai_compat.get(provider) {
            tracing::debug!(kind = "openai_compat", "fetching model metadata");
            return LlmBackend::fetch_model_info(client, request)
                .await
                .map_err(ConfiguredLlmError::OpenAiCompat);
        }
        if let Some(client) = self.inner.gemini.get(provider) {
            tracing::debug!(kind = "gemini", "fetching model metadata");
            return LlmBackend::fetch_model_info(client, request)
                .await
                .map_err(ConfiguredLlmError::Gemini);
        }
        if let Some(client) = self.inner.xai.get(provider) {
            tracing::debug!(kind = "xai", "fetching model metadata");
            return LlmBackend::fetch_model_info(client, request)
                .await
                .map_err(ConfiguredLlmError::Xai);
        }
        tracing::warn!("requested provider is missing from registry");
        Err(ConfiguredLlmError::Missing(provider.clone()))
    }
}

/// Concrete named image-generation provider registry.
#[derive(Debug, Clone)]
pub struct ConfiguredImageGenerators {
    inner: Arc<ConfiguredImageGeneratorsInner>,
}

#[derive(Debug, Default)]
struct ConfiguredImageGeneratorsInner {
    gemini: BTreeMap<ProviderName, chudbot_gemini::GeminiClient>,
    openai: BTreeMap<ProviderName, chudbot_openai::OpenAiClient>,
    xai: BTreeMap<ProviderName, chudbot_xai::XaiClient>,
}

impl Default for ConfiguredImageGenerators {
    fn default() -> Self {
        Self {
            inner: Arc::new(ConfiguredImageGeneratorsInner::default()),
        }
    }
}

impl ConfiguredImageGenerators {
    #[tracing::instrument(
        name = "image_registry.from_config",
        skip_all,
        fields(providers = config.len())
    )]
    fn from_config(config: &BTreeMap<ProviderName, ImageProviderConfig>) -> Self {
        let mut providers = ConfiguredImageGeneratorsInner::default();
        for (name, provider) in config {
            match provider {
                ImageProviderConfig::OpenAi {
                    api_key,
                    base_url,
                    pricing,
                } => {
                    let mut client = chudbot_openai::OpenAiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    if !pricing.is_empty() {
                        client = client.with_image_pricing(pricing.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "openai",
                        base_url_override = base_url.is_some(),
                        pricing_overrides = pricing.len(),
                        "registered image provider"
                    );
                    providers.openai.insert(name.clone(), client);
                }
                ImageProviderConfig::Xai { api_key, base_url } => {
                    let mut client = chudbot_xai::XaiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "xai",
                        base_url_override = base_url.is_some(),
                        "registered image provider"
                    );
                    providers.xai.insert(name.clone(), client);
                }
                ImageProviderConfig::Gemini { api_key, base_url } => {
                    let mut client = chudbot_gemini::GeminiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "gemini",
                        base_url_override = base_url.is_some(),
                        "registered image provider"
                    );
                    providers.gemini.insert(name.clone(), client);
                }
            }
        }
        Self {
            inner: Arc::new(providers),
        }
    }

    pub(crate) fn configured_count(&self) -> usize {
        self.inner.gemini.len() + self.inner.openai.len() + self.inner.xai.len()
    }
}

impl ImageGeneratorRegistry for ConfiguredImageGenerators {
    type Error = ConfiguredImageError;

    fn contains_generator(&self, provider: &ProviderName) -> bool {
        let contains = self.inner.gemini.contains_key(provider)
            || self.inner.openai.contains_key(provider)
            || self.inner.xai.contains_key(provider);
        tracing::trace!(provider = %provider, contains, "checking image provider registry");
        contains
    }

    #[tracing::instrument(
        name = "image_registry.generate",
        skip_all,
        fields(provider = %provider, model = ?request.model.as_ref())
    )]
    async fn generate_image(
        &self,
        provider: &ProviderName,
        request: ImageRequest,
    ) -> Result<GeneratedImage, Self::Error> {
        if let Some(client) = self.inner.openai.get(provider) {
            tracing::debug!(kind = "openai", "dispatching image generation");
            return ImageGenerator::generate_image(client, request)
                .await
                .map_err(ConfiguredImageError::OpenAi);
        }
        if let Some(client) = self.inner.gemini.get(provider) {
            tracing::debug!(kind = "gemini", "dispatching image generation");
            return ImageGenerator::generate_image(client, request)
                .await
                .map_err(ConfiguredImageError::Gemini);
        }
        if let Some(client) = self.inner.xai.get(provider) {
            tracing::debug!(kind = "xai", "dispatching image generation");
            return ImageGenerator::generate_image(client, request)
                .await
                .map_err(ConfiguredImageError::Xai);
        }
        tracing::warn!("requested image provider is missing from registry");
        Err(ConfiguredImageError::Missing(provider.clone()))
    }
}

/// Concrete named video-generation provider registry.
#[derive(Debug, Clone)]
pub struct ConfiguredVideoGenerators {
    inner: Arc<ConfiguredVideoGeneratorsInner>,
}

#[derive(Debug, Default)]
struct ConfiguredVideoGeneratorsInner {
    gemini: BTreeMap<ProviderName, chudbot_gemini::GeminiClient>,
    xai: BTreeMap<ProviderName, chudbot_xai::XaiClient>,
}

impl Default for ConfiguredVideoGenerators {
    fn default() -> Self {
        Self {
            inner: Arc::new(ConfiguredVideoGeneratorsInner::default()),
        }
    }
}

impl ConfiguredVideoGenerators {
    #[tracing::instrument(
        name = "video_registry.from_config",
        skip_all,
        fields(providers = config.len())
    )]
    fn from_config(config: &BTreeMap<ProviderName, VideoProviderConfig>) -> Self {
        let mut providers = ConfiguredVideoGeneratorsInner::default();
        for (name, provider) in config {
            match provider {
                VideoProviderConfig::Xai { api_key, base_url } => {
                    let mut client = chudbot_xai::XaiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "xai",
                        base_url_override = base_url.is_some(),
                        "registered video provider"
                    );
                    providers.xai.insert(name.clone(), client);
                }
                VideoProviderConfig::Gemini { api_key, base_url } => {
                    let mut client = chudbot_gemini::GeminiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "gemini",
                        base_url_override = base_url.is_some(),
                        "registered video provider"
                    );
                    providers.gemini.insert(name.clone(), client);
                }
            }
        }
        Self {
            inner: Arc::new(providers),
        }
    }

    pub(crate) fn configured_count(&self) -> usize {
        self.inner.gemini.len() + self.inner.xai.len()
    }
}

impl VideoGeneratorRegistry for ConfiguredVideoGenerators {
    type Error = ConfiguredVideoError;

    fn contains_generator(&self, provider: &ProviderName) -> bool {
        let contains =
            self.inner.gemini.contains_key(provider) || self.inner.xai.contains_key(provider);
        tracing::trace!(provider = %provider, contains, "checking video provider registry");
        contains
    }

    #[tracing::instrument(
        name = "video_registry.submit",
        skip_all,
        fields(provider = %provider, model = ?request.model.as_ref())
    )]
    async fn submit_video(
        &self,
        provider: &ProviderName,
        request: VideoRequest,
    ) -> Result<VideoJobId, Self::Error> {
        if let Some(client) = self.inner.xai.get(provider) {
            tracing::debug!(kind = "xai", "dispatching video submit");
            return VideoGenerator::submit_video(client, request)
                .await
                .map_err(ConfiguredVideoError::Xai);
        }
        if let Some(client) = self.inner.gemini.get(provider) {
            tracing::debug!(kind = "gemini", "dispatching video submit");
            return VideoGenerator::submit_video(client, request)
                .await
                .map_err(ConfiguredVideoError::Gemini);
        }
        tracing::warn!("requested video provider is missing from registry");
        Err(ConfiguredVideoError::Missing(provider.clone()))
    }

    #[tracing::instrument(name = "video_registry.check", skip_all, fields(provider = %provider, job = %job))]
    async fn check_video(
        &self,
        provider: &ProviderName,
        job: VideoJobId,
    ) -> Result<VideoJobStatus, Self::Error> {
        if let Some(client) = self.inner.xai.get(provider) {
            return VideoGenerator::check_video(client, job)
                .await
                .map_err(ConfiguredVideoError::Xai);
        }
        if let Some(client) = self.inner.gemini.get(provider) {
            return VideoGenerator::check_video(client, job)
                .await
                .map_err(ConfiguredVideoError::Gemini);
        }
        tracing::warn!("requested video provider is missing from registry");
        Err(ConfiguredVideoError::Missing(provider.clone()))
    }

    #[tracing::instrument(name = "video_registry.download", skip_all, fields(provider = %provider))]
    async fn download_video(
        &self,
        provider: &ProviderName,
        url: String,
    ) -> Result<Vec<u8>, Self::Error> {
        if let Some(client) = self.inner.xai.get(provider) {
            return VideoGenerator::download_video(client, url)
                .await
                .map_err(ConfiguredVideoError::Xai);
        }
        if let Some(client) = self.inner.gemini.get(provider) {
            return VideoGenerator::download_video(client, url)
                .await
                .map_err(ConfiguredVideoError::Gemini);
        }
        tracing::warn!("requested video provider is missing from registry");
        Err(ConfiguredVideoError::Missing(provider.clone()))
    }
}

/// Concrete named audio transcription provider registry.
#[derive(Debug, Clone)]
pub struct ConfiguredAudioTranscribers {
    inner: Arc<ConfiguredAudioTranscribersInner>,
}

#[derive(Debug, Default)]
struct ConfiguredAudioTranscribersInner {
    xai: BTreeMap<ProviderName, chudbot_xai::XaiClient>,
}

impl Default for ConfiguredAudioTranscribers {
    fn default() -> Self {
        Self {
            inner: Arc::new(ConfiguredAudioTranscribersInner::default()),
        }
    }
}

impl ConfiguredAudioTranscribers {
    #[tracing::instrument(
        name = "audio_registry.from_config",
        skip_all,
        fields(providers = config.len())
    )]
    fn from_config(config: &BTreeMap<ProviderName, AudioProviderConfig>) -> Self {
        let mut providers = ConfiguredAudioTranscribersInner::default();
        for (name, provider) in config {
            match provider {
                AudioProviderConfig::Xai { api_key, base_url } => {
                    let mut client = chudbot_xai::XaiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "xai",
                        base_url_override = base_url.is_some(),
                        "registered audio transcription provider"
                    );
                    providers.xai.insert(name.clone(), client);
                }
            }
        }
        Self {
            inner: Arc::new(providers),
        }
    }

    pub(crate) fn configured_count(&self) -> usize {
        self.inner.xai.len()
    }
}

impl AudioTranscriberRegistry for ConfiguredAudioTranscribers {
    type Error = ConfiguredAudioError;

    fn contains_transcriber(&self, provider: &ProviderName) -> bool {
        let contains = self.inner.xai.contains_key(provider);
        tracing::trace!(provider = %provider, contains, "checking audio provider registry");
        contains
    }

    #[tracing::instrument(
        name = "audio_registry.transcribe",
        skip_all,
        fields(provider = %provider, model = ?request.model.as_ref())
    )]
    async fn transcribe_audio(
        &self,
        provider: &ProviderName,
        request: AudioTranscriptionRequest,
    ) -> Result<AudioTranscription, Self::Error> {
        if let Some(client) = self.inner.xai.get(provider) {
            tracing::debug!(kind = "xai", "dispatching audio transcription");
            return AudioTranscriber::transcribe_audio(client, request)
                .await
                .map_err(ConfiguredAudioError::Xai);
        }
        tracing::warn!("requested audio provider is missing from registry");
        Err(ConfiguredAudioError::Missing(provider.clone()))
    }
}
