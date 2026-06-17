use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use chudbot_api::{
    AudioTranscriber, AudioTranscriberRegistry, AudioTranscription, AudioTranscriptionRequest,
    GeneratedImage, ImageGenerator, ImageGeneratorRegistry, ImageRequest, LlmBackend,
    LlmProviderRegistry, ModelId, ModelInfo, ModelInfoRequest, ModelStep, ModelStepRequest,
    ProviderName, VideoGenerator, VideoGeneratorRegistry, VideoJobId, VideoJobStatus, VideoRequest,
};
use chudbot_asset_local::LocalMediaStore;
use chudbot_web::{EventBus, WebConfig};
use moka::future::Cache;
use serde_json::json;

use crate::config::{
    AudioProviderConfig, ImageProviderConfig, LlmModelInfoConfig, LlmProviderConfig, RuntimeConfig,
    VideoProviderConfig,
};
use crate::errors::{
    BinError, ConfiguredAudioError, ConfiguredImageError, ConfiguredLlmError, ConfiguredVideoError,
};

const MODEL_INFO_CACHE_TTL: Duration = Duration::from_secs(6 * 60 * 60);

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
    model_info: BTreeMap<ProviderName, BTreeMap<ModelId, ModelInfo>>,
    model_info_cache: ModelInfoCache,
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
            let model_info = configured_model_info(provider);
            let model_info_fallbacks = model_info.len();
            match provider {
                LlmProviderConfig::Anthropic {
                    api_key,
                    base_url,
                    pricing,
                    ..
                } => {
                    let mut client =
                        chudbot_anthropic::AnthropicClient::new(name.clone(), api_key.clone());
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
                        model_info_fallbacks,
                        "registered LLM provider"
                    );
                    providers.anthropic.insert(name.clone(), client);
                }
                LlmProviderConfig::OpenAi {
                    api_key,
                    base_url,
                    pricing,
                    ..
                } => {
                    let mut client =
                        chudbot_openai::OpenAiClient::new(name.clone(), api_key.clone());
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
                        model_info_fallbacks,
                        "registered LLM provider"
                    );
                    providers.openai.insert(name.clone(), client);
                }
                LlmProviderConfig::OpenAiCompat {
                    base_url, api_key, ..
                } => {
                    let mut client = chudbot_openai_compat::OpenAiCompatClient::new(
                        name.clone(),
                        base_url.clone(),
                    );
                    if let Some(api_key) = api_key {
                        client = client.with_api_key(api_key.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "openai_compat",
                        auth_configured = api_key.is_some(),
                        model_info_fallbacks,
                        "registered LLM provider"
                    );
                    providers.openai_compat.insert(name.clone(), client);
                }
                LlmProviderConfig::Gemini {
                    api_key, base_url, ..
                } => {
                    let mut client =
                        chudbot_gemini::GeminiClient::new(name.clone(), api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "gemini",
                        base_url_override = base_url.is_some(),
                        model_info_fallbacks,
                        "registered LLM provider"
                    );
                    providers.gemini.insert(name.clone(), client);
                }
                LlmProviderConfig::Xai {
                    api_key, base_url, ..
                } => {
                    let mut client = chudbot_xai::XaiClient::new(name.clone(), api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "xai",
                        base_url_override = base_url.is_some(),
                        model_info_fallbacks,
                        "registered LLM provider"
                    );
                    providers.xai.insert(name.clone(), client);
                }
            }
            if !model_info.is_empty() {
                providers.model_info.insert(name.clone(), model_info);
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

    async fn fetch_remote_model_info(
        &self,
        provider: &ProviderName,
        request: ModelInfoRequest,
    ) -> Result<Option<ModelInfo>, ConfiguredLlmError> {
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
        if let Some(info) = self.configured_model_info(provider, &request.model) {
            tracing::debug!("using configured model metadata");
            return Ok(Some(info));
        }

        let cache_key = ModelInfoCacheKey::new(provider, &request);
        if let Some(info) = self.inner.model_info_cache.get(&cache_key).await {
            tracing::debug!(
                cache_hit = true,
                available = info.is_some(),
                "using cached model metadata"
            );
            return Ok(info);
        }

        let info = self.fetch_remote_model_info(provider, request).await?;
        self.inner
            .model_info_cache
            .insert(cache_key, info.clone())
            .await;
        Ok(info)
    }
}

impl ConfiguredLlmProviders {
    fn configured_model_info(&self, provider: &ProviderName, model: &ModelId) -> Option<ModelInfo> {
        self.inner.model_info.get(provider)?.get(model).cloned()
    }
}

fn configured_model_info(provider: &LlmProviderConfig) -> BTreeMap<ModelId, ModelInfo> {
    provider
        .model_info()
        .iter()
        .filter_map(|(model, info)| {
            configured_model_info_entry(model, info).map(|info| (model.clone(), info))
        })
        .collect()
}

fn configured_model_info_entry(model: &ModelId, info: &LlmModelInfoConfig) -> Option<ModelInfo> {
    if info.context_window_tokens.is_none() && info.max_output_tokens.is_none() {
        return None;
    }

    Some(ModelInfo {
        id: model.clone(),
        context_window_tokens: info.context_window_tokens,
        max_output_tokens: info.max_output_tokens,
        raw: Some(json!({
            "source": "config",
            "context_window_tokens": info.context_window_tokens,
            "max_output_tokens": info.max_output_tokens,
        })),
    })
}

struct ModelInfoCache {
    entries: Cache<ModelInfoCacheKey, Option<ModelInfo>>,
}

impl ModelInfoCache {
    async fn get(&self, key: &ModelInfoCacheKey) -> Option<Option<ModelInfo>> {
        self.entries.get(key).await
    }

    async fn insert(&self, key: ModelInfoCacheKey, info: Option<ModelInfo>) {
        self.entries.insert(key, info).await;
    }
}

impl Default for ModelInfoCache {
    fn default() -> Self {
        Self {
            entries: Cache::builder()
                .name("llm-model-info")
                .time_to_live(MODEL_INFO_CACHE_TTL)
                .max_capacity(1024)
                .build(),
        }
    }
}

impl std::fmt::Debug for ModelInfoCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelInfoCache")
            .field("entry_count", &self.entries.entry_count())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ModelInfoCacheKey {
    provider: ProviderName,
    model: ModelId,
    provider_options: Option<String>,
}

impl ModelInfoCacheKey {
    fn new(provider: &ProviderName, request: &ModelInfoRequest) -> Self {
        Self {
            provider: provider.clone(),
            model: request.model.clone(),
            provider_options: request
                .provider_options
                .as_ref()
                .map(|options| options.value.to_string()),
        }
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
                    let mut client =
                        chudbot_openai::OpenAiClient::new(name.clone(), api_key.clone());
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
                    let mut client = chudbot_xai::XaiClient::new(name.clone(), api_key.clone());
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
                    let mut client =
                        chudbot_gemini::GeminiClient::new(name.clone(), api_key.clone());
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
                    let mut client = chudbot_xai::XaiClient::new(name.clone(), api_key.clone());
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
                    let mut client =
                        chudbot_gemini::GeminiClient::new(name.clone(), api_key.clone());
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
                    let mut client = chudbot_xai::XaiClient::new(name.clone(), api_key.clone());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_llm_provider_name_is_recorded_backend_name() {
        let provider = ProviderName::new("claude");
        let config = BTreeMap::from([(
            provider.clone(),
            LlmProviderConfig::Anthropic {
                api_key: "test-key".to_string(),
                base_url: None,
                pricing: BTreeMap::new(),
                model_info: BTreeMap::new(),
            },
        )]);

        let registry = ConfiguredLlmProviders::from_config(&config);
        let client = registry
            .inner
            .anthropic
            .get(&provider)
            .expect("configured anthropic client");

        assert_eq!(LlmBackend::backend_name(client), &provider);
    }

    #[test]
    fn configured_llm_model_info_fallback_is_loaded_for_every_provider_kind() {
        let provider = ProviderName::new("claude");
        let xai_provider = ProviderName::new("grok");
        let openai_provider = ProviderName::new("openai");
        let compat_provider = ProviderName::new("local");
        let gemini_provider = ProviderName::new("gemini");
        let model = ModelId::new("claude-haiku-4-5-20251001");
        let model_info = || {
            BTreeMap::from([(
                model.clone(),
                LlmModelInfoConfig {
                    context_window_tokens: Some(200_000),
                    max_output_tokens: Some(8_192),
                },
            )])
        };
        let config = BTreeMap::from([
            (
                provider.clone(),
                LlmProviderConfig::Anthropic {
                    api_key: "test-key".to_string(),
                    base_url: None,
                    pricing: BTreeMap::new(),
                    model_info: model_info(),
                },
            ),
            (
                xai_provider.clone(),
                LlmProviderConfig::Xai {
                    api_key: "test-key".to_string(),
                    base_url: None,
                    model_info: model_info(),
                },
            ),
            (
                openai_provider.clone(),
                LlmProviderConfig::OpenAi {
                    api_key: "test-key".to_string(),
                    base_url: None,
                    pricing: BTreeMap::new(),
                    model_info: model_info(),
                },
            ),
            (
                compat_provider.clone(),
                LlmProviderConfig::OpenAiCompat {
                    base_url: "http://127.0.0.1:1234/v1".to_string(),
                    api_key: None,
                    model_info: model_info(),
                },
            ),
            (
                gemini_provider.clone(),
                LlmProviderConfig::Gemini {
                    api_key: "test-key".to_string(),
                    base_url: None,
                    model_info: model_info(),
                },
            ),
        ]);

        let registry = ConfiguredLlmProviders::from_config(&config);
        for provider in [
            &provider,
            &xai_provider,
            &openai_provider,
            &compat_provider,
            &gemini_provider,
        ] {
            let info = registry
                .configured_model_info(provider, &model)
                .expect("configured model info");
            assert_eq!(info.id, model);
            assert_eq!(info.context_window_tokens, Some(200_000));
            assert_eq!(info.max_output_tokens, Some(8_192));
        }
    }

    #[tokio::test]
    async fn configured_llm_model_info_is_returned_without_provider_lookup() {
        let provider = ProviderName::new("local");
        let model = ModelId::new("local-model");
        let config = BTreeMap::from([(
            provider.clone(),
            LlmProviderConfig::OpenAiCompat {
                base_url: "http://127.0.0.1:1/v1".to_string(),
                api_key: None,
                model_info: BTreeMap::from([(
                    model.clone(),
                    LlmModelInfoConfig {
                        context_window_tokens: Some(131_072),
                        max_output_tokens: Some(8_192),
                    },
                )]),
            },
        )]);

        let registry = ConfiguredLlmProviders::from_config(&config);
        let info = LlmProviderRegistry::fetch_model_info(
            &registry,
            &provider,
            ModelInfoRequest {
                model: model.clone(),
                provider_options: None,
            },
        )
        .await
        .expect("model info lookup")
        .expect("configured model info");

        assert_eq!(info.id, model);
        assert_eq!(info.context_window_tokens, Some(131_072));
        assert_eq!(info.max_output_tokens, Some(8_192));
    }

    #[tokio::test]
    async fn model_info_cache_stores_hits_and_misses() {
        let cache = ModelInfoCache::default();
        let provider = ProviderName::new("provider");
        let model = ModelId::new("model");
        let key = ModelInfoCacheKey::new(
            &provider,
            &ModelInfoRequest {
                model: model.clone(),
                provider_options: None,
            },
        );
        let info = ModelInfo {
            id: model.clone(),
            context_window_tokens: Some(200_000),
            max_output_tokens: Some(8_192),
            raw: None,
        };

        assert!(cache.get(&key).await.is_none());
        cache.insert(key.clone(), Some(info.clone())).await;
        let cached = cache
            .get(&key)
            .await
            .expect("cached entry")
            .expect("cached model info");
        assert_eq!(cached.id, info.id);
        assert_eq!(cached.context_window_tokens, info.context_window_tokens);
        assert_eq!(cached.max_output_tokens, info.max_output_tokens);

        let missing_key = ModelInfoCacheKey::new(
            &provider,
            &ModelInfoRequest {
                model: ModelId::new("missing-model"),
                provider_options: None,
            },
        );
        cache.insert(missing_key.clone(), None).await;
        assert!(matches!(cache.get(&missing_key).await, Some(None)));
    }

    #[test]
    fn model_info_cache_key_includes_provider_options() {
        let provider = ProviderName::new("provider");
        let model = ModelId::new("model");
        let without_options = ModelInfoCacheKey::new(
            &provider,
            &ModelInfoRequest {
                model: model.clone(),
                provider_options: None,
            },
        );
        let with_options = ModelInfoCacheKey::new(
            &provider,
            &ModelInfoRequest {
                model,
                provider_options: Some(chudbot_api::ProviderOptions {
                    value: json!({ "region": "us-east-1" }),
                }),
            },
        );

        assert_ne!(without_options, with_options);
    }

    #[test]
    fn configured_media_provider_names_are_recorded_backend_names() {
        let image_provider = ProviderName::new("grok_images");
        let image_config = BTreeMap::from([(
            image_provider.clone(),
            ImageProviderConfig::Xai {
                api_key: "test-key".to_string(),
                base_url: None,
            },
        )]);
        let images = ConfiguredImageGenerators::from_config(&image_config);
        let image_client = images
            .inner
            .xai
            .get(&image_provider)
            .expect("configured xai image client");
        assert_eq!(ImageGenerator::backend_name(image_client), &image_provider);

        let video_provider = ProviderName::new("grok_video");
        let video_config = BTreeMap::from([(
            video_provider.clone(),
            VideoProviderConfig::Xai {
                api_key: "test-key".to_string(),
                base_url: None,
            },
        )]);
        let videos = ConfiguredVideoGenerators::from_config(&video_config);
        let video_client = videos
            .inner
            .xai
            .get(&video_provider)
            .expect("configured xai video client");
        assert_eq!(VideoGenerator::backend_name(video_client), &video_provider);

        let audio_provider = ProviderName::new("grok_audio");
        let audio_config = BTreeMap::from([(
            audio_provider.clone(),
            AudioProviderConfig::Xai {
                api_key: "test-key".to_string(),
                base_url: None,
            },
        )]);
        let audio = ConfiguredAudioTranscribers::from_config(&audio_config);
        let audio_client = audio
            .inner
            .xai
            .get(&audio_provider)
            .expect("configured xai audio client");
        assert_eq!(
            AudioTranscriber::backend_name(audio_client),
            &audio_provider
        );
    }
}
