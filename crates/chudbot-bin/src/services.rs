//! Binary-owned service construction.
//!
//! This module is the launcher-side bridge from TOML-shaped `RuntimeConfig` to
//! concrete runtime services. Startup builds local media/web plumbing and named
//! provider registries here; `chudbot-bot` later assembles conversation agents
//! from those registries through [`ConfiguredBotRuntime`].

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use chudbot_api::{
    AudioTranscriber, AudioTranscriberRegistry, AudioTranscription, AudioTranscriptionRequest,
    BoxedMediaRef, CreateMedia, GeneratedImage, ImageGenerator, ImageGeneratorRegistry,
    ImageRequest, LlmBackend, LlmProviderRegistry, MediaCategory, MediaError, MediaStore, MediaUri,
    ModelId, ModelInfo, ModelInfoRequest, ModelStep, ModelStepRequest, ProviderName,
    VideoGenerator, VideoGeneratorRegistry, VideoJobId, VideoJobStatus, VideoRequest,
};
use chudbot_asset_local::LocalMediaStore;
use chudbot_asset_s3::S3MediaStore;
use chudbot_bot::BotRuntimeTypes;
use chudbot_storage_sqlx::SqlxStorage;
use chudbot_web::{EventBus, WebConfig, WebRuntimeTypes};
use moka::future::Cache;
use serde_json::json;

use crate::config::{
    AudioProviderConfig, ImageProviderConfig, LlmModelInfoConfig, LlmProviderConfig, RuntimeConfig,
    StorageConfig, VideoProviderConfig,
};
use crate::errors::{
    BinError, ConfiguredAudioError, ConfiguredImageError, ConfiguredLlmError, ConfiguredVideoError,
};
use crate::platforms::ConfiguredMessagePlatforms;

const MODEL_INFO_CACHE_TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// Config-derived services built before storage and platform sockets connect.
///
/// Postgres and platform clients are opened later in `main`; this bundle covers
/// the pure/local services that can be constructed directly from configuration.
#[derive(Debug)]
pub struct BootstrapServices {
    /// LLM provider registry used when building agent model backends.
    pub llms: ConfiguredLlmProviders,
    /// Image generation registry used by configured agent image tools.
    pub images: ConfiguredImageGenerators,
    /// Video generation registry used by configured agent video tools.
    pub videos: ConfiguredVideoGenerators,
    /// Audio transcription registry used by configured agent transcription tools.
    pub audio: ConfiguredAudioTranscribers,
    /// Media store shared by bot tools and web routes.
    pub media_store: ConfiguredMediaStore,
    /// In-process event bus for trace-viewer live updates.
    pub events: EventBus,
    /// Viewer-facing web configuration derived from runtime settings.
    pub web: WebConfig,
}

/// Concrete runtime type bundle used by the bot and web generic code.
///
/// This associates the binary's concrete registries, storage implementation,
/// media store, and event bus with the provider-neutral runtime traits. Agent
/// assembly happens in `chudbot-bot`; these associated types tell it which
/// concrete services it will receive.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConfiguredBotRuntime;

impl BotRuntimeTypes for ConfiguredBotRuntime {
    type Platforms = ConfiguredMessagePlatforms;
    type Storage = SqlxStorage;
    type Media = ConfiguredMediaStore;
    type Llms = ConfiguredLlmProviders;
    type Images = ConfiguredImageGenerators;
    type Videos = ConfiguredVideoGenerators;
    type Audio = ConfiguredAudioTranscribers;
    type Events = EventBus;
}

impl WebRuntimeTypes for ConfiguredBotRuntime {
    type Storage = <Self as BotRuntimeTypes>::Storage;
    type Media = <Self as BotRuntimeTypes>::Media;
    type Llms = <Self as BotRuntimeTypes>::Llms;
}

impl BootstrapServices {
    /// Build the config-owned services used by both the bot runtime and web viewer.
    ///
    /// The flow is intentionally shallow: wire local media paths, turn provider
    /// tables into named registries, then create the in-process web support.
    #[tracing::instrument(
        name = "services.build",
        skip_all,
        fields(
            storage_kind = storage_kind(config),
            frontend_dir = %config.web.frontend_dir.display(),
        )
    )]
    pub(crate) fn build(config: &RuntimeConfig) -> Result<Self, BinError> {
        // Media links should resolve through the public web surface. A storage
        // override wins, otherwise generated links point at the bot web base URL.
        let media_store =
            ConfiguredMediaStore::from_config(&config.storage, config.bot.web_base_url.clone());

        // Provider names come from the TOML table keys. Agent assembly later
        // routes model and tool calls through these registries by that name.
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
            "built bootstrap services"
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

/// Runtime-selected media store backend.
#[derive(Debug, Clone)]
pub enum ConfiguredMediaStore {
    /// Local filesystem media store.
    Local(LocalMediaStore),
    /// S3-compatible media store.
    S3(S3MediaStore),
}

impl ConfiguredMediaStore {
    fn from_config(config: &StorageConfig, fallback_public_base_url: String) -> Self {
        match config {
            StorageConfig::Local(config) => Self::Local(LocalMediaStore::new(
                config.images_dir.clone(),
                config.videos_dir.clone(),
                config.audio_dir.clone(),
                config.avatars_dir.clone(),
                config
                    .public_base_url
                    .clone()
                    .or(Some(fallback_public_base_url)),
            )),
            StorageConfig::S3(config) => Self::S3(S3MediaStore::new(
                config.bucket.clone(),
                config.region.clone(),
                config.endpoint_url.clone(),
                config.force_path_style,
                config
                    .public_base_url
                    .clone()
                    .or(Some(fallback_public_base_url)),
            )),
        }
    }
}

impl MediaStore for ConfiguredMediaStore {
    async fn create_media(&self, input: CreateMedia) -> Result<BoxedMediaRef, MediaError> {
        match self {
            Self::Local(store) => store.create_media(input).await,
            Self::S3(store) => store.create_media(input).await,
        }
    }

    async fn media_from_uri(&self, uri: &MediaUri) -> Result<BoxedMediaRef, MediaError> {
        match self {
            Self::Local(store) => store.media_from_uri(uri).await,
            Self::S3(store) => store.media_from_uri(uri).await,
        }
    }

    async fn media_from_name(
        &self,
        category: MediaCategory,
        name: &str,
    ) -> Result<BoxedMediaRef, MediaError> {
        match self {
            Self::Local(store) => store.media_from_name(category, name).await,
            Self::S3(store) => store.media_from_name(category, name).await,
        }
    }
}

fn storage_kind(config: &RuntimeConfig) -> &'static str {
    match &config.storage {
        StorageConfig::Local(_) => "local",
        StorageConfig::S3(_) => "s3",
    }
}

/// Concrete named LLM provider registry for all implemented LLM backends.
///
/// Each configured `[llm.<name>]` entry becomes one concrete client stored under
/// `<name>`. `RoutedLlmBackend` in `chudbot-bot` carries the selected name for
/// an agent and calls this registry for each model step.
#[derive(Debug, Clone)]
pub struct ConfiguredLlmProviders {
    inner: Arc<ConfiguredLlmProvidersInner>,
}

#[derive(Debug, Default)]
struct ConfiguredLlmProvidersInner {
    /// Provider clients grouped by backend crate so dispatch can call concrete implementations.
    anthropic: BTreeMap<ProviderName, chudbot_anthropic::AnthropicClient>,
    gemini: BTreeMap<ProviderName, chudbot_gemini::GeminiClient>,
    openai: BTreeMap<ProviderName, chudbot_openai::OpenAiClient>,
    openai_compat: BTreeMap<ProviderName, chudbot_openai_compat::OpenAiCompatClient>,
    xai: BTreeMap<ProviderName, chudbot_xai::XaiClient>,
    /// Operator-supplied model metadata used before provider discovery.
    model_info: BTreeMap<ProviderName, BTreeMap<ModelId, ModelInfo>>,
    /// Short-lived cache for provider-discovered metadata and provider misses.
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
    /// Convert named LLM config entries into concrete clients and metadata fallbacks.
    ///
    /// The match arms are the boundary where generic TOML knobs become
    /// provider-specific builder calls such as base URL overrides, pricing
    /// tables, and optional API keys.
    #[tracing::instrument(
        name = "llm_registry.from_config",
        skip_all,
        fields(providers = config.len())
    )]
    fn from_config(config: &BTreeMap<ProviderName, LlmProviderConfig>) -> Self {
        let mut providers = ConfiguredLlmProvidersInner::default();
        for (name, provider) in config {
            // Model metadata has the same shape for every backend, so extract it
            // before branching into provider-specific client construction.
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
                    // OpenAI-compatible gateways are often local or self-hosted:
                    // the API root is required, while authentication is optional.
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

    /// Fetch metadata from the named backend after config overrides and cache lookup miss.
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
        // Agents call through `RoutedLlmBackend`; by this point the provider
        // name is the route key and the request shape is already final.
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
        // Configured metadata wins over provider discovery so operators can
        // patch missing or inaccurate provider-reported limits.
        if let Some(info) = self.configured_model_info(provider, &request.model) {
            tracing::debug!("using configured model metadata");
            return Ok(Some(info));
        }

        // Cache both hits and misses; some providers cannot report metadata and
        // should not be asked again on every agent construction path.
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

/// Collect configured metadata entries that contain at least one usable limit.
fn configured_model_info(provider: &LlmProviderConfig) -> BTreeMap<ModelId, ModelInfo> {
    provider
        .model_info()
        .iter()
        .filter_map(|(model, info)| {
            configured_model_info_entry(model, info).map(|info| (model.clone(), info))
        })
        .collect()
}

/// Convert one config entry into provider-neutral model metadata.
fn configured_model_info_entry(model: &ModelId, info: &LlmModelInfoConfig) -> Option<ModelInfo> {
    // Empty entries are ignored so partially prepared config tables do not
    // mask provider discovery with an all-`None` override.
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

/// Cache for provider-discovered model metadata, including unsupported models.
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
    /// Serialized provider options that can affect metadata, such as gateway routing.
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
///
/// Agent image bindings name one of these providers and a default model; the
/// routed tool adapter applies that model before calling this registry.
#[derive(Debug, Clone)]
pub struct ConfiguredImageGenerators {
    inner: Arc<ConfiguredImageGeneratorsInner>,
}

#[derive(Debug, Default)]
struct ConfiguredImageGeneratorsInner {
    /// Provider clients grouped by backend crate for direct generator dispatch.
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
    /// Convert named image provider config into concrete generator clients.
    ///
    /// This is the option-mapping boundary for image-specific knobs such as
    /// base URL overrides and OpenAI image pricing overrides.
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
        // The routed image tool already applied any binding-level default
        // model; this registry only selects the named concrete provider.
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
///
/// Video tools route submit, poll, and download operations through the same
/// provider name so async provider job IDs stay attached to their backend.
#[derive(Debug, Clone)]
pub struct ConfiguredVideoGenerators {
    inner: Arc<ConfiguredVideoGeneratorsInner>,
}

#[derive(Debug, Default)]
struct ConfiguredVideoGeneratorsInner {
    /// Provider clients grouped by backend crate for direct video dispatch.
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
    /// Convert named video provider config into concrete generator clients.
    ///
    /// Video providers currently share the same option shape: API key plus an
    /// optional base URL override.
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
        // The returned job ID is opaque; callers keep using this provider name
        // for later status checks and downloads.
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
///
/// The registry shape mirrors image/video even though xAI is the only current
/// audio backend, keeping agent transcription bindings provider-neutral.
#[derive(Debug, Clone)]
pub struct ConfiguredAudioTranscribers {
    inner: Arc<ConfiguredAudioTranscribersInner>,
}

#[derive(Debug, Default)]
struct ConfiguredAudioTranscribersInner {
    /// Provider clients grouped by backend crate for direct transcription dispatch.
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
    /// Convert named audio transcription config into concrete transcriber clients.
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
        // Agent bindings have already chosen the provider and optional model;
        // this layer only fans out to the concrete transcriber implementation.
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
