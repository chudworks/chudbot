//! TOML-facing runtime configuration for the `chudbot-bin` process.
//!
//! This module is the boundary between `config.toml` and runtime construction.
//! It owns the root process sections and concrete provider/platform service
//! registries, while `chudbot-bot` owns the agent-first config that routes
//! turns to those registries. Loading deliberately keeps the original TOML
//! source text beside the deserialized values so `check-config` can report
//! aggregated, span-aware diagnostics instead of stopping at the first semantic
//! error.
//!
//! Provider-specific per-model knobs, such as OpenAI reasoning settings or
//! local gateway `extra_body` payloads, are not decoded by this module. They
//! live under `bot.agents.<name>.model.provider_options.value` as an opaque
//! JSON value and are interpreted by the already-routed provider backend.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chudbot_api::{ModelId, PrivacyMode, ProviderName, SamplingNumber};
use chudbot_bot::{BotConfig, MemoryConfig};
use chudbot_web::WebConfig;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing_subscriber::EnvFilter;

use crate::VERSION;
use crate::diagnostics::{ConfigSource, ConfigValidationReport, validate_runtime_config};
use crate::errors::{BinError, ConfigError};

/// Full process configuration deserialized from `config.toml`.
///
/// The root shape is part of the public operator contract documented in
/// `config.example.toml`: `[bot]` defines agents and their bindings, while
/// `[llm]`, `[image]`, `[video]`, `[audio]`, and `[platforms]` define named
/// runtime services those agents can reference. Keep this type broadly
/// deserializable; stale and unknown config keys are reported by the diagnostics
/// validator so operators get all actionable errors in one run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Postgres database connection config.
    pub database: DatabaseConfig,
    /// Process logging/tracing config.
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Bot agent/platform binding config.
    pub bot: BotConfig,
    /// User-memory runtime config.
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Deployment fallback privacy mode before a guild stores an override.
    #[serde(default = "default_privacy")]
    pub default_privacy: PrivacyMode,
    /// Named LLM provider configs keyed by provider registry name.
    #[serde(default)]
    pub llm: BTreeMap<ProviderName, LlmProviderConfig>,
    /// Named image-generation provider configs keyed by provider registry name.
    #[serde(default)]
    pub image: BTreeMap<ProviderName, ImageProviderConfig>,
    /// Named video-generation provider configs keyed by provider registry name.
    #[serde(default)]
    pub video: BTreeMap<ProviderName, VideoProviderConfig>,
    /// Named audio transcription provider configs keyed by provider registry name.
    #[serde(default)]
    pub audio: BTreeMap<ProviderName, AudioProviderConfig>,
    /// Named message platform configs keyed by platform registry name.
    #[serde(default)]
    pub platforms: BTreeMap<chudbot_api::PlatformName, MessagePlatformConfig>,
    /// Web viewer config.
    pub web: WebRuntimeConfig,
    /// Media storage backend config.
    #[serde(default)]
    pub storage: StorageConfig,
}

/// Parsed runtime config paired with the source text used for diagnostics.
///
/// `serve`, `migrate`, and `check-config` all start from the same parse path.
/// Keeping the source here lets validation report errors against the exact TOML
/// the operator supplied, including nested tables owned by other crates.
#[derive(Debug, Clone)]
pub(crate) struct LoadedRuntimeConfig {
    /// Deserialized runtime config.
    pub(crate) config: RuntimeConfig,
    /// Original TOML source.
    pub(crate) source: ConfigSource,
}

/// Process logging/tracing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Tracing filter expression, e.g. `info` or
    /// `info,chudbot=debug`.
    #[serde(default = "default_log_filter")]
    pub filter: String,
    /// Output format.
    #[serde(default)]
    pub format: LogFormat,
    /// Whether ANSI color/style codes are emitted.
    #[serde(default = "default_log_ansi")]
    pub ansi: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            filter: default_log_filter(),
            format: LogFormat::default(),
            ansi: default_log_ansi(),
        }
    }
}

impl LoggingConfig {
    /// Parse the configured tracing filter after TOML validation.
    pub(crate) fn filter(&self) -> Result<EnvFilter, LoggingFilterError> {
        EnvFilter::try_new(&self.filter).map_err(|source| LoggingFilterError {
            filter: self.filter.clone(),
            source,
        })
    }
}

#[derive(Debug, Error)]
#[error("invalid logging filter `{filter}`")]
pub struct LoggingFilterError {
    filter: String,
    #[source]
    source: tracing_subscriber::filter::ParseError,
}

/// Log output format.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// Pretty human-readable logs.
    #[default]
    Pretty,
    /// Compact line-oriented logs.
    Compact,
    /// JSON logs.
    Json,
}

fn default_log_filter() -> String {
    "info".to_string()
}

fn default_log_ansi() -> bool {
    true
}

fn default_privacy() -> PrivacyMode {
    PrivacyMode::OptIn
}

impl RuntimeConfig {
    /// Load config from TOML and retain source text for diagnostics.
    ///
    /// This only performs file I/O, TOML parsing, and default normalization that
    /// needs process state. Semantic checks are deferred to [`Self::validate_all`]
    /// so commands can render one aggregated diagnostics report.
    #[tracing::instrument(name = "config.load", skip_all, fields(path = %path.display()))]
    pub(crate) fn load_with_source(path: &Path) -> Result<LoadedRuntimeConfig, ConfigError> {
        // Read the source once and keep it alive for both parse errors and the
        // later span-aware semantic validator.
        tracing::debug!("reading config file");
        let content = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        tracing::debug!(bytes = content.len(), "read config file");

        // Serde owns syntax and type-shape failures. Unknown/stale keys remain
        // on the diagnostics path so `check-config` can report them together
        // with cross-reference and semantic errors.
        let mut config: Self = match toml::from_str(&content) {
            Ok(config) => config,
            Err(source) => {
                return Err(ConfigError::Parse {
                    path: path.to_path_buf(),
                    content: content.into_boxed_str(),
                    source: Box::new(source),
                });
            }
        };

        // The bot prompt includes a version label. If config omits it, use the
        // binary build version without making operators duplicate deployment
        // metadata in TOML.
        if config.bot.version.is_empty() {
            config.bot.version = VERSION.to_string();
            tracing::debug!(version = VERSION, "defaulted bot version from binary");
        }

        let source = ConfigSource::new(path.to_path_buf(), content);
        config.apply_sampling_source_literals(&source);

        tracing::info!(
            agents = config.bot.agents.len(),
            llm_providers = config.llm.len(),
            image_providers = config.image.len(),
            video_providers = config.video.len(),
            audio_providers = config.audio.len(),
            platforms = config.platforms.len(),
            "loaded runtime config"
        );
        Ok(LoadedRuntimeConfig { config, source })
    }

    fn apply_sampling_source_literals(&mut self, source: &ConfigSource) {
        for (agent_name, agent) in &mut self.bot.agents {
            apply_sampling_source_literal(
                source,
                agent_name.as_str(),
                "temperature",
                &mut agent.model.sampling.temperature,
            );
            apply_sampling_source_literal(
                source,
                agent_name.as_str(),
                "top_p",
                &mut agent.model.sampling.top_p,
            );
        }
    }

    /// Validate config and return all static diagnostics with TOML spans.
    ///
    /// The validator checks root keys, nested bot config owned by
    /// `chudbot-bot`, provider/platform service maps, cross-references, and
    /// simple value constraints before any runtime services are constructed.
    pub(crate) fn validate_all(&self, source: &ConfigSource) -> Result<(), ConfigValidationReport> {
        validate_runtime_config(self, source)
    }

    /// Check the database URL immediately before commands that need Postgres.
    pub(crate) fn validate_database(&self) -> Result<(), BinError> {
        if self.database.url.trim().is_empty() {
            tracing::warn!("database URL is empty");
            return Err(BinError::MissingDatabaseUrl);
        }
        Ok(())
    }
}

fn apply_sampling_source_literal(
    source: &ConfigSource,
    agent_name: &str,
    field: &str,
    target: &mut Option<SamplingNumber>,
) {
    if target.is_none() {
        return;
    }
    let keys = ["bot", "agents", agent_name, "model", "sampling", field];
    if let Some(raw) = source.source_for_keys(&keys)
        && let Ok(number) = SamplingNumber::from_json_number_literal(raw)
    {
        *target = Some(number);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_literals_preserve_user_source_spelling() {
        let input = r#"
[database]
url = "postgres://localhost/chudbot"

[web]
title_prefix = "Chudbot"
frontend_dir = "frontend-build"

[bot]
web_base_url = "http://localhost:1860"
default_agent = "default"

[bot.agents.default]
provider = "grok"
system_prompt = "hi"

[bot.agents.default.model]
id = "grok-test"

[bot.agents.default.model.sampling]
temperature = 1.30
top_p = 0.950
"#;
        let mut config = toml::from_str::<RuntimeConfig>(input).unwrap();
        let source = ConfigSource::new(PathBuf::from("config.test.toml"), input.to_string());

        config.apply_sampling_source_literals(&source);
        let sampling = &config.bot.agents["default"].model.sampling;

        assert_eq!(
            serde_json::to_string(sampling.temperature.as_ref().unwrap()).unwrap(),
            "1.30"
        );
        assert_eq!(
            serde_json::to_string(sampling.top_p.as_ref().unwrap()).unwrap(),
            "0.950"
        );
    }
}

/// Postgres database connection settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Standard `postgres://user:pass@host/db` URL.
    pub url: String,
}

/// Web listener plus trace-viewer presentation config.
///
/// This is the TOML shape. [`WebRuntimeConfig::viewer_config`] converts it into
/// the narrower `chudbot-web` config after process defaults such as `VERSION`
/// and `[bot].web_base_url` are available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebRuntimeConfig {
    /// Socket address to listen on.
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Browser tab title prefix.
    pub title_prefix: String,
    /// Directory containing the built frontend bundle.
    pub frontend_dir: PathBuf,
    /// Optional favicon served at /favicon.ico.
    #[serde(default)]
    pub favicon_path: Option<PathBuf>,
    /// Public origin used for absolute URLs in link-preview metadata. Falls
    /// back to `[bot].web_base_url` when omitted.
    #[serde(default)]
    pub public_base_url: Option<String>,
    /// Optional link-preview thumbnail served at /og-image.
    #[serde(default)]
    pub og_image_path: Option<PathBuf>,
    /// Whether access logs trust proxy-provided client IP headers.
    #[serde(default = "default_trust_forwarded_for")]
    pub trust_forwarded_for: bool,
}

impl WebRuntimeConfig {
    /// Convert TOML config into the web crate's runtime view.
    pub(crate) fn viewer_config(&self, fallback_public_base_url: &str) -> WebConfig {
        WebConfig {
            title_prefix: self.title_prefix.clone(),
            version: VERSION.to_string(),
            frontend_dir: self.frontend_dir.clone(),
            favicon_path: self.favicon_path.clone(),
            // Link-preview metadata needs an absolute public origin. Let the
            // web section override it, otherwise reuse the bot's viewer URL so
            // older configs keep working.
            public_base_url: self
                .public_base_url
                .clone()
                .or_else(|| Some(fallback_public_base_url.to_string())),
            og_image_path: self.og_image_path.clone(),
            trust_forwarded_for: self.trust_forwarded_for,
        }
    }
}

fn default_listen() -> String {
    "127.0.0.1:1860".to_string()
}

fn default_trust_forwarded_for() -> bool {
    true
}

/// Media storage backend config.
///
/// Existing configs omit `kind` and continue to use local filesystem storage.
/// S3 is selected with `kind = "s3"` and keeps the same `file://...` media URI
/// surface while persisting bytes in the configured bucket.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StorageConfig {
    /// Local filesystem storage.
    Local(LocalStorageConfig),
    /// S3-compatible object storage.
    S3(S3StorageConfig),
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self::Local(LocalStorageConfig::default())
    }
}

impl<'de> Deserialize<'de> for StorageConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawStorageConfig::deserialize(deserializer)?;
        match raw.kind {
            StorageKind::Local => Ok(Self::Local(LocalStorageConfig {
                images_dir: raw.images_dir,
                videos_dir: raw.videos_dir,
                audio_dir: raw.audio_dir,
                avatars_dir: raw.avatars_dir,
                guild_icons_dir: raw.guild_icons_dir,
                public_base_url: raw.public_base_url,
            })),
            StorageKind::S3 => Ok(Self::S3(S3StorageConfig {
                bucket: raw.bucket.unwrap_or_default(),
                region: raw.region,
                endpoint_url: raw.endpoint_url,
                force_path_style: raw.force_path_style,
                public_base_url: raw.public_base_url,
            })),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StorageKind {
    /// Local filesystem storage.
    #[default]
    Local,
    /// S3-compatible object storage.
    S3,
}

#[derive(Debug, Deserialize)]
struct RawStorageConfig {
    #[serde(default)]
    kind: StorageKind,
    #[serde(default = "default_images_dir")]
    images_dir: PathBuf,
    #[serde(default = "default_videos_dir")]
    videos_dir: PathBuf,
    #[serde(default = "default_audio_dir")]
    audio_dir: PathBuf,
    #[serde(default = "default_avatars_dir")]
    avatars_dir: PathBuf,
    #[serde(default = "default_guild_icons_dir")]
    guild_icons_dir: PathBuf,
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    region: Option<String>,
    #[serde(default)]
    endpoint_url: Option<String>,
    #[serde(default)]
    force_path_style: bool,
    #[serde(default)]
    public_base_url: Option<String>,
}

/// Local media storage directories and public URL base.
///
/// The filesystem-backed media store keeps each media class in a separate
/// directory but exposes one optional public origin for generated viewer links.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalStorageConfig {
    /// Image directory.
    #[serde(default = "default_images_dir")]
    pub images_dir: PathBuf,
    /// Video directory.
    #[serde(default = "default_videos_dir")]
    pub videos_dir: PathBuf,
    /// Audio directory.
    #[serde(default = "default_audio_dir")]
    pub audio_dir: PathBuf,
    /// Avatar directory.
    #[serde(default = "default_avatars_dir")]
    pub avatars_dir: PathBuf,
    /// Guild icon directory.
    #[serde(default = "default_guild_icons_dir")]
    pub guild_icons_dir: PathBuf,
    /// Public base URL for media, usually the same host as the web viewer.
    #[serde(default)]
    pub public_base_url: Option<String>,
}

impl Default for LocalStorageConfig {
    fn default() -> Self {
        Self {
            images_dir: default_images_dir(),
            videos_dir: default_videos_dir(),
            audio_dir: default_audio_dir(),
            avatars_dir: default_avatars_dir(),
            guild_icons_dir: default_guild_icons_dir(),
            public_base_url: None,
        }
    }
}

fn default_images_dir() -> PathBuf {
    PathBuf::from("images")
}

fn default_videos_dir() -> PathBuf {
    PathBuf::from("videos")
}

fn default_audio_dir() -> PathBuf {
    PathBuf::from("audio")
}

fn default_avatars_dir() -> PathBuf {
    PathBuf::from("avatars")
}

fn default_guild_icons_dir() -> PathBuf {
    PathBuf::from("guild-icons")
}

/// S3-compatible media storage config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3StorageConfig {
    /// Bucket that stores chudbot media objects.
    #[serde(default)]
    pub bucket: String,
    /// Optional AWS region override. When omitted, the AWS SDK region provider
    /// chain uses `AWS_REGION`, shared config profiles, or runtime metadata.
    #[serde(default)]
    pub region: Option<String>,
    /// Optional S3 endpoint URL for compatible APIs such as MinIO, R2, or
    /// LocalStack.
    #[serde(default)]
    pub endpoint_url: Option<String>,
    /// Force path-style addressing for S3-compatible APIs that do not support
    /// virtual-hosted bucket names.
    #[serde(default)]
    pub force_path_style: bool,
    /// Public URL base for media. If omitted, runtime wiring falls back to
    /// `[bot].web_base_url` so the web server can proxy stored media.
    #[serde(default)]
    pub public_base_url: Option<String>,
}

/// Concrete LLM service config for one entry in `[llm.<name>]`.
///
/// The map key is the runtime provider name referenced by agents; the `kind`
/// tag selects which backend implementation to construct. Multiple entries may
/// use the same kind with different credentials, base URLs, pricing overrides,
/// or model metadata fallbacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LlmProviderConfig {
    /// xAI provider.
    Xai {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
        /// Optional local directory for pretty-printed xAI request/response dumps.
        #[serde(default)]
        dump_dir: Option<PathBuf>,
        /// Optional per-model metadata fallback.
        #[serde(default)]
        model_info: BTreeMap<ModelId, LlmModelInfoConfig>,
    },
    /// OpenAI provider.
    #[serde(rename = "openai")]
    OpenAi {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
        /// Optional per-model text-token pricing overrides.
        #[serde(default)]
        pricing: BTreeMap<ModelId, chudbot_openai::OpenAiTokenPricing>,
        /// Optional per-model metadata fallback.
        #[serde(default)]
        model_info: BTreeMap<ModelId, LlmModelInfoConfig>,
    },
    /// Anthropic provider.
    Anthropic {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
        /// Optional per-model text-token pricing overrides.
        #[serde(default)]
        pricing: BTreeMap<ModelId, chudbot_anthropic::AnthropicTokenPricing>,
        /// Optional per-model metadata fallback.
        #[serde(default)]
        model_info: BTreeMap<ModelId, LlmModelInfoConfig>,
    },
    /// OpenAI-compatible chat-completions provider, usually a local gateway.
    #[serde(rename = "openai_compat")]
    OpenAiCompat {
        /// Base URL.
        base_url: String,
        /// Optional API key.
        #[serde(default)]
        api_key: Option<String>,
        /// Optional per-model metadata fallback.
        #[serde(default)]
        model_info: BTreeMap<ModelId, LlmModelInfoConfig>,
    },
    /// Google Gemini API provider.
    Gemini {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
        /// Optional per-model metadata fallback.
        #[serde(default)]
        model_info: BTreeMap<ModelId, LlmModelInfoConfig>,
    },
}

impl LlmProviderConfig {
    /// Return static model metadata fallbacks independent of provider kind.
    ///
    /// Services use these entries before asking the remote provider, which is
    /// useful for local gateways or models whose context limits are not exposed
    /// by the upstream API.
    pub(crate) fn model_info(&self) -> &BTreeMap<ModelId, LlmModelInfoConfig> {
        match self {
            Self::Xai { model_info, .. }
            | Self::OpenAi { model_info, .. }
            | Self::Anthropic { model_info, .. }
            | Self::OpenAiCompat { model_info, .. }
            | Self::Gemini { model_info, .. } => model_info,
        }
    }
}

/// Configured fallback metadata for one LLM model id.
///
/// These values are optional because some providers can fetch equivalent model
/// metadata remotely. Configured values take precedence when present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmModelInfoConfig {
    /// Maximum context-window tokens accepted by the model.
    #[serde(default)]
    pub context_window_tokens: Option<u64>,
    /// Maximum output tokens the model can produce, when known separately.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
}

/// Concrete image-generation service config for one entry in `[image.<name>]`.
///
/// Agent image-generation bindings reference the map key, not the provider
/// kind, so deployments can expose several image services at once.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageProviderConfig {
    /// OpenAI image generation provider.
    #[serde(rename = "openai")]
    OpenAi {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
        /// Optional per-model image-token pricing overrides.
        #[serde(default)]
        pricing: BTreeMap<ModelId, chudbot_openai::OpenAiImagePricing>,
    },
    /// xAI image generation provider.
    Xai {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
    /// Google Gemini API image generation provider.
    Gemini {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
}

/// Concrete video-generation service config for one entry in `[video.<name>]`.
///
/// Video rate limits live on agent tool bindings because the same provider can
/// be exposed with different policy to different agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VideoProviderConfig {
    /// xAI video generation provider.
    Xai {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
    /// Google Gemini API Veo video generation provider.
    Gemini {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
}

/// Concrete audio transcription service config for one entry in `[audio.<name>]`.
///
/// Agent transcription bindings reference these names and may optionally supply
/// model and wake-word behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AudioProviderConfig {
    /// xAI speech-to-text provider.
    Xai {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
}

/// Concrete message-platform service config for one entry in `[platforms.<name>]`.
///
/// `[bot.platforms.<name>]` binds these platform services to default agents.
/// The platform map stays separate from bot routing so future platform adapters
/// can be added without changing the agent config contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MessagePlatformConfig {
    /// Discord platform backed by the `chudbot-discord` Twilight adapter.
    Discord {
        /// Bot token.
        token: String,
        /// Deprecated. Commands now register globally so every installed guild
        /// can use them.
        #[serde(default)]
        dev_guild_id: Option<String>,
    },
}
