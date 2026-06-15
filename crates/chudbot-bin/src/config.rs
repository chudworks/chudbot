use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use chudbot_api::{ModelId, PrivacyMode, ProviderName};
use chudbot_bot::{BotConfig, MemoryConfig};
use chudbot_web::WebConfig;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing_subscriber::EnvFilter;

use crate::VERSION;
use crate::errors::{BinError, ConfigError};

/// Full process configuration.
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
    /// Named LLM provider configs.
    #[serde(default)]
    pub llm: BTreeMap<ProviderName, LlmProviderConfig>,
    /// Named image-generation provider configs.
    #[serde(default)]
    pub image: BTreeMap<ProviderName, ImageProviderConfig>,
    /// Named video-generation provider configs.
    #[serde(default)]
    pub video: BTreeMap<ProviderName, VideoProviderConfig>,
    /// Named audio transcription provider configs.
    #[serde(default)]
    pub audio: BTreeMap<ProviderName, AudioProviderConfig>,
    /// Named message platform configs.
    #[serde(default)]
    pub platforms: BTreeMap<chudbot_api::PlatformName, MessagePlatformConfig>,
    /// Web viewer config.
    pub web: WebRuntimeConfig,
    /// Local media storage config.
    #[serde(default)]
    pub storage: LocalStorageConfig,
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
    /// Load config from TOML.
    #[tracing::instrument(name = "config.load", skip_all, fields(path = %path.display()))]
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        tracing::debug!("reading config file");
        let content = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        tracing::debug!(bytes = content.len(), "read config file");
        let mut config: Self = toml::from_str(&content).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        if config.bot.version.is_empty() {
            config.bot.version = VERSION.to_string();
            tracing::debug!(version = VERSION, "defaulted bot version from binary");
        }
        tracing::info!(
            agents = config.bot.agents.len(),
            llm_providers = config.llm.len(),
            image_providers = config.image.len(),
            video_providers = config.video.len(),
            audio_providers = config.audio.len(),
            platforms = config.platforms.len(),
            "loaded runtime config"
        );
        Ok(config)
    }

    /// Validate cross references.
    #[tracing::instrument(
        name = "config.validate",
        skip_all,
        fields(
            agents = self.bot.agents.len(),
            llm_providers = self.llm.len(),
            image_providers = self.image.len(),
            video_providers = self.video.len(),
            audio_providers = self.audio.len(),
            platforms = self.platforms.len(),
        )
    )]
    pub fn validate(&self) -> Result<(), BinError> {
        self.validate_database()?;
        self.logging.filter()?;
        self.bot.validate()?;
        self.memory.compaction_interval_seconds()?;
        self.memory.diary_backfill_window_seconds()?;
        self.memory.diary_interval_seconds()?;

        let provider_names = self.llm.keys().collect::<BTreeSet<_>>();
        let image_provider_names = self.image.keys().collect::<BTreeSet<_>>();
        let video_provider_names = self.video.keys().collect::<BTreeSet<_>>();
        let audio_provider_names = self.audio.keys().collect::<BTreeSet<_>>();
        if self.memory.enabled {
            let memory_agents = self
                .memory
                .resolved_agent_providers(&self.bot.agents, self.bot.limits);
            for (agent, provider) in memory_agents {
                if !provider_names.contains(&provider) {
                    tracing::warn!(
                        agent = %agent,
                        provider = %provider,
                        "memory agent references missing provider config"
                    );
                    return Err(BinError::MissingMemoryProviderConfig { agent, provider });
                }
            }
        }
        for (agent_name, agent) in &self.bot.agents {
            if !provider_names.contains(&agent.provider) {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %agent.provider,
                    "agent references missing provider config"
                );
                return Err(BinError::MissingProviderConfig {
                    agent: agent_name.clone(),
                    provider: agent.provider.clone(),
                });
            }
            if let Some(binding) = &agent.image_generation
                && !image_provider_names.contains(&binding.provider)
            {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %binding.provider,
                    model = %binding.model,
                    "agent references missing image provider config"
                );
                return Err(BinError::MissingImageProviderConfig {
                    agent: agent_name.clone(),
                    provider: binding.provider.clone(),
                });
            }
            if let Some(binding) = &agent.video_generation
                && !video_provider_names.contains(&binding.provider)
            {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %binding.provider,
                    model = %binding.model,
                    "agent references missing video provider config"
                );
                return Err(BinError::MissingVideoProviderConfig {
                    agent: agent_name.clone(),
                    provider: binding.provider.clone(),
                });
            }
            if let Some(binding) = &agent.audio_transcription
                && !audio_provider_names.contains(&binding.provider)
            {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %binding.provider,
                    model = ?binding.model.as_ref(),
                    "agent references missing audio provider config"
                );
                return Err(BinError::MissingAudioProviderConfig {
                    agent: agent_name.clone(),
                    provider: binding.provider.clone(),
                });
            }
        }

        for platform in self.bot.platforms.keys() {
            if !self.platforms.contains_key(platform) {
                tracing::warn!(
                    platform = %platform,
                    "bot platform binding references missing platform config"
                );
                return Err(BinError::MissingPlatformConfig {
                    platform: platform.clone(),
                });
            }
        }

        SocketAddr::from_str(&self.web.listen)?;
        tracing::info!("runtime config validated");
        Ok(())
    }

    pub(crate) fn validate_database(&self) -> Result<(), BinError> {
        if self.database.url.trim().is_empty() {
            tracing::warn!("database URL is empty");
            return Err(BinError::MissingDatabaseUrl);
        }
        Ok(())
    }
}

/// Postgres database connection settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Standard `postgres://user:pass@host/db` URL.
    pub url: String,
}

/// Web listener plus viewer config.
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
    pub(crate) fn viewer_config(&self, fallback_public_base_url: &str) -> WebConfig {
        WebConfig {
            title_prefix: self.title_prefix.clone(),
            version: VERSION.to_string(),
            frontend_dir: self.frontend_dir.clone(),
            favicon_path: self.favicon_path.clone(),
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

/// Local storage directories.
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

/// Named LLM provider config.
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
    },
    /// OpenAI-compatible provider placeholder.
    #[serde(rename = "openai_compat")]
    OpenAiCompat {
        /// Base URL.
        base_url: String,
        /// Optional API key.
        #[serde(default)]
        api_key: Option<String>,
    },
}

/// Named image-generation provider config.
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
}

/// Named video-generation provider config.
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
}

/// Named audio transcription provider config.
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

/// Named message platform config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MessagePlatformConfig {
    /// Discord platform placeholder.
    Discord {
        /// Bot token.
        token: String,
        /// Deprecated. Commands now register globally so every installed guild
        /// can use them.
        #[serde(default)]
        dev_guild_id: Option<String>,
    },
}
