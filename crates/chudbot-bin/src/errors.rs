use std::path::PathBuf;

use chudbot_api::ProviderName;
use thiserror::Error;
use tokio::task::JoinError;

use crate::config::LoggingFilterError;

/// Errors from the concrete provider registry.
#[derive(Debug, Error)]
pub enum ConfiguredLlmError {
    /// Provider was referenced but not implemented/configured.
    #[error("provider `{0}` is not available in the 2.0 runtime")]
    Missing(ProviderName),
    /// Anthropic request failed.
    #[error(transparent)]
    Anthropic(#[from] chudbot_anthropic::AnthropicError),
    /// OpenAI request failed.
    #[error(transparent)]
    OpenAi(#[from] chudbot_openai::OpenAiError),
    /// OpenAI-compatible request failed.
    #[error(transparent)]
    OpenAiCompat(#[from] chudbot_openai_compat::OpenAiCompatError),
    /// xAI request failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors from the concrete image-generation registry.
#[derive(Debug, Error)]
pub enum ConfiguredImageError {
    /// Provider was referenced but not implemented/configured.
    #[error("image provider `{0}` is not available in the 2.0 runtime")]
    Missing(ProviderName),
    /// OpenAI image generation failed.
    #[error(transparent)]
    OpenAi(#[from] chudbot_openai::OpenAiError),
    /// xAI image generation failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors from the concrete video-generation registry.
#[derive(Debug, Error)]
pub enum ConfiguredVideoError {
    /// Provider was referenced but not implemented/configured.
    #[error("video provider `{0}` is not available in the 2.0 runtime")]
    Missing(ProviderName),
    /// xAI video generation failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors from the concrete audio transcription registry.
#[derive(Debug, Error)]
pub enum ConfiguredAudioError {
    /// Provider was referenced but not implemented/configured.
    #[error("audio provider `{0}` is not available in the 2.0 runtime")]
    Missing(ProviderName),
    /// xAI audio transcription failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors from the concrete message-platform registry.
#[derive(Debug, Error)]
pub enum ConfiguredPlatformError {
    /// No platform exists for a requested platform name.
    #[error("message platform `{0}` is not available in the 2.0 runtime")]
    Missing(chudbot_api::PlatformName),
    /// The registry is empty.
    #[error("no message platforms are configured")]
    Empty,
    /// All event pump tasks stopped.
    #[error("all message platform event streams are closed")]
    EventsClosed,
    /// A platform event pump panicked.
    #[error("message platform `{platform}` event pump panicked: {message}")]
    EventPumpPanic {
        /// Platform name.
        platform: chudbot_api::PlatformName,
        /// Panic payload.
        message: String,
    },
    /// Discord platform failed.
    #[error(transparent)]
    Discord(#[from] chudbot_discord::DiscordError),
}

/// Top-level binary errors.
#[derive(Debug, Error)]
pub enum BinError {
    /// Config load failed.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Bot config failed validation.
    #[error(transparent)]
    Bot(#[from] chudbot_bot::BotError),
    /// Web server failed.
    #[error(transparent)]
    Web(#[from] chudbot_web::WebServerError),
    /// SQLx storage failed.
    #[error(transparent)]
    Storage(#[from] chudbot_storage_sqlx::SqlxStorageError),
    /// Platform setup failed.
    #[error(transparent)]
    Platform(#[from] ConfiguredPlatformError),
    /// Logging filter failed validation.
    #[error(transparent)]
    LoggingFilter(#[from] LoggingFilterError),
    /// Memory config failed validation.
    #[error(transparent)]
    MemoryConfig(#[from] chudbot_bot::memory::MemoryConfigError),
    /// A service task failed to join.
    #[error("{task} service task join failed: {source}")]
    TaskJoin {
        /// Service name.
        task: &'static str,
        /// Join error.
        source: JoinError,
    },
    /// Agent provider has no matching `[llm.<name>]` config.
    #[error("agent `{agent}` uses provider `{provider}` but no matching [llm] entry exists")]
    MissingProviderConfig {
        /// Agent name.
        agent: String,
        /// Provider name.
        provider: ProviderName,
    },
    /// Memory provider has no matching `[llm.<name>]` config.
    #[error("memory agent `{agent}` uses provider `{provider}` but no matching [llm] entry exists")]
    MissingMemoryProviderConfig {
        /// Memory agent name.
        agent: String,
        /// Provider name.
        provider: ProviderName,
    },
    /// Agent image provider has no matching `[image.<name>]` config.
    #[error(
        "agent `{agent}` uses image provider `{provider}` but no matching [image] entry exists"
    )]
    MissingImageProviderConfig {
        /// Agent name.
        agent: String,
        /// Provider name.
        provider: ProviderName,
    },
    /// Agent video provider has no matching `[video.<name>]` config.
    #[error(
        "agent `{agent}` uses video provider `{provider}` but no matching [video] entry exists"
    )]
    MissingVideoProviderConfig {
        /// Agent name.
        agent: String,
        /// Provider name.
        provider: ProviderName,
    },
    /// Agent audio provider has no matching `[audio.<name>]` config.
    #[error(
        "agent `{agent}` uses audio provider `{provider}` but no matching [audio] entry exists"
    )]
    MissingAudioProviderConfig {
        /// Agent name.
        agent: String,
        /// Provider name.
        provider: ProviderName,
    },
    /// Platform binding has no matching platform config.
    #[error("platform `{platform}` is bound in [bot.platforms] but has no [platforms] entry")]
    MissingPlatformConfig {
        /// Platform name.
        platform: chudbot_api::PlatformName,
    },
    /// Database URL was omitted.
    #[error("database.url must not be empty")]
    MissingDatabaseUrl,
    /// Listen address failed to parse.
    #[error("invalid web listen address: {0}")]
    Listen(#[from] std::net::AddrParseError),
}

/// Config parse errors.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Config file could not be read.
    #[error("could not read config file {}", path.display())]
    Read {
        /// Path.
        path: PathBuf,
        /// Source error.
        #[source]
        source: std::io::Error,
    },
    /// Config file could not be parsed.
    #[error("could not parse config file {}", path.display())]
    Parse {
        /// Path.
        path: PathBuf,
        /// Source error.
        #[source]
        source: toml::de::Error,
    },
}
