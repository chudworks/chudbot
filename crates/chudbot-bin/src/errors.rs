use std::path::PathBuf;

use chudbot_api::ProviderName;
use thiserror::Error;
use tokio::task::JoinError;

use crate::config::LoggingFilterError;
use crate::diagnostics::ConfigValidationReport;

/// Errors from the concrete provider registry.
#[derive(Debug, Error)]
pub enum ConfiguredLlmError {
    /// Provider was referenced but not implemented/configured.
    #[error("provider `{0}` is not available in this runtime")]
    Missing(ProviderName),
    /// Anthropic request failed.
    #[error(transparent)]
    Anthropic(#[from] chudbot_anthropic::AnthropicError),
    /// Gemini request failed.
    #[error(transparent)]
    Gemini(#[from] chudbot_gemini::GeminiError),
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
    #[error("image provider `{0}` is not available in this runtime")]
    Missing(ProviderName),
    /// Gemini image generation failed.
    #[error(transparent)]
    Gemini(#[from] chudbot_gemini::GeminiError),
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
    #[error("video provider `{0}` is not available in this runtime")]
    Missing(ProviderName),
    /// Gemini video generation failed.
    #[error(transparent)]
    Gemini(#[from] chudbot_gemini::GeminiError),
    /// xAI video generation failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors from the concrete audio transcription registry.
#[derive(Debug, Error)]
pub enum ConfiguredAudioError {
    /// Provider was referenced but not implemented/configured.
    #[error("audio provider `{0}` is not available in this runtime")]
    Missing(ProviderName),
    /// xAI audio transcription failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors from the concrete message-platform registry.
#[derive(Debug, Error)]
pub enum ConfiguredPlatformError {
    /// No platform exists for a requested platform name.
    #[error("message platform `{0}` is not available in this runtime")]
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
    /// Config validation failed.
    #[error(transparent)]
    ConfigValidation(#[from] ConfigValidationReport),
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
        /// Source TOML.
        content: Box<str>,
        /// Source error.
        #[source]
        source: Box<toml::de::Error>,
    },
}
