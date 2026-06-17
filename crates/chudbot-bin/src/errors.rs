//! Error taxonomy for the binary crate.
//!
//! The binary owns concrete provider/platform wiring, config loading, and
//! process lifecycle. These errors keep those boundaries explicit: registry
//! errors adapt concrete implementations to `chudbot-api` traits, `ConfigError`
//! preserves raw TOML for diagnostics, and `BinError` is the single top-level
//! error type reported by `main`.

use std::path::PathBuf;

use chudbot_api::ProviderName;
use thiserror::Error;
use tokio::task::JoinError;

use crate::config::LoggingFilterError;
use crate::diagnostics::ConfigValidationReport;

/// Errors returned by the named LLM provider registry.
///
/// The registry is built from `[llm.<name>]` config and dispatches each request
/// to the matching concrete provider crate. Provider variants stay transparent
/// so callers retain the original source chain.
#[derive(Debug, Error)]
pub enum ConfiguredLlmError {
    /// A referenced provider name has no registered LLM client.
    #[error("provider `{0}` is not available in this runtime")]
    Missing(ProviderName),
    /// Anthropic model metadata or model-step request failed.
    #[error(transparent)]
    Anthropic(#[from] chudbot_anthropic::AnthropicError),
    /// Gemini model metadata or model-step request failed.
    #[error(transparent)]
    Gemini(#[from] chudbot_gemini::GeminiError),
    /// OpenAI model metadata or model-step request failed.
    #[error(transparent)]
    OpenAi(#[from] chudbot_openai::OpenAiError),
    /// OpenAI-compatible model metadata or model-step request failed.
    #[error(transparent)]
    OpenAiCompat(#[from] chudbot_openai_compat::OpenAiCompatError),
    /// xAI model metadata or model-step request failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors returned by the named image-generation registry.
///
/// This groups all implemented image backends behind the
/// `ImageGeneratorRegistry` associated error while preserving concrete provider
/// failures for diagnostics.
#[derive(Debug, Error)]
pub enum ConfiguredImageError {
    /// A referenced provider name has no registered image generator.
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

/// Errors returned by the named video-generation registry.
///
/// The same error type covers video submit, status check, and download calls so
/// long-running video jobs can bubble failures through one API boundary.
#[derive(Debug, Error)]
pub enum ConfiguredVideoError {
    /// A referenced provider name has no registered video generator.
    #[error("video provider `{0}` is not available in this runtime")]
    Missing(ProviderName),
    /// Gemini video submit, status, or download failed.
    #[error(transparent)]
    Gemini(#[from] chudbot_gemini::GeminiError),
    /// xAI video submit, status, or download failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors returned by the named audio transcription registry.
///
/// Audio currently has fewer backends than text/image/video, but keeps the same
/// registry error shape so bot tools can handle all media providers uniformly.
#[derive(Debug, Error)]
pub enum ConfiguredAudioError {
    /// A referenced provider name has no registered audio transcriber.
    #[error("audio provider `{0}` is not available in this runtime")]
    Missing(ProviderName),
    /// xAI audio transcription failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors returned by the named message-platform registry.
///
/// Platform errors cover both synchronous registry lookups and the async event
/// pumps that forward platform events into the bot runtime.
#[derive(Debug, Error)]
pub enum ConfiguredPlatformError {
    /// A requested platform name has no registered platform client.
    #[error("message platform `{0}` is not available in this runtime")]
    Missing(chudbot_api::PlatformName),
    /// Event polling was requested before any platform was configured.
    #[error("no message platforms are configured")]
    Empty,
    /// All platform event pump senders closed before another event arrived.
    #[error("all message platform event streams are closed")]
    EventsClosed,
    /// A spawned platform event pump panicked and was converted into an error event.
    #[error("message platform `{platform}` event pump panicked: {message}")]
    EventPumpPanic {
        /// Platform whose event pump panicked.
        platform: chudbot_api::PlatformName,
        /// Human-readable panic payload.
        message: String,
    },
    /// Discord platform setup, command, message, or event operation failed.
    #[error(transparent)]
    Discord(#[from] chudbot_discord::DiscordError),
}

/// Top-level error for CLI commands and process startup/runtime.
///
/// Variants mirror the binary phases so `main` can special-case rich config
/// diagnostics and otherwise print the standard error source chain.
#[derive(Debug, Error)]
pub enum BinError {
    /// Config file I/O or TOML deserialization failed.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Span-aware semantic config validation failed.
    #[error(transparent)]
    ConfigValidation(#[from] ConfigValidationReport),
    /// Bot orchestration or bot config validation failed.
    #[error(transparent)]
    Bot(#[from] chudbot_bot::BotError),
    /// Web viewer server setup or runtime failed.
    #[error(transparent)]
    Web(#[from] chudbot_web::WebServerError),
    /// Postgres storage setup, migrations, or runtime operation failed.
    #[error(transparent)]
    Storage(#[from] chudbot_storage_sqlx::SqlxStorageError),
    /// Message platform setup or runtime dispatch failed.
    #[error(transparent)]
    Platform(#[from] ConfiguredPlatformError),
    /// Logging filter failed validation before tracing was initialized.
    #[error(transparent)]
    LoggingFilter(#[from] LoggingFilterError),
    /// User-memory runtime config failed validation.
    #[error(transparent)]
    MemoryConfig(#[from] chudbot_bot::memory::MemoryConfigError),
    /// A spawned long-running service task failed to join cleanly.
    #[error("{task} service task join failed: {source}")]
    TaskJoin {
        /// Service task label used in logs and error output.
        task: &'static str,
        /// Tokio join error preserving cancellation or panic details.
        source: JoinError,
    },
    /// Database URL was omitted for a command that needs Postgres.
    #[error("database.url must not be empty")]
    MissingDatabaseUrl,
    /// Web listen address could not be parsed into a socket address.
    #[error("invalid web listen address: {0}")]
    Listen(#[from] std::net::AddrParseError),
}

/// Errors from reading and deserializing the TOML config file.
///
/// Semantic config validation is reported separately by
/// `ConfigValidationReport`; this type stops at file I/O and TOML parser
/// failures. Parse errors retain the original source text so the CLI can render
/// a useful location-aware diagnostic.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Config file could not be read.
    #[error("could not read config file {}", path.display())]
    Read {
        /// Config path requested by the CLI.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// Config file could not be parsed or deserialized as `RuntimeConfig`.
    #[error("could not parse config file {}", path.display())]
    Parse {
        /// Config path requested by the CLI.
        path: PathBuf,
        /// Original TOML source retained for stderr diagnostic rendering.
        content: Box<str>,
        /// Underlying TOML parser/deserializer error.
        #[source]
        source: Box<toml::de::Error>,
    },
}
