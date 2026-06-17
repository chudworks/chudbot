//! Bot-level error reporting for the orchestration crate.
//!
//! `chudbot-bot` sits between platform adapters, storage, provider
//! registries, and command/turn logic. This module keeps those failures in one
//! public error type so callers such as `chudbot-bin` can report them without
//! the bot crate exposing concrete Discord, SQLx, or provider-client errors.

use crate::prelude::*;

/// Errors that can stop bot orchestration before a user-facing action is ready.
///
/// The variants intentionally describe the boundary that failed rather than the
/// underlying implementation type. Platform and storage adapters are converted
/// to strings at the crate boundary; configuration and runtime-invariant
/// failures keep structured fields so logs and tests can identify the broken
/// agent, provider, conversation, or turn.
#[derive(Debug, Error)]
pub enum BotError {
    /// Messaging platform operation failed while handling an event or reply.
    #[error("platform error: {message}")]
    Platform {
        /// Display text from the concrete platform adapter error.
        message: String,
    },
    /// Persistent storage operation failed.
    #[error("storage error: {message}")]
    Storage {
        /// Display text from the concrete storage adapter error.
        message: String,
    },
    /// Config or command selection names an agent absent from the bot config.
    #[error("agent `{name}` is not configured")]
    MissingAgent {
        /// Requested or default agent name.
        name: String,
    },
    /// Agent config exposes a subagent tool whose target agent is absent.
    #[error("agent `{agent}` references missing subagent `{subagent}`")]
    MissingSubagent {
        /// Agent that owns the invalid subagent binding.
        agent: String,
        /// Target agent name from the subagent binding.
        subagent: String,
    },
    /// Retry storage result did not include the turn that was requested.
    #[error("retry turn `{turn_id}` was not present in the loaded conversation")]
    MissingRetryTurn {
        /// Turn id that initiated retry handling.
        turn_id: TurnId,
    },
    /// Storage could not reload a conversation that another record referenced.
    #[error("conversation `{conversation_id}` was not found")]
    MissingConversation {
        /// Conversation id that should already exist.
        conversation_id: ConversationId,
    },
    /// Agent references an LLM provider that is absent from the runtime registry.
    #[error("agent `{agent}` uses provider `{provider}` but that provider is not configured")]
    MissingProvider {
        /// Agent whose model backend could not be built.
        agent: String,
        /// Provider registry key from the agent config.
        provider: ProviderName,
    },
    /// Agent references an image generator absent from the runtime registry.
    #[error(
        "agent `{agent}` uses image provider `{provider}` but that generator is not configured"
    )]
    MissingImageGenerator {
        /// Agent whose image tool binding could not be enabled.
        agent: String,
        /// Image provider registry key from the agent config.
        provider: ProviderName,
    },
    /// Agent references a video generator absent from the runtime registry.
    #[error(
        "agent `{agent}` uses video provider `{provider}` but that generator is not configured"
    )]
    MissingVideoGenerator {
        /// Agent whose video tool binding could not be enabled.
        agent: String,
        /// Video provider registry key from the agent config.
        provider: ProviderName,
    },
    /// Agent references an audio transcriber absent from the runtime registry.
    #[error(
        "agent `{agent}` uses audio provider `{provider}` but that transcriber is not configured"
    )]
    MissingAudioTranscriber {
        /// Agent whose transcription tool binding could not be enabled.
        agent: String,
        /// Audio provider registry key from the agent config.
        provider: ProviderName,
    },
    /// Agent media or transcription binding failed semantic validation.
    #[error("agent `{agent}` has invalid `{field}` binding: {message}")]
    InvalidGenerationBinding {
        /// Agent that owns the malformed binding.
        agent: String,
        /// Config field being validated, such as `image_generation`.
        field: &'static str,
        /// Operator-facing explanation of the invalid binding value.
        message: String,
    },
    /// Agent graph would recurse through subagent tool construction.
    #[error("agent `{name}` recursively references itself through subagents")]
    RecursiveAgent {
        /// Agent name observed twice in the current construction stack.
        name: String,
    },
    /// Slash-command or platform-command input could not be resolved.
    #[error("command input: {0}")]
    CommandInput(String),
    /// One-shot model operation failed outside the normal conversation turn.
    #[error("model operation failed: {message}")]
    Model {
        /// Provider or response-shape failure text from the background job.
        message: String,
    },
    /// Avatar image fetch or upload failed while refreshing agent avatars.
    #[error("avatar download failed: {0}")]
    AvatarDownload(String),
}

/// Convert a platform-adapter error into the bot crate's public error type.
pub(crate) fn platform_error(error: impl std::fmt::Display) -> BotError {
    // Store display text at the boundary so concrete platform errors do not
    // leak into `chudbot-bot`'s public API.
    BotError::Platform {
        message: error.to_string(),
    }
}

/// Convert a storage-adapter error into the bot crate's public error type.
pub(crate) fn storage_error(error: impl std::fmt::Display) -> BotError {
    // Storage backends stay behind `chudbot-api`; this wrapper keeps the bot
    // error stable even if the SQL implementation changes.
    BotError::Storage {
        message: error.to_string(),
    }
}
