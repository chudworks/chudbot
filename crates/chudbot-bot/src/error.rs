//! Bot-level error types and adapters from platform/storage failures.

use crate::prelude::*;

#[derive(Debug, Error)]
pub enum BotError {
    /// Messaging platform failed.
    #[error("platform error: {message}")]
    Platform {
        /// Error message.
        message: String,
    },
    /// Storage failed.
    #[error("storage error: {message}")]
    Storage {
        /// Error message.
        message: String,
    },
    /// Configured agent is missing.
    #[error("agent `{name}` is not configured")]
    MissingAgent {
        /// Agent name.
        name: String,
    },
    /// A subagent binding points at an unknown agent.
    #[error("agent `{agent}` references missing subagent `{subagent}`")]
    MissingSubagent {
        /// Parent agent name.
        agent: String,
        /// Missing subagent name.
        subagent: String,
    },
    /// Retry storage result did not include the requested turn.
    #[error("retry turn `{turn_id}` was not present in the loaded conversation")]
    MissingRetryTurn {
        /// Turn id.
        turn_id: TurnId,
    },
    /// Storage could not reload a conversation that was just referenced.
    #[error("conversation `{conversation_id}` was not found")]
    MissingConversation {
        /// Conversation id.
        conversation_id: ConversationId,
    },
    /// Agent references an unavailable provider.
    #[error("agent `{agent}` uses provider `{provider}` but that provider is not configured")]
    MissingProvider {
        /// Agent name.
        agent: String,
        /// Missing provider.
        provider: ProviderName,
    },
    /// Agent references an unavailable image generator.
    #[error(
        "agent `{agent}` uses image provider `{provider}` but that generator is not configured"
    )]
    MissingImageGenerator {
        /// Agent name.
        agent: String,
        /// Missing image provider.
        provider: ProviderName,
    },
    /// Agent references an unavailable video generator.
    #[error(
        "agent `{agent}` uses video provider `{provider}` but that generator is not configured"
    )]
    MissingVideoGenerator {
        /// Agent name.
        agent: String,
        /// Missing video provider.
        provider: ProviderName,
    },
    /// Agent references an unavailable audio transcriber.
    #[error(
        "agent `{agent}` uses audio provider `{provider}` but that transcriber is not configured"
    )]
    MissingAudioTranscriber {
        /// Agent name.
        agent: String,
        /// Missing audio provider.
        provider: ProviderName,
    },
    /// Agent media-generation binding is malformed.
    #[error("agent `{agent}` has invalid `{field}` binding: {message}")]
    InvalidGenerationBinding {
        /// Agent name.
        agent: String,
        /// Config field.
        field: &'static str,
        /// Error detail.
        message: String,
    },
    /// Agent graph is recursive.
    #[error("agent `{name}` recursively references itself through subagents")]
    RecursiveAgent {
        /// Agent name.
        name: String,
    },
    /// Command input could not be resolved.
    #[error("command input: {0}")]
    CommandInput(String),
    /// One-shot model operation failed.
    #[error("model operation failed: {message}")]
    Model {
        /// Error message.
        message: String,
    },
    /// Avatar download failed.
    #[error("avatar download failed: {0}")]
    AvatarDownload(String),
}

pub(crate) fn platform_error(error: impl std::fmt::Display) -> BotError {
    BotError::Platform {
        message: error.to_string(),
    }
}

pub(crate) fn storage_error(error: impl std::fmt::Display) -> BotError {
    BotError::Storage {
        message: error.to_string(),
    }
}
