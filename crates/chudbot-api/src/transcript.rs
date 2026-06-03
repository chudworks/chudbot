//! Runtime model transcript types.

use serde::{Deserialize, Serialize};

use crate::ids::ProviderName;
use crate::media::BoxedMediaRef;
use crate::tool::{ClientToolCall, ClientToolResult};

/// Full model-facing transcript.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    /// Stable transcript id. Providers may use this for prompt-cache routing.
    pub id: Option<String>,
    /// System/developer instructions for this transcript.
    pub instructions: Option<String>,
    /// Ordered turns.
    pub turns: Vec<TranscriptTurn>,
}

impl Transcript {
    /// Empty transcript.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a transcript from one user text turn.
    pub fn from_user_text(text: impl Into<String>) -> Self {
        let mut transcript = Self::new();
        transcript.push(TranscriptTurn::text(TurnRole::User, text.into()));
        transcript
    }

    /// Append a message.
    pub fn push(&mut self, turn: TranscriptTurn) {
        self.turns.push(turn);
    }
}

/// One model-facing transcript turn.
#[derive(Debug, Clone)]
pub struct TranscriptTurn {
    /// Turn role.
    pub role: TurnRole,
    /// Ordered content blocks.
    pub blocks: Vec<ContentBlock>,
    /// Opaque metadata owned by the caller/provider.
    pub metadata: serde_json::Value,
}

impl TranscriptTurn {
    /// Convenience constructor for a one-text-block message.
    pub fn text(role: TurnRole, text: impl Into<String>) -> Self {
        Self {
            role,
            blocks: vec![ContentBlock::Text { text: text.into() }],
            metadata: serde_json::Value::Null,
        }
    }
}

/// Model-facing turn role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnRole {
    /// End user.
    User,
    /// Assistant/model.
    Assistant,
}

/// Content block inside a model message.
#[derive(Debug, Clone)]
pub enum ContentBlock {
    /// Plain text.
    Text {
        /// Text content.
        text: String,
    },
    /// Image/video/media reference.
    Media {
        /// Media reference.
        media: BoxedMediaRef,
    },
    /// Assistant-requested client tool invocation.
    ClientToolCall(ClientToolCall),
    /// User-provided client tool result.
    ClientToolResult(ClientToolResult),
    /// Opaque provider continuation state, replayed only to the provider that
    /// emitted it.
    Continuation(ProviderContinuation),
}

/// Opaque provider-specific continuation state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderContinuation {
    /// Provider that emitted this continuation.
    pub provider: ProviderName,
    /// Provider-specific payload.
    pub data: serde_json::Value,
}
