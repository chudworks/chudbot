//! Conversation domain types. These mirror the Postgres schema and are
//! the source of truth for the web viewer's data model.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

/// A conversation between a user and Grok. Identified by a UUID that is
/// surfaced in the web viewer URL. Created implicitly when a user
/// mentions the bot outside of an existing conversation context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// Stable identifier; appears in the web viewer URL.
    pub id: Uuid,
    /// When the conversation was opened.
    pub created_at: OffsetDateTime,
    /// Discord user who initiated it.
    pub created_by_user_id: u64,
    /// Discord channel the first message lived in.
    pub discord_channel_id: u64,
    /// Optional human-readable title (e.g. inferred from the first prompt).
    pub title: Option<String>,
}

/// One user→assistant exchange within a conversation. A conversation is
/// an ordered list of turns. Each turn records exactly what was fed to
/// Grok (see `context_items` table) and the resulting answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Stable identifier.
    pub id: Uuid,
    /// Owning conversation.
    pub conversation_id: Uuid,
    /// Zero-based index within the conversation.
    pub index: i32,
    /// The user's prompt text.
    pub user_content: String,
    /// Grok's answer text (None while the turn is still in flight).
    pub assistant_content: Option<String>,
    /// When the turn was started.
    pub created_at: OffsetDateTime,
}
