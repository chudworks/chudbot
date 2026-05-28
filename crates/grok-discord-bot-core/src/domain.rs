//! Conversation domain types. These mirror the Postgres schema and are
//! the source of truth for the web viewer's data model.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::llm::ToolCallRecord;

/// A conversation between a user and the LLM. Identified by a UUID
/// surfaced in the web viewer URL. Created when a user mentions the bot
/// outside any existing conversation context.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Conversation {
    /// Stable identifier; appears in the web viewer URL.
    pub id: Uuid,
    /// When the conversation was opened.
    pub created_at: OffsetDateTime,
    /// Discord guild ID (server). Zero for DMs.
    pub discord_guild_id: i64,
    /// Discord channel ID where the first message lives.
    pub discord_channel_id: i64,
    /// Discord user ID that initiated the conversation.
    pub created_by_user_id: i64,
    /// Discord message ID of the very first user prompt.
    pub root_discord_message_id: i64,
    /// Optional human-readable title (inferred from first prompt).
    pub title: Option<String>,
    /// LLM provider identifier (e.g. `xai/grok-4.1-fast`).
    pub model: String,
}

/// One user→assistant exchange within a conversation. A conversation is
/// an ordered list of turns. Each turn captures exactly what was fed to
/// the model (via [`ContextItem`] rows) and the resulting answer.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Turn {
    /// Stable identifier.
    pub id: Uuid,
    /// Owning conversation.
    pub conversation_id: Uuid,
    /// Zero-based index within the conversation.
    pub turn_index: i32,
    /// When the turn was started.
    pub created_at: OffsetDateTime,
    /// When the turn completed (success or failure).
    pub completed_at: Option<OffsetDateTime>,
    /// Discord message ID of the user's prompt.
    pub user_discord_message_id: i64,
    /// Raw text of the user's prompt (mentions stripped).
    pub user_content: String,
    /// Discord message ID of the bot's reply (None until posted).
    pub assistant_discord_message_id: Option<i64>,
    /// Final answer text from the model.
    pub assistant_content: Option<String>,
    /// `pending` | `completed` | `failed`.
    pub status: String,
    /// Error message if `status = 'failed'`.
    pub error: Option<String>,
    /// Persona name active when this turn ran. `None` for turns
    /// written before the personas feature shipped.
    pub persona_name: Option<String>,
}

/// One row in `context_items`: a single message snapshot that was sent
/// to the LLM for a given turn. Recorded so the viewer can show the
/// exact context the model saw, not a recomputed one.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ContextItem {
    /// Position in the prompt (0-based).
    pub position: i32,
    /// Where the content came from (`system`, `discord:msg:<id>`,
    /// `turn:<uuid>:user|assistant`).
    pub source: String,
    /// Role assigned in the chat sequence. Lowercase string
    /// (`system` / `user` / `assistant`) to keep the DB boundary
    /// schema-flexible.
    pub role: String,
    /// Verbatim text sent to the model.
    pub content: String,
    /// Original Discord message ID, when applicable.
    pub discord_message_id: Option<i64>,
}

/// One outstanding (or completed) video generation job submitted to
/// xAI. Mutated as `check_video_status` polls; surfaces to the
/// viewer alongside the tool call trace.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct VideoJob {
    /// Stable identifier.
    pub id: Uuid,
    /// Owning turn.
    pub turn_id: Uuid,
    /// xAI's request_id; the key used for polling.
    pub request_id: String,
    /// Prompt the job was submitted with.
    pub prompt: String,
    /// `pending` | `done` | `failed` | `expired`.
    pub status: String,
    /// `file://videos/<uuid>.mp4` URI once status flips to `done`.
    pub video_uri: Option<String>,
    /// When the submit call succeeded.
    pub submitted_at: OffsetDateTime,
    /// When status reached a terminal state.
    pub completed_at: Option<OffsetDateTime>,
    /// Upstream error message if `status = 'failed'` or `'expired'`.
    pub error: Option<String>,
}

/// Aggregated read-model for the web viewer: a conversation plus all of
/// its turns, each with its context items and tool calls.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationView {
    /// Conversation row.
    pub conversation: Conversation,
    /// Turns, ordered by [`Turn::turn_index`] ascending.
    pub turns: Vec<TurnView>,
}

/// One turn plus its context and tool calls. Used only for rendering.
#[derive(Debug, Clone, Serialize)]
pub struct TurnView {
    /// Turn row.
    pub turn: Turn,
    /// Context items fed to the model, ordered by `position` ascending.
    pub context: Vec<ContextItem>,
    /// Server-side tool calls, ordered by `ordinal` ascending.
    pub tool_calls: Vec<ToolCallRecord>,
}
