//! Conversation domain types. These mirror the Postgres schema and are
//! the source of truth for the web viewer's data model.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::llm::ToolCallRecord;

/// Serde helpers that move Discord snowflake IDs across the JSON API as
/// **strings** rather than numbers.
///
/// Discord IDs are 64-bit integers in the ~10^18 range, well past
/// JavaScript's `Number.MAX_SAFE_INTEGER` (2^53 ≈ 9×10^15). A snowflake
/// emitted as a JSON *number* is silently rounded the moment the browser
/// runs `JSON.parse` — e.g. `…023356` arrives as `…023300` — which then
/// fails to match the exact string key serde uses for `HashMap<i64, _>`
/// (JSON object keys are always strings). Emitting every snowflake as a
/// quoted string keeps it lossless end to end, matching Discord's own
/// API. The struct fields stay `i64`; `sqlx::FromRow` is unaffected, so
/// only the wire format changes.
mod snowflake {
    use serde::{Deserialize, Deserializer, Serializer};

    /// Lenient on-the-wire shape: we always *emit* a quoted string, but
    /// *accept* a bare integer too so hand-written fixtures and any
    /// older payloads still deserialize.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Str(String),
        Int(i64),
    }

    impl Repr {
        fn into_i64<E>(self) -> Result<i64, E>
        where
            E: serde::de::Error,
        {
            match self {
                Repr::Str(s) => s.parse().map_err(serde::de::Error::custom),
                Repr::Int(i) => Ok(i),
            }
        }
    }

    pub fn serialize<S>(id: &i64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // `collect_str` writes the `Display` form (decimal digits) as a
        // JSON string with no intermediate allocation.
        serializer.collect_str(id)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<i64, D::Error>
    where
        D: Deserializer<'de>,
    {
        Repr::deserialize(deserializer)?.into_i64()
    }

    /// Same as the parent module but for `Option<i64>`: `None` stays
    /// JSON `null`, `Some(n)` becomes the quoted decimal string.
    pub mod option {
        use super::Repr;
        use serde::{Deserialize, Deserializer, Serializer};

        pub fn serialize<S>(id: &Option<i64>, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            match id {
                Some(v) => serializer.collect_str(v),
                None => serializer.serialize_none(),
            }
        }

        pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
        where
            D: Deserializer<'de>,
        {
            match Option::<Repr>::deserialize(deserializer)? {
                Some(r) => r.into_i64().map(Some),
                None => Ok(None),
            }
        }
    }
}

/// A conversation between a user and the LLM. Identified by a UUID
/// surfaced in the web viewer URL. Created when a user mentions the bot
/// outside any existing conversation context.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Conversation {
    /// Stable identifier; appears in the web viewer URL.
    pub id: Uuid,
    /// When the conversation was opened.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Discord guild ID (server). Zero for DMs.
    #[serde(with = "snowflake")]
    pub discord_guild_id: i64,
    /// Discord channel ID where the first message lives.
    #[serde(with = "snowflake")]
    pub discord_channel_id: i64,
    /// Discord user ID that initiated the conversation.
    #[serde(with = "snowflake")]
    pub created_by_user_id: i64,
    /// Discord message ID of the very first user prompt.
    #[serde(with = "snowflake")]
    pub root_discord_message_id: i64,
    /// Optional human-readable title (inferred from first prompt).
    pub title: Option<String>,
    /// When the background titler last successfully populated `title`.
    /// `None` if it never ran (or hasn't yet — titles are generated
    /// asynchronously after the first turn completes).
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub title_generated_at: Option<OffsetDateTime>,
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
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// When the turn completed (success or failure).
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub completed_at: Option<OffsetDateTime>,
    /// Discord message ID of the user's prompt.
    #[serde(with = "snowflake")]
    pub user_discord_message_id: i64,
    /// Raw text of the user's prompt (mentions stripped).
    pub user_content: String,
    /// Discord message ID of the bot's reply (None until posted).
    #[serde(with = "snowflake::option", default)]
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
    /// Discord user id of whoever sent the prompt. `None` for turns
    /// written before the identity-tracking feature shipped.
    #[serde(with = "snowflake::option", default)]
    pub discord_user_id: Option<i64>,
    /// Display name (or username if no display name was set) of that
    /// user *at the time the turn was recorded* — names can change but
    /// the historical attribution stays pinned to the turn.
    pub discord_user_name: Option<String>,
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
    #[serde(with = "snowflake::option", default)]
    pub discord_message_id: Option<i64>,
}

/// One image to replay into a later turn's model context. Produced by
/// [`crate::Db::load_conversation_image_uris`] from both user-uploaded
/// image `context_items` and `generate_image` tool-call outputs, so the
/// model can still "see" images from earlier turns. Not persisted — the
/// per-turn viewer trace reads its rows directly; this is purely the
/// in-memory replay set.
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct ReplayImage {
    /// Turn the image belonged to; its bytes get re-attached to that
    /// turn's user message on replay.
    pub turn_id: Uuid,
    /// Turn index, used only to order images chronologically so the
    /// most-recent-N cap drops the oldest first.
    pub turn_index: i32,
    /// Stored media URI (`file://images/<uuid>.<ext>`).
    pub uri: String,
}

/// Cached identity of one Discord user the bot has interacted with.
/// Backs the web viewer's per-message avatar + name rendering. Rows
/// are upserted from every `MessageCreate` event so this row tracks
/// the *current* known identity (not historical — turns carry their
/// own historical name snapshot).
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DiscordUser {
    /// Discord user id (snowflake) cast to i64 for Postgres.
    #[serde(with = "snowflake")]
    pub id: i64,
    /// Username (the global, lowercase one, e.g. "chud").
    pub username: String,
    /// Per-server display name or global name override. `None` means
    /// the user has no display name and the viewer should fall back to
    /// `username`.
    pub display_name: Option<String>,
    /// Discord avatar hash. `None` means the user has the default
    /// (auto-generated) avatar; the fetcher renders that via the
    /// `embed/avatars/{(id >> 22) % 6}.png` endpoint instead.
    pub avatar_hash: Option<String>,
    /// Filesystem path (relative to `storage.avatars_dir`) of the
    /// cached avatar. `None` until the fetcher has stored something.
    pub avatar_local_path: Option<String>,
    /// When the fetcher last successfully wrote a file for this user.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub last_avatar_fetched_at: Option<OffsetDateTime>,
    /// When this row was last touched by an inbound Discord event.
    #[serde(with = "time::serde::rfc3339")]
    pub last_seen_at: OffsetDateTime,
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
    #[serde(with = "time::serde::rfc3339")]
    pub submitted_at: OffsetDateTime,
    /// When status reached a terminal state.
    #[serde(with = "time::serde::rfc3339::option", default)]
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
    /// All Discord users whose ids appear on any turn in this view,
    /// keyed by user id. Lets the frontend render avatars + names
    /// without an N+1 fetch per turn.
    pub users: HashMap<i64, DiscordUser>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use time::macros::datetime;

    // Locks the wire format for timestamps: the JS frontend parses
    // these with `new Date(...)`, which only accepts ISO-8601 strings.
    // Without the `time/serde-well-known` feature + rfc3339 annotations
    // the default serde repr is a 9-element integer array, which
    // `new Date(...)` reads as NaN and crashes the viewer. This test
    // fails loudly if either is ever dropped.
    #[test]
    fn timestamps_serialize_as_rfc3339_strings() {
        let conv = Conversation {
            id: Uuid::nil(),
            created_at: datetime!(2026-05-28 17:11:51 UTC),
            discord_guild_id: 0,
            discord_channel_id: 0,
            created_by_user_id: 0,
            root_discord_message_id: 0,
            title: None,
            title_generated_at: Some(datetime!(2026-05-28 17:12:00 UTC)),
            model: "test".to_string(),
        };
        let json = serde_json::to_value(&conv).unwrap();
        assert_eq!(json["created_at"], "2026-05-28T17:11:51Z");
        assert_eq!(json["title_generated_at"], "2026-05-28T17:12:00Z");
        // A null option must stay null (not an array, not absent).
        let conv2 = Conversation {
            title_generated_at: None,
            ..conv
        };
        assert!(serde_json::to_value(&conv2).unwrap()["title_generated_at"].is_null());
    }

    // Discord snowflakes must cross the JSON boundary as strings. As
    // JSON *numbers* they exceed JS's 2^53 safe-integer range and get
    // silently rounded (…356 → …300) the moment the browser runs
    // `JSON.parse`. This test fails loudly if the `snowflake` serde
    // annotations are ever dropped.
    #[test]
    fn snowflakes_serialize_as_strings() {
        // A real id that is NOT representable exactly as an f64 — this
        // is precisely why a JSON number would corrupt it.
        let id: i64 = 1335037364980023356;
        assert_ne!(id as f64 as i64, id, "test id must lose precision as f64");

        let conv = Conversation {
            id: Uuid::nil(),
            created_at: datetime!(2026-05-28 17:11:51 UTC),
            discord_guild_id: 384888918006562800,
            discord_channel_id: 1508906237700477200,
            created_by_user_id: id,
            root_discord_message_id: 1509612651402100700,
            title: None,
            title_generated_at: None,
            model: "test".to_string(),
        };
        let json = serde_json::to_value(&conv).unwrap();
        assert_eq!(json["created_by_user_id"], "1335037364980023356");
        assert!(json["discord_guild_id"].is_string());
        assert!(json["discord_channel_id"].is_string());
        assert!(json["root_discord_message_id"].is_string());

        // Round-trips back to the exact i64 (lossless, symmetric).
        let back: Conversation = serde_json::from_value(json).unwrap();
        assert_eq!(back.created_by_user_id, id);
    }

    // The `users` map key, `DiscordUser.id`, and `Turn.discord_user_id`
    // must all serialize to the SAME exact string, or the frontend's
    // `users[turn.discord_user_id]` lookup misses and the avatar/name
    // don't render. This is the exact bug that prompted the fix.
    #[test]
    fn view_user_key_matches_turn_and_id() {
        let uid: i64 = 1335037364980023356;
        let user = DiscordUser {
            id: uid,
            username: "joe".to_string(),
            display_name: Some("Robert".to_string()),
            avatar_hash: None,
            avatar_local_path: None,
            last_avatar_fetched_at: None,
            last_seen_at: datetime!(2026-05-28 17:11:51 UTC),
        };
        let turn = Turn {
            id: Uuid::nil(),
            conversation_id: Uuid::nil(),
            turn_index: 0,
            created_at: datetime!(2026-05-28 17:11:51 UTC),
            completed_at: None,
            user_discord_message_id: 1509612651402100700,
            user_content: "hi".to_string(),
            assistant_discord_message_id: None,
            assistant_content: None,
            status: "completed".to_string(),
            error: None,
            persona_name: None,
            discord_user_id: Some(uid),
            discord_user_name: Some("Robert".to_string()),
        };
        let view = ConversationView {
            conversation: Conversation {
                id: Uuid::nil(),
                created_at: datetime!(2026-05-28 17:11:51 UTC),
                discord_guild_id: 0,
                discord_channel_id: 0,
                created_by_user_id: uid,
                root_discord_message_id: 0,
                title: None,
                title_generated_at: None,
                model: "test".to_string(),
            },
            turns: vec![TurnView {
                turn,
                context: vec![],
                tool_calls: vec![],
            }],
            users: HashMap::from([(uid, user)]),
        };
        let json = serde_json::to_value(&view).unwrap();
        let key = "1335037364980023356";
        // Map key is the exact snowflake string...
        assert!(json["users"].get(key).is_some(), "users map key must be the exact id");
        // ...and equals both the embedded id and the turn's user id.
        assert_eq!(json["users"][key]["id"], key);
        assert_eq!(json["turns"][0]["turn"]["discord_user_id"], key);
        // A `None` snowflake option stays JSON null (not "null", not 0).
        assert!(json["turns"][0]["turn"]["assistant_discord_message_id"].is_null());
    }
}
