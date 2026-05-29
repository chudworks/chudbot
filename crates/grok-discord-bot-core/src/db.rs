//! Postgres data layer. Thin wrapper around [`sqlx::PgPool`] with helpers
//! for the conversation lifecycle: creating conversations, recording
//! turns + their context + tool calls, looking up conversations by
//! Discord message id, and reading the aggregated view for the web viewer.

use sqlx::PgPool;
use sqlx::migrate::Migrator;
use thiserror::Error;
use uuid::Uuid;

use crate::config::PrivacyMode;
use crate::domain::{
    AppVersion, ContextItem, Conversation, ConversationView, DiscordUser, ReplayImage, Turn,
    TurnView, VideoJob,
};
use crate::llm::ToolCallRecord;

/// Migrations baked in at compile time from the workspace's
/// `migrations/` directory. Run via [`Db::migrate`].
pub static MIGRATOR: Migrator = sqlx::migrate!("../../migrations");

/// Errors returned by [`Db`].
#[derive(Debug, Error)]
pub enum DbError {
    /// Underlying sqlx error (network, protocol, query).
    #[error("sqlx: {0}")]
    Sqlx(#[from] sqlx::Error),
    /// Migration runner error.
    #[error("migrate: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    /// JSON (de)serialization of a tool call's request/response payload.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Connection pool + query helpers. Cheap to clone (pool is internally
/// `Arc`d).
#[derive(Debug, Clone)]
pub struct Db {
    pool: PgPool,
}

impl Db {
    /// Connect to Postgres at `url`.
    pub async fn connect(url: &str) -> Result<Self, DbError> {
        let pool = PgPool::connect(url).await?;
        Ok(Self { pool })
    }

    /// Borrow the underlying pool (for tests or one-off queries).
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Run pending migrations.
    pub async fn migrate(&self) -> Result<(), DbError> {
        MIGRATOR.run(&self.pool).await?;
        Ok(())
    }

    /// Resolve the ordered "vN" version row for the running build,
    /// inserting it the first time this build is ever seen. Called once
    /// at `serve` startup with `env!("GIT_VERSION")`.
    ///
    /// SELECT-then-INSERT, deliberately *not* an `ON CONFLICT` upsert:
    /// the SERIAL `id` is the user-facing version number and must stay
    /// gap-free, but `ON CONFLICT DO NOTHING` still burns a sequence
    /// value on every conflict (Postgres allocates it before detecting
    /// the conflict, and sequences don't roll back). Reading first and
    /// only inserting for a never-seen build consumes a number exactly
    /// once per real version. Startup is effectively single-writer, so
    /// the read→insert race is vanishingly unlikely; the `UNIQUE`
    /// constraint still guarantees correctness, and a lost race surfaces
    /// as a unique-violation error rather than a duplicate row.
    pub async fn register_app_version(&self, git_version: &str) -> Result<AppVersion, DbError> {
        if let Some(existing) = sqlx::query_as::<_, AppVersion>(
            "SELECT id, git_version, first_seen FROM app_versions WHERE git_version = $1",
        )
        .bind(git_version)
        .fetch_optional(&self.pool)
        .await?
        {
            return Ok(existing);
        }

        let inserted = sqlx::query_as::<_, AppVersion>(
            "INSERT INTO app_versions (git_version) VALUES ($1) \
             RETURNING id, git_version, first_seen",
        )
        .bind(git_version)
        .fetch_one(&self.pool)
        .await?;
        Ok(inserted)
    }

    /// Create a new conversation row.
    pub async fn create_conversation(
        &self,
        discord_guild_id: i64,
        discord_channel_id: i64,
        created_by_user_id: i64,
        root_discord_message_id: i64,
        model: &str,
        title: Option<&str>,
    ) -> Result<Conversation, DbError> {
        let id = Uuid::new_v4();
        let conv = sqlx::query_as::<_, Conversation>(
            "INSERT INTO conversations \
               (id, discord_guild_id, discord_channel_id, created_by_user_id, \
                root_discord_message_id, title, model) \
             VALUES ($1, $2, $3, $4, $5, $6, $7) \
             RETURNING id, created_at, discord_guild_id, discord_channel_id, \
               created_by_user_id, root_discord_message_id, title, title_generated_at, \
               model, stopped_at, stopped_by_user_id",
        )
        .bind(id)
        .bind(discord_guild_id)
        .bind(discord_channel_id)
        .bind(created_by_user_id)
        .bind(root_discord_message_id)
        .bind(title)
        .bind(model)
        .fetch_one(&self.pool)
        .await?;
        Ok(conv)
    }

    /// Fetch a conversation by id.
    pub async fn get_conversation(&self, id: Uuid) -> Result<Option<Conversation>, DbError> {
        let conv = sqlx::query_as::<_, Conversation>(
            "SELECT id, created_at, discord_guild_id, discord_channel_id, \
              created_by_user_id, root_discord_message_id, title, title_generated_at, model, \
              stopped_at, stopped_by_user_id \
             FROM conversations WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(conv)
    }

    /// Pause the bot in a conversation: stamp `stopped_at = now()` and
    /// record the admin who did it. Idempotent — re-stopping an already
    /// stopped conversation just refreshes the timestamp/attribution.
    /// Triggered by an admin's 🛑 reaction; see the bot's reaction
    /// handler. Returns the number of rows touched (0 if the id is
    /// unknown).
    pub async fn stop_conversation(
        &self,
        id: Uuid,
        stopped_by_user_id: i64,
    ) -> Result<u64, DbError> {
        let result = sqlx::query(
            "UPDATE conversations SET stopped_at = now(), stopped_by_user_id = $2 WHERE id = $1",
        )
        .bind(id)
        .bind(stopped_by_user_id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Resume a paused conversation: clear `stopped_at` /
    /// `stopped_by_user_id` back to NULL. Idempotent. Triggered when an
    /// admin removes their 🛑 reaction.
    pub async fn resume_conversation(&self, id: Uuid) -> Result<u64, DbError> {
        let result = sqlx::query(
            "UPDATE conversations SET stopped_at = NULL, stopped_by_user_id = NULL WHERE id = $1",
        )
        .bind(id)
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    /// Look up which conversation a Discord message belongs to, if any.
    /// Used to decide whether an @mention continues an existing
    /// conversation (via Discord reply) or starts a new one.
    pub async fn lookup_conversation_by_message(
        &self,
        discord_message_id: i64,
    ) -> Result<Option<Uuid>, DbError> {
        let row: Option<(Uuid,)> = sqlx::query_as(
            "SELECT conversation_id FROM message_links WHERE discord_message_id = $1",
        )
        .bind(discord_message_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|(id,)| id))
    }

    /// Look up which turn a Discord message belongs to, if any. Every
    /// message_link (the user's @mention and the bot's replies/status
    /// posts) points at its turn, so this resolves a 🔄 reaction on
    /// *either* side of a turn back to the turn id.
    pub async fn lookup_turn_by_message(
        &self,
        discord_message_id: i64,
    ) -> Result<Option<Uuid>, DbError> {
        let row: Option<(Uuid,)> =
            sqlx::query_as("SELECT turn_id FROM message_links WHERE discord_message_id = $1")
                .bind(discord_message_id)
                .fetch_optional(&self.pool)
                .await?;
        Ok(row.map(|(id,)| id))
    }

    /// Fetch a single turn by id.
    pub async fn get_turn(&self, turn_id: Uuid) -> Result<Option<Turn>, DbError> {
        let turn = sqlx::query_as::<_, Turn>(
            "SELECT id, conversation_id, turn_index, created_at, completed_at, \
               user_discord_message_id, user_content, assistant_discord_message_id, \
               assistant_content, status, error, persona_name, version_id, \
               discord_user_id, discord_user_name, provider_state \
             FROM turns WHERE id = $1",
        )
        .bind(turn_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(turn)
    }

    /// Start a new turn in `conversation_id`. Assigns the next
    /// `turn_index` atomically and pins the Discord user's identity to
    /// the row so historical attribution survives username changes.
    pub async fn start_turn(
        &self,
        conversation_id: Uuid,
        user_discord_message_id: i64,
        user_content: &str,
        discord_user_id: i64,
        discord_user_name: &str,
        version_id: i32,
    ) -> Result<Turn, DbError> {
        let id = Uuid::new_v4();
        let turn = sqlx::query_as::<_, Turn>(
            "INSERT INTO turns \
               (id, conversation_id, turn_index, user_discord_message_id, user_content, \
                discord_user_id, discord_user_name, version_id) \
             VALUES ($1, $2, \
               COALESCE((SELECT MAX(turn_index) + 1 FROM turns WHERE conversation_id = $2), 0), \
               $3, $4, $5, $6, $7) \
             RETURNING id, conversation_id, turn_index, created_at, completed_at, \
               user_discord_message_id, user_content, assistant_discord_message_id, \
               assistant_content, status, error, persona_name, version_id, \
               discord_user_id, discord_user_name, provider_state",
        )
        .bind(id)
        .bind(conversation_id)
        .bind(user_discord_message_id)
        .bind(user_content)
        .bind(discord_user_id)
        .bind(discord_user_name)
        .bind(version_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(turn)
    }

    /// Mark a turn as completed and persist the assistant's reply.
    pub async fn complete_turn(
        &self,
        turn_id: Uuid,
        assistant_content: &str,
        assistant_discord_message_id: i64,
        provider_state: Option<&serde_json::Value>,
    ) -> Result<(), DbError> {
        sqlx::query(
            "UPDATE turns \
             SET status = 'completed', \
                 completed_at = now(), \
                 assistant_content = $2, \
                 assistant_discord_message_id = $3, \
                 provider_state = $4 \
             WHERE id = $1",
        )
        .bind(turn_id)
        .bind(assistant_content)
        .bind(assistant_discord_message_id)
        .bind(provider_state)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Stamp the persona that answered a turn. Written before
    /// completion so the model used for the run is recoverable even
    /// when the turn later fails.
    pub async fn set_turn_persona(&self, turn_id: Uuid, persona_name: &str) -> Result<(), DbError> {
        sqlx::query("UPDATE turns SET persona_name = $2 WHERE id = $1")
            .bind(turn_id)
            .bind(persona_name)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Mark a turn as failed and persist the error.
    pub async fn fail_turn(&self, turn_id: Uuid, error: &str) -> Result<(), DbError> {
        sqlx::query(
            "UPDATE turns \
             SET status = 'failed', completed_at = now(), error = $2 \
             WHERE id = $1",
        )
        .bind(turn_id)
        .bind(error)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Mark a turn as `cancelled` — an admin hit 🛑 mid-flight and the
    /// agent loop was aborted before a reply was posted. Distinct from
    /// `failed` on purpose: a cancelled turn is intentional, so it gets
    /// no 🔄 retry affordance (`reset_turn_for_retry` only acts on
    /// `failed`) and the viewer can style it differently. The reason is
    /// stored in `error` for the trace.
    pub async fn cancel_turn(&self, turn_id: Uuid, reason: &str) -> Result<(), DbError> {
        sqlx::query(
            "UPDATE turns \
             SET status = 'cancelled', completed_at = now(), error = $2 \
             WHERE id = $1",
        )
        .bind(turn_id)
        .bind(reason)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Mark a turn as failed but also persist whatever reply we managed
    /// to post. Used when a turn produced a user-facing message (e.g. a
    /// "⚠️ image generation failed" notice, possibly with partial model
    /// text) yet still counts as a failure — the viewer then shows the
    /// error AND any salvaged content, and the Discord message is linked.
    pub async fn fail_turn_with_reply(
        &self,
        turn_id: Uuid,
        error: &str,
        assistant_content: Option<&str>,
        assistant_discord_message_id: Option<i64>,
    ) -> Result<(), DbError> {
        sqlx::query(
            "UPDATE turns \
             SET status = 'failed', completed_at = now(), error = $2, \
                 assistant_content = $3, assistant_discord_message_id = $4 \
             WHERE id = $1",
        )
        .bind(turn_id)
        .bind(error)
        .bind(assistant_content)
        .bind(assistant_discord_message_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Reset a failed turn back to `pending` so it can be re-run, but
    /// ONLY if it is both currently `failed` AND the latest turn in its
    /// conversation. Returns `true` when a row was reset. The combined
    /// guard is atomic, so a double 🔄 (or a stale reaction on an older
    /// turn) is a no-op — retrying a mid-conversation turn would
    /// invalidate the turns built on top of it.
    pub async fn reset_turn_for_retry(
        &self,
        turn_id: Uuid,
        conversation_id: Uuid,
    ) -> Result<bool, DbError> {
        let res = sqlx::query(
            "UPDATE turns \
             SET status = 'pending', error = NULL, assistant_content = NULL, \
                 assistant_discord_message_id = NULL, completed_at = NULL, \
                 provider_state = NULL \
             WHERE id = $1 AND status = 'failed' \
               AND turn_index = (SELECT MAX(turn_index) FROM turns \
                                  WHERE conversation_id = $2)",
        )
        .bind(turn_id)
        .bind(conversation_id)
        .execute(&self.pool)
        .await?;
        Ok(res.rows_affected() == 1)
    }

    /// Delete all tool-call rows for a turn. Called before a retry re-runs
    /// the turn so the fresh calls don't collide on the `(turn_id,
    /// ordinal)` unique constraint with the failed attempt's rows.
    pub async fn delete_turn_tool_calls(&self, turn_id: Uuid) -> Result<(), DbError> {
        sqlx::query("DELETE FROM tool_calls WHERE turn_id = $1")
            .bind(turn_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Discord message ids the bot posted for a turn (its replies and
    /// status messages — every `message_links` role except `user`). Used
    /// on retry to clean up the prior (failed) reply before posting fresh.
    pub async fn assistant_message_ids_for_turn(&self, turn_id: Uuid) -> Result<Vec<i64>, DbError> {
        let rows: Vec<(i64,)> = sqlx::query_as(
            "SELECT discord_message_id FROM message_links \
             WHERE turn_id = $1 AND role LIKE 'assistant%'",
        )
        .bind(turn_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(|(id,)| id).collect())
    }

    /// Load a single turn's persisted context items in position order.
    /// On retry these are the turn's NOVEL inputs (the user's @mention,
    /// any quoted message, and image-attachment rows) — prior-turn text
    /// and replay images are reconstructed separately.
    pub async fn load_turn_context(&self, turn_id: Uuid) -> Result<Vec<ContextItem>, DbError> {
        let items = sqlx::query_as::<_, ContextItem>(
            "SELECT position, source, role, content, discord_message_id \
             FROM context_items WHERE turn_id = $1 ORDER BY position ASC",
        )
        .bind(turn_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(items)
    }

    /// Append a context item used in a turn's prompt.
    pub async fn record_context_item(
        &self,
        turn_id: Uuid,
        item: &ContextItem,
    ) -> Result<(), DbError> {
        sqlx::query(
            "INSERT INTO context_items (turn_id, position, source, role, content, discord_message_id) \
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(turn_id)
        .bind(item.position)
        .bind(&item.source)
        .bind(&item.role)
        .bind(&item.content)
        .bind(item.discord_message_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Snapshot the fully-composed system prompt sent to the model for a
    /// turn. Stored once per turn for the web viewer; `ON CONFLICT` keeps
    /// it idempotent if a turn is ever re-stamped. Lives in its own table
    /// so the history hot path never loads this large text.
    pub async fn record_turn_system_prompt(
        &self,
        turn_id: Uuid,
        content: &str,
    ) -> Result<(), DbError> {
        sqlx::query(
            "INSERT INTO turn_system_prompts (turn_id, content) VALUES ($1, $2) \
             ON CONFLICT (turn_id) DO UPDATE SET content = EXCLUDED.content",
        )
        .bind(turn_id)
        .bind(content)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Persist a tool call performed during a turn.
    pub async fn record_tool_call(
        &self,
        turn_id: Uuid,
        ordinal: i32,
        record: &ToolCallRecord,
    ) -> Result<(), DbError> {
        sqlx::query(
            "INSERT INTO tool_calls (turn_id, ordinal, tool_name, request, response) \
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(turn_id)
        .bind(ordinal)
        .bind(&record.tool_name)
        .bind(&record.request)
        .bind(&record.response)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Record that a Discord message belongs to a given turn / conversation.
    /// Used to resolve "is this @mention a continuation?" on the next
    /// message that arrives.
    pub async fn record_message_link(
        &self,
        discord_message_id: i64,
        discord_guild_id: i64,
        conversation_id: Uuid,
        turn_id: Uuid,
        role: &str,
    ) -> Result<(), DbError> {
        sqlx::query(
            "INSERT INTO message_links \
               (discord_message_id, discord_guild_id, conversation_id, turn_id, role) \
             VALUES ($1, $2, $3, $4, $5) \
             ON CONFLICT (discord_message_id) DO NOTHING",
        )
        .bind(discord_message_id)
        .bind(discord_guild_id)
        .bind(conversation_id)
        .bind(turn_id)
        .bind(role)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Set a conversation's title and stamp `title_generated_at = now()`.
    /// Called by the background titler after the first turn completes.
    pub async fn set_conversation_title(
        &self,
        conversation_id: Uuid,
        title: &str,
    ) -> Result<(), DbError> {
        sqlx::query(
            "UPDATE conversations \
             SET title = $2, title_generated_at = now() \
             WHERE id = $1",
        )
        .bind(conversation_id)
        .bind(title)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Upsert a Discord user's identity. Username / display name /
    /// avatar hash are overwritten with the latest values; the local
    /// avatar path and fetched-at timestamp are *preserved* (only the
    /// avatar fetcher writes those, via [`Self::mark_avatar_fetched`]).
    ///
    /// Returns the post-upsert row so the caller can compare against
    /// what it sent and decide whether to enqueue an avatar refetch.
    pub async fn upsert_discord_user(
        &self,
        id: i64,
        username: &str,
        display_name: Option<&str>,
        avatar_hash: Option<&str>,
    ) -> Result<DiscordUser, DbError> {
        let user = sqlx::query_as::<_, DiscordUser>(
            "INSERT INTO discord_users (id, username, display_name, avatar_hash) \
             VALUES ($1, $2, $3, $4) \
             ON CONFLICT (id) DO UPDATE \
               SET username = EXCLUDED.username, \
                   display_name = EXCLUDED.display_name, \
                   avatar_hash = EXCLUDED.avatar_hash, \
                   last_seen_at = now() \
             RETURNING id, username, display_name, avatar_hash, \
               avatar_local_path, last_avatar_fetched_at, last_seen_at",
        )
        .bind(id)
        .bind(username)
        .bind(display_name)
        .bind(avatar_hash)
        .fetch_one(&self.pool)
        .await?;
        Ok(user)
    }

    /// Fetch a Discord user by id.
    pub async fn get_discord_user(&self, id: i64) -> Result<Option<DiscordUser>, DbError> {
        let user = sqlx::query_as::<_, DiscordUser>(
            "SELECT id, username, display_name, avatar_hash, \
               avatar_local_path, last_avatar_fetched_at, last_seen_at \
             FROM discord_users WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(user)
    }

    /// Record that the avatar for `user_id` was just fetched and saved
    /// to `local_path` (relative to `storage.avatars_dir`).
    pub async fn mark_avatar_fetched(&self, user_id: i64, local_path: &str) -> Result<(), DbError> {
        sqlx::query(
            "UPDATE discord_users \
             SET avatar_local_path = $2, last_avatar_fetched_at = now() \
             WHERE id = $1",
        )
        .bind(user_id)
        .bind(local_path)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Load all completed turns of a conversation in index order.
    /// Used by the bot to build chat history when continuing a thread.
    pub async fn load_conversation_history(
        &self,
        conversation_id: Uuid,
    ) -> Result<Vec<Turn>, DbError> {
        let turns = sqlx::query_as::<_, Turn>(
            "SELECT id, conversation_id, turn_index, created_at, completed_at, \
               user_discord_message_id, user_content, assistant_discord_message_id, \
               assistant_content, status, error, persona_name, version_id, \
               discord_user_id, discord_user_name, provider_state \
             FROM turns \
             WHERE conversation_id = $1 AND status = 'completed' \
             ORDER BY turn_index ASC",
        )
        .bind(conversation_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(turns)
    }

    /// Load every replayable image in a conversation, across all
    /// completed turns, in chronological (turn-index) order. Two
    /// sources are merged:
    ///   - user-uploaded attachments, stored as image `context_items`
    ///     (`source LIKE 'discord:msg:%:image:%'`, `content` = URI);
    ///   - model-generated images, recovered from `generate_image`
    ///     tool-call outputs (`response->>'image_uri'`).
    ///
    /// The bot re-attaches these to the model context on later turns so
    /// the model can still see earlier images (history otherwise carries
    /// only text). Image rows for prior turns are never persisted, so
    /// this query can't feed on its own output — it only reads the
    /// genuinely-novel uploads/generations each turn recorded.
    pub async fn load_conversation_image_uris(
        &self,
        conversation_id: Uuid,
    ) -> Result<Vec<ReplayImage>, DbError> {
        let images = sqlx::query_as::<_, ReplayImage>(
            "SELECT t.id AS turn_id, t.turn_index AS turn_index, ci.content AS uri \
               FROM context_items ci \
               JOIN turns t ON t.id = ci.turn_id \
              WHERE t.conversation_id = $1 AND t.status = 'completed' \
                    AND ci.source LIKE 'discord:msg:%:image:%' \
             UNION ALL \
             SELECT t.id AS turn_id, t.turn_index AS turn_index, \
                    tc.response->>'image_uri' AS uri \
               FROM tool_calls tc \
               JOIN turns t ON t.id = tc.turn_id \
              WHERE t.conversation_id = $1 AND t.status = 'completed' \
                    AND tc.tool_name = 'generate_image' \
                    AND tc.response->>'image_uri' IS NOT NULL \
             ORDER BY turn_index ASC",
        )
        .bind(conversation_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(images)
    }

    /// Set a user's privacy preference for a specific guild. `true` =
    /// opt in (Grok may see their messages as quoted-message context);
    /// `false` = opt out (default; messages excluded from context).
    /// Preferences are per-guild — a user can opt in on one server
    /// without affecting their state on another.
    pub async fn set_user_privacy(
        &self,
        discord_guild_id: i64,
        discord_user_id: i64,
        opted_in: bool,
    ) -> Result<(), DbError> {
        sqlx::query(
            "INSERT INTO user_privacy \
               (discord_guild_id, discord_user_id, opted_in, updated_at) \
             VALUES ($1, $2, $3, now()) \
             ON CONFLICT (discord_guild_id, discord_user_id) DO UPDATE \
               SET opted_in = EXCLUDED.opted_in, updated_at = now()",
        )
        .bind(discord_guild_id)
        .bind(discord_user_id)
        .bind(opted_in)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Look up a user's privacy preference for a guild. Returns
    /// `Some(bool)` if the user has ever toggled it in that guild, or
    /// `None` if no row exists (treated as opted-out by the bot).
    pub async fn get_user_privacy(
        &self,
        discord_guild_id: i64,
        discord_user_id: i64,
    ) -> Result<Option<bool>, DbError> {
        let row: Option<(bool,)> = sqlx::query_as(
            "SELECT opted_in FROM user_privacy \
             WHERE discord_guild_id = $1 AND discord_user_id = $2",
        )
        .bind(discord_guild_id)
        .bind(discord_user_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|(b,)| b))
    }

    /// Convenience wrapper around [`Self::get_user_privacy`] that
    /// returns the bot-effective answer: missing → false.
    pub async fn user_opted_in(
        &self,
        discord_guild_id: i64,
        discord_user_id: i64,
    ) -> Result<bool, DbError> {
        Ok(self
            .get_user_privacy(discord_guild_id, discord_user_id)
            .await?
            .unwrap_or(false))
    }

    /// Get the privacy mode configured for a guild. Returns `None` if
    /// no row exists — callers should fall back to the config-supplied
    /// default in that case.
    pub async fn get_guild_privacy_mode(
        &self,
        discord_guild_id: i64,
    ) -> Result<Option<PrivacyMode>, DbError> {
        let row: Option<(serde_json::Value,)> =
            sqlx::query_as("SELECT privacy_mode FROM guild_settings WHERE discord_guild_id = $1")
                .bind(discord_guild_id)
                .fetch_optional(&self.pool)
                .await?;
        match row {
            Some((value,)) => Ok(Some(serde_json::from_value(value)?)),
            None => Ok(None),
        }
    }

    /// Convenience: get the effective privacy mode, falling back to
    /// `fallback` when no DB row exists.
    pub async fn guild_privacy_mode_or(
        &self,
        discord_guild_id: i64,
        fallback: &PrivacyMode,
    ) -> Result<PrivacyMode, DbError> {
        Ok(self
            .get_guild_privacy_mode(discord_guild_id)
            .await?
            .unwrap_or_else(|| fallback.clone()))
    }

    /// Replace the guild's privacy mode (upsert).
    pub async fn set_guild_privacy_mode(
        &self,
        discord_guild_id: i64,
        mode: &PrivacyMode,
    ) -> Result<(), DbError> {
        let json = serde_json::to_value(mode)?;
        sqlx::query(
            "INSERT INTO guild_settings (discord_guild_id, privacy_mode, updated_at) \
             VALUES ($1, $2, now()) \
             ON CONFLICT (discord_guild_id) DO UPDATE \
               SET privacy_mode = EXCLUDED.privacy_mode, updated_at = now()",
        )
        .bind(discord_guild_id)
        .bind(json)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Record a freshly-submitted video generation job. Called from the
    /// `start_video_generation` tool after xAI returns a `request_id`.
    pub async fn create_video_job(
        &self,
        turn_id: Uuid,
        request_id: &str,
        prompt: &str,
    ) -> Result<VideoJob, DbError> {
        let id = Uuid::new_v4();
        let job = sqlx::query_as::<_, VideoJob>(
            "INSERT INTO video_jobs (id, turn_id, request_id, prompt) \
             VALUES ($1, $2, $3, $4) \
             RETURNING id, turn_id, request_id, prompt, status, video_uri, \
               submitted_at, completed_at, error",
        )
        .bind(id)
        .bind(turn_id)
        .bind(request_id)
        .bind(prompt)
        .fetch_one(&self.pool)
        .await?;
        Ok(job)
    }

    /// Look up a job by its xAI `request_id`. The `check_video_status`
    /// tool uses this to associate a polling response with its row.
    pub async fn get_video_job(&self, request_id: &str) -> Result<Option<VideoJob>, DbError> {
        let row = sqlx::query_as::<_, VideoJob>(
            "SELECT id, turn_id, request_id, prompt, status, video_uri, \
               submitted_at, completed_at, error \
             FROM video_jobs WHERE request_id = $1",
        )
        .bind(request_id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row)
    }

    /// Update a job's status. Used both for terminal transitions (done /
    /// failed / expired with completion timestamp) and for noop status
    /// snapshots from `check_video_status` polls.
    pub async fn update_video_job_status(
        &self,
        request_id: &str,
        status: &str,
        video_uri: Option<&str>,
        error: Option<&str>,
    ) -> Result<(), DbError> {
        let terminal = matches!(status, "done" | "failed" | "expired");
        sqlx::query(
            "UPDATE video_jobs \
             SET status = $2, \
                 video_uri = COALESCE($3, video_uri), \
                 error = COALESCE($4, error), \
                 completed_at = CASE WHEN $5 THEN now() ELSE completed_at END \
             WHERE request_id = $1",
        )
        .bind(request_id)
        .bind(status)
        .bind(video_uri)
        .bind(error)
        .bind(terminal)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Resolve which persona name applies for a given Discord call
    /// site. Tries the most specific scope first (per-conversation), then
    /// falls back through user-in-guild → channel → guild. Returns
    /// `None` when nothing matches; callers should then fall back to
    /// the config's `default_persona`.
    ///
    /// Each branch is a single PK probe against `persona_selections`,
    /// so the worst case is four cheap lookups per turn. We don't try
    /// to coalesce these into one query: the table is tiny, the keys
    /// have different shapes, and the early-return semantics keep the
    /// code obvious.
    pub async fn resolve_persona(
        &self,
        conversation_id: Option<Uuid>,
        guild_id: Option<i64>,
        channel_id: i64,
        user_id: i64,
    ) -> Result<Option<String>, DbError> {
        if let Some(cid) = conversation_id
            && let Some(name) = self
                .get_persona_selection("conversation", &cid.to_string())
                .await?
        {
            return Ok(Some(name));
        }
        if let Some(gid) = guild_id {
            let user_key = format!("{gid}:{user_id}");
            if let Some(name) = self.get_persona_selection("user", &user_key).await? {
                return Ok(Some(name));
            }
        }
        if let Some(name) = self
            .get_persona_selection("channel", &channel_id.to_string())
            .await?
        {
            return Ok(Some(name));
        }
        if let Some(gid) = guild_id
            && let Some(name) = self
                .get_persona_selection("guild", &gid.to_string())
                .await?
        {
            return Ok(Some(name));
        }
        Ok(None)
    }

    /// Read a single `persona_selections` row by composite key.
    pub async fn get_persona_selection(
        &self,
        scope: &str,
        key: &str,
    ) -> Result<Option<String>, DbError> {
        let row: Option<(String,)> = sqlx::query_as(
            "SELECT persona_name FROM persona_selections WHERE scope = $1 AND key = $2",
        )
        .bind(scope)
        .bind(key)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|(n,)| n))
    }

    /// Set (upsert) the persona for a given scope key. The set of
    /// valid `scope` values is `conversation | user | channel | guild`;
    /// `key` shape depends on scope (see migrations/0005_personas.sql).
    pub async fn set_persona_selection(
        &self,
        scope: &str,
        key: &str,
        persona_name: &str,
    ) -> Result<(), DbError> {
        sqlx::query(
            "INSERT INTO persona_selections (scope, key, persona_name, updated_at) \
             VALUES ($1, $2, $3, now()) \
             ON CONFLICT (scope, key) DO UPDATE \
               SET persona_name = EXCLUDED.persona_name, updated_at = now()",
        )
        .bind(scope)
        .bind(key)
        .bind(persona_name)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Remove a persona override. Returns `true` if a row was deleted.
    pub async fn clear_persona_selection(&self, scope: &str, key: &str) -> Result<bool, DbError> {
        let result = sqlx::query("DELETE FROM persona_selections WHERE scope = $1 AND key = $2")
            .bind(scope)
            .bind(key)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected() > 0)
    }

    /// Aggregate read model for the web viewer.
    pub async fn fetch_conversation_view(
        &self,
        id: Uuid,
    ) -> Result<Option<ConversationView>, DbError> {
        let Some(conversation) = self.get_conversation(id).await? else {
            return Ok(None);
        };

        let turns = sqlx::query_as::<_, Turn>(
            "SELECT id, conversation_id, turn_index, created_at, completed_at, \
               user_discord_message_id, user_content, assistant_discord_message_id, \
               assistant_content, status, error, persona_name, version_id, \
               discord_user_id, discord_user_name, provider_state \
             FROM turns WHERE conversation_id = $1 ORDER BY turn_index ASC",
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await?;

        let mut turn_views = Vec::with_capacity(turns.len());
        let mut user_ids: Vec<i64> = Vec::new();
        let mut version_ids: Vec<i32> = Vec::new();
        for turn in turns {
            if let Some(uid) = turn.discord_user_id
                && !user_ids.contains(&uid)
            {
                user_ids.push(uid);
            }
            if let Some(vid) = turn.version_id
                && !version_ids.contains(&vid)
            {
                version_ids.push(vid);
            }
            let context = sqlx::query_as::<_, ContextItem>(
                "SELECT position, source, role, content, discord_message_id \
                 FROM context_items WHERE turn_id = $1 ORDER BY position ASC",
            )
            .bind(turn.id)
            .fetch_all(&self.pool)
            .await?;

            let tool_call_rows: Vec<(String, serde_json::Value, serde_json::Value)> =
                sqlx::query_as(
                    "SELECT tool_name, request, response FROM tool_calls \
                     WHERE turn_id = $1 ORDER BY ordinal ASC",
                )
                .bind(turn.id)
                .fetch_all(&self.pool)
                .await?;
            let tool_calls = tool_call_rows
                .into_iter()
                .map(|(tool_name, request, response)| ToolCallRecord {
                    tool_name,
                    request,
                    response,
                })
                .collect();

            // Viewer-only: the composed system prompt snapshot, if one was
            // recorded for this turn (absent on legacy turns).
            let system_prompt: Option<String> =
                sqlx::query_scalar("SELECT content FROM turn_system_prompts WHERE turn_id = $1")
                    .bind(turn.id)
                    .fetch_optional(&self.pool)
                    .await?;

            turn_views.push(TurnView {
                turn,
                system_prompt,
                context,
                tool_calls,
            });
        }

        // Pull every referenced user in one query so the frontend can
        // resolve avatars + names without an N+1.
        let mut users: std::collections::HashMap<i64, DiscordUser> =
            std::collections::HashMap::new();
        if !user_ids.is_empty() {
            let rows = sqlx::query_as::<_, DiscordUser>(
                "SELECT id, username, display_name, avatar_hash, \
                   avatar_local_path, last_avatar_fetched_at, last_seen_at \
                 FROM discord_users WHERE id = ANY($1)",
            )
            .bind(&user_ids)
            .fetch_all(&self.pool)
            .await?;
            for u in rows {
                users.insert(u.id, u);
            }
        }

        // Resolve every build referenced by a turn in one query, so the
        // viewer can render "vN" with the commit string on hover without
        // an N+1. Mirrors the `users` batch above.
        let mut versions: std::collections::HashMap<i32, AppVersion> =
            std::collections::HashMap::new();
        if !version_ids.is_empty() {
            let rows = sqlx::query_as::<_, AppVersion>(
                "SELECT id, git_version, first_seen FROM app_versions WHERE id = ANY($1)",
            )
            .bind(&version_ids)
            .fetch_all(&self.pool)
            .await?;
            for v in rows {
                versions.insert(v.id, v);
            }
        }

        Ok(Some(ConversationView {
            conversation,
            turns: turn_views,
            users,
            versions,
        }))
    }
}
