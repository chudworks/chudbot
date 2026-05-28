//! Postgres data layer. Thin wrapper around [`sqlx::PgPool`] with helpers
//! for the conversation lifecycle: creating conversations, recording
//! turns + their context + tool calls, looking up conversations by
//! Discord message id, and reading the aggregated view for the web viewer.

use sqlx::PgPool;
use sqlx::migrate::Migrator;
use thiserror::Error;
use uuid::Uuid;

use crate::config::PrivacyMode;
use crate::domain::{ContextItem, Conversation, ConversationView, Turn, TurnView, VideoJob};
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
               created_by_user_id, root_discord_message_id, title, model",
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
              created_by_user_id, root_discord_message_id, title, model \
             FROM conversations WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(conv)
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

    /// Start a new turn in `conversation_id`. Assigns the next
    /// `turn_index` atomically.
    pub async fn start_turn(
        &self,
        conversation_id: Uuid,
        user_discord_message_id: i64,
        user_content: &str,
    ) -> Result<Turn, DbError> {
        let id = Uuid::new_v4();
        let turn = sqlx::query_as::<_, Turn>(
            "INSERT INTO turns \
               (id, conversation_id, turn_index, user_discord_message_id, user_content) \
             VALUES ($1, $2, \
               COALESCE((SELECT MAX(turn_index) + 1 FROM turns WHERE conversation_id = $2), 0), \
               $3, $4) \
             RETURNING id, conversation_id, turn_index, created_at, completed_at, \
               user_discord_message_id, user_content, assistant_discord_message_id, \
               assistant_content, status, error, persona_name",
        )
        .bind(id)
        .bind(conversation_id)
        .bind(user_discord_message_id)
        .bind(user_content)
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
    ) -> Result<(), DbError> {
        sqlx::query(
            "UPDATE turns \
             SET status = 'completed', \
                 completed_at = now(), \
                 assistant_content = $2, \
                 assistant_discord_message_id = $3 \
             WHERE id = $1",
        )
        .bind(turn_id)
        .bind(assistant_content)
        .bind(assistant_discord_message_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Stamp the persona that answered a turn. Written before
    /// completion so the model used for the run is recoverable even
    /// when the turn later fails.
    pub async fn set_turn_persona(
        &self,
        turn_id: Uuid,
        persona_name: &str,
    ) -> Result<(), DbError> {
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

    /// Load all completed turns of a conversation in index order.
    /// Used by the bot to build chat history when continuing a thread.
    pub async fn load_conversation_history(
        &self,
        conversation_id: Uuid,
    ) -> Result<Vec<Turn>, DbError> {
        let turns = sqlx::query_as::<_, Turn>(
            "SELECT id, conversation_id, turn_index, created_at, completed_at, \
               user_discord_message_id, user_content, assistant_discord_message_id, \
               assistant_content, status, error, persona_name \
             FROM turns \
             WHERE conversation_id = $1 AND status = 'completed' \
             ORDER BY turn_index ASC",
        )
        .bind(conversation_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(turns)
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
        let row: Option<(serde_json::Value,)> = sqlx::query_as(
            "SELECT privacy_mode FROM guild_settings WHERE discord_guild_id = $1",
        )
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
    pub async fn get_video_job(
        &self,
        request_id: &str,
    ) -> Result<Option<VideoJob>, DbError> {
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
        if let Some(cid) = conversation_id {
            if let Some(name) = self
                .get_persona_selection("conversation", &cid.to_string())
                .await?
            {
                return Ok(Some(name));
            }
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
        if let Some(gid) = guild_id {
            if let Some(name) = self
                .get_persona_selection("guild", &gid.to_string())
                .await?
            {
                return Ok(Some(name));
            }
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
    pub async fn clear_persona_selection(
        &self,
        scope: &str,
        key: &str,
    ) -> Result<bool, DbError> {
        let result = sqlx::query(
            "DELETE FROM persona_selections WHERE scope = $1 AND key = $2",
        )
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
               assistant_content, status, error, persona_name \
             FROM turns WHERE conversation_id = $1 ORDER BY turn_index ASC",
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await?;

        let mut turn_views = Vec::with_capacity(turns.len());
        for turn in turns {
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

            turn_views.push(TurnView {
                turn,
                context,
                tool_calls,
            });
        }

        Ok(Some(ConversationView {
            conversation,
            turns: turn_views,
        }))
    }
}
