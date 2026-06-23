//! SQLx/Postgres storage for chudbot.
//!
//! This crate owns the database boundary for the bot runtime. It
//! intentionally uses runtime-checked SQLx queries so normal builds do not
//! require a live `DATABASE_URL`.

use std::collections::BTreeMap;

use chudbot_api::{
    AgentSelection, BeginTurn, BotStorage, ChannelLink, ChannelRef, ContextItem, Conversation,
    ConversationId, ConversationLookup, ConversationSnapshot, ConversationStop,
    CountActiveVideoGenerations, CreateVideoJob, ExternalId, FinishTurn, GuildProfile,
    MediaCategory, MediaUri, MemoryJobCompletion, MemoryJobKind, MemoryJobSchedule,
    MemoryTurnWindow, MessageLink, MessageRef, ModelId, ModelStepKind, ModelStepTrace,
    NewUserMemoryDiaryEntry, NewUserMemoryDocumentRevision, NewUserMemoryEvent, PlatformName,
    ProviderName, ResolveAgent, RetryTurn, SaveTurnInput, StoredUserProfile, StoredVideoJob,
    ToolTrace, Turn, TurnAsset, TurnId, TurnRole, TurnSnapshot, TurnStatus, UpdateVideoJob,
    UsageCostGrouping, UsageCostQuery, UsageCostRow, UsageCostScope, UsageRecord, UsageSubject,
    UserMemoryAudioTranscription, UserMemoryDiaryEntry, UserMemoryDocument, UserMemoryEvent,
    UserMemoryEventKind, UserMemoryImageContext, UserMemoryJob, UserMemoryKey, UserMemoryTurn,
    UserProfile, UserRef, canonical_stored_media_uri, is_stored_media_uri, parse_stored_media_uri,
};
use serde_json::Value;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Postgres, Row, Transaction};
use thiserror::Error;
use time::OffsetDateTime;
use uuid::Uuid;

static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("../../migrations");

/// Postgres-backed bot storage.
#[derive(Debug, Clone)]
pub struct SqlxStorage {
    pool: PgPool,
    app_version_id: Option<i32>,
}

/// Registered build version row.
#[derive(Debug, Clone)]
pub struct AppVersion {
    /// Human-facing ordered version number.
    pub id: i32,
    /// Full `git describe --tags --always --dirty` string.
    pub git_version: String,
    /// First time this build was seen by this database.
    pub first_seen_at: OffsetDateTime,
}

impl SqlxStorage {
    /// Connect to Postgres.
    #[tracing::instrument(name = "storage_sqlx.connect", skip_all)]
    pub async fn connect(database_url: &str) -> Result<Self, SqlxStorageError> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await?;
        tracing::info!("connected SQLx storage");
        Ok(Self {
            pool,
            app_version_id: None,
        })
    }

    /// Construct from an existing pool.
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            app_version_id: None,
        }
    }

    /// Stamp newly written conversations, turns, and attempts with an app
    /// version row resolved by [`Self::register_app_version`].
    pub fn with_app_version_id(mut self, app_version_id: i32) -> Self {
        self.app_version_id = Some(app_version_id);
        self
    }

    /// Borrow the underlying pool.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Run embedded workspace-root migrations.
    #[tracing::instrument(name = "storage_sqlx.migrate", skip_all)]
    pub async fn run_migrations(&self) -> Result<(), SqlxStorageError> {
        MIGRATOR.run(&self.pool).await?;
        tracing::info!("database migrations complete");
        Ok(())
    }

    /// Resolve or insert the ordered version row for the running build.
    ///
    /// This deliberately selects before inserting instead of using an upsert:
    /// `app_versions.id` is the user-facing `vN` number, and Postgres
    /// sequences advance even when `ON CONFLICT DO NOTHING` rejects a row.
    #[tracing::instrument(
        name = "storage_sqlx.register_app_version",
        skip_all,
        fields(git_version)
    )]
    pub async fn register_app_version(
        &self,
        git_version: &str,
    ) -> Result<AppVersion, SqlxStorageError> {
        if let Some(row) = sqlx::query(
            "SELECT id, git_version, first_seen_at \
               FROM app_versions \
              WHERE git_version = $1",
        )
        .bind(git_version)
        .fetch_optional(&self.pool)
        .await?
        {
            return Ok(app_version_from_row(row));
        }

        let row = sqlx::query(
            "INSERT INTO app_versions (git_version) \
             VALUES ($1) \
             RETURNING id, git_version, first_seen_at",
        )
        .bind(git_version)
        .fetch_one(&self.pool)
        .await?;
        Ok(app_version_from_row(row))
    }

    async fn load_snapshot(
        &self,
        conversation_id: ConversationId,
    ) -> Result<Option<ConversationSnapshot>, SqlxStorageError> {
        let Some(conversation) = self.load_conversation_row(conversation_id).await? else {
            return Ok(None);
        };
        let turns = self.load_turn_snapshots(conversation_id).await?;
        Ok(Some(ConversationSnapshot {
            conversation,
            turns,
        }))
    }

    async fn load_conversation_row(
        &self,
        conversation_id: ConversationId,
    ) -> Result<Option<Conversation>, SqlxStorageError> {
        let row = sqlx::query(
            "SELECT id, created_at, message_provider, channel, created_by_user_key, \
                    root_message_provider, root_message_channel, root_message, agent_name, \
                    llm_provider, llm_model, system_instructions, title, stopped_at, \
                    stopped_by_provider, stopped_by_user_key \
               FROM conversations WHERE id = $1",
        )
        .bind(conversation_id.0)
        .fetch_optional(&self.pool)
        .await?;
        row.map(conversation_from_row).transpose()
    }

    async fn load_turn_snapshots(
        &self,
        conversation_id: ConversationId,
    ) -> Result<Vec<TurnSnapshot>, SqlxStorageError> {
        let rows = sqlx::query(
            "SELECT t.id, t.ordinal, t.history_cutoff, t.response_ordinal, t.created_at, \
                    t.user_message_created_at, t.completed_at, t.user_message_provider, \
                    t.user_message_channel, t.user_message, t.user_key, t.user_display_name, \
                    t.user_content, t.assistant_message_provider, t.assistant_message_channel, \
                    t.assistant_message, t.assistant_content, t.status, t.error, \
                    t.app_version_id, ta.agent_name, ta.llm_provider, ta.llm_model \
               FROM turns t \
               LEFT JOIN LATERAL ( \
                    SELECT agent_name, llm_provider, llm_model \
                      FROM turn_attempts \
                     WHERE turn_id = t.id \
                     ORDER BY attempt_ordinal DESC \
                     LIMIT 1 \
               ) ta ON true \
              WHERE t.conversation_id = $1 \
              ORDER BY t.ordinal",
        )
        .bind(conversation_id.0)
        .fetch_all(&self.pool)
        .await?;

        let mut turns = Vec::with_capacity(rows.len());
        for row in rows {
            let turn_id = TurnId(row.get("id"));
            let attempt_id = self.latest_attempt_id(turn_id).await?;
            let system_instructions = match attempt_id {
                Some(id) => Some(self.attempt_system_instructions(id).await?),
                None => None,
            };
            let context = match attempt_id {
                Some(id) => self.load_context(id).await?,
                None => Vec::new(),
            };
            let tool_trace = match attempt_id {
                Some(id) => self.load_tool_trace(id).await?,
                None => Vec::new(),
            };
            let model_steps = match attempt_id {
                Some(id) => self.load_model_steps(id).await?,
                None => Vec::new(),
            };
            let replay_assets = self.load_turn_assets(turn_id).await?;
            let usage = self.load_usage_for_turn(turn_id).await?;
            turns.push(TurnSnapshot {
                turn: turn_from_row(&row)?,
                system_instructions,
                context,
                tool_trace,
                model_steps,
                replay_assets,
                usage,
            });
        }
        Ok(turns)
    }

    async fn latest_attempt_id(&self, turn_id: TurnId) -> Result<Option<Uuid>, SqlxStorageError> {
        let id = sqlx::query_scalar(
            "SELECT id FROM turn_attempts WHERE turn_id = $1 ORDER BY attempt_ordinal DESC LIMIT 1",
        )
        .bind(turn_id.0)
        .fetch_optional(&self.pool)
        .await?;
        Ok(id)
    }

    async fn latest_attempt_id_required(&self, turn_id: TurnId) -> Result<Uuid, SqlxStorageError> {
        self.latest_attempt_id(turn_id)
            .await?
            .ok_or(SqlxStorageError::MissingAttempt { turn_id })
    }

    async fn attempt_system_instructions(
        &self,
        attempt_id: Uuid,
    ) -> Result<String, SqlxStorageError> {
        Ok(
            sqlx::query_scalar("SELECT system_instructions FROM turn_attempts WHERE id = $1")
                .bind(attempt_id)
                .fetch_one(&self.pool)
                .await?,
        )
    }

    async fn load_context(&self, attempt_id: Uuid) -> Result<Vec<ContextItem>, SqlxStorageError> {
        let rows = sqlx::query(
            "SELECT ordinal, source, role, content, message_provider, channel, message \
               FROM turn_attempt_context_items \
              WHERE attempt_id = $1 \
              ORDER BY ordinal",
        )
        .bind(attempt_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                Ok(ContextItem {
                    position: row.get("ordinal"),
                    source: row.get("source"),
                    role: row.get("role"),
                    content: row.get("content"),
                    message: optional_message_ref(
                        row.get::<Option<String>, _>("message_provider"),
                        row.get::<Option<String>, _>("channel"),
                        row.get::<Option<String>, _>("message"),
                    )?,
                })
            })
            .collect()
    }

    async fn load_tool_trace(&self, attempt_id: Uuid) -> Result<Vec<ToolTrace>, SqlxStorageError> {
        let rows = sqlx::query(
            "SELECT trace FROM turn_attempt_tool_traces WHERE attempt_id = $1 ORDER BY ordinal",
        )
        .bind(attempt_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| serde_json::from_value(row.get("trace")).map_err(SqlxStorageError::Json))
            .collect()
    }

    async fn load_model_steps(
        &self,
        attempt_id: Uuid,
    ) -> Result<Vec<ModelStepTrace>, SqlxStorageError> {
        let rows = sqlx::query(
            "SELECT ordinal, step_kind, llm_provider, llm_model, continuation \
               FROM turn_attempt_model_steps \
              WHERE attempt_id = $1 \
              ORDER BY ordinal",
        )
        .bind(attempt_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(model_step_from_row).collect()
    }

    async fn load_turn_assets(&self, turn_id: TurnId) -> Result<Vec<TurnAsset>, SqlxStorageError> {
        let rows = sqlx::query(
            "SELECT a.media_uri, a.source, m.mime_type \
               FROM turn_assets a \
               LEFT JOIN media_assets m ON m.uri = a.media_uri \
              WHERE a.turn_id = $1 AND a.replayable \
              ORDER BY a.ordinal",
        )
        .bind(turn_id.0)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|row| TurnAsset {
                uri: MediaUri::new(row.get::<String, _>("media_uri")),
                turn_id,
                source: row.get("source"),
                mime_type: row.get("mime_type"),
            })
            .collect())
    }

    async fn load_usage_for_turn(
        &self,
        turn_id: TurnId,
    ) -> Result<Vec<UsageRecord>, SqlxStorageError> {
        let rows = sqlx::query("SELECT raw FROM usage_records WHERE turn_id = $1 ORDER BY id")
            .bind(turn_id.0)
            .fetch_all(&self.pool)
            .await?;
        rows.into_iter()
            .filter_map(|row| row.get::<Option<Value>, _>("raw"))
            .map(|value| serde_json::from_value(value).map_err(SqlxStorageError::Json))
            .collect()
    }
}

impl BotStorage for SqlxStorage {
    type Error = SqlxStorageError;

    async fn load_conversation(
        &self,
        lookup: ConversationLookup,
    ) -> Result<Option<ConversationSnapshot>, Self::Error> {
        let id = match lookup {
            ConversationLookup::Id { id } => Some(id),
            ConversationLookup::Message { message } => self
                .conversation_id_for_message(&message)
                .await?
                .map(ConversationId),
            ConversationLookup::Channel { channel } => self
                .conversation_id_for_channel(&channel)
                .await?
                .map(ConversationId),
        };
        let Some(id) = id else { return Ok(None) };
        self.load_snapshot(id).await
    }

    async fn open_conversation(
        &self,
        input: chudbot_api::OpenConversation,
    ) -> Result<ConversationSnapshot, Self::Error> {
        let mut tx = self.pool.begin().await?;
        let id = ConversationId::new();
        let channel = upsert_channel(&mut tx, &input.channel).await?;
        upsert_user(&mut tx, &input.created_by, None).await?;
        upsert_message(
            &mut tx,
            &input.root_message,
            Some(input.created_by.user_id.as_str()),
            None,
        )
        .await?;
        sqlx::query(
            "INSERT INTO conversations \
               (id, message_provider, channel, created_by_user_key, root_message_provider, \
                root_message_channel, root_message, agent_name, llm_provider, llm_model, \
                system_instructions, title, created_app_version_id) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
        )
        .bind(id.0)
        .bind(input.channel.platform.as_str())
        .bind(channel)
        .bind(input.created_by.user_id.as_str())
        .bind(input.root_message.platform.as_str())
        .bind(channel_key_from_message(&input.root_message))
        .bind(input.root_message.message_id.as_str())
        .bind(&input.agent_name)
        .bind(input.provider.as_str())
        .bind(input.initial_model.as_str())
        .bind(&input.system_instructions)
        .bind(&input.title)
        .bind(self.app_version_id)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        self.load_snapshot(id)
            .await?
            .ok_or(SqlxStorageError::MissingConversation {
                conversation_id: id,
            })
    }

    async fn begin_turn(&self, input: BeginTurn) -> Result<Turn, Self::Error> {
        let mut tx = self.pool.begin().await?;
        let conv =
            sqlx::query("SELECT id, next_turn_ordinal FROM conversations WHERE id = $1 FOR UPDATE")
                .bind(input.conversation_id.0)
                .fetch_one(&mut *tx)
                .await?;
        let ordinal: i64 = conv.get("next_turn_ordinal");
        let history_cutoff: Option<i64> = sqlx::query_scalar(
            "SELECT MAX(response_ordinal) \
               FROM turns \
              WHERE conversation_id = $1 \
                AND status = 'completed' \
                AND response_ordinal IS NOT NULL \
                AND completed_at <= $2",
        )
        .bind(input.conversation_id.0)
        .bind(input.user_message_created_at)
        .fetch_one(&mut *tx)
        .await?;
        upsert_user(&mut tx, &input.user, Some(&input.user_display_name)).await?;
        upsert_message(
            &mut tx,
            &input.user_message,
            Some(input.user.user_id.as_str()),
            Some(&input.user_content),
        )
        .await?;
        let id = TurnId::new();
        let row = sqlx::query(
            "INSERT INTO turns \
               (id, conversation_id, ordinal, history_cutoff, status, user_message_created_at, \
                user_message_provider, user_message_channel, user_message, user_key, \
                user_display_name, user_content, app_version_id) \
             VALUES ($1, $2, $3, $4, 'pending', $5, $6, $7, $8, $9, $10, $11, $12) \
             RETURNING id, ordinal, history_cutoff, response_ordinal, created_at, \
                       user_message_created_at, completed_at, user_message_provider, \
                       user_message_channel, user_message, user_key, user_display_name, \
                       user_content, assistant_message_provider, assistant_message_channel, \
                       assistant_message, assistant_content, status, error, app_version_id, \
                       NULL::text AS agent_name, NULL::text AS llm_provider, \
                       NULL::text AS llm_model",
        )
        .bind(id.0)
        .bind(input.conversation_id.0)
        .bind(ordinal)
        .bind(history_cutoff)
        .bind(input.user_message_created_at)
        .bind(input.user_message.platform.as_str())
        .bind(channel_key_from_message(&input.user_message))
        .bind(input.user_message.message_id.as_str())
        .bind(input.user.user_id.as_str())
        .bind(&input.user_display_name)
        .bind(&input.user_content)
        .bind(self.app_version_id)
        .fetch_one(&mut *tx)
        .await?;
        sqlx::query(
            "UPDATE conversations SET next_turn_ordinal = next_turn_ordinal + 1 WHERE id = $1",
        )
        .bind(input.conversation_id.0)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        turn_from_row(&row)
    }

    async fn save_turn_input(&self, input: SaveTurnInput) -> Result<(), Self::Error> {
        let mut tx = self.pool.begin().await?;
        let turn = sqlx::query(
            "SELECT t.id, t.conversation_id, c.id AS conversation_id_check \
               FROM turns t JOIN conversations c ON c.id = t.conversation_id \
              WHERE t.id = $1",
        )
        .bind(input.turn_id.0)
        .fetch_one(&mut *tx)
        .await?;
        let conversation_id: Uuid = turn.get("conversation_id");
        let attempt_ordinal: i32 = sqlx::query_scalar(
            "SELECT COALESCE(MAX(attempt_ordinal) + 1, 0) FROM turn_attempts WHERE turn_id = $1",
        )
        .bind(input.turn_id.0)
        .fetch_one(&mut *tx)
        .await?;
        let attempt_id = Uuid::new_v4();
        sqlx::query("UPDATE turns SET app_version_id = COALESCE($2, app_version_id) WHERE id = $1")
            .bind(input.turn_id.0)
            .bind(self.app_version_id)
            .execute(&mut *tx)
            .await?;
        sqlx::query(
            "INSERT INTO turn_attempts \
               (id, turn_id, attempt_ordinal, agent_name, llm_provider, llm_model, \
                system_instructions, app_version_id) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(attempt_id)
        .bind(input.turn_id.0)
        .bind(attempt_ordinal)
        .bind(&input.agent_name)
        .bind(input.provider.as_str())
        .bind(input.model.as_str())
        .bind(&input.system_instructions)
        .bind(self.app_version_id)
        .execute(&mut *tx)
        .await?;
        for item in input.context {
            insert_context_item(&mut tx, input.turn_id, attempt_id, item).await?;
        }
        if let Some(transcript) = input.transcript {
            insert_transcript(&mut tx, attempt_id, transcript).await?;
        }
        tx.commit().await?;
        tracing::debug!(
            conversation = %conversation_id,
            turn = %input.turn_id,
            attempt = %attempt_id,
            attempt_ordinal,
            "saved turn input"
        );
        Ok(())
    }

    async fn append_tool_trace(
        &self,
        turn_id: TurnId,
        ordinal: i32,
        trace: ToolTrace,
    ) -> Result<(), Self::Error> {
        let attempt_id = self.latest_attempt_id_required(turn_id).await?;
        let fields = tool_trace_fields(&trace)?;
        let media_asset = tool_trace_media_asset(&fields);
        let tool_trace_id: i64 = sqlx::query_scalar(
            "INSERT INTO turn_attempt_tool_traces \
               (attempt_id, ordinal, trace_kind, tool_name, provider, tool_use_id, \
                is_error, request, response, trace) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) \
             RETURNING id",
        )
        .bind(attempt_id)
        .bind(ordinal)
        .bind(fields.trace_kind)
        .bind(fields.tool_name)
        .bind(fields.provider)
        .bind(fields.tool_use_id)
        .bind(fields.is_error)
        .bind(fields.request)
        .bind(fields.response)
        .bind(serde_json::to_value(trace)?)
        .fetch_one(&self.pool)
        .await?;
        if let Some((uri, source)) = media_asset {
            let mut tx = self.pool.begin().await?;
            insert_turn_asset(
                &mut tx,
                TurnAssetInsert {
                    turn_id,
                    attempt_id: Some(attempt_id),
                    uri: &uri,
                    source: &source,
                    context_item_id: None,
                    tool_trace_id: Some(tool_trace_id),
                    ordinal,
                },
            )
            .await?;
            tx.commit().await?;
        }
        Ok(())
    }

    async fn append_model_step_trace(
        &self,
        turn_id: TurnId,
        trace: ModelStepTrace,
    ) -> Result<(), Self::Error> {
        let attempt_id = self.latest_attempt_id_required(turn_id).await?;
        sqlx::query(
            "INSERT INTO turn_attempt_model_steps \
               (attempt_id, ordinal, step_kind, llm_provider, llm_model, continuation) \
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(attempt_id)
        .bind(trace.ordinal)
        .bind(model_step_kind_str(trace.kind))
        .bind(trace.provider.as_str())
        .bind(trace.model.as_str())
        .bind(optional_json(&trace.continuation)?)
        .execute(&self.pool)
        .await?;
        tracing::trace!(
            turn = %turn_id,
            attempt = %attempt_id,
            ordinal = trace.ordinal,
            kind = model_step_kind_str(trace.kind),
            provider = %trace.provider,
            model = %trace.model,
            has_continuation = trace.continuation.is_some(),
            "persisted model step trace"
        );
        Ok(())
    }

    async fn link_message(&self, link: MessageLink) -> Result<(), Self::Error> {
        let mut tx = self.pool.begin().await?;
        upsert_channel_from_message(&mut tx, &link.message).await?;
        upsert_message(&mut tx, &link.message, None, None).await?;
        let attempt_id = self.latest_attempt_id(link.turn_id).await?;
        upsert_message_link(
            &mut tx,
            &link.message,
            link.conversation_id,
            link.turn_id,
            attempt_id,
            &link.role,
        )
        .await?;
        tx.commit().await?;
        tracing::debug!(
            platform = %link.message.platform,
            guild = ?link.message.guild_id.as_ref().map(ExternalId::as_str),
            channel = %link.message.channel_id,
            message = %link.message.message_id,
            conversation = %link.conversation_id,
            turn = %link.turn_id,
            attempt = ?attempt_id,
            role = %link.role,
            "linked platform message to conversation"
        );
        Ok(())
    }

    async fn link_channel(&self, link: ChannelLink) -> Result<(), Self::Error> {
        let mut tx = self.pool.begin().await?;
        let channel = upsert_channel(&mut tx, &link.channel).await?;
        sqlx::query(
            "INSERT INTO channel_links \
               (message_provider, channel, conversation_id, turn_id, role) \
             VALUES ($1, $2, $3, $4, $5) \
             ON CONFLICT (message_provider, channel, role) DO UPDATE \
               SET conversation_id = EXCLUDED.conversation_id, turn_id = EXCLUDED.turn_id",
        )
        .bind(link.channel.platform.as_str())
        .bind(channel)
        .bind(link.conversation_id.0)
        .bind(link.turn_id.0)
        .bind(&link.role)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    async fn load_message_link(
        &self,
        message: MessageRef,
    ) -> Result<Option<MessageLink>, Self::Error> {
        let row = sqlx::query(
            "SELECT conversation_id, turn_id, role \
               FROM message_links \
              WHERE message_provider = $1 AND channel = $2 AND message = $3",
        )
        .bind(message.platform.as_str())
        .bind(channel_key_from_message(&message))
        .bind(message.message_id.as_str())
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(|row| MessageLink {
            message,
            conversation_id: ConversationId(row.get("conversation_id")),
            turn_id: TurnId(row.get("turn_id")),
            role: row.get("role"),
        }))
    }

    async fn load_message_links_for_turn(
        &self,
        turn_id: TurnId,
    ) -> Result<Vec<MessageLink>, Self::Error> {
        let rows = sqlx::query(
            "SELECT message_provider, channel, message, conversation_id, role \
               FROM message_links \
              WHERE turn_id = $1 \
              ORDER BY linked_at, message",
        )
        .bind(turn_id.0)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| {
                Ok(MessageLink {
                    message: message_ref(
                        &row.get::<String, _>("message_provider"),
                        &row.get::<String, _>("channel"),
                        row.get::<String, _>("message"),
                    )?,
                    conversation_id: ConversationId(row.get("conversation_id")),
                    turn_id,
                    role: row.get("role"),
                })
            })
            .collect()
    }

    async fn finish_turn(&self, input: FinishTurn) -> Result<(), Self::Error> {
        match input {
            FinishTurn::Completed {
                turn_id,
                assistant_content,
                assistant_message,
                usage,
            } => {
                let mut tx = self.pool.begin().await?;
                let turn =
                    sqlx::query("SELECT conversation_id FROM turns WHERE id = $1 FOR UPDATE")
                        .bind(turn_id.0)
                        .fetch_one(&mut *tx)
                        .await?;
                let conversation_id: Uuid = turn.get("conversation_id");
                sqlx::query("SELECT id FROM conversations WHERE id = $1 FOR UPDATE")
                    .bind(conversation_id)
                    .fetch_one(&mut *tx)
                    .await?;
                let response_ordinal: i64 = sqlx::query_scalar(
                    "SELECT next_response_ordinal FROM conversations WHERE id = $1",
                )
                .bind(conversation_id)
                .fetch_one(&mut *tx)
                .await?;
                upsert_message(&mut tx, &assistant_message, None, Some(&assistant_content)).await?;
                sqlx::query(
                    "UPDATE turns \
                        SET status = 'completed', completed_at = now(), response_ordinal = $2, \
                            assistant_message_provider = $3, assistant_message_channel = $4, \
                            assistant_message = $5, assistant_content = $6, error = NULL \
                      WHERE id = $1",
                )
                .bind(turn_id.0)
                .bind(response_ordinal)
                .bind(assistant_message.platform.as_str())
                .bind(channel_key_from_message(&assistant_message))
                .bind(assistant_message.message_id.as_str())
                .bind(&assistant_content)
                .execute(&mut *tx)
                .await?;
                sqlx::query(
                    "UPDATE conversations SET next_response_ordinal = next_response_ordinal + 1 WHERE id = $1",
                )
                .bind(conversation_id)
                .execute(&mut *tx)
                .await?;
                update_latest_attempt(
                    &mut tx,
                    turn_id,
                    "completed",
                    Some(&assistant_message),
                    Some(&assistant_content),
                    None,
                )
                .await?;
                insert_usage(
                    &mut tx,
                    ConversationId(conversation_id),
                    Some(turn_id),
                    usage,
                )
                .await?;
                tx.commit().await?;
            }
            FinishTurn::Failed {
                turn_id,
                error,
                assistant_content,
                assistant_message,
                usage,
            } => {
                let mut tx = self.pool.begin().await?;
                let conversation_id = conversation_for_turn(&mut tx, turn_id).await?;
                if let Some(message) = &assistant_message {
                    upsert_message(&mut tx, message, None, assistant_content.as_deref()).await?;
                }
                sqlx::query(
                    "UPDATE turns \
                        SET status = 'failed', completed_at = now(), assistant_message_provider = $2, \
                            assistant_message_channel = $3, assistant_message = $4, \
                            assistant_content = $5, error = $6 \
                      WHERE id = $1",
                )
                .bind(turn_id.0)
                .bind(assistant_message.as_ref().map(|m| m.platform.as_str()))
                .bind(assistant_message.as_ref().map(channel_key_from_message))
                .bind(assistant_message.as_ref().map(|m| m.message_id.as_str()))
                .bind(&assistant_content)
                .bind(&error)
                .execute(&mut *tx)
                .await?;
                update_latest_attempt(
                    &mut tx,
                    turn_id,
                    "failed",
                    assistant_message.as_ref(),
                    assistant_content.as_deref(),
                    Some(&error),
                )
                .await?;
                insert_usage(&mut tx, conversation_id, Some(turn_id), usage).await?;
                tx.commit().await?;
            }
            FinishTurn::Cancelled {
                turn_id,
                reason,
                usage,
            } => {
                let mut tx = self.pool.begin().await?;
                let conversation_id = conversation_for_turn(&mut tx, turn_id).await?;
                sqlx::query(
                    "UPDATE turns SET status = 'cancelled', completed_at = now(), error = $2 WHERE id = $1",
                )
                .bind(turn_id.0)
                .bind(&reason)
                .execute(&mut *tx)
                .await?;
                update_latest_attempt(&mut tx, turn_id, "cancelled", None, None, Some(&reason))
                    .await?;
                insert_usage(&mut tx, conversation_id, Some(turn_id), usage).await?;
                tx.commit().await?;
            }
        }
        Ok(())
    }

    async fn prepare_retry(&self, turn_id: TurnId) -> Result<Option<RetryTurn>, Self::Error> {
        let row = sqlx::query(
            "UPDATE turns \
                SET status = 'pending', completed_at = NULL, assistant_message_provider = NULL, \
                    assistant_message_channel = NULL, assistant_message = NULL, \
                    assistant_content = NULL, error = NULL \
              FROM conversations \
              WHERE turns.id = $1 AND turns.status = 'failed' \
                AND conversations.id = turns.conversation_id \
                AND conversations.stopped_at IS NULL \
              RETURNING turns.conversation_id",
        )
        .bind(turn_id.0)
        .fetch_optional(&self.pool)
        .await?;
        let Some(row) = row else { return Ok(None) };
        let conversation_id = ConversationId(row.get("conversation_id"));
        let Some(conversation) = self.load_snapshot(conversation_id).await? else {
            return Ok(None);
        };
        Ok(Some(RetryTurn {
            conversation,
            turn_id,
        }))
    }

    async fn set_conversation_stop(&self, input: ConversationStop) -> Result<bool, Self::Error> {
        let result = match input {
            ConversationStop::Stop {
                conversation_id,
                stopped_by,
            } => {
                sqlx::query(
                    "UPDATE conversations \
                        SET stopped_at = COALESCE(stopped_at, now()), stopped_by_provider = $2, \
                            stopped_by_user_key = $3 \
                      WHERE id = $1",
                )
                .bind(conversation_id.0)
                .bind(stopped_by.platform.as_str())
                .bind(stopped_by.user_id.as_str())
                .execute(&self.pool)
                .await?
            }
            ConversationStop::Resume { conversation_id } => {
                sqlx::query(
                    "UPDATE conversations \
                        SET stopped_at = NULL, stopped_by_provider = NULL, stopped_by_user_key = NULL \
                      WHERE id = $1",
                )
                .bind(conversation_id.0)
                .execute(&self.pool)
                .await?
            }
        };
        Ok(result.rows_affected() > 0)
    }

    async fn resolve_agent(&self, input: ResolveAgent) -> Result<Option<String>, Self::Error> {
        if let Some(conversation_id) = input.conversation_id
            && let Some(agent) = sqlx::query_scalar(
                "SELECT agent_name FROM conversation_agent_selections WHERE conversation_id = $1",
            )
            .bind(conversation_id.0)
            .fetch_optional(&self.pool)
            .await?
        {
            return Ok(Some(agent));
        }
        let provider = input.message_provider.as_str();
        if let Some(guild) = &input.guild_key {
            let scope = guild_scope(guild);
            if let Some(agent) = sqlx::query_scalar(
                "SELECT agent_name FROM user_agent_selections \
                  WHERE message_provider = $1 AND channel = $2 AND user_key = $3",
            )
            .bind(provider)
            .bind(&scope)
            .bind(&input.user_key)
            .fetch_optional(&self.pool)
            .await?
            {
                return Ok(Some(agent));
            }
        }
        let channel = if let Some(guild) = &input.guild_key {
            format!("guild:{guild}:channel:{}", input.channel_key)
        } else {
            format!("channel:{}", input.channel_key)
        };
        if let Some(agent) = sqlx::query_scalar(
            "SELECT agent_name FROM channel_agent_selections \
              WHERE message_provider = $1 AND channel = $2",
        )
        .bind(provider)
        .bind(&channel)
        .fetch_optional(&self.pool)
        .await?
        {
            return Ok(Some(agent));
        }
        if input.guild_key.is_some() {
            let legacy_channel = format!("channel:{}", input.channel_key);
            if let Some(agent) = sqlx::query_scalar(
                "SELECT agent_name FROM channel_agent_selections \
                  WHERE message_provider = $1 AND channel = $2",
            )
            .bind(provider)
            .bind(&legacy_channel)
            .fetch_optional(&self.pool)
            .await?
            {
                return Ok(Some(agent));
            }
        }
        if let Some(guild) = &input.guild_key
            && let Some(agent) = sqlx::query_scalar(
                "SELECT agent_name FROM channel_agent_selections \
                  WHERE message_provider = $1 AND channel = $2",
            )
            .bind(provider)
            .bind(guild_scope(guild))
            .fetch_optional(&self.pool)
            .await?
        {
            return Ok(Some(agent));
        }
        sqlx::query_scalar(
            "SELECT agent_name FROM provider_agent_selections WHERE message_provider = $1",
        )
        .bind(provider)
        .fetch_optional(&self.pool)
        .await
        .map_err(SqlxStorageError::Sqlx)
    }

    async fn load_agent_selection(
        &self,
        selection: AgentSelection,
    ) -> Result<Option<String>, Self::Error> {
        match selection {
            AgentSelection::Conversation { conversation_id } => {
                sqlx::query_scalar(
                    "SELECT agent_name FROM conversation_agent_selections \
                      WHERE conversation_id = $1",
                )
                .bind(conversation_id.0)
                .fetch_optional(&self.pool)
                .await
            }
            AgentSelection::User {
                message_provider,
                guild_key,
                user_key,
            } => {
                sqlx::query_scalar(
                    "SELECT agent_name FROM user_agent_selections \
                      WHERE message_provider = $1 AND channel = $2 AND user_key = $3",
                )
                .bind(message_provider.as_str())
                .bind(guild_scope(&guild_key))
                .bind(&user_key)
                .fetch_optional(&self.pool)
                .await
            }
            AgentSelection::Channel {
                message_provider,
                guild_key,
                channel_key,
            } => {
                sqlx::query_scalar(
                    "SELECT agent_name FROM channel_agent_selections \
                      WHERE message_provider = $1 AND channel = $2",
                )
                .bind(message_provider.as_str())
                .bind(selection_channel_key(guild_key.as_deref(), &channel_key))
                .fetch_optional(&self.pool)
                .await
            }
            AgentSelection::Guild {
                message_provider,
                guild_key,
            } => {
                sqlx::query_scalar(
                    "SELECT agent_name FROM channel_agent_selections \
                      WHERE message_provider = $1 AND channel = $2",
                )
                .bind(message_provider.as_str())
                .bind(guild_scope(&guild_key))
                .fetch_optional(&self.pool)
                .await
            }
            AgentSelection::Platform { message_provider } => {
                sqlx::query_scalar(
                    "SELECT agent_name FROM provider_agent_selections \
                      WHERE message_provider = $1",
                )
                .bind(message_provider.as_str())
                .fetch_optional(&self.pool)
                .await
            }
        }
        .map_err(SqlxStorageError::Sqlx)
    }

    async fn set_agent_selection(
        &self,
        selection: AgentSelection,
        agent_name: String,
    ) -> Result<(), Self::Error> {
        match selection {
            AgentSelection::Conversation { conversation_id } => {
                sqlx::query(
                    "INSERT INTO conversation_agent_selections (conversation_id, agent_name) \
                     VALUES ($1, $2) \
                     ON CONFLICT (conversation_id) DO UPDATE \
                       SET agent_name = EXCLUDED.agent_name",
                )
                .bind(conversation_id.0)
                .bind(&agent_name)
                .execute(&self.pool)
                .await?;
            }
            AgentSelection::User {
                message_provider,
                guild_key,
                user_key,
            } => {
                sqlx::query(
                    "INSERT INTO user_agent_selections \
                       (message_provider, channel, user_key, agent_name) \
                     VALUES ($1, $2, $3, $4) \
                     ON CONFLICT (message_provider, channel, user_key) DO UPDATE \
                       SET agent_name = EXCLUDED.agent_name",
                )
                .bind(message_provider.as_str())
                .bind(guild_scope(&guild_key))
                .bind(&user_key)
                .bind(&agent_name)
                .execute(&self.pool)
                .await?;
            }
            AgentSelection::Channel {
                message_provider,
                guild_key,
                channel_key,
            } => {
                sqlx::query(
                    "INSERT INTO channel_agent_selections \
                       (message_provider, channel, agent_name) \
                     VALUES ($1, $2, $3) \
                     ON CONFLICT (message_provider, channel) DO UPDATE \
                       SET agent_name = EXCLUDED.agent_name",
                )
                .bind(message_provider.as_str())
                .bind(selection_channel_key(guild_key.as_deref(), &channel_key))
                .bind(&agent_name)
                .execute(&self.pool)
                .await?;
            }
            AgentSelection::Guild {
                message_provider,
                guild_key,
            } => {
                sqlx::query(
                    "INSERT INTO channel_agent_selections \
                       (message_provider, channel, agent_name) \
                     VALUES ($1, $2, $3) \
                     ON CONFLICT (message_provider, channel) DO UPDATE \
                       SET agent_name = EXCLUDED.agent_name",
                )
                .bind(message_provider.as_str())
                .bind(guild_scope(&guild_key))
                .bind(&agent_name)
                .execute(&self.pool)
                .await?;
            }
            AgentSelection::Platform { message_provider } => {
                sqlx::query(
                    "INSERT INTO provider_agent_selections (message_provider, agent_name) \
                     VALUES ($1, $2) \
                     ON CONFLICT (message_provider) DO UPDATE \
                       SET agent_name = EXCLUDED.agent_name",
                )
                .bind(message_provider.as_str())
                .bind(&agent_name)
                .execute(&self.pool)
                .await?;
            }
        }
        Ok(())
    }

    async fn clear_agent_selection(&self, selection: AgentSelection) -> Result<bool, Self::Error> {
        let result = match selection {
            AgentSelection::Conversation { conversation_id } => {
                sqlx::query("DELETE FROM conversation_agent_selections WHERE conversation_id = $1")
                    .bind(conversation_id.0)
                    .execute(&self.pool)
                    .await?
            }
            AgentSelection::User {
                message_provider,
                guild_key,
                user_key,
            } => {
                sqlx::query(
                    "DELETE FROM user_agent_selections \
                  WHERE message_provider = $1 AND channel = $2 AND user_key = $3",
                )
                .bind(message_provider.as_str())
                .bind(guild_scope(&guild_key))
                .bind(&user_key)
                .execute(&self.pool)
                .await?
            }
            AgentSelection::Channel {
                message_provider,
                guild_key,
                channel_key,
            } => {
                sqlx::query(
                    "DELETE FROM channel_agent_selections \
                  WHERE message_provider = $1 AND channel = $2",
                )
                .bind(message_provider.as_str())
                .bind(selection_channel_key(guild_key.as_deref(), &channel_key))
                .execute(&self.pool)
                .await?
            }
            AgentSelection::Guild {
                message_provider,
                guild_key,
            } => {
                sqlx::query(
                    "DELETE FROM channel_agent_selections \
                  WHERE message_provider = $1 AND channel = $2",
                )
                .bind(message_provider.as_str())
                .bind(guild_scope(&guild_key))
                .execute(&self.pool)
                .await?
            }
            AgentSelection::Platform { message_provider } => {
                sqlx::query("DELETE FROM provider_agent_selections WHERE message_provider = $1")
                    .bind(message_provider.as_str())
                    .execute(&self.pool)
                    .await?
            }
        };
        Ok(result.rows_affected() > 0)
    }

    async fn upsert_guild(&self, guild: GuildProfile) -> Result<(), Self::Error> {
        sqlx::query(
            "INSERT INTO platform_channels \
               (message_provider, channel, channel_kind, display_name, icon_hash, icon_url, last_seen_at) \
             VALUES ($1, $2, 'workspace', $3, $4, $5, now()) \
             ON CONFLICT (message_provider, channel) DO UPDATE \
               SET channel_kind = 'workspace', \
                   display_name = EXCLUDED.display_name, \
                   icon_media_uri = CASE \
                       WHEN EXCLUDED.icon_hash IS NOT NULL \
                            AND platform_channels.icon_hash IS NOT DISTINCT FROM EXCLUDED.icon_hash \
                       THEN platform_channels.icon_media_uri \
                       ELSE NULL \
                   END, \
                   icon_hash = EXCLUDED.icon_hash, \
                   icon_url = EXCLUDED.icon_url, \
                   last_seen_at = now()",
        )
        .bind(guild.platform.as_str())
        .bind(guild_scope(guild.guild_id.as_str()))
        .bind(&guild.name)
        .bind(&guild.icon_hash)
        .bind(&guild.icon_url)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn load_guild_icon(
        &self,
        platform: PlatformName,
        guild_id: ExternalId,
    ) -> Result<Option<MediaUri>, Self::Error> {
        sqlx::query_scalar::<_, Option<String>>(
            "SELECT icon_media_uri FROM platform_channels \
              WHERE message_provider = $1 AND channel = $2",
        )
        .bind(platform.as_str())
        .bind(guild_scope(guild_id.as_str()))
        .fetch_optional(&self.pool)
        .await
        .map(|value| value.flatten().map(MediaUri::new))
        .map_err(SqlxStorageError::Sqlx)
    }

    async fn set_guild_icon(
        &self,
        platform: PlatformName,
        guild_id: ExternalId,
        icon_hash: String,
        icon: MediaUri,
    ) -> Result<(), Self::Error> {
        let icon = canonical_media_uri(&icon)?;
        let mut tx = self.pool.begin().await?;
        upsert_media_asset(&mut tx, icon.as_str()).await?;
        let result = sqlx::query(
            "UPDATE platform_channels \
                SET icon_media_uri = $4, last_seen_at = now() \
              WHERE message_provider = $1 AND channel = $2 AND icon_hash = $3",
        )
        .bind(platform.as_str())
        .bind(guild_scope(guild_id.as_str()))
        .bind(&icon_hash)
        .bind(icon.as_str())
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        if result.rows_affected() == 0 {
            tracing::trace!(
                platform = %platform,
                guild = %guild_id,
                icon_hash,
                uri = %icon,
                "skipped stale guild icon cache update"
            );
        }
        Ok(())
    }

    async fn upsert_user(&self, user: UserProfile) -> Result<(), Self::Error> {
        let mut tx = self.pool.begin().await?;
        upsert_user(
            &mut tx,
            &user.id,
            Some(display_name_for_user(&user).as_str()),
        )
        .await?;
        sqlx::query(
            "UPDATE platform_users \
                SET username = $3, \
                    display_name = $4, \
                    avatar_media_uri = CASE \
                        WHEN $5 IS NOT NULL \
                             AND platform_users.avatar_url IS NOT DISTINCT FROM $5 \
                        THEN platform_users.avatar_media_uri \
                        ELSE NULL \
                    END, \
                    avatar_url = $5, \
                    is_bot = $6, \
                    last_seen_at = now() \
              WHERE message_provider = $1 AND user_key = $2",
        )
        .bind(user.id.platform.as_str())
        .bind(user.id.user_id.as_str())
        .bind(&user.username)
        .bind(&user.display_name)
        .bind(&user.avatar_url)
        .bind(user.is_bot)
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        Ok(())
    }

    async fn load_user_avatar(&self, user: UserRef) -> Result<Option<MediaUri>, Self::Error> {
        sqlx::query_scalar::<_, Option<String>>(
            "SELECT avatar_media_uri FROM platform_users \
              WHERE message_provider = $1 AND user_key = $2",
        )
        .bind(user.platform.as_str())
        .bind(user.user_id.as_str())
        .fetch_optional(&self.pool)
        .await
        .map(|value| value.flatten().map(MediaUri::new))
        .map_err(SqlxStorageError::Sqlx)
    }

    async fn set_user_avatar(
        &self,
        user: UserRef,
        avatar_url: String,
        avatar: MediaUri,
    ) -> Result<(), Self::Error> {
        let avatar = canonical_media_uri(&avatar)?;
        let mut tx = self.pool.begin().await?;
        upsert_user(&mut tx, &user, None).await?;
        upsert_media_asset(&mut tx, avatar.as_str()).await?;
        let result = sqlx::query(
            "UPDATE platform_users \
                SET avatar_media_uri = $4, last_seen_at = now() \
              WHERE message_provider = $1 AND user_key = $2 AND avatar_url = $3",
        )
        .bind(user.platform.as_str())
        .bind(user.user_id.as_str())
        .bind(&avatar_url)
        .bind(avatar.as_str())
        .execute(&mut *tx)
        .await?;
        tx.commit().await?;
        if result.rows_affected() == 0 {
            tracing::trace!(
                platform = %user.platform,
                user = %user.user_id,
                uri = %avatar,
                "skipped stale avatar cache update"
            );
        }
        Ok(())
    }

    async fn load_user_profiles(
        &self,
        users: Vec<UserRef>,
    ) -> Result<Vec<StoredUserProfile>, Self::Error> {
        let mut profiles = Vec::with_capacity(users.len());
        for user in users {
            let row = sqlx::query(
                "SELECT username, display_name, avatar_url, avatar_media_uri, is_bot \
                   FROM platform_users \
                  WHERE message_provider = $1 AND user_key = $2",
            )
            .bind(user.platform.as_str())
            .bind(user.user_id.as_str())
            .fetch_optional(&self.pool)
            .await?;
            let Some(row) = row else {
                continue;
            };
            profiles.push(StoredUserProfile {
                profile: UserProfile {
                    id: user,
                    username: row.get("username"),
                    name: None,
                    display_name: row.get("display_name"),
                    avatar_url: row.get("avatar_url"),
                    is_bot: row.get("is_bot"),
                },
                avatar: row
                    .get::<Option<String>, _>("avatar_media_uri")
                    .map(MediaUri::new),
            });
        }
        Ok(profiles)
    }

    async fn set_conversation_title(
        &self,
        conversation_id: ConversationId,
        title: String,
    ) -> Result<(), Self::Error> {
        sqlx::query(
            "UPDATE conversations SET title = $2, title_generated_at = now() WHERE id = $1",
        )
        .bind(conversation_id.0)
        .bind(&title)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn create_video_job(&self, input: CreateVideoJob) -> Result<StoredVideoJob, Self::Error> {
        let id = Uuid::new_v4();
        let attempt_id = self.latest_attempt_id(input.turn_id).await?;
        sqlx::query(
            "INSERT INTO video_jobs \
               (id, turn_id, attempt_id, video_provider, provider_job_id, prompt) \
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(id)
        .bind(input.turn_id.0)
        .bind(attempt_id)
        .bind(input.provider.as_str())
        .bind(&input.provider_job_id)
        .bind(&input.prompt)
        .execute(&self.pool)
        .await?;
        Ok(StoredVideoJob {
            turn_id: input.turn_id,
            provider: input.provider,
            provider_job_id: input.provider_job_id,
            prompt: input.prompt,
            status: "pending".to_string(),
            output_uri: None,
            error: None,
        })
    }

    async fn update_video_job(&self, input: UpdateVideoJob) -> Result<(), Self::Error> {
        let output_uri = input
            .output_uri
            .as_ref()
            .map(canonical_media_uri)
            .transpose()?;
        let mut tx = self.pool.begin().await?;
        if let Some(uri) = &output_uri {
            upsert_media_asset(&mut tx, uri.as_str()).await?;
        }
        let updated: Option<(Uuid, Option<Uuid>)> = sqlx::query_as(
            "UPDATE video_jobs \
               SET status = $2, output_uri = COALESCE($3, output_uri), error = $4, \
                   completed_at = CASE WHEN $2 <> 'pending' THEN COALESCE(completed_at, now()) ELSE completed_at END \
             WHERE video_provider = $1 AND provider_job_id = $5 \
             RETURNING turn_id, attempt_id",
        )
        .bind(input.provider.as_str())
        .bind(&input.status)
        .bind(output_uri.as_ref().map(MediaUri::as_str))
        .bind(&input.error)
        .bind(&input.provider_job_id)
        .fetch_optional(&mut *tx)
        .await?;
        if let (Some(uri), Some((turn_id, attempt_id))) = (&output_uri, updated) {
            insert_turn_asset(
                &mut tx,
                TurnAssetInsert {
                    turn_id: TurnId(turn_id),
                    attempt_id,
                    uri: uri.as_str(),
                    source: "video_job",
                    context_item_id: None,
                    tool_trace_id: None,
                    ordinal: i32::MAX,
                },
            )
            .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    async fn count_active_video_generations(
        &self,
        input: CountActiveVideoGenerations,
    ) -> Result<u64, Self::Error> {
        let interval_seconds = i64::try_from(input.interval_seconds).unwrap_or(i64::MAX);
        let scope = input
            .scope_id
            .as_ref()
            .map(|scope| guild_scope(scope.as_str()));
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*)::BIGINT \
               FROM video_jobs v \
               JOIN turns t ON t.id = v.turn_id \
              WHERE t.user_message_provider = $1 \
                AND ( \
                    ($2::text IS NULL AND t.user_message_channel NOT LIKE 'guild:%') \
                    OR ($2::text IS NOT NULL AND t.user_message_channel LIKE $2 || ':%') \
                ) \
                AND ( \
                    v.status = 'pending' \
                    OR ( \
                        v.status = 'done' \
                        AND v.output_uri IS NOT NULL \
                        AND v.completed_at IS NOT NULL \
                    ) \
                ) \
                AND ( \
                    CASE \
                        WHEN v.status = 'pending' THEN v.submitted_at \
                        ELSE v.completed_at \
                    END \
                ) >= now() - ($3::double precision * interval '1 second')",
        )
        .bind(input.platform.as_str())
        .bind(scope.as_deref())
        .bind(interval_seconds)
        .fetch_one(&self.pool)
        .await?;
        Ok(u64::try_from(count).unwrap_or(0))
    }

    async fn usage_cost_report(
        &self,
        query: UsageCostQuery,
    ) -> Result<Vec<UsageCostRow>, Self::Error> {
        let (guild, channel): (Option<&str>, Option<String>) = match &query.scope {
            UsageCostScope::All => (None, None),
            UsageCostScope::Guild { guild_id } => (Some(guild_id.as_str()), None),
            UsageCostScope::Channel {
                guild_id,
                channel_id,
            } => (
                None,
                Some(selection_channel_key(guild_id.as_deref(), channel_id)),
            ),
        };
        let limit = i64::from(query.limit.max(1));
        // Safe to assert: the SQL is assembled from compile-time fragments
        // only; all caller data is bound as parameters.
        let rows = sqlx::query(sqlx::AssertSqlSafe(usage_cost_report_sql(query.group_by)))
            .bind(query.platform.as_str())
            .bind(guild)
            .bind(channel)
            .bind(query.since)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;
        Ok(rows.into_iter().map(usage_cost_row_from_row).collect())
    }

    async fn load_user_memory_document(
        &self,
        key: UserMemoryKey,
    ) -> Result<Option<UserMemoryDocument>, Self::Error> {
        let row = sqlx::query(
            "SELECT message_provider, scope_key, subject_user_key, revision, markdown, \
                    last_compacted_at, source_event_cutoff, source_diary_cutoff, \
                    created_at, created_at AS updated_at \
               FROM user_memory_document_versions \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
              ORDER BY revision DESC \
              LIMIT 1",
        )
        .bind(key.platform.as_str())
        .bind(&key.scope_key)
        .bind(&key.user_key)
        .fetch_optional(&self.pool)
        .await?;
        row.map(document_from_row).transpose()
    }

    async fn append_user_memory_event(
        &self,
        event: NewUserMemoryEvent,
    ) -> Result<UserMemoryEvent, Self::Error> {
        let id = Uuid::new_v4();
        let row = sqlx::query(
            "INSERT INTO user_memory_events \
               (id, message_provider, scope_key, subject_user_key, actor_user_key, kind, body, \
                tags, confidence, source_conversation_id, source_turn_id, source_tool_trace_id, \
                supersedes_event_id) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13) \
             RETURNING id, message_provider, scope_key, subject_user_key, actor_user_key, kind, \
                       body, tags, confidence, source_conversation_id, source_turn_id, \
                       source_tool_trace_id, supersedes_event_id, created_at, updated_at",
        )
        .bind(id)
        .bind(event.key.platform.as_str())
        .bind(&event.key.scope_key)
        .bind(&event.key.user_key)
        .bind(&event.actor_user_key)
        .bind(memory_event_kind_as_str(event.kind))
        .bind(&event.body)
        .bind(serde_json::to_value(&event.tags)?)
        .bind(event.confidence)
        .bind(event.source_conversation_id.map(|id| id.0))
        .bind(event.source_turn_id.map(|id| id.0))
        .bind(event.source_tool_trace_id)
        .bind(event.supersedes_event_id)
        .fetch_one(&self.pool)
        .await?;
        memory_event_from_row(row)
    }

    async fn list_pending_memory_events(
        &self,
        key: UserMemoryKey,
        since: Option<OffsetDateTime>,
    ) -> Result<Vec<UserMemoryEvent>, Self::Error> {
        let rows = sqlx::query(
            "SELECT id, message_provider, scope_key, subject_user_key, actor_user_key, kind, \
                    body, tags, confidence, source_conversation_id, source_turn_id, \
                    source_tool_trace_id, supersedes_event_id, created_at, updated_at \
               FROM user_memory_events \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
                AND ($4::timestamptz IS NULL OR created_at > $4) \
              ORDER BY created_at, id",
        )
        .bind(key.platform.as_str())
        .bind(&key.scope_key)
        .bind(&key.user_key)
        .bind(since)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(memory_event_from_row).collect()
    }

    async fn list_pending_memory_diary_entries(
        &self,
        key: UserMemoryKey,
        since: Option<OffsetDateTime>,
    ) -> Result<Vec<UserMemoryDiaryEntry>, Self::Error> {
        let rows = sqlx::query(
            "SELECT id, message_provider, scope_key, subject_user_key, window_start, window_end, \
                    source_turn_ids, markdown, agent_name, llm_provider, llm_model, usage, \
                    created_at, updated_at \
               FROM user_memory_diary_entries \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
                AND ($4::timestamptz IS NULL OR created_at > $4) \
              ORDER BY created_at, id",
        )
        .bind(key.platform.as_str())
        .bind(&key.scope_key)
        .bind(&key.user_key)
        .bind(since)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(diary_entry_from_row).collect()
    }

    async fn list_recent_memory_diary_entries(
        &self,
        key: UserMemoryKey,
        limit: u32,
    ) -> Result<Vec<UserMemoryDiaryEntry>, Self::Error> {
        let mut rows = sqlx::query(
            "SELECT id, message_provider, scope_key, subject_user_key, window_start, window_end, \
                    source_turn_ids, markdown, agent_name, llm_provider, llm_model, usage, \
                    created_at, updated_at \
               FROM user_memory_diary_entries \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
              ORDER BY created_at DESC, id DESC \
              LIMIT $4",
        )
        .bind(key.platform.as_str())
        .bind(&key.scope_key)
        .bind(&key.user_key)
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await?;
        rows.reverse();
        rows.into_iter().map(diary_entry_from_row).collect()
    }

    async fn save_user_memory_diary_entry(
        &self,
        entry: NewUserMemoryDiaryEntry,
    ) -> Result<UserMemoryDiaryEntry, Self::Error> {
        let id = Uuid::new_v4();
        let source_turn_ids = entry
            .source_turn_ids
            .iter()
            .map(|turn_id| turn_id.0)
            .collect::<Vec<_>>();
        let row = sqlx::query(
            "INSERT INTO user_memory_diary_entries \
               (id, message_provider, scope_key, subject_user_key, window_start, window_end, \
                source_turn_ids, markdown, agent_name, llm_provider, llm_model, usage) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) \
             RETURNING id, message_provider, scope_key, subject_user_key, window_start, \
                       window_end, source_turn_ids, markdown, agent_name, llm_provider, \
                       llm_model, usage, created_at, updated_at",
        )
        .bind(id)
        .bind(entry.key.platform.as_str())
        .bind(&entry.key.scope_key)
        .bind(&entry.key.user_key)
        .bind(entry.window_start)
        .bind(entry.window_end)
        .bind(source_turn_ids)
        .bind(&entry.markdown)
        .bind(&entry.agent_name)
        .bind(entry.llm_provider.as_str())
        .bind(entry.llm_model.as_str())
        .bind(serde_json::to_value(&entry.usage)?)
        .fetch_one(&self.pool)
        .await?;
        diary_entry_from_row(row)
    }

    async fn save_user_memory_document_revision(
        &self,
        document: NewUserMemoryDocumentRevision,
    ) -> Result<UserMemoryDocument, Self::Error> {
        let mut tx = self.pool.begin().await?;
        let existing_revision: Option<i64> = sqlx::query_scalar(
            "SELECT revision FROM user_memory_document_versions \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
              ORDER BY revision DESC \
              LIMIT 1 \
              FOR UPDATE",
        )
        .bind(document.key.platform.as_str())
        .bind(&document.key.scope_key)
        .bind(&document.key.user_key)
        .fetch_optional(&mut *tx)
        .await?;
        let revision = existing_revision.unwrap_or(0) + 1;
        let version_id = Uuid::new_v4();
        let row = sqlx::query(
            "INSERT INTO user_memory_document_versions \
               (id, message_provider, scope_key, subject_user_key, revision, markdown, \
                source_event_ids, source_diary_entry_ids, agent_name, llm_provider, llm_model, \
                usage, last_compacted_at, source_event_cutoff, source_diary_cutoff) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, now(), $13, $14) \
             RETURNING message_provider, scope_key, subject_user_key, revision, markdown, \
                       last_compacted_at, source_event_cutoff, source_diary_cutoff, \
                       created_at, created_at AS updated_at",
        )
        .bind(version_id)
        .bind(document.key.platform.as_str())
        .bind(&document.key.scope_key)
        .bind(&document.key.user_key)
        .bind(revision)
        .bind(&document.markdown)
        .bind(&document.source_event_ids)
        .bind(&document.source_diary_entry_ids)
        .bind(&document.agent_name)
        .bind(document.llm_provider.as_str())
        .bind(document.llm_model.as_str())
        .bind(serde_json::to_value(&document.usage)?)
        .bind(document.source_event_cutoff)
        .bind(document.source_diary_cutoff)
        .fetch_one(&mut *tx)
        .await?;
        tx.commit().await?;
        document_from_row(row)
    }

    async fn enqueue_due_memory_jobs(
        &self,
        schedule: MemoryJobSchedule,
    ) -> Result<u64, Self::Error> {
        let mut inserted = 0u64;
        let diary_window_seconds =
            i64::try_from(schedule.diary_window_seconds.max(1)).unwrap_or(i64::MAX);
        sqlx::query(
            "UPDATE user_memory_jobs active_jobs \
                SET status = 'failed', completed_at = COALESCE(completed_at, $1), \
                    leased_by = NULL, leased_until = NULL, \
                    error = COALESCE(error, 'diary window already reached terminal state') \
              WHERE active_jobs.kind = 'diary' \
                AND (active_jobs.status = 'pending' \
                     OR (active_jobs.status = 'running' AND active_jobs.leased_until < $1)) \
                AND active_jobs.window_start IS NOT NULL \
                AND active_jobs.window_end IS NOT NULL \
                AND ( \
                     EXISTS ( \
                         SELECT 1 \
                           FROM user_memory_diary_entries diary_entry \
                          WHERE diary_entry.message_provider = active_jobs.message_provider \
                            AND diary_entry.scope_key = active_jobs.scope_key \
                            AND diary_entry.subject_user_key = active_jobs.subject_user_key \
                            AND diary_entry.window_start = active_jobs.window_start \
                            AND diary_entry.window_end = active_jobs.window_end \
                     ) \
                     OR EXISTS ( \
                         SELECT 1 \
                           FROM user_memory_jobs terminal_jobs \
                          WHERE terminal_jobs.kind = 'diary' \
                            AND terminal_jobs.status IN ('completed', 'failed') \
                            AND terminal_jobs.id <> active_jobs.id \
                            AND terminal_jobs.message_provider = active_jobs.message_provider \
                            AND terminal_jobs.scope_key = active_jobs.scope_key \
                            AND terminal_jobs.subject_user_key = active_jobs.subject_user_key \
                            AND terminal_jobs.window_start = active_jobs.window_start \
                            AND terminal_jobs.window_end = active_jobs.window_end \
                     ) \
                )",
        )
        .bind(schedule.now)
        .execute(&self.pool)
        .await?;
        let diary_rows = sqlx::query(
            "SELECT diary_windows.message_provider, diary_windows.scope_key, \
                    diary_windows.subject_user_key, diary_windows.window_start, \
                    diary_windows.window_start + ($2::double precision * INTERVAL '1 second') AS window_end \
               FROM ( \
                    SELECT candidate_turns.message_provider, candidate_turns.scope_key, \
                           candidate_turns.subject_user_key, \
                           CASE \
                             WHEN candidate_turns.latest_window_end IS NULL \
                               OR candidate_turns.latest_window_end < $1 \
                             THEN candidate_turns.first_completed_at \
                             ELSE candidate_turns.latest_window_end + ( \
                               GREATEST( \
                                 ceil(EXTRACT(EPOCH FROM (candidate_turns.first_completed_at - candidate_turns.latest_window_end))::double precision / $2::double precision)::bigint - 1, \
                                 0::bigint \
                               )::double precision * $2::double precision * INTERVAL '1 second' \
                             ) \
                           END AS window_start \
                      FROM ( \
                           SELECT t.user_message_provider AS message_provider, \
                                  CASE \
                                    WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                    THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                    ELSE 'global' \
                                  END AS scope_key, \
                                  t.user_key AS subject_user_key, \
                                  latest_diary.window_end AS latest_window_end, \
                                  MIN(t.completed_at) AS first_completed_at \
                             FROM turns t \
                             LEFT JOIN ( \
                                  SELECT message_provider, scope_key, subject_user_key, \
                                         MAX(window_end) AS window_end \
                                    FROM ( \
                                         SELECT message_provider, scope_key, subject_user_key, \
                                                window_end \
                                           FROM user_memory_diary_entries \
                                          UNION ALL \
                                         SELECT message_provider, scope_key, subject_user_key, \
                                                window_end \
                                           FROM user_memory_jobs \
                                          WHERE kind = 'diary' \
                                            AND status IN ('completed', 'failed') \
                                            AND window_end IS NOT NULL \
                                    ) processed_diary_windows \
                                   GROUP BY message_provider, scope_key, subject_user_key \
                             ) latest_diary \
                               ON latest_diary.message_provider = t.user_message_provider \
                              AND latest_diary.scope_key = CASE \
                                    WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                    THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                    ELSE 'global' \
                                  END \
                              AND latest_diary.subject_user_key = t.user_key \
                            WHERE t.status = 'completed' \
                              AND t.completed_at IS NOT NULL \
                              AND t.completed_at >= $1 \
                              AND (latest_diary.window_end IS NULL \
                                   OR latest_diary.window_end < $1 \
                                   OR t.completed_at >= latest_diary.window_end) \
                            GROUP BY t.user_message_provider, \
                                  CASE \
                                    WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                    THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                    ELSE 'global' \
                                  END, \
                                  t.user_key, \
                                  latest_diary.window_end \
                      ) candidate_turns \
               ) diary_windows \
              WHERE diary_windows.window_start <= $3",
        )
        .bind(schedule.diary_cutoff)
        .bind(diary_window_seconds)
        .bind(schedule.diary_due_before)
        .fetch_all(&self.pool)
        .await?;
        for row in diary_rows {
            let key = UserMemoryKey {
                platform: PlatformName::new(row.get::<String, _>("message_provider")),
                scope_key: row.get("scope_key"),
                user_key: row.get("subject_user_key"),
            };
            let memory_key = key.memory_key();
            let result = sqlx::query(
                "INSERT INTO user_memory_jobs \
                   (id, kind, message_provider, scope_key, subject_user_key, memory_key, \
                    window_start, window_end, status, next_run_at, dedupe_key) \
                 VALUES ($1, 'diary', $2, $3, $4, $5, $6, $7, 'pending', $8, $9) \
                 ON CONFLICT DO NOTHING",
            )
            .bind(Uuid::new_v4())
            .bind(key.platform.as_str())
            .bind(&key.scope_key)
            .bind(&key.user_key)
            .bind(&memory_key)
            .bind(row.get::<OffsetDateTime, _>("window_start"))
            .bind(row.get::<OffsetDateTime, _>("window_end"))
            .bind(schedule.now)
            .bind(format!("diary:{memory_key}"))
            .execute(&self.pool)
            .await?;
            inserted += result.rows_affected();
        }

        let compact_rows = sqlx::query(
            "SELECT pending_sources.message_provider, pending_sources.scope_key, \
                    pending_sources.subject_user_key \
               FROM ( \
                    SELECT source.message_provider, source.scope_key, source.subject_user_key, \
                           latest_document.last_compacted_at \
                      FROM ( \
                           SELECT e.message_provider, e.scope_key, e.subject_user_key, \
                                  e.created_at, TRUE AS is_event \
                             FROM user_memory_events e \
                            UNION ALL \
                           SELECT de.message_provider, de.scope_key, de.subject_user_key, \
                                  de.created_at, FALSE AS is_event \
                             FROM user_memory_diary_entries de \
                      ) source \
                      LEFT JOIN ( \
                           SELECT DISTINCT ON (message_provider, scope_key, subject_user_key) \
                                  message_provider, scope_key, subject_user_key, last_compacted_at, \
                                  source_event_cutoff, source_diary_cutoff \
                             FROM user_memory_document_versions \
                            ORDER BY message_provider, scope_key, subject_user_key, revision DESC \
                      ) latest_document \
                        ON latest_document.message_provider = source.message_provider \
                       AND latest_document.scope_key = source.scope_key \
                       AND latest_document.subject_user_key = source.subject_user_key \
                     WHERE (source.is_event \
                            AND source.created_at > COALESCE(latest_document.source_event_cutoff, '-infinity'::timestamptz)) \
                        OR (NOT source.is_event \
                            AND source.created_at > COALESCE(latest_document.source_diary_cutoff, '-infinity'::timestamptz)) \
               ) pending_sources \
               LEFT JOIN ( \
                    SELECT DISTINCT message_provider, scope_key, subject_user_key \
                      FROM user_memory_jobs \
                     WHERE kind = 'diary' \
                       AND status IN ('pending', 'running') \
               ) active_diary \
                 ON active_diary.message_provider = pending_sources.message_provider \
                AND active_diary.scope_key = pending_sources.scope_key \
                AND active_diary.subject_user_key = pending_sources.subject_user_key \
               LEFT JOIN ( \
                    SELECT diary_windows.message_provider, diary_windows.scope_key, \
                           diary_windows.subject_user_key \
                      FROM ( \
                           SELECT candidate_turns.message_provider, candidate_turns.scope_key, \
                                  candidate_turns.subject_user_key, \
                                  CASE \
                                    WHEN candidate_turns.latest_window_end IS NULL \
                                      OR candidate_turns.latest_window_end < $2 \
                                    THEN candidate_turns.first_completed_at \
                                    ELSE candidate_turns.latest_window_end + ( \
                                      GREATEST( \
                                        ceil(EXTRACT(EPOCH FROM (candidate_turns.first_completed_at - candidate_turns.latest_window_end))::double precision / $4::double precision)::bigint - 1, \
                                        0::bigint \
                                      )::double precision * $4::double precision * INTERVAL '1 second' \
                                    ) \
                                  END AS window_start \
                             FROM ( \
                                  SELECT t.user_message_provider AS message_provider, \
                                         CASE \
                                           WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                           THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                           ELSE 'global' \
                                         END AS scope_key, \
                                         t.user_key AS subject_user_key, \
                                         latest_diary.window_end AS latest_window_end, \
                                         MIN(t.completed_at) AS first_completed_at \
                                    FROM turns t \
                                    LEFT JOIN ( \
                                         SELECT message_provider, scope_key, subject_user_key, \
                                                MAX(window_end) AS window_end \
                                           FROM ( \
                                                SELECT message_provider, scope_key, subject_user_key, \
                                                       window_end \
                                                  FROM user_memory_diary_entries \
                                                 UNION ALL \
                                                SELECT message_provider, scope_key, subject_user_key, \
                                                       window_end \
                                                  FROM user_memory_jobs \
                                                 WHERE kind = 'diary' \
                                                   AND status IN ('completed', 'failed') \
                                                   AND window_end IS NOT NULL \
                                           ) processed_diary_windows \
                                          GROUP BY message_provider, scope_key, subject_user_key \
                                    ) latest_diary \
                                      ON latest_diary.message_provider = t.user_message_provider \
                                     AND latest_diary.scope_key = CASE \
                                           WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                           THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                           ELSE 'global' \
                                         END \
                                     AND latest_diary.subject_user_key = t.user_key \
                                   WHERE t.status = 'completed' \
                                     AND t.completed_at IS NOT NULL \
                                     AND t.completed_at >= $2 \
                                     AND (latest_diary.window_end IS NULL \
                                          OR latest_diary.window_end < $2 \
                                          OR t.completed_at >= latest_diary.window_end) \
                                   GROUP BY t.user_message_provider, \
                                         CASE \
                                           WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                           THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                           ELSE 'global' \
                                         END, \
                                         t.user_key, \
                                         latest_diary.window_end \
                             ) candidate_turns \
                      ) diary_windows \
                     WHERE diary_windows.window_start <= $3 \
               ) due_diary \
                 ON due_diary.message_provider = pending_sources.message_provider \
                AND due_diary.scope_key = pending_sources.scope_key \
                AND due_diary.subject_user_key = pending_sources.subject_user_key \
              WHERE active_diary.message_provider IS NULL \
                AND due_diary.message_provider IS NULL \
              GROUP BY pending_sources.message_provider, pending_sources.scope_key, \
                       pending_sources.subject_user_key, pending_sources.last_compacted_at \
             HAVING COALESCE(pending_sources.last_compacted_at, '-infinity'::timestamptz) <= $1",
        )
        .bind(schedule.compact_due_before)
        .bind(schedule.diary_cutoff)
        .bind(schedule.diary_due_before)
        .bind(diary_window_seconds)
        .fetch_all(&self.pool)
        .await?;
        for row in compact_rows {
            let key = UserMemoryKey {
                platform: PlatformName::new(row.get::<String, _>("message_provider")),
                scope_key: row.get("scope_key"),
                user_key: row.get("subject_user_key"),
            };
            let memory_key = key.memory_key();
            let result = sqlx::query(
                "INSERT INTO user_memory_jobs \
                   (id, kind, message_provider, scope_key, subject_user_key, memory_key, \
                    status, next_run_at, dedupe_key) \
                 VALUES ($1, 'compact', $2, $3, $4, $5, 'pending', $6, $7) \
                 ON CONFLICT DO NOTHING",
            )
            .bind(Uuid::new_v4())
            .bind(key.platform.as_str())
            .bind(&key.scope_key)
            .bind(&key.user_key)
            .bind(&memory_key)
            .bind(schedule.now)
            .bind(format!("compact:{memory_key}"))
            .execute(&self.pool)
            .await?;
            inserted += result.rows_affected();
        }
        Ok(inserted)
    }

    async fn claim_memory_jobs(
        &self,
        worker_id: String,
        limit: u32,
        lease_until: OffsetDateTime,
    ) -> Result<Vec<UserMemoryJob>, Self::Error> {
        let rows = sqlx::query(
            "UPDATE user_memory_jobs j \
                SET status = 'running', attempts = attempts + 1, leased_by = $2, \
                    leased_until = $3, started_at = COALESCE(started_at, now()), \
                    completed_at = NULL, error = NULL \
               FROM ( \
                    SELECT picked.id \
                      FROM user_memory_jobs picked \
                      JOIN ( \
                           SELECT candidate.id, \
                                  row_number() OVER ( \
                                      PARTITION BY candidate.memory_key \
                                      ORDER BY candidate.next_run_at, candidate.created_at \
                                  ) AS rn \
                             FROM user_memory_jobs candidate \
                            WHERE candidate.next_run_at <= now() \
                              AND (candidate.status = 'pending' \
                                   OR (candidate.status = 'running' AND candidate.leased_until < now())) \
                              AND NOT EXISTS ( \
                                   SELECT 1 FROM user_memory_jobs active \
                                    WHERE active.memory_key = candidate.memory_key \
                                      AND active.status = 'running' \
                                      AND active.leased_until >= now() \
                                      AND active.id <> candidate.id \
                              ) \
                      ) candidates ON candidates.id = picked.id \
                     WHERE candidates.rn = 1 \
                     ORDER BY picked.next_run_at, picked.created_at \
                     LIMIT $1 \
                     FOR UPDATE OF picked SKIP LOCKED \
               ) picked_jobs \
              WHERE j.id = picked_jobs.id \
              RETURNING j.id, j.kind, j.message_provider, j.scope_key, j.subject_user_key, \
                        j.memory_key, j.window_start, j.window_end, j.attempts, \
                        j.leased_by, j.leased_until, j.dedupe_key",
        )
        .bind(i64::from(limit))
        .bind(&worker_id)
        .bind(lease_until)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(memory_job_from_row).collect()
    }

    async fn finish_memory_job(&self, completion: MemoryJobCompletion) -> Result<(), Self::Error> {
        match completion {
            MemoryJobCompletion::Completed { job_id } => {
                sqlx::query(
                    "UPDATE user_memory_jobs \
                        SET status = 'completed', completed_at = now(), leased_by = NULL, \
                            leased_until = NULL, error = NULL \
                      WHERE id = $1",
                )
                .bind(job_id)
                .execute(&self.pool)
                .await?;
            }
            MemoryJobCompletion::Retry {
                job_id,
                error,
                next_run_at,
            } => {
                sqlx::query(
                    "UPDATE user_memory_jobs \
                        SET status = 'pending', next_run_at = $2, leased_by = NULL, \
                            leased_until = NULL, error = $3 \
                      WHERE id = $1",
                )
                .bind(job_id)
                .bind(next_run_at)
                .bind(&error)
                .execute(&self.pool)
                .await?;
            }
            MemoryJobCompletion::Failed { job_id, error } => {
                sqlx::query(
                    "UPDATE user_memory_jobs \
                        SET status = 'failed', completed_at = now(), leased_by = NULL, \
                            leased_until = NULL, error = $2 \
                      WHERE id = $1",
                )
                .bind(job_id)
                .bind(&error)
                .execute(&self.pool)
                .await?;
            }
        }
        Ok(())
    }

    async fn load_memory_turn_window(
        &self,
        window: MemoryTurnWindow,
    ) -> Result<Vec<UserMemoryTurn>, Self::Error> {
        let rows = sqlx::query(
            "SELECT conversation_id, id AS turn_id, completed_at, user_display_name, \
                    user_content, assistant_content \
               FROM turns \
              WHERE user_message_provider = $1 \
                AND user_key = $2 \
                AND CASE \
                      WHEN user_message_channel LIKE 'guild:%:channel:%' \
                      THEN 'guild:' || split_part(user_message_channel, ':', 2) \
                      ELSE 'global' \
                    END = $3 \
                AND status = 'completed' \
                AND completed_at IS NOT NULL \
                AND completed_at >= $4 \
                AND completed_at < $5 \
              ORDER BY completed_at, ordinal \
              LIMIT $6",
        )
        .bind(window.key.platform.as_str())
        .bind(&window.key.user_key)
        .bind(&window.key.scope_key)
        .bind(window.window_start)
        .bind(window.window_end)
        .bind(i64::from(window.max_turns))
        .fetch_all(&self.pool)
        .await?;
        let mut turns = rows
            .into_iter()
            .map(|row| UserMemoryTurn {
                conversation_id: ConversationId(row.get("conversation_id")),
                turn_id: TurnId(row.get("turn_id")),
                completed_at: row.get("completed_at"),
                user_display_name: row.get("user_display_name"),
                user_content: row.get("user_content"),
                assistant_content: row.get("assistant_content"),
                image_context: Vec::new(),
                audio_transcriptions: Vec::new(),
            })
            .collect::<Vec<_>>();
        let turn_ids = turns.iter().map(|turn| turn.turn_id.0).collect::<Vec<_>>();
        let mut images_by_turn = load_memory_image_context(&self.pool, &turn_ids).await?;
        let mut audio_by_turn = load_memory_audio_transcriptions(&self.pool, &turn_ids).await?;
        for turn in &mut turns {
            turn.image_context = images_by_turn.remove(&turn.turn_id).unwrap_or_default();
            turn.audio_transcriptions = audio_by_turn.remove(&turn.turn_id).unwrap_or_default();
        }
        Ok(turns)
    }
}

async fn load_memory_image_context(
    pool: &PgPool,
    turn_ids: &[Uuid],
) -> Result<BTreeMap<TurnId, Vec<UserMemoryImageContext>>, SqlxStorageError> {
    if turn_ids.is_empty() {
        return Ok(BTreeMap::new());
    }
    let rows = sqlx::query(
        "SELECT a.turn_id, a.media_uri, a.source, m.mime_type \
           FROM turn_assets a \
           LEFT JOIN media_assets m ON m.uri = a.media_uri \
          WHERE a.turn_id = ANY($1) \
            AND a.replayable \
            AND (m.category = 'image' OR m.mime_type LIKE 'image/%' \
                 OR a.media_uri LIKE 'media://images/%' OR a.media_uri LIKE 'file://images/%') \
          ORDER BY a.turn_id, a.ordinal, a.id",
    )
    .bind(turn_ids)
    .fetch_all(pool)
    .await?;
    let mut out = BTreeMap::<TurnId, Vec<UserMemoryImageContext>>::new();
    for row in rows {
        let turn_id = TurnId(row.get("turn_id"));
        out.entry(turn_id)
            .or_default()
            .push(UserMemoryImageContext {
                image_uri: MediaUri::new(row.get::<String, _>("media_uri")),
                source: row.get("source"),
                mime_type: row.get("mime_type"),
            });
    }
    Ok(out)
}

async fn load_memory_audio_transcriptions(
    pool: &PgPool,
    turn_ids: &[Uuid],
) -> Result<BTreeMap<TurnId, Vec<UserMemoryAudioTranscription>>, SqlxStorageError> {
    if turn_ids.is_empty() {
        return Ok(BTreeMap::new());
    }
    let rows = sqlx::query(
        "SELECT ta.turn_id, tt.id AS tool_trace_id, tt.request, tt.response \
           FROM turn_attempt_tool_traces tt \
           JOIN turn_attempts ta ON ta.id = tt.attempt_id \
          WHERE ta.turn_id = ANY($1) \
            AND ta.status = 'completed' \
            AND tt.trace_kind = 'client' \
            AND tt.tool_name = 'transcribe_audio' \
            AND COALESCE(tt.is_error, false) = false \
          ORDER BY ta.turn_id, tt.ordinal",
    )
    .bind(turn_ids)
    .fetch_all(pool)
    .await?;
    let mut out = BTreeMap::<TurnId, Vec<UserMemoryAudioTranscription>>::new();
    for row in rows {
        let turn_id = TurnId(row.get("turn_id"));
        let Some(transcription) = memory_audio_transcription_from_tool_row(&row) else {
            continue;
        };
        out.entry(turn_id).or_default().push(transcription);
    }
    Ok(out)
}

fn memory_audio_transcription_from_tool_row(
    row: &sqlx::postgres::PgRow,
) -> Option<UserMemoryAudioTranscription> {
    let response = row.get::<Option<Value>, _>("response")?;
    let request = row.get::<Option<Value>, _>("request");
    memory_audio_transcription_from_values(row.get("tool_trace_id"), request.as_ref(), &response)
}

fn memory_audio_transcription_from_values(
    tool_trace_id: i64,
    request: Option<&Value>,
    response: &Value,
) -> Option<UserMemoryAudioTranscription> {
    let text = response
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())?
        .to_string();
    Some(UserMemoryAudioTranscription {
        tool_trace_id,
        audio_uri: request.and_then(audio_uri_from_tool_request),
        text,
        language: optional_non_empty_string(response.get("language")),
        duration_seconds: response
            .get("duration_seconds")
            .and_then(Value::as_f64)
            .filter(|duration| duration.is_finite()),
    })
}

fn audio_uri_from_tool_request(request: &Value) -> Option<String> {
    request
        .get("input")
        .and_then(|input| input.get("audio_uri").or_else(|| input.get("audio")))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|uri| !uri.is_empty())
        .and_then(|uri| canonical_media_uri_string(uri).ok())
}

fn optional_non_empty_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

impl SqlxStorage {
    async fn conversation_id_for_message(
        &self,
        message: &MessageRef,
    ) -> Result<Option<Uuid>, SqlxStorageError> {
        sqlx::query_scalar(
            "SELECT conversation_id FROM message_links \
              WHERE message_provider = $1 AND channel = $2 AND message = $3",
        )
        .bind(message.platform.as_str())
        .bind(channel_key_from_message(message))
        .bind(message.message_id.as_str())
        .fetch_optional(&self.pool)
        .await
        .map_err(SqlxStorageError::Sqlx)
    }

    async fn conversation_id_for_channel(
        &self,
        channel: &ChannelRef,
    ) -> Result<Option<Uuid>, SqlxStorageError> {
        sqlx::query_scalar(
            "SELECT conversation_id FROM channel_links \
              WHERE message_provider = $1 AND channel = $2 \
              ORDER BY linked_at DESC LIMIT 1",
        )
        .bind(channel.platform.as_str())
        .bind(channel_key(channel))
        .fetch_optional(&self.pool)
        .await
        .map_err(SqlxStorageError::Sqlx)
    }
}

/// Storage errors.
#[derive(Debug, Error)]
pub enum SqlxStorageError {
    /// SQLx query failed.
    #[error("sqlx: {0}")]
    Sqlx(#[from] sqlx::Error),
    /// Migration failed.
    #[error("migration: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    /// JSON encode/decode failed.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// A referenced conversation was missing.
    #[error("conversation `{conversation_id}` was not found")]
    MissingConversation {
        /// Conversation id.
        conversation_id: ConversationId,
    },
    /// A turn has no saved attempt.
    #[error("turn `{turn_id}` has no saved attempt")]
    MissingAttempt {
        /// Turn id.
        turn_id: TurnId,
    },
    /// Stored platform reference was malformed.
    #[error("invalid platform reference: {0}")]
    InvalidReference(String),
    /// Stored model step kind was malformed.
    #[error("invalid model step kind: {0}")]
    InvalidModelStepKind(String),
    /// Stored media URI was malformed.
    #[error("invalid media uri: {0}")]
    InvalidMediaUri(String),
}

struct ToolTraceFields {
    trace_kind: &'static str,
    tool_name: Option<String>,
    provider: Option<String>,
    tool_use_id: Option<String>,
    is_error: Option<bool>,
    request: Option<Value>,
    response: Option<Value>,
}

fn tool_trace_fields(trace: &ToolTrace) -> Result<ToolTraceFields, SqlxStorageError> {
    match trace {
        ToolTrace::Client { trace } => Ok(ToolTraceFields {
            trace_kind: "client",
            tool_name: Some(trace.call.name.to_string()),
            provider: None,
            tool_use_id: Some(trace.call.id.to_string()),
            is_error: Some(trace.result.is_error),
            request: Some(serde_json::to_value(&trace.call)?),
            response: Some(normalized_client_tool_response(trace)),
        }),
        ToolTrace::Server { tool } => Ok(ToolTraceFields {
            trace_kind: "server",
            tool_name: Some(tool.name.to_string()),
            provider: Some(tool.provider.to_string()),
            tool_use_id: tool.id.clone(),
            is_error: None,
            request: None,
            response: Some(tool.raw.clone()),
        }),
        ToolTrace::Grounding { metadata } => Ok(ToolTraceFields {
            trace_kind: "grounding",
            tool_name: None,
            provider: Some(metadata.provider.to_string()),
            tool_use_id: None,
            is_error: None,
            request: None,
            response: Some(metadata.raw.clone()),
        }),
    }
}

fn tool_trace_media_asset(fields: &ToolTraceFields) -> Option<(String, String)> {
    let response = fields.response.as_ref()?;
    let uri = media_uri_from_value(response)?;
    Some((
        uri,
        fields
            .tool_name
            .clone()
            .unwrap_or_else(|| "tool_trace".to_string()),
    ))
}

fn normalized_client_tool_response(trace: &chudbot_api::ClientToolTrace) -> Value {
    match &trace.result.content {
        chudbot_api::ClientToolResultContent::Json { value } => value.clone(),
        chudbot_api::ClientToolResultContent::Text { text } => {
            if media_uri_from_value(&trace.trace_response).is_some() {
                trace.trace_response.clone()
            } else {
                serde_json::json!({ "text": text })
            }
        }
    }
}

fn media_uri_from_value(value: &Value) -> Option<String> {
    if let Some(uri) = value
        .get("uri")
        .or_else(|| value.get("image_uri"))
        .or_else(|| value.get("video_uri"))
        .and_then(Value::as_str)
        .filter(|uri| is_stored_media_uri(uri))
        .and_then(|uri| canonical_media_uri_string(uri).ok())
    {
        return Some(uri);
    }

    match value {
        Value::Object(object) => {
            for key in ["value", "content", "result", "trace_response"] {
                if let Some(uri) = object.get(key).and_then(media_uri_from_value) {
                    return Some(uri);
                }
            }
            object.values().find_map(media_uri_from_value)
        }
        Value::Array(values) => values.iter().find_map(media_uri_from_value),
        _ => None,
    }
}

async fn upsert_channel(
    tx: &mut Transaction<'_, Postgres>,
    channel: &ChannelRef,
) -> Result<String, SqlxStorageError> {
    let key = channel_key(channel);
    let parent = channel
        .guild_id
        .as_ref()
        .map(|guild| guild_scope(guild.as_str()));
    if let Some(parent) = &parent {
        sqlx::query(
            "INSERT INTO platform_channels (message_provider, channel, channel_kind) \
             VALUES ($1, $2, 'workspace') ON CONFLICT DO NOTHING",
        )
        .bind(channel.platform.as_str())
        .bind(parent)
        .execute(&mut **tx)
        .await?;
    }
    sqlx::query(
        "INSERT INTO platform_channels (message_provider, channel, parent_channel, channel_kind, last_seen_at) \
         VALUES ($1, $2, $3, 'channel', now()) \
         ON CONFLICT (message_provider, channel) DO UPDATE SET last_seen_at = now()",
    )
    .bind(channel.platform.as_str())
    .bind(&key)
    .bind(parent)
    .execute(&mut **tx)
    .await?;
    Ok(key)
}

async fn upsert_channel_from_message(
    tx: &mut Transaction<'_, Postgres>,
    message: &MessageRef,
) -> Result<String, SqlxStorageError> {
    upsert_channel(tx, &channel_from_message(message)).await
}

async fn upsert_user(
    tx: &mut Transaction<'_, Postgres>,
    user: &UserRef,
    display_name: Option<&str>,
) -> Result<(), SqlxStorageError> {
    sqlx::query(
        "INSERT INTO platform_users \
           (message_provider, user_key, username, display_name, last_seen_at) \
         VALUES ($1, $2, $3, $4, now()) \
         ON CONFLICT (message_provider, user_key) DO UPDATE \
           SET display_name = COALESCE(EXCLUDED.display_name, platform_users.display_name), \
               last_seen_at = now()",
    )
    .bind(user.platform.as_str())
    .bind(user.user_id.as_str())
    .bind(display_name.unwrap_or_else(|| user.user_id.as_str()))
    .bind(display_name)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn upsert_message(
    tx: &mut Transaction<'_, Postgres>,
    message: &MessageRef,
    author_user_key: Option<&str>,
    content: Option<&str>,
) -> Result<(), SqlxStorageError> {
    upsert_channel_from_message(tx, message).await?;
    sqlx::query(
        "INSERT INTO platform_messages \
           (message_provider, channel, message, author_user_key, content) \
         VALUES ($1, $2, $3, $4, $5) \
         ON CONFLICT (message_provider, channel, message) DO UPDATE \
           SET author_user_key = COALESCE(EXCLUDED.author_user_key, platform_messages.author_user_key), \
               content = COALESCE(EXCLUDED.content, platform_messages.content)",
    )
    .bind(message.platform.as_str())
    .bind(channel_key_from_message(message))
    .bind(message.message_id.as_str())
    .bind(author_user_key)
    .bind(content)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn insert_context_item(
    tx: &mut Transaction<'_, Postgres>,
    turn_id: TurnId,
    attempt_id: Uuid,
    item: ContextItem,
) -> Result<(), SqlxStorageError> {
    let (provider, channel, message) = item
        .message
        .as_ref()
        .map(|message| {
            (
                Some(message.platform.as_str().to_string()),
                Some(channel_key_from_message(message)),
                Some(message.message_id.as_str().to_string()),
            )
        })
        .unwrap_or((None, None, None));
    let media_uri = if is_stored_media_uri(&item.content) {
        Some(canonical_media_uri_string(&item.content)?)
    } else {
        None
    };
    if let Some(uri) = media_uri.as_deref() {
        upsert_media_asset(tx, uri).await?;
    }
    let context_item_id: i64 = sqlx::query_scalar(
        "INSERT INTO turn_attempt_context_items \
           (attempt_id, ordinal, source, role, content, message_provider, channel, message, media_uri) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) \
         RETURNING id",
    )
    .bind(attempt_id)
    .bind(item.position)
    .bind(&item.source)
    .bind(&item.role)
    .bind(&item.content)
    .bind(provider)
    .bind(channel)
    .bind(message)
    .bind(media_uri.as_deref())
    .fetch_one(&mut **tx)
    .await?;
    if let Some(uri) = media_uri.as_deref() {
        insert_turn_asset(
            tx,
            TurnAssetInsert {
                turn_id,
                attempt_id: Some(attempt_id),
                uri,
                source: &item.source,
                context_item_id: Some(context_item_id),
                tool_trace_id: None,
                ordinal: item.position,
            },
        )
        .await?;
    }
    Ok(())
}

struct TurnAssetInsert<'a> {
    turn_id: TurnId,
    attempt_id: Option<Uuid>,
    uri: &'a str,
    source: &'a str,
    context_item_id: Option<i64>,
    tool_trace_id: Option<i64>,
    ordinal: i32,
}

async fn insert_turn_asset(
    tx: &mut Transaction<'_, Postgres>,
    asset: TurnAssetInsert<'_>,
) -> Result<(), SqlxStorageError> {
    let uri = canonical_media_uri_string(asset.uri)?;
    upsert_media_asset(tx, &uri).await?;
    sqlx::query(
        "INSERT INTO turn_assets \
           (turn_id, attempt_id, media_uri, source, replayable, context_item_id, tool_trace_id, ordinal) \
         VALUES ($1, $2, $3, $4, true, $5, $6, $7) \
         ON CONFLICT (turn_id, media_uri, source) DO UPDATE \
           SET attempt_id = COALESCE(EXCLUDED.attempt_id, turn_assets.attempt_id), \
               context_item_id = COALESCE(EXCLUDED.context_item_id, turn_assets.context_item_id), \
               tool_trace_id = COALESCE(EXCLUDED.tool_trace_id, turn_assets.tool_trace_id), \
               ordinal = EXCLUDED.ordinal, replayable = true",
    )
    .bind(asset.turn_id.0)
    .bind(asset.attempt_id)
    .bind(uri.as_str())
    .bind(asset.source)
    .bind(asset.context_item_id)
    .bind(asset.tool_trace_id)
    .bind(asset.ordinal)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn insert_transcript(
    tx: &mut Transaction<'_, Postgres>,
    attempt_id: Uuid,
    transcript: chudbot_api::Transcript,
) -> Result<(), SqlxStorageError> {
    for (turn_ordinal, turn) in transcript.turns.into_iter().enumerate() {
        let role = match turn.role {
            TurnRole::User => "user",
            TurnRole::Assistant => "assistant",
        };
        let message_id: i64 = sqlx::query_scalar(
            "INSERT INTO turn_attempt_input_messages (attempt_id, ordinal, role, metadata) \
             VALUES ($1, $2, $3, $4) RETURNING id",
        )
        .bind(attempt_id)
        .bind(i32::try_from(turn_ordinal).unwrap_or(i32::MAX))
        .bind(role)
        .bind(turn.metadata)
        .fetch_one(&mut **tx)
        .await?;
        for (block_ordinal, block) in turn.blocks.into_iter().enumerate() {
            insert_input_block(
                tx,
                message_id,
                i32::try_from(block_ordinal).unwrap_or(i32::MAX),
                block,
            )
            .await?;
        }
    }
    Ok(())
}

async fn insert_input_block(
    tx: &mut Transaction<'_, Postgres>,
    input_message_id: i64,
    ordinal: i32,
    block: chudbot_api::ContentBlock,
) -> Result<(), SqlxStorageError> {
    let (kind, text, media_uri, payload) = match block {
        chudbot_api::ContentBlock::Text { text } => ("text", Some(text), None, Value::Null),
        chudbot_api::ContentBlock::Media { media } => {
            let mut metadata = media.metadata().clone();
            metadata.uri = canonical_media_uri(&metadata.uri)?;
            let uri = metadata.uri.to_string();
            upsert_media_asset(tx, &uri).await?;
            ("media", None, Some(uri), serde_json::to_value(metadata)?)
        }
        chudbot_api::ContentBlock::ClientToolCall(call) => {
            ("client_tool_call", None, None, serde_json::to_value(call)?)
        }
        chudbot_api::ContentBlock::ClientToolResult(result) => (
            "client_tool_result",
            None,
            None,
            serde_json::to_value(result)?,
        ),
        chudbot_api::ContentBlock::Continuation(continuation) => (
            "continuation",
            None,
            None,
            serde_json::to_value(continuation)?,
        ),
    };
    sqlx::query(
        "INSERT INTO turn_attempt_input_blocks \
           (input_message_id, ordinal, block_kind, text_content, media_uri, payload) \
         VALUES ($1, $2, $3, $4, $5, $6)",
    )
    .bind(input_message_id)
    .bind(ordinal)
    .bind(kind)
    .bind(text)
    .bind(media_uri)
    .bind(payload)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn upsert_media_asset(
    tx: &mut Transaction<'_, Postgres>,
    uri: &str,
) -> Result<(), SqlxStorageError> {
    let uri = canonical_media_uri_string(uri)?;
    let (category, name, mime) = media_parts(&uri);
    sqlx::query(
        "INSERT INTO media_assets (uri, category, name, mime_type, size_bytes) \
         VALUES ($1, $2, $3, $4, 0) ON CONFLICT DO NOTHING",
    )
    .bind(uri.as_str())
    .bind(category)
    .bind(name)
    .bind(mime)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn conversation_for_turn(
    tx: &mut Transaction<'_, Postgres>,
    turn_id: TurnId,
) -> Result<ConversationId, SqlxStorageError> {
    let id = sqlx::query_scalar("SELECT conversation_id FROM turns WHERE id = $1")
        .bind(turn_id.0)
        .fetch_one(&mut **tx)
        .await?;
    Ok(ConversationId(id))
}

async fn update_latest_attempt(
    tx: &mut Transaction<'_, Postgres>,
    turn_id: TurnId,
    status: &str,
    assistant_message: Option<&MessageRef>,
    assistant_content: Option<&str>,
    error: Option<&str>,
) -> Result<(), SqlxStorageError> {
    let attempt_id: Uuid = sqlx::query_scalar(
        "SELECT id FROM turn_attempts WHERE turn_id = $1 ORDER BY attempt_ordinal DESC LIMIT 1",
    )
    .bind(turn_id.0)
    .fetch_one(&mut **tx)
    .await?;
    sqlx::query(
        "UPDATE turn_attempts \
            SET status = $2, completed_at = now(), assistant_message_provider = $3, \
                assistant_message_channel = $4, assistant_message = $5, assistant_content = $6, \
                error = $7 \
          WHERE id = $1",
    )
    .bind(attempt_id)
    .bind(status)
    .bind(assistant_message.map(|m| m.platform.as_str()))
    .bind(assistant_message.map(channel_key_from_message))
    .bind(assistant_message.map(|m| m.message_id.as_str()))
    .bind(assistant_content)
    .bind(error)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn upsert_message_link(
    tx: &mut Transaction<'_, Postgres>,
    message: &MessageRef,
    conversation_id: ConversationId,
    turn_id: TurnId,
    attempt_id: Option<Uuid>,
    role: &str,
) -> Result<(), SqlxStorageError> {
    sqlx::query(
        "INSERT INTO message_links \
           (message_provider, channel, message, conversation_id, turn_id, attempt_id, role) \
         VALUES ($1, $2, $3, $4, $5, $6, $7) \
         ON CONFLICT (message_provider, channel, message) DO UPDATE \
           SET conversation_id = EXCLUDED.conversation_id, turn_id = EXCLUDED.turn_id, \
               attempt_id = EXCLUDED.attempt_id, role = EXCLUDED.role",
    )
    .bind(message.platform.as_str())
    .bind(channel_key_from_message(message))
    .bind(message.message_id.as_str())
    .bind(conversation_id.0)
    .bind(turn_id.0)
    .bind(attempt_id)
    .bind(role)
    .execute(&mut **tx)
    .await?;
    Ok(())
}

async fn insert_usage(
    tx: &mut Transaction<'_, Postgres>,
    conversation_id: ConversationId,
    turn_id: Option<TurnId>,
    records: Vec<UsageRecord>,
) -> Result<(), SqlxStorageError> {
    let attempt_id: Option<Uuid> = match turn_id {
        Some(turn_id) => sqlx::query_scalar(
            "SELECT id FROM turn_attempts WHERE turn_id = $1 ORDER BY attempt_ordinal DESC LIMIT 1",
        )
        .bind(turn_id.0)
        .fetch_optional(&mut **tx)
        .await?,
        None => None,
    };
    for record in records {
        let (subject_kind, subject_name) = usage_subject(&record.subject);
        let raw = serde_json::to_value(&record)?;
        sqlx::query(
            "INSERT INTO usage_records \
               (conversation_id, turn_id, attempt_id, provider, model, subject_kind, subject_name, \
                input_tokens, cached_tokens, output_tokens, reasoning_tokens, total_tokens, \
                cost_amount, cost_unit, cost_estimated, raw) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)",
        )
        .bind(conversation_id.0)
        .bind(turn_id.map(|id| id.0))
        .bind(attempt_id)
        .bind(record.provider.as_str())
        .bind(record.model.as_ref().map(ModelId::as_str))
        .bind(subject_kind)
        .bind(subject_name)
        .bind(record.input_tokens.and_then(|v| i64::try_from(v).ok()))
        .bind(
            record
                .cached_input_tokens
                .and_then(|v| i64::try_from(v).ok()),
        )
        .bind(record.output_tokens.and_then(|v| i64::try_from(v).ok()))
        .bind(record.reasoning_tokens.and_then(|v| i64::try_from(v).ok()))
        .bind(record.total_tokens.and_then(|v| i64::try_from(v).ok()))
        .bind(record.cost.as_ref().map(|cost| cost.amount.as_str()))
        .bind(record.cost.as_ref().map(|cost| cost.unit.as_str()))
        .bind(record.cost.as_ref().map(|cost| cost.estimated))
        .bind(raw)
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

fn memory_event_kind_as_str(kind: UserMemoryEventKind) -> &'static str {
    match kind {
        UserMemoryEventKind::Remember => "remember",
        UserMemoryEventKind::Correction => "correction",
        UserMemoryEventKind::Forget => "forget",
        UserMemoryEventKind::DiaryObservation => "diary_observation",
        UserMemoryEventKind::OperatorNote => "operator_note",
    }
}

fn memory_event_kind_from_str(kind: &str) -> Result<UserMemoryEventKind, SqlxStorageError> {
    match kind {
        "remember" => Ok(UserMemoryEventKind::Remember),
        "correction" => Ok(UserMemoryEventKind::Correction),
        "forget" => Ok(UserMemoryEventKind::Forget),
        "diary_observation" => Ok(UserMemoryEventKind::DiaryObservation),
        "operator_note" => Ok(UserMemoryEventKind::OperatorNote),
        other => Err(SqlxStorageError::InvalidReference(format!(
            "unknown user memory event kind `{other}`"
        ))),
    }
}

fn memory_job_kind_from_str(kind: &str) -> Result<MemoryJobKind, SqlxStorageError> {
    match kind {
        "diary" => Ok(MemoryJobKind::Diary),
        "compact" => Ok(MemoryJobKind::Compact),
        other => Err(SqlxStorageError::InvalidReference(format!(
            "unknown user memory job kind `{other}`"
        ))),
    }
}

fn user_memory_key(provider: String, scope_key: String, user_key: String) -> UserMemoryKey {
    UserMemoryKey {
        platform: PlatformName::new(provider),
        scope_key,
        user_key,
    }
}

fn memory_event_from_row(row: sqlx::postgres::PgRow) -> Result<UserMemoryEvent, SqlxStorageError> {
    let tags = serde_json::from_value(row.get::<Value, _>("tags"))?;
    let kind = memory_event_kind_from_str(&row.get::<String, _>("kind"))?;
    Ok(UserMemoryEvent {
        id: row.get("id"),
        key: user_memory_key(
            row.get("message_provider"),
            row.get("scope_key"),
            row.get("subject_user_key"),
        ),
        actor_user_key: row.get("actor_user_key"),
        kind,
        body: row.get("body"),
        tags,
        confidence: row.get("confidence"),
        source_conversation_id: row
            .get::<Option<Uuid>, _>("source_conversation_id")
            .map(ConversationId),
        source_turn_id: row.get::<Option<Uuid>, _>("source_turn_id").map(TurnId),
        source_tool_trace_id: row.get("source_tool_trace_id"),
        supersedes_event_id: row.get("supersedes_event_id"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}

fn diary_entry_from_row(
    row: sqlx::postgres::PgRow,
) -> Result<UserMemoryDiaryEntry, SqlxStorageError> {
    let source_turn_ids = row
        .get::<Vec<Uuid>, _>("source_turn_ids")
        .into_iter()
        .map(TurnId)
        .collect();
    let usage = serde_json::from_value(row.get::<Value, _>("usage"))?;
    Ok(UserMemoryDiaryEntry {
        id: row.get("id"),
        key: user_memory_key(
            row.get("message_provider"),
            row.get("scope_key"),
            row.get("subject_user_key"),
        ),
        window_start: row.get("window_start"),
        window_end: row.get("window_end"),
        source_turn_ids,
        markdown: row.get("markdown"),
        agent_name: row.get("agent_name"),
        llm_provider: ProviderName::new(row.get::<String, _>("llm_provider")),
        llm_model: ModelId::new(row.get::<String, _>("llm_model")),
        usage,
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}

fn document_from_row(row: sqlx::postgres::PgRow) -> Result<UserMemoryDocument, SqlxStorageError> {
    Ok(UserMemoryDocument {
        key: user_memory_key(
            row.get("message_provider"),
            row.get("scope_key"),
            row.get("subject_user_key"),
        ),
        revision: row.get("revision"),
        markdown: row.get("markdown"),
        last_compacted_at: row.get("last_compacted_at"),
        source_event_cutoff: row.get("source_event_cutoff"),
        source_diary_cutoff: row.get("source_diary_cutoff"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}

fn memory_job_from_row(row: sqlx::postgres::PgRow) -> Result<UserMemoryJob, SqlxStorageError> {
    let kind = memory_job_kind_from_str(&row.get::<String, _>("kind"))?;
    Ok(UserMemoryJob {
        id: row.get("id"),
        kind,
        key: user_memory_key(
            row.get("message_provider"),
            row.get("scope_key"),
            row.get("subject_user_key"),
        ),
        memory_key: row.get("memory_key"),
        window_start: row.get("window_start"),
        window_end: row.get("window_end"),
        attempts: row.get("attempts"),
        leased_by: row.get("leased_by"),
        leased_until: row.get("leased_until"),
        dedupe_key: row.get("dedupe_key"),
    })
}

fn conversation_from_row(row: sqlx::postgres::PgRow) -> Result<Conversation, SqlxStorageError> {
    let provider: String = row.get("message_provider");
    let channel: String = row.get("channel");
    let stopped_by_provider: Option<String> = row.get("stopped_by_provider");
    let stopped_by_user_key: Option<String> = row.get("stopped_by_user_key");
    Ok(Conversation {
        id: ConversationId(row.get("id")),
        created_at: row.get("created_at"),
        channel: channel_ref(&provider, &channel)?,
        created_by: user_ref(
            &provider,
            &channel,
            row.get::<String, _>("created_by_user_key"),
        ),
        root_message: message_ref(
            &row.get::<String, _>("root_message_provider"),
            &row.get::<String, _>("root_message_channel"),
            row.get::<String, _>("root_message"),
        )?,
        initial_model: ModelId::new(row.get::<String, _>("llm_model")),
        agent_name: row.get("agent_name"),
        provider: ProviderName::new(row.get::<String, _>("llm_provider")),
        system_instructions: row.get("system_instructions"),
        title: row.get("title"),
        stopped_at: row.get("stopped_at"),
        stopped_by: stopped_by_provider
            .zip(stopped_by_user_key)
            .map(|(provider, user)| user_ref(&provider, &channel, user)),
    })
}

fn app_version_from_row(row: sqlx::postgres::PgRow) -> AppVersion {
    AppVersion {
        id: row.get("id"),
        git_version: row.get("git_version"),
        first_seen_at: row.get("first_seen_at"),
    }
}

fn turn_from_row(row: &sqlx::postgres::PgRow) -> Result<Turn, SqlxStorageError> {
    let status: String = row.get("status");
    Ok(Turn {
        id: TurnId(row.get("id")),
        ordinal: row.get("ordinal"),
        history_cutoff: row.get("history_cutoff"),
        response_ordinal: row.get("response_ordinal"),
        created_at: row.get("created_at"),
        user_message_created_at: row.get("user_message_created_at"),
        completed_at: row.get("completed_at"),
        user_message: message_ref(
            &row.get::<String, _>("user_message_provider"),
            &row.get::<String, _>("user_message_channel"),
            row.get::<String, _>("user_message"),
        )?,
        user: user_ref(
            &row.get::<String, _>("user_message_provider"),
            &row.get::<String, _>("user_message_channel"),
            row.get::<String, _>("user_key"),
        ),
        user_display_name: row.get("user_display_name"),
        user_content: row.get("user_content"),
        assistant_message: optional_message_ref(
            row.get("assistant_message_provider"),
            row.get("assistant_message_channel"),
            row.get("assistant_message"),
        )?,
        assistant_content: row.get("assistant_content"),
        status: status_from_str(&status),
        error: row.get("error"),
        agent_name: row.get("agent_name"),
        provider: row
            .get::<Option<String>, _>("llm_provider")
            .map(ProviderName::new),
        model: row.get::<Option<String>, _>("llm_model").map(ModelId::new),
        app_version_id: row.get("app_version_id"),
    })
}

fn model_step_from_row(row: sqlx::postgres::PgRow) -> Result<ModelStepTrace, SqlxStorageError> {
    let kind: String = row.get("step_kind");
    let continuation = row
        .get::<Option<Value>, _>("continuation")
        .map(serde_json::from_value)
        .transpose()?;
    Ok(ModelStepTrace {
        ordinal: row.get("ordinal"),
        kind: model_step_kind_from_str(&kind)?,
        provider: ProviderName::new(row.get::<String, _>("llm_provider")),
        model: ModelId::new(row.get::<String, _>("llm_model")),
        continuation,
    })
}

fn status_from_str(status: &str) -> TurnStatus {
    match status {
        "completed" => TurnStatus::Completed,
        "failed" => TurnStatus::Failed,
        "cancelled" => TurnStatus::Cancelled,
        _ => TurnStatus::Pending,
    }
}

fn model_step_kind_from_str(kind: &str) -> Result<ModelStepKind, SqlxStorageError> {
    match kind {
        "final" => Ok(ModelStepKind::Final),
        "continue" => Ok(ModelStepKind::Continue),
        "client_tools" => Ok(ModelStepKind::ClientTools),
        other => Err(SqlxStorageError::InvalidModelStepKind(other.to_string())),
    }
}

fn model_step_kind_str(kind: ModelStepKind) -> &'static str {
    match kind {
        ModelStepKind::Final => "final",
        ModelStepKind::Continue => "continue",
        ModelStepKind::ClientTools => "client_tools",
    }
}

fn optional_json<T>(value: &Option<T>) -> Result<Option<Value>, SqlxStorageError>
where
    T: serde::Serialize,
{
    value
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(SqlxStorageError::Json)
}

fn optional_message_ref(
    provider: Option<String>,
    channel: Option<String>,
    message: Option<String>,
) -> Result<Option<MessageRef>, SqlxStorageError> {
    match (provider, channel, message) {
        (Some(provider), Some(channel), Some(message)) => {
            message_ref(&provider, &channel, message).map(Some)
        }
        _ => Ok(None),
    }
}

fn message_ref(
    provider: &str,
    channel: &str,
    message: String,
) -> Result<MessageRef, SqlxStorageError> {
    let channel = channel_ref(provider, channel)?;
    Ok(MessageRef {
        platform: channel.platform,
        guild_id: channel.guild_id,
        channel_id: channel.channel_id,
        message_id: ExternalId::new(message),
    })
}

fn channel_ref(provider: &str, channel: &str) -> Result<ChannelRef, SqlxStorageError> {
    let parts = channel.split(':').collect::<Vec<_>>();
    if parts.len() >= 4 && parts[0] == "guild" && parts[2] == "channel" {
        let channel_id = if parts.len() >= 6 && parts[4] == "thread" {
            parts[5]
        } else {
            parts[3]
        };
        return Ok(ChannelRef {
            platform: PlatformName::new(provider),
            guild_id: Some(ExternalId::new(parts[1])),
            channel_id: ExternalId::new(channel_id),
        });
    }
    if parts.len() == 2 && parts[0] == "channel" {
        return Ok(ChannelRef {
            platform: PlatformName::new(provider),
            guild_id: None,
            channel_id: ExternalId::new(parts[1]),
        });
    }
    Ok(ChannelRef {
        platform: PlatformName::new(provider),
        guild_id: None,
        channel_id: ExternalId::new(channel),
    })
}

fn user_ref(provider: &str, channel: &str, user_key: String) -> UserRef {
    let guild_id = channel
        .strip_prefix("guild:")
        .and_then(|rest| rest.split(':').next())
        .map(ExternalId::new);
    UserRef {
        platform: PlatformName::new(provider),
        guild_id,
        user_id: ExternalId::new(user_key),
    }
}

fn channel_from_message(message: &MessageRef) -> ChannelRef {
    ChannelRef {
        platform: message.platform.clone(),
        guild_id: message.guild_id.clone(),
        channel_id: message.channel_id.clone(),
    }
}

fn channel_key_from_message(message: &MessageRef) -> String {
    channel_key(&channel_from_message(message))
}

fn channel_key(channel: &ChannelRef) -> String {
    match &channel.guild_id {
        Some(guild) => format!(
            "guild:{}:channel:{}",
            guild.as_str(),
            channel.channel_id.as_str()
        ),
        None => format!("channel:{}", channel.channel_id.as_str()),
    }
}

fn guild_scope(guild: &str) -> String {
    format!("guild:{guild}")
}

fn selection_channel_key(guild: Option<&str>, channel: &str) -> String {
    if let Some(guild) = guild {
        return format!("guild:{guild}:channel:{channel}");
    }
    format!("channel:{channel}")
}

fn canonical_media_uri(uri: &MediaUri) -> Result<MediaUri, SqlxStorageError> {
    canonical_stored_media_uri(uri)
        .map_err(|_| SqlxStorageError::InvalidMediaUri(uri.as_str().to_string()))
}

fn canonical_media_uri_string(uri: &str) -> Result<String, SqlxStorageError> {
    canonical_media_uri(&MediaUri::new(uri)).map(|uri| uri.to_string())
}

fn media_parts(uri: &str) -> (&'static str, String, &'static str) {
    let Ok(parsed) = parse_stored_media_uri(&MediaUri::new(uri)) else {
        return ("other", uri.to_string(), "application/octet-stream");
    };
    match parsed.category {
        MediaCategory::Image => ("image", parsed.name, "image/png"),
        MediaCategory::Video => ("video", parsed.name, "video/mp4"),
        MediaCategory::Audio => ("audio", parsed.name, "audio/ogg"),
        MediaCategory::Avatar => ("avatar", parsed.name, "image/png"),
        MediaCategory::GuildIcon => ("guild_icon", parsed.name, "image/png"),
        MediaCategory::Other(_) => ("other", parsed.name, "application/octet-stream"),
    }
}

/// Common row shape for turn usage and background memory-job usage.
///
/// `chan` is the storage channel key (`guild:<g>:channel:<c>` or
/// `channel:<c>`); memory rows use the `memory` sentinel so channel-scoped
/// queries exclude them while guild/user groupings still attribute them.
const USAGE_COST_SOURCE_SQL: &str = "\
    SELECT c.message_provider AS message_provider, \
           CASE WHEN COALESCE(t.user_message_channel, c.channel) LIKE 'guild:%:channel:%' \
                THEN split_part(COALESCE(t.user_message_channel, c.channel), ':', 2) \
           END AS guild_key, \
           COALESCE(t.user_message_channel, c.channel) AS chan, \
           COALESCE(t.user_key, c.created_by_user_key) AS user_key, \
           COALESCE(a.agent_name, c.agent_name) AS agent_name, \
           u.provider AS provider, \
           u.model AS model, \
           COALESCE(u.subject_kind || ':' || u.subject_name, u.subject_kind) AS subject, \
           u.input_tokens AS input_tokens, \
           u.cached_tokens AS cached_tokens, \
           u.output_tokens AS output_tokens, \
           u.reasoning_tokens AS reasoning_tokens, \
           u.total_tokens AS total_tokens, \
           u.cost_amount AS cost_amount, \
           u.cost_unit AS cost_unit, \
           u.cost_estimated AS cost_estimated, \
           u.conversation_id AS conversation_id, \
           u.turn_id AS turn_id, \
           u.created_at AS created_at \
      FROM usage_records u \
      JOIN conversations c ON c.id = u.conversation_id \
      LEFT JOIN turns t ON t.id = u.turn_id \
      LEFT JOIN turn_attempts a ON a.id = u.attempt_id \
    UNION ALL \
    SELECT m.message_provider, \
           CASE WHEN m.scope_key LIKE 'guild:%' THEN substr(m.scope_key, 7) END, \
           'memory', \
           m.subject_user_key, \
           m.agent_name, \
           rec.value ->> 'provider', \
           rec.value ->> 'model', \
           m.subject, \
           (rec.value ->> 'input_tokens')::bigint, \
           (rec.value ->> 'cached_input_tokens')::bigint, \
           (rec.value ->> 'output_tokens')::bigint, \
           (rec.value ->> 'reasoning_tokens')::bigint, \
           (rec.value ->> 'total_tokens')::bigint, \
           rec.value -> 'cost' ->> 'amount', \
           rec.value -> 'cost' ->> 'unit', \
           (rec.value -> 'cost' ->> 'estimated')::boolean, \
           NULL::uuid, \
           NULL::uuid, \
           m.created_at \
      FROM ( \
           SELECT de.message_provider, de.scope_key, de.subject_user_key, de.agent_name, \
                  de.usage, de.created_at, 'memory_diary' AS subject \
             FROM user_memory_diary_entries de \
           UNION ALL \
           SELECT dv.message_provider, dv.scope_key, dv.subject_user_key, dv.agent_name, \
                  dv.usage, dv.created_at, 'memory_compact' \
             FROM user_memory_document_versions dv \
           ) m \
      CROSS JOIN LATERAL jsonb_array_elements(m.usage) rec(value)";

fn usage_cost_report_sql(group_by: UsageCostGrouping) -> String {
    const GROUPED: &str = "GROUP BY 1 ORDER BY cost_usd_numeric DESC NULLS LAST, records DESC";
    let (key, label, label_join, group_order) = match group_by {
        UsageCostGrouping::Total => ("NULL::text", "MAX(NULL::text)", "", ""),
        UsageCostGrouping::Guild => (
            "COALESCE(src.guild_key, 'direct')",
            "MAX(NULL::text)",
            "",
            GROUPED,
        ),
        UsageCostGrouping::Channel => ("src.chan", "MAX(NULL::text)", "", GROUPED),
        UsageCostGrouping::User => (
            "src.user_key",
            "MAX(COALESCE(pu.display_name, pu.username))",
            "LEFT JOIN platform_users pu \
               ON pu.message_provider = src.message_provider AND pu.user_key = src.user_key ",
            GROUPED,
        ),
        UsageCostGrouping::Agent => ("src.agent_name", "MAX(NULL::text)", "", GROUPED),
        UsageCostGrouping::Provider => ("src.provider", "MAX(NULL::text)", "", GROUPED),
        UsageCostGrouping::Model => (
            "COALESCE(src.provider || '/' || src.model, src.provider)",
            "MAX(NULL::text)",
            "",
            GROUPED,
        ),
        UsageCostGrouping::Kind => ("src.subject", "MAX(NULL::text)", "", GROUPED),
    };
    format!(
        "SELECT {key} AS key, \
                {label} AS label, \
                COUNT(*) AS records, \
                COUNT(DISTINCT src.conversation_id) AS conversations, \
                COUNT(DISTINCT src.turn_id) AS turns, \
                COALESCE(SUM(src.input_tokens), 0)::bigint AS input_tokens, \
                COALESCE(SUM(src.cached_tokens), 0)::bigint AS cached_tokens, \
                COALESCE(SUM(src.output_tokens), 0)::bigint AS output_tokens, \
                COALESCE(SUM(src.reasoning_tokens), 0)::bigint AS reasoning_tokens, \
                COALESCE(SUM(src.total_tokens), 0)::bigint AS total_tokens, \
                SUM(CASE src.cost_unit \
                    WHEN 'usd_ticks' THEN src.cost_amount::numeric / 10000000000::numeric \
                    WHEN 'usd' THEN src.cost_amount::numeric \
                END) AS cost_usd_numeric, \
                trim_scale(round(SUM(CASE src.cost_unit \
                    WHEN 'usd_ticks' THEN src.cost_amount::numeric / 10000000000::numeric \
                    WHEN 'usd' THEN src.cost_amount::numeric \
                END), 6))::text AS cost_usd, \
                COALESCE(BOOL_OR(src.cost_estimated) \
                    FILTER (WHERE src.cost_unit IN ('usd', 'usd_ticks')), false) AS cost_estimated, \
                COUNT(*) FILTER (WHERE src.cost_amount IS NULL OR src.cost_unit IS NULL \
                    OR src.cost_unit NOT IN ('usd', 'usd_ticks')) AS unpriced_records \
           FROM ({USAGE_COST_SOURCE_SQL}) src \
           {label_join}\
          WHERE src.message_provider = $1 \
            AND ($2::text IS NULL OR src.guild_key = $2) \
            AND ($3::text IS NULL OR src.chan = $3) \
            AND ($4::timestamptz IS NULL OR src.created_at >= $4) \
          {group_order} \
          LIMIT $5"
    )
}

fn usage_cost_row_from_row(row: sqlx::postgres::PgRow) -> UsageCostRow {
    fn non_negative(value: i64) -> u64 {
        u64::try_from(value).unwrap_or(0)
    }
    UsageCostRow {
        key: row.get("key"),
        label: row.get("label"),
        records: non_negative(row.get("records")),
        conversations: non_negative(row.get("conversations")),
        turns: non_negative(row.get("turns")),
        input_tokens: non_negative(row.get("input_tokens")),
        cached_input_tokens: non_negative(row.get("cached_tokens")),
        output_tokens: non_negative(row.get("output_tokens")),
        reasoning_tokens: non_negative(row.get("reasoning_tokens")),
        total_tokens: non_negative(row.get("total_tokens")),
        cost_usd: row.get("cost_usd"),
        cost_estimated: row.get("cost_estimated"),
        unpriced_records: non_negative(row.get("unpriced_records")),
    }
}

fn usage_subject(subject: &UsageSubject) -> (&'static str, Option<String>) {
    match subject {
        UsageSubject::ModelStep => ("model_step", None),
        UsageSubject::ServerTool { name } => ("server_tool", Some(name.to_string())),
        UsageSubject::ClientTool { name } => ("client_tool", Some(name.to_string())),
        UsageSubject::SubAgent { name } => ("sub_agent", Some(name.to_string())),
        UsageSubject::ImageGeneration => ("image_generation", None),
        UsageSubject::VideoGeneration => ("video_generation", None),
        UsageSubject::AudioTranscription => ("audio_transcription", None),
    }
}

fn display_name_for_user(user: &UserProfile) -> String {
    user.display_name
        .clone()
        .or_else(|| user.name.clone())
        .unwrap_or_else(|| user.username.clone())
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use test_case::test_case;

    use super::*;

    #[test_case(UsageCostGrouping::Total, "NULL::text AS key" ; "total key is null")]
    #[test_case(UsageCostGrouping::Guild, "COALESCE(src.guild_key, 'direct') AS key" ; "guild key")]
    #[test_case(UsageCostGrouping::Channel, "src.chan AS key" ; "channel key")]
    #[test_case(UsageCostGrouping::User, "src.user_key AS key" ; "user key")]
    #[test_case(UsageCostGrouping::Agent, "src.agent_name AS key" ; "agent key")]
    #[test_case(UsageCostGrouping::Provider, "src.provider AS key" ; "provider key")]
    #[test_case(
        UsageCostGrouping::Model,
        "COALESCE(src.provider || '/' || src.model, src.provider) AS key" ; "model key"
    )]
    #[test_case(UsageCostGrouping::Kind, "src.subject AS key" ; "kind key")]
    fn usage_cost_report_sql_selects_group_key(group_by: UsageCostGrouping, key_expr: &str) {
        let sql = usage_cost_report_sql(group_by);
        assert!(sql.starts_with(&format!("SELECT {key_expr}")), "{sql}");
    }

    #[test_case(UsageCostGrouping::Total, false ; "total is ungrouped")]
    #[test_case(UsageCostGrouping::Guild, true ; "guild is grouped")]
    #[test_case(UsageCostGrouping::User, true ; "user is grouped")]
    fn usage_cost_report_sql_groups_when_keyed(group_by: UsageCostGrouping, grouped: bool) {
        let sql = usage_cost_report_sql(group_by);
        assert_eq!(sql.contains("GROUP BY 1"), grouped, "{sql}");
        assert_eq!(
            sql.contains("ORDER BY cost_usd_numeric DESC"),
            grouped,
            "{sql}"
        );
    }

    #[test]
    fn usage_cost_report_sql_joins_user_labels_only_for_user_grouping() {
        let user_sql = usage_cost_report_sql(UsageCostGrouping::User);
        assert!(
            user_sql.contains("LEFT JOIN platform_users pu"),
            "{user_sql}"
        );
        let guild_sql = usage_cost_report_sql(UsageCostGrouping::Guild);
        assert!(!guild_sql.contains("platform_users"), "{guild_sql}");
    }

    #[test]
    fn usage_cost_report_sql_binds_all_filters() {
        let sql = usage_cost_report_sql(UsageCostGrouping::Total);
        for filter in [
            "src.message_provider = $1",
            "$2::text IS NULL OR src.guild_key = $2",
            "$3::text IS NULL OR src.chan = $3",
            "$4::timestamptz IS NULL OR src.created_at >= $4",
            "LIMIT $5",
        ] {
            assert!(sql.contains(filter), "missing `{filter}` in {sql}");
        }
    }

    #[test]
    fn tool_trace_media_asset_canonicalizes_nested_legacy_client_result_uri() {
        let fields = ToolTraceFields {
            trace_kind: "client",
            tool_name: Some("generate_image".to_string()),
            provider: None,
            tool_use_id: Some("call-1".to_string()),
            is_error: Some(false),
            request: None,
            response: Some(json!({
                "tool_use_id": "call-1",
                "content": {
                    "kind": "json",
                    "value": {
                        "uri": "file://images/generated.png",
                        "public_url": "https://example.invalid/generated.png"
                    }
                },
                "is_error": false
            })),
        };

        assert_eq!(
            tool_trace_media_asset(&fields),
            Some((
                "media://images/generated.png".to_string(),
                "generate_image".to_string()
            ))
        );
    }

    #[test]
    fn tool_trace_media_asset_ignores_public_urls() {
        let fields = ToolTraceFields {
            trace_kind: "client",
            tool_name: Some("generate_video".to_string()),
            provider: None,
            tool_use_id: Some("call-1".to_string()),
            is_error: Some(false),
            request: None,
            response: Some(json!({
                "uri": "https://example.invalid/generated.mp4",
                "public_url": "https://example.invalid/generated.mp4"
            })),
        };

        assert_eq!(tool_trace_media_asset(&fields), None);
    }

    #[test]
    fn memory_audio_transcription_parses_tool_request_and_response() {
        let request = json!({
            "id": "call-1",
            "name": "transcribe_audio",
            "input": {
                "audio_uri": "file://audio/voice.ogg"
            }
        });
        let response = json!({
            "text": "I am allergic to coconut.",
            "language": "en",
            "duration_seconds": 3.25,
            "words": [
                {
                    "text": "Ignore",
                    "start": 0.0,
                    "end": 0.25
                }
            ]
        });

        let transcription =
            memory_audio_transcription_from_values(42, Some(&request), &response).unwrap();

        assert_eq!(transcription.tool_trace_id, 42);
        assert_eq!(
            transcription.audio_uri.as_deref(),
            Some("media://audio/voice.ogg")
        );
        assert_eq!(transcription.text, "I am allergic to coconut.");
        assert_eq!(transcription.language.as_deref(), Some("en"));
        assert_eq!(transcription.duration_seconds, Some(3.25));
    }
}
