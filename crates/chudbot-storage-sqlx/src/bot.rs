//! Bot storage writes, platform records, and the `BotStorage` implementation.

use chudbot_api::{
    AgentSelection, BeginTurn, BotStorage, ChannelLink, ChannelRef, ContextItem, ConversationId,
    ConversationLookup, ConversationSnapshot, ConversationStop, CountActiveVideoGenerations,
    CreateVideoJob, ExternalId, FinishTurn, GuildProfile, MediaCategory, MediaUri,
    MemoryJobCompletion, MemoryJobSchedule, MemoryTurnWindow, MessageLink, MessageRef, ModelId,
    ModelStepKind, ModelStepTrace, NewUserMemoryDiaryEntry, NewUserMemoryDocumentRevision,
    NewUserMemoryEvent, PlatformName, ResolveAgent, RetryTurn, SaveTurnInput, StoredUserProfile,
    StoredVideoJob, ToolTrace, Turn, TurnId, TurnRole, UpdateVideoJob, UsageCostGrouping,
    UsageCostQuery, UsageCostRow, UsageCostScope, UsageRecord, UsageSubject, UserMemoryDiaryEntry,
    UserMemoryDocument, UserMemoryEvent, UserMemoryJob, UserMemoryKey, UserMemoryTurn, UserProfile,
    UserRef, canonical_stored_media_uri, is_stored_media_uri, parse_stored_media_uri,
};
use serde_json::Value;
use sqlx::{Postgres, Row, Transaction};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::snapshots::turn_from_row;
use crate::{SqlxStorage, SqlxStorageError};

impl SqlxStorage {
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
                title, created_app_version_id) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
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
                app_version_id, explicit_retry_user_key) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(attempt_id)
        .bind(input.turn_id.0)
        .bind(attempt_ordinal)
        .bind(&input.agent_name)
        .bind(input.provider.as_str())
        .bind(input.model.as_str())
        .bind(self.app_version_id)
        .bind(
            input
                .explicit_retry_user
                .as_ref()
                .map(|user| user.user_id.as_str()),
        )
        .execute(&mut *tx)
        .await?;
        for item in input.context {
            insert_context_item(&mut tx, input.turn_id, attempt_id, item).await?;
        }
        insert_transcript(&mut tx, attempt_id, input.transcript).await?;
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
        SqlxStorage::load_user_memory_document(self, key).await
    }

    async fn append_user_memory_event(
        &self,
        event: NewUserMemoryEvent,
    ) -> Result<UserMemoryEvent, Self::Error> {
        SqlxStorage::append_user_memory_event(self, event).await
    }

    async fn list_pending_memory_events(
        &self,
        key: UserMemoryKey,
        since: Option<OffsetDateTime>,
    ) -> Result<Vec<UserMemoryEvent>, Self::Error> {
        SqlxStorage::list_pending_memory_events(self, key, since).await
    }

    async fn list_pending_memory_diary_entries(
        &self,
        key: UserMemoryKey,
        since: Option<OffsetDateTime>,
    ) -> Result<Vec<UserMemoryDiaryEntry>, Self::Error> {
        SqlxStorage::list_pending_memory_diary_entries(self, key, since).await
    }

    async fn list_recent_memory_diary_entries(
        &self,
        key: UserMemoryKey,
        limit: u32,
    ) -> Result<Vec<UserMemoryDiaryEntry>, Self::Error> {
        SqlxStorage::list_recent_memory_diary_entries(self, key, limit).await
    }

    async fn save_user_memory_diary_entry(
        &self,
        entry: NewUserMemoryDiaryEntry,
    ) -> Result<UserMemoryDiaryEntry, Self::Error> {
        SqlxStorage::save_user_memory_diary_entry(self, entry).await
    }

    async fn save_user_memory_document_revision(
        &self,
        document: NewUserMemoryDocumentRevision,
    ) -> Result<UserMemoryDocument, Self::Error> {
        SqlxStorage::save_user_memory_document_revision(self, document).await
    }

    async fn enqueue_due_memory_jobs(
        &self,
        schedule: MemoryJobSchedule,
    ) -> Result<u64, Self::Error> {
        SqlxStorage::enqueue_due_memory_jobs(self, schedule).await
    }

    async fn claim_memory_jobs(
        &self,
        worker_id: String,
        limit: u32,
        lease_until: OffsetDateTime,
    ) -> Result<Vec<UserMemoryJob>, Self::Error> {
        SqlxStorage::claim_memory_jobs(self, worker_id, limit, lease_until).await
    }

    async fn finish_memory_job(&self, completion: MemoryJobCompletion) -> Result<(), Self::Error> {
        SqlxStorage::finish_memory_job(self, completion).await
    }

    async fn load_memory_turn_window(
        &self,
        window: MemoryTurnWindow,
    ) -> Result<Vec<UserMemoryTurn>, Self::Error> {
        SqlxStorage::load_memory_turn_window(self, window).await
    }
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
            TurnRole::System => "system",
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

pub(super) fn optional_message_ref(
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

pub(super) fn message_ref(
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

pub(super) fn channel_ref(provider: &str, channel: &str) -> Result<ChannelRef, SqlxStorageError> {
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

pub(super) fn user_ref(provider: &str, channel: &str, user_key: String) -> UserRef {
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

pub(super) fn canonical_media_uri_string(uri: &str) -> Result<String, SqlxStorageError> {
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
}
