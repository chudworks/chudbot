//! Conversation snapshots reconstructed from stored turns and attempts.

use std::collections::BTreeMap;

use chudbot_api::{
    AGENT_INSTRUCTIONS_PART_KEY_METADATA_KEY, AGENT_INSTRUCTIONS_PART_METADATA_KEY,
    AGENT_INSTRUCTIONS_PART_ORDINAL_METADATA_KEY, AgentInstructionPartSnapshot,
    AgentInstructionSnapshot, ContextItem, Conversation, ConversationId, ConversationSnapshot,
    MediaUri, ModelId, ModelStepKind, ModelStepTrace, ProviderName, ToolTrace, Turn, TurnAsset,
    TurnId, TurnSnapshot, TurnStatus, UsageRecord,
};
use serde_json::Value;
use sqlx::Row;
use uuid::Uuid;

use crate::bot::{channel_ref, message_ref, optional_message_ref, user_ref};
use crate::{SqlxStorage, SqlxStorageError};

#[derive(Debug, Clone, Copy)]
struct PromptStateBoundary {
    turn_id: TurnId,
    turn_ordinal: i64,
    attempt_ordinal: Option<i32>,
}

#[derive(Debug, Clone)]
struct StoredPromptMarker {
    turn_ordinal: i64,
    attempt_ordinal: i32,
    part_key: Option<String>,
    part_ordinal: i32,
    text: String,
}

impl SqlxStorage {
    pub(super) async fn load_snapshot(
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

    pub(super) async fn load_conversation_row(
        &self,
        conversation_id: ConversationId,
    ) -> Result<Option<Conversation>, SqlxStorageError> {
        let row = sqlx::query(
            "SELECT id, created_at, message_provider, channel, created_by_user_key, \
                    root_message_provider, root_message_channel, root_message, agent_name, \
                    llm_provider, llm_model, title, stopped_at, stopped_by_provider, \
                    stopped_by_user_key \
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
                    t.app_version_id, ta.id AS attempt_id, ta.attempt_ordinal AS attempt_ordinal, \
                    ( \
                        SELECT b.text_content \
                          FROM turns prompt_turn \
                          JOIN turn_attempts prompt_attempt \
                            ON prompt_attempt.turn_id = prompt_turn.id \
                          JOIN turn_attempt_input_messages m \
                            ON m.attempt_id = prompt_attempt.id \
                          JOIN turn_attempt_input_blocks b \
                            ON b.input_message_id = m.id \
                         WHERE prompt_turn.conversation_id = t.conversation_id \
                           AND ( \
                               prompt_turn.ordinal < t.ordinal \
                               OR ( \
                                   prompt_turn.ordinal = t.ordinal \
                                   AND ta.attempt_ordinal IS NOT NULL \
                                   AND prompt_attempt.attempt_ordinal <= ta.attempt_ordinal \
                               ) \
                           ) \
                           AND m.role = 'system' \
                           AND m.metadata->>'agent_instructions' = 'true' \
                           AND b.block_kind = 'text' \
                         ORDER BY prompt_turn.ordinal DESC, \
                                  prompt_attempt.attempt_ordinal DESC, \
                                  m.ordinal DESC, \
                                  b.ordinal \
                         LIMIT 1 \
                    ) AS attempt_agent_instruction_text, \
                    ta.agent_name, ta.llm_provider, ta.llm_model \
               FROM turns t \
               LEFT JOIN LATERAL ( \
                    SELECT a.id, a.attempt_ordinal, a.agent_name, a.llm_provider, a.llm_model \
                      FROM turn_attempts a \
                     WHERE a.turn_id = t.id \
                     ORDER BY a.attempt_ordinal DESC \
                     LIMIT 1 \
               ) ta ON true \
              WHERE t.conversation_id = $1 \
              ORDER BY t.ordinal",
        )
        .bind(conversation_id.0)
        .fetch_all(&self.pool)
        .await?;

        struct TurnRow {
            turn_id: TurnId,
            attempt_id: Option<Uuid>,
            turn_ordinal: i64,
            attempt_ordinal: Option<i32>,
            legacy_agent_instruction_text: Option<String>,
            row: sqlx::postgres::PgRow,
        }

        let mut turn_rows = Vec::with_capacity(rows.len());
        for row in rows {
            let turn_id = TurnId(row.get("id"));
            turn_rows.push(TurnRow {
                turn_id,
                attempt_id: row.get("attempt_id"),
                turn_ordinal: row.get("ordinal"),
                attempt_ordinal: row.get("attempt_ordinal"),
                legacy_agent_instruction_text: row.get("attempt_agent_instruction_text"),
                row,
            });
        }

        let prompt_boundaries = turn_rows
            .iter()
            .map(|turn| PromptStateBoundary {
                turn_id: turn.turn_id,
                turn_ordinal: turn.turn_ordinal,
                attempt_ordinal: turn.attempt_ordinal,
            })
            .collect::<Vec<_>>();
        let mut agent_instruction_state_by_turn = self
            .load_stored_agent_instruction_state(conversation_id, &prompt_boundaries)
            .await?;
        let turn_ids = turn_rows
            .iter()
            .map(|turn| turn.turn_id.0)
            .collect::<Vec<_>>();
        let attempt_ids = turn_rows
            .iter()
            .filter_map(|turn| turn.attempt_id)
            .collect::<Vec<_>>();

        let mut context_by_attempt = self.load_context_for_attempts(&attempt_ids).await?;
        let mut tool_trace_by_attempt = self.load_tool_trace_for_attempts(&attempt_ids).await?;
        let mut model_steps_by_attempt = self.load_model_steps_for_attempts(&attempt_ids).await?;
        let mut assets_by_turn = self.load_assets_for_turns(&turn_ids).await?;
        let mut usage_by_turn = self
            .load_usage_for_turns(conversation_id, &turn_ids)
            .await?;

        let mut turns = Vec::with_capacity(turn_rows.len());
        for turn_row in turn_rows {
            let context = turn_row
                .attempt_id
                .and_then(|id| context_by_attempt.remove(&id))
                .unwrap_or_default();
            let tool_trace = turn_row
                .attempt_id
                .and_then(|id| tool_trace_by_attempt.remove(&id))
                .unwrap_or_default();
            let model_steps = turn_row
                .attempt_id
                .and_then(|id| model_steps_by_attempt.remove(&id))
                .unwrap_or_default();
            let replay_assets = assets_by_turn.remove(&turn_row.turn_id).unwrap_or_default();
            let usage = usage_by_turn.remove(&turn_row.turn_id).unwrap_or_default();
            let agent_instructions = agent_instruction_state_by_turn
                .remove(&turn_row.turn_id)
                .flatten()
                .or_else(|| {
                    turn_row
                        .legacy_agent_instruction_text
                        .map(|text| AgentInstructionSnapshot::LegacyText { text })
                });
            turns.push(TurnSnapshot {
                turn: turn_from_row(&turn_row.row)?,
                agent_instructions,
                context,
                tool_trace,
                model_steps,
                replay_assets,
                usage,
            });
        }
        Ok(turns)
    }

    async fn load_stored_agent_instruction_state(
        &self,
        conversation_id: ConversationId,
        prompt_boundaries: &[PromptStateBoundary],
    ) -> Result<BTreeMap<TurnId, Option<AgentInstructionSnapshot>>, SqlxStorageError> {
        if prompt_boundaries.is_empty() {
            return Ok(BTreeMap::new());
        }
        // Agent instruction rows are prompt change markers. A legacy marker
        // replaces the whole prompt; a labeled marker replaces only that part.
        // The reducer below replays those markers in database order to expose
        // the effective prompt state at each requested turn boundary.
        let rows = sqlx::query(
            "SELECT t.ordinal AS turn_ordinal, ta.attempt_ordinal, m.ordinal AS message_ordinal, \
                    m.metadata, prompt.text_content \
               FROM turns t \
               JOIN turn_attempts ta ON ta.turn_id = t.id \
               JOIN turn_attempt_input_messages m ON m.attempt_id = ta.id \
               JOIN LATERAL ( \
                    SELECT b.text_content \
                      FROM turn_attempt_input_blocks b \
                     WHERE b.input_message_id = m.id \
                       AND b.block_kind = 'text' \
                       AND b.text_content IS NOT NULL \
                     ORDER BY b.ordinal \
                     LIMIT 1 \
               ) prompt ON true \
              WHERE t.conversation_id = $1 \
                AND m.role = 'system' \
                AND m.metadata->>'agent_instructions' = 'true' \
              ORDER BY t.ordinal, ta.attempt_ordinal, m.ordinal",
        )
        .bind(conversation_id.0)
        .fetch_all(&self.pool)
        .await?;

        let mut markers = Vec::with_capacity(rows.len());
        for row in rows {
            let metadata: Value = row.get("metadata");
            let part_key = metadata
                .get(AGENT_INSTRUCTIONS_PART_KEY_METADATA_KEY)
                .and_then(Value::as_str)
                .map(str::to_string);
            let part_ordinal = metadata
                .get(AGENT_INSTRUCTIONS_PART_ORDINAL_METADATA_KEY)
                .and_then(Value::as_i64)
                .and_then(|value| i32::try_from(value).ok())
                .unwrap_or_else(|| row.get("message_ordinal"));
            let is_part = metadata
                .get(AGENT_INSTRUCTIONS_PART_METADATA_KEY)
                .and_then(Value::as_bool)
                .unwrap_or(false);
            markers.push(StoredPromptMarker {
                turn_ordinal: row.get("turn_ordinal"),
                attempt_ordinal: row.get("attempt_ordinal"),
                part_key: is_part.then_some(part_key).flatten(),
                part_ordinal,
                text: row.get("text_content"),
            });
        }

        let mut states = BTreeMap::new();
        for boundary in prompt_boundaries {
            states.insert(
                boundary.turn_id,
                stored_agent_instruction_state_at_boundary(&markers, *boundary),
            );
        }
        Ok(states)
    }

    async fn load_context_for_attempts(
        &self,
        attempt_ids: &[Uuid],
    ) -> Result<BTreeMap<Uuid, Vec<ContextItem>>, SqlxStorageError> {
        if attempt_ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        let rows = sqlx::query(
            "SELECT attempt_id, ordinal, source, role, content, message_provider, channel, message \
               FROM turn_attempt_context_items \
              WHERE attempt_id = ANY($1) \
              ORDER BY attempt_id, ordinal",
        )
        .bind(attempt_ids)
        .fetch_all(&self.pool)
        .await?;
        let mut out = BTreeMap::<Uuid, Vec<ContextItem>>::new();
        for row in rows {
            let attempt_id = row.get("attempt_id");
            out.entry(attempt_id).or_default().push(ContextItem {
                position: row.get("ordinal"),
                source: row.get("source"),
                role: row.get("role"),
                content: row.get("content"),
                message: optional_message_ref(
                    row.get::<Option<String>, _>("message_provider"),
                    row.get::<Option<String>, _>("channel"),
                    row.get::<Option<String>, _>("message"),
                )?,
            });
        }
        Ok(out)
    }

    async fn load_tool_trace_for_attempts(
        &self,
        attempt_ids: &[Uuid],
    ) -> Result<BTreeMap<Uuid, Vec<ToolTrace>>, SqlxStorageError> {
        if attempt_ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        let rows = sqlx::query(
            "SELECT attempt_id, trace \
               FROM turn_attempt_tool_traces \
              WHERE attempt_id = ANY($1) \
              ORDER BY attempt_id, ordinal",
        )
        .bind(attempt_ids)
        .fetch_all(&self.pool)
        .await?;
        let mut out = BTreeMap::<Uuid, Vec<ToolTrace>>::new();
        for row in rows {
            let attempt_id = row.get("attempt_id");
            let trace = serde_json::from_value(row.get("trace")).map_err(SqlxStorageError::Json)?;
            out.entry(attempt_id).or_default().push(trace);
        }
        Ok(out)
    }

    async fn load_model_steps_for_attempts(
        &self,
        attempt_ids: &[Uuid],
    ) -> Result<BTreeMap<Uuid, Vec<ModelStepTrace>>, SqlxStorageError> {
        if attempt_ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        let rows = sqlx::query(
            "SELECT attempt_id, ordinal, step_kind, llm_provider, llm_model, continuation \
               FROM turn_attempt_model_steps \
              WHERE attempt_id = ANY($1) \
              ORDER BY attempt_id, ordinal",
        )
        .bind(attempt_ids)
        .fetch_all(&self.pool)
        .await?;
        let mut out = BTreeMap::<Uuid, Vec<ModelStepTrace>>::new();
        for row in rows {
            let attempt_id = row.get("attempt_id");
            let step = model_step_from_row(row)?;
            out.entry(attempt_id).or_default().push(step);
        }
        Ok(out)
    }

    async fn load_assets_for_turns(
        &self,
        turn_ids: &[Uuid],
    ) -> Result<BTreeMap<TurnId, Vec<TurnAsset>>, SqlxStorageError> {
        if turn_ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        let rows = sqlx::query(
            "SELECT a.turn_id, a.media_uri, a.source, m.mime_type \
               FROM turn_assets a \
               LEFT JOIN media_assets m ON m.uri = a.media_uri \
              WHERE a.turn_id = ANY($1) AND a.replayable \
              ORDER BY a.turn_id, a.ordinal, a.id",
        )
        .bind(turn_ids)
        .fetch_all(&self.pool)
        .await?;
        let mut out = BTreeMap::<TurnId, Vec<TurnAsset>>::new();
        for row in rows {
            let turn_id = TurnId(row.get("turn_id"));
            out.entry(turn_id).or_default().push(TurnAsset {
                uri: MediaUri::new(row.get::<String, _>("media_uri")),
                turn_id,
                source: row.get("source"),
                mime_type: row.get("mime_type"),
            });
        }
        Ok(out)
    }

    async fn load_usage_for_turns(
        &self,
        conversation_id: ConversationId,
        turn_ids: &[Uuid],
    ) -> Result<BTreeMap<TurnId, Vec<UsageRecord>>, SqlxStorageError> {
        if turn_ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        let rows = sqlx::query(
            "SELECT turn_id, raw \
               FROM usage_records \
              WHERE conversation_id = $1 \
                AND turn_id = ANY($2) \
              ORDER BY turn_id, id",
        )
        .bind(conversation_id.0)
        .bind(turn_ids)
        .fetch_all(&self.pool)
        .await?;
        let mut out = BTreeMap::<TurnId, Vec<UsageRecord>>::new();
        for row in rows {
            let Some(raw) = row.get::<Option<Value>, _>("raw") else {
                continue;
            };
            let turn_id = TurnId(row.get("turn_id"));
            let usage = serde_json::from_value(raw).map_err(SqlxStorageError::Json)?;
            out.entry(turn_id).or_default().push(usage);
        }
        Ok(out)
    }
}

fn stored_agent_instruction_state_at_boundary(
    markers: &[StoredPromptMarker],
    boundary: PromptStateBoundary,
) -> Option<AgentInstructionSnapshot> {
    let mut legacy_text = None;
    let mut modern_parts = BTreeMap::<String, AgentInstructionPartSnapshot>::new();
    let mut using_modern_parts = false;
    // Replay markers in stored transcript order. Re-inserting a labeled part
    // by key leaves the latest value for that prompt category at this boundary.
    for marker in markers
        .iter()
        .filter(|marker| prompt_marker_is_visible_at_boundary(marker, boundary))
    {
        if let Some(key) = &marker.part_key {
            using_modern_parts = true;
            legacy_text = None;
            modern_parts.insert(
                key.clone(),
                AgentInstructionPartSnapshot {
                    key: key.clone(),
                    ordinal: marker.part_ordinal,
                    text: marker.text.clone(),
                },
            );
        } else {
            using_modern_parts = false;
            modern_parts.clear();
            legacy_text = Some(marker.text.clone());
        }
    }

    if using_modern_parts {
        let mut parts = modern_parts.into_values().collect::<Vec<_>>();
        parts.sort_by_key(|part| part.ordinal);
        Some(AgentInstructionSnapshot::Parts { parts })
    } else {
        legacy_text.map(|text| AgentInstructionSnapshot::LegacyText { text })
    }
}

fn prompt_marker_is_visible_at_boundary(
    marker: &StoredPromptMarker,
    boundary: PromptStateBoundary,
) -> bool {
    marker.turn_ordinal < boundary.turn_ordinal
        || (marker.turn_ordinal == boundary.turn_ordinal
            && boundary
                .attempt_ordinal
                .is_some_and(|attempt_ordinal| marker.attempt_ordinal <= attempt_ordinal))
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
        title: row.get("title"),
        stopped_at: row.get("stopped_at"),
        stopped_by: stopped_by_provider
            .zip(stopped_by_user_key)
            .map(|(provider, user)| user_ref(&provider, &channel, user)),
    })
}
pub(super) fn turn_from_row(row: &sqlx::postgres::PgRow) -> Result<Turn, SqlxStorageError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_agent_instruction_state_keeps_latest_marker_per_part_key() {
        let markers = vec![
            StoredPromptMarker {
                turn_ordinal: 1,
                attempt_ordinal: 0,
                part_key: Some("operator_policy".to_string()),
                part_ordinal: 0,
                text: "policy".to_string(),
            },
            StoredPromptMarker {
                turn_ordinal: 1,
                attempt_ordinal: 0,
                part_key: Some("bot_version".to_string()),
                part_ordinal: 1,
                text: "The version of Chudbot is v1.\n".to_string(),
            },
            StoredPromptMarker {
                turn_ordinal: 2,
                attempt_ordinal: 0,
                part_key: Some("bot_version".to_string()),
                part_ordinal: 1,
                text: "The version of Chudbot was updated to v2.\n".to_string(),
            },
            StoredPromptMarker {
                turn_ordinal: 3,
                attempt_ordinal: 0,
                part_key: Some("bot_version".to_string()),
                part_ordinal: 1,
                text: "The version of Chudbot was updated to v3.\n".to_string(),
            },
        ];

        let state = stored_agent_instruction_state_at_boundary(
            &markers,
            PromptStateBoundary {
                turn_id: TurnId::new(),
                turn_ordinal: 4,
                attempt_ordinal: None,
            },
        );

        let Some(AgentInstructionSnapshot::Parts { parts }) = state else {
            panic!("expected labeled prompt parts");
        };
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].key, "operator_policy");
        assert_eq!(parts[0].text, "policy");
        assert_eq!(parts[1].key, "bot_version");
        assert_eq!(parts[1].text, "The version of Chudbot was updated to v3.\n");
    }
}
