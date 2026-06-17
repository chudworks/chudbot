//! User-memory client tools.

use super::*;

use chudbot_api::{
    NewUserMemoryEvent, UserMemoryDiaryEntry, UserMemoryEvent, UserMemoryEventKind, UserMemoryKey,
};
use time::format_description::well_known::Rfc3339;

pub(crate) use crate::memory::{
    FORGET_USER_MEMORY_TOOL, LOOKUP_USER_MEMORY_TOOL, REMEMBER_USER_MEMORY_TOOL,
};

const LOOKUP_DIARY_ENTRY_LIMIT: u32 = 3;
const EMPTY_MEMORY: &str = "(no stored memory)";

/// Shared context for user-memory client tools during one turn.
#[derive(Debug, Clone)]
pub(crate) struct MemoryToolContext {
    base_key: UserMemoryKey,
    actor_user_key: String,
    actor_display_name: String,
    conversation_id: ConversationId,
    turn_id: TurnId,
}

impl MemoryToolContext {
    pub(crate) fn new(
        base_key: UserMemoryKey,
        actor_display_name: String,
        conversation_id: ConversationId,
        turn_id: TurnId,
    ) -> Self {
        Self {
            actor_user_key: base_key.user_key.clone(),
            base_key,
            actor_display_name,
            conversation_id,
            turn_id,
        }
    }

    fn target_key(&self, input: &serde_json::Value) -> Result<UserMemoryKey, MemoryToolError> {
        let Some(target) = input
            .get("target_user_id")
            .and_then(serde_json::Value::as_str)
            .map(normalize_target_user_id)
            .transpose()?
        else {
            return Ok(self.base_key.clone());
        };
        Ok(UserMemoryKey {
            user_key: target,
            ..self.base_key.clone()
        })
    }
}

pub(crate) fn lookup_user_memory_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Look up the compact remembered profile and recent un-compacted memory events for the current user or another user id in this server.".to_string(),
        input_schema: lookup_schema(),
    }
}

pub(crate) fn remember_user_memory_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Remember a stable preference, relationship, project, correction, recurring fact, or running joke for the current user or a target user id in this server.".to_string(),
        input_schema: remember_schema(),
    }
}

pub(crate) fn forget_user_memory_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Record that a remembered fact should be forgotten or no longer used for the current user or a target user id in this server.".to_string(),
        input_schema: forget_schema(),
    }
}

#[tracing::instrument(
    name = "tool.user_memory.lookup",
    skip_all,
    fields(tool_call = %call.id)
)]
pub(crate) async fn lookup_user_memory<S>(
    storage: &S,
    context: &MemoryToolContext,
    call: ClientToolCall,
) -> Result<ClientToolOutput, MemoryToolError>
where
    S: BotStorage,
{
    let key = context.target_key(&call.input)?;
    let document = storage
        .load_user_memory_document(key.clone())
        .await
        .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
    let since = document
        .as_ref()
        .and_then(|document| document.source_event_cutoff);
    let events = storage
        .list_pending_memory_events(key.clone(), since)
        .await
        .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
    let diary_entries = storage
        .list_recent_memory_diary_entries(key.clone(), LOOKUP_DIARY_ENTRY_LIMIT)
        .await
        .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
    tracing::debug!(
        message_provider = %key.platform,
        scope_key = %key.scope_key,
        target_user_id = %key.user_key,
        found_profile = document.is_some(),
        recent_events = events.len(),
        recent_diary_entries = diary_entries.len(),
        "looked up user memory"
    );
    let value = serde_json::json!({
        "message_provider": key.platform,
        "target_user_id": key.user_key,
        "scope_key": key.scope_key,
        "profile_found": document.is_some(),
        "profile_revision": document.as_ref().map(|document| document.revision),
        "profile": document
            .as_ref()
            .map(|document| document.markdown.as_str())
            .unwrap_or(EMPTY_MEMORY),
        "recent_events": events.iter().map(memory_event_trace).collect::<Vec<_>>(),
        "recent_diary_entries": diary_entries
            .iter()
            .map(memory_diary_entry_trace)
            .collect::<Vec<_>>(),
    });
    Ok(ClientToolOutput {
        result: ClientToolResultContent::Json {
            value: value.clone(),
        },
        media: Vec::new(),
        is_error: false,
        trace_response: value,
        usage: Vec::new(),
    })
}

#[tracing::instrument(
    name = "tool.user_memory.remember",
    skip_all,
    fields(tool_call = %call.id)
)]
pub(crate) async fn remember_user_memory<S>(
    storage: &S,
    context: &MemoryToolContext,
    call: ClientToolCall,
) -> Result<ClientToolOutput, MemoryToolError>
where
    S: BotStorage,
{
    let key = context.target_key(&call.input)?;
    let memory = required_string(&call.input, "memory")?;
    let tags = optional_string_array(&call.input, "tags")?;
    let confidence = optional_f32(&call.input, "confidence")?;
    let event = storage
        .append_user_memory_event(NewUserMemoryEvent {
            key: key.clone(),
            actor_user_key: Some(context.actor_user_key.clone()),
            kind: UserMemoryEventKind::Remember,
            body: memory,
            tags,
            confidence,
            source_conversation_id: Some(context.conversation_id),
            source_turn_id: Some(context.turn_id),
            source_tool_trace_id: None,
            supersedes_event_id: None,
        })
        .await
        .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
    let text = if key.user_key == context.actor_user_key {
        format!(
            "Remembered for {} in this server.",
            context.actor_display_name
        )
    } else {
        format!("Remembered for user `{}` in this server.", key.user_key)
    };
    Ok(ClientToolOutput {
        result: ClientToolResultContent::Text { text },
        media: Vec::new(),
        is_error: false,
        trace_response: memory_event_trace(&event),
        usage: Vec::new(),
    })
}

#[tracing::instrument(
    name = "tool.user_memory.forget",
    skip_all,
    fields(tool_call = %call.id)
)]
pub(crate) async fn forget_user_memory<S>(
    storage: &S,
    context: &MemoryToolContext,
    call: ClientToolCall,
) -> Result<ClientToolOutput, MemoryToolError>
where
    S: BotStorage,
{
    let key = context.target_key(&call.input)?;
    let query = required_string(&call.input, "query")?;
    let reason = call
        .input
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let body = match reason {
        Some(reason) => format!("{query}\n\nReason: {reason}"),
        None => query,
    };
    let event = storage
        .append_user_memory_event(NewUserMemoryEvent {
            key: key.clone(),
            actor_user_key: Some(context.actor_user_key.clone()),
            kind: UserMemoryEventKind::Forget,
            body,
            tags: Vec::new(),
            confidence: None,
            source_conversation_id: Some(context.conversation_id),
            source_turn_id: Some(context.turn_id),
            source_tool_trace_id: None,
            supersedes_event_id: None,
        })
        .await
        .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
    let text = if key.user_key == context.actor_user_key {
        format!(
            "Recorded a forget request for {} in this server.",
            context.actor_display_name
        )
    } else {
        format!(
            "Recorded a forget request for user `{}` in this server.",
            key.user_key
        )
    };
    Ok(ClientToolOutput {
        result: ClientToolResultContent::Text { text },
        media: Vec::new(),
        is_error: false,
        trace_response: memory_event_trace(&event),
        usage: Vec::new(),
    })
}

/// Errors from memory client tools.
#[derive(Debug, Error)]
pub(crate) enum MemoryToolError {
    /// Tool input was invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Storage operation failed.
    #[error("storage error: {0}")]
    Storage(String),
}

fn lookup_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "properties": {
            "target_user_id": {
                "type": "string",
                "description": "Optional platform user id. Defaults to the current author."
            }
        },
        "additionalProperties": false
    }))
}

fn remember_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "properties": {
            "target_user_id": {
                "type": "string",
                "description": "Optional platform user id. Defaults to the current author."
            },
            "memory": {
                "type": "string",
                "description": "Stable useful fact to remember."
            },
            "tags": {
                "type": "array",
                "items": { "type": "string" }
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            }
        },
        "required": ["memory"],
        "additionalProperties": false
    }))
}

fn forget_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "properties": {
            "target_user_id": {
                "type": "string",
                "description": "Optional platform user id. Defaults to the current author."
            },
            "query": {
                "type": "string",
                "description": "Description of what should be forgotten or no longer used."
            },
            "reason": {
                "type": "string"
            }
        },
        "required": ["query"],
        "additionalProperties": false
    }))
}

fn required_string(input: &serde_json::Value, field: &str) -> Result<String, MemoryToolError> {
    input
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| MemoryToolError::InvalidInput(format!("`{field}` is required")))
}

fn optional_string_array(
    input: &serde_json::Value,
    field: &str,
) -> Result<Vec<String>, MemoryToolError> {
    let Some(value) = input.get(field) else {
        return Ok(Vec::new());
    };
    let Some(values) = value.as_array() else {
        return Err(MemoryToolError::InvalidInput(format!(
            "`{field}` must be an array of strings"
        )));
    };
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .ok_or_else(|| {
                    MemoryToolError::InvalidInput(format!(
                        "`{field}` must contain only non-empty strings"
                    ))
                })
        })
        .collect()
}

fn optional_f32(input: &serde_json::Value, field: &str) -> Result<Option<f32>, MemoryToolError> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    let Some(value) = value.as_f64() else {
        return Err(MemoryToolError::InvalidInput(format!(
            "`{field}` must be a number"
        )));
    };
    if !(0.0..=1.0).contains(&value) {
        return Err(MemoryToolError::InvalidInput(format!(
            "`{field}` must be between 0 and 1"
        )));
    }
    Ok(Some(value as f32))
}

fn normalize_target_user_id(input: &str) -> Result<String, MemoryToolError> {
    let trimmed = input.trim();
    let unwrapped = trimmed
        .strip_prefix("<@!")
        .and_then(|value| value.strip_suffix('>'))
        .or_else(|| {
            trimmed
                .strip_prefix("<@")
                .and_then(|value| value.strip_suffix('>'))
        })
        .unwrap_or(trimmed)
        .trim();
    if unwrapped.is_empty() {
        return Err(MemoryToolError::InvalidInput(
            "`target_user_id` cannot be empty".to_string(),
        ));
    }
    Ok(unwrapped.to_string())
}

fn memory_event_trace(event: &UserMemoryEvent) -> serde_json::Value {
    serde_json::json!({
        "id": event.id,
        "message_provider": event.key.platform,
        "target_user_id": event.key.user_key,
        "scope_key": event.key.scope_key,
        "kind": event_kind_label(event.kind),
        "body": event.body,
        "tags": event.tags,
        "confidence": event.confidence,
        "created_at": timestamp_rfc3339(event.created_at),
    })
}

fn memory_diary_entry_trace(entry: &UserMemoryDiaryEntry) -> serde_json::Value {
    serde_json::json!({
        "id": entry.id,
        "window_start": timestamp_rfc3339(entry.window_start),
        "window_end": timestamp_rfc3339(entry.window_end),
        "created_at": timestamp_rfc3339(entry.created_at),
        "markdown": entry.markdown,
    })
}

fn timestamp_rfc3339(timestamp: OffsetDateTime) -> String {
    timestamp
        .format(&Rfc3339)
        .unwrap_or_else(|_| timestamp.to_string())
}

fn event_kind_label(kind: UserMemoryEventKind) -> &'static str {
    match kind {
        UserMemoryEventKind::Remember => "remember",
        UserMemoryEventKind::Correction => "correction",
        UserMemoryEventKind::Forget => "forget",
        UserMemoryEventKind::DiaryObservation => "diary_observation",
        UserMemoryEventKind::OperatorNote => "operator_note",
    }
}

#[cfg(test)]
mod tests {
    use chudbot_api::{
        ConversationId, ModelId, PlatformName, TurnId, UserMemoryDiaryEntry, UserMemoryEvent,
        UserMemoryEventKind, UserMemoryKey,
    };
    use time::macros::datetime;

    use super::*;
    use crate::memory::MEMORY_DIARY_AGENT;

    #[test]
    fn memory_event_trace_serializes_created_at_as_rfc3339_string() {
        let key = UserMemoryKey {
            platform: PlatformName::new("discord"),
            scope_key: "guild:guild-1".to_string(),
            user_key: "user-1".to_string(),
        };
        let event = UserMemoryEvent {
            id: ConversationId::new().0,
            key,
            actor_user_key: Some("user-1".to_string()),
            kind: UserMemoryEventKind::Remember,
            body: "Richie likes Israel.".to_string(),
            tags: vec!["server_lore".to_string()],
            confidence: None,
            source_conversation_id: None,
            source_turn_id: None,
            source_tool_trace_id: None,
            supersedes_event_id: None,
            created_at: datetime!(2026-06-03 22:27:01.816929 UTC),
            updated_at: datetime!(2026-06-03 22:27:01.816929 UTC),
        };

        let value = memory_event_trace(&event);

        assert_eq!(
            value["created_at"].as_str(),
            Some("2026-06-03T22:27:01.816929Z")
        );
    }

    #[test]
    fn memory_diary_entry_trace_serializes_compact_rfc3339_entry() {
        let key = UserMemoryKey {
            platform: PlatformName::new("discord"),
            scope_key: "guild:guild-1".to_string(),
            user_key: "user-1".to_string(),
        };
        let entry = UserMemoryDiaryEntry {
            id: ConversationId::new().0,
            key,
            window_start: datetime!(2026-06-03 00:00:00 UTC),
            window_end: datetime!(2026-06-04 00:00:00 UTC),
            source_turn_ids: vec![TurnId::new()],
            markdown: "- Chud prefers concise status updates.".to_string(),
            agent_name: MEMORY_DIARY_AGENT.to_string(),
            llm_provider: ProviderName::new("xai"),
            llm_model: ModelId::new("grok-4.3"),
            usage: Vec::new(),
            created_at: datetime!(2026-06-04 00:01:02.123456 UTC),
            updated_at: datetime!(2026-06-04 00:01:02.123456 UTC),
        };

        let value = memory_diary_entry_trace(&entry);

        assert_eq!(value.as_object().map(|object| object.len()), Some(5));
        assert_eq!(value["window_start"].as_str(), Some("2026-06-03T00:00:00Z"));
        assert_eq!(value["window_end"].as_str(), Some("2026-06-04T00:00:00Z"));
        assert_eq!(
            value["created_at"].as_str(),
            Some("2026-06-04T00:01:02.123456Z")
        );
        assert_eq!(
            value["markdown"].as_str(),
            Some("- Chud prefers concise status updates.")
        );
    }

    #[test]
    fn normalizes_discord_mention_target_ids() {
        assert_eq!(
            normalize_target_user_id("<@!123456789012345678>").unwrap(),
            "123456789012345678"
        );
        assert_eq!(
            normalize_target_user_id("<@123456789012345678>").unwrap(),
            "123456789012345678"
        );
    }
}
