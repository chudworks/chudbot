//! User-memory client tools.
//!
//! The model sees these as ordinary client tools, but they are append-only
//! inputs to the memory system. Lookup reads the compact profile plus recent
//! unfused context. Remember and forget append events that the background
//! memory runtime later folds into profile revisions.
//!
//! The runtime injects persistent memory notes for conversation participants
//! as system turns (see `BotRuntime::participant_memory_context_items`), so
//! `crate::memory::PROMPT_GUIDANCE` tells memory-enabled agents to use lookup
//! only for refreshes and uncovered users, to remember durable facts, and to
//! record explicit forget requests. The schemas and descriptions in this
//! module are the provider-neutral tool contract backing that guidance.

use super::*;

use chudbot_api::{
    NewUserMemoryEvent, UserMemoryDiaryEntry, UserMemoryEvent, UserMemoryEventKind, UserMemoryKey,
};
use time::format_description::well_known::Rfc3339;

pub(crate) use crate::memory::{
    FORGET_USER_MEMORY_TOOL, LOOKUP_USER_MEMORY_TOOL, REMEMBER_USER_MEMORY_TOOL,
};

/// Number of most-recent diary entries returned by `lookup_user_memory`.
const LOOKUP_DIARY_ENTRY_LIMIT: u32 = 3;
/// Sentinel profile text used when no compact memory document exists yet.
const EMPTY_MEMORY: &str = "(no stored memory)";

/// Shared context for user-memory client tools during one turn.
///
/// The base key points at the current message author in the current platform
/// scope. Tool input may override only the target user id; platform and scope
/// stay fixed so a model cannot write memory outside the current server/global
/// memory namespace.
#[derive(Debug, Clone)]
pub(crate) struct MemoryToolContext {
    base_key: UserMemoryKey,
    actor_user_key: String,
    actor_display_name: String,
    conversation_id: ConversationId,
    turn_id: TurnId,
}

impl MemoryToolContext {
    /// Capture the actor, turn, and default memory target for one agent turn.
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

    /// Resolve the input target into the exact storage key for this call.
    ///
    /// Missing `target_user_id` means "the current author." Discord mention
    /// syntax is accepted for convenience, but only the user component is
    /// replaced; the current platform and scope are preserved.
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
            platform: self.base_key.platform.clone(),
            scope_key: self.base_key.scope_key.clone(),
            user_key: target,
        })
    }
}

/// Errors from memory client tools.
///
/// These are execution failures, not successful tool results with
/// `is_error = true`. `RuntimeToolExecutor` stringifies them through
/// `runtime_tool_execution_error`, and the agent loop wraps the error text into
/// the model-facing failed tool result and trace JSON.
#[derive(Debug, Error)]
pub(crate) enum MemoryToolError {
    /// Tool input was invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Storage operation failed.
    #[error("storage error: {0}")]
    Storage(String),
}

impl MemoryToolContext {
    /// Human wording for the memory scope used in tool descriptions.
    pub(crate) fn scope_phrase(&self) -> &'static str {
        if self.base_key.scope_key.starts_with("guild:") {
            "in this server"
        } else {
            "in direct messages"
        }
    }
}

/// Model-facing spec for reading a user's memory state.
///
/// `notes_auto_injected` is true for top-level turns, where the runtime
/// injects persistent participant memory notes as system messages; the
/// description then frames this tool as a refresh/fallback path. The result
/// payload is JSON with identity fields, compact profile metadata/content,
/// pending memory events, and the latest diary entries.
pub(crate) fn lookup_user_memory_spec(
    context: &MemoryToolContext,
    notes_auto_injected: bool,
) -> ClientToolSpec {
    let scope = context.scope_phrase();
    let description = if notes_auto_injected {
        format!(
            "Re-read the remembered profile, pending memory events, and recent diary entries \
             for a user {scope}. Memory notes for conversation participants are injected \
             automatically, so call this only when a user asks you to re-load their memory, \
             when an injected note seems out of date, or for a user who has no note in this \
             conversation. Skip bots."
        )
    } else {
        format!(
            "Look up the remembered profile, pending memory events, and recent diary entries \
             for the current user or another user id {scope}. Call it at most once per user \
             per conversation."
        )
    };
    ClientToolSpec {
        description,
        input_schema: lookup_schema(),
    }
}

/// Model-facing spec for appending a durable memory event.
///
/// This does not rewrite the compact profile immediately. It records a
/// `remember` event that the memory compactor later merges into a profile
/// revision.
pub(crate) fn remember_user_memory_spec(context: &MemoryToolContext) -> ClientToolSpec {
    let scope = context.scope_phrase();
    ClientToolSpec {
        description: format!(
            "Remember one stable fact about a user {scope}: a preference, relationship, \
             project, correction, recurring fact, or running joke. Defaults to the current \
             message author; pass target_user_id to remember a fact about someone else."
        ),
        input_schema: remember_schema(),
    }
}

/// Model-facing spec for appending a forget/tombstone event.
///
/// Forget requests are stored as events too. The compactor interprets them when
/// deciding what should be removed or no longer trusted in the compact profile.
pub(crate) fn forget_user_memory_spec(context: &MemoryToolContext) -> ClientToolSpec {
    let scope = context.scope_phrase();
    ClientToolSpec {
        description: format!(
            "Record that a remembered fact about a user {scope} should be forgotten or no \
             longer used. Defaults to the current message author; pass target_user_id to \
             target someone else."
        ),
        input_schema: forget_schema(),
    }
}

/// Load the compact memory document plus recent unfused context for one user.
///
/// Successful outputs are JSON and are mirrored into `trace_response`. The
/// payload shape is:
///
/// - `message_provider`, `scope_key`, and `target_user_id`: resolved storage key.
/// - `profile_found` and `profile_revision`: compact document state.
/// - `profile`: compact markdown or `(no stored memory)`.
/// - `recent_events`: pending events after the document cutoff, serialized with
///   RFC3339 `created_at`.
/// - `recent_diary_entries`: latest diary summaries, oldest-to-newest within the
///   bounded slice returned by storage.
///
/// Invalid input and storage failures are returned as `Err`; the runtime
/// executor converts them into model-facing tool errors with `is_error = true`.
/// Load the model-facing memory payload for one user.
///
/// Shared by the `lookup_user_memory` tool and the automatic author-memory
/// context preload so both surfaces present the same JSON shape.
pub(crate) async fn user_memory_payload<S>(
    storage: &S,
    key: &UserMemoryKey,
) -> Result<serde_json::Value, MemoryToolError>
where
    S: BotStorage,
{
    let document = storage
        .load_user_memory_document(key.clone())
        .await
        .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
    // Pending events are those not yet represented by the compact document.
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
        "loaded user memory payload"
    );
    Ok(serde_json::json!({
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
    }))
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
    let value = user_memory_payload(storage, &key).await?;
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

/// Append a `remember` event for the current or targeted user.
///
/// The model receives a short text confirmation. The persisted trace stores the
/// event JSON, including tags, confidence, and creation time. Successful memory
/// tools do not attach media or emit usage records; failures leave this function
/// as `Err` and are converted by the executor into tool errors.
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
    // The event preserves provenance so compaction can explain where it came from.
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

/// Append a forget request for the current or targeted user.
///
/// The request body is the model-supplied description of the fact to stop
/// using, optionally followed by a reason paragraph. Like
/// `remember_user_memory`, this records an event for the compactor rather than
/// directly editing the compact profile. The advertised field is `memory` to
/// mirror the remember tool; `query` remains an accepted legacy alias.
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
    let memory = required_string_with_alias(&call.input, "memory", "query")?;
    let reason = call
        .input
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let body = match reason {
        Some(reason) => format!("{memory}\n\nReason: {reason}"),
        None => memory,
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

/// Shared schema for the optional memory target field.
///
/// The description teaches the model where real ids live in the message JSON
/// because passing usernames instead of ids is the most common misuse; the
/// runtime rejects non-numeric targets in [`normalize_target_user_id`].
fn target_user_id_schema() -> ToolInputValueSchema {
    ToolInputValueSchema::string().description(
        "Numeric platform user id of the user this applies to. Omit for the current message \
         author. Copy the id from `author.id` or `mentioned_users[].id` in the message JSON; \
         `<@id>` mention syntax is also accepted. Usernames and display names are not \
         accepted.",
    )
}

/// JSON Schema for the lookup tool input.
///
/// The only accepted field is optional `target_user_id`; absent input reads the
/// current author captured in `MemoryToolContext`.
fn lookup_schema() -> ToolInputSchema {
    ToolInputSchema::object([ToolInputField::optional(
        "target_user_id",
        target_user_id_schema(),
    )])
}

/// JSON Schema for appending a `remember` event.
///
/// `memory` is required and must be non-empty after trimming. The parser still
/// tolerates legacy `tags`/`confidence` metadata, but they are no longer
/// advertised: the compactor never reads them, so offering them only gave the
/// model extra ways to construct invalid input.
fn remember_schema() -> ToolInputSchema {
    ToolInputSchema::object([
        ToolInputField::optional("target_user_id", target_user_id_schema()),
        ToolInputField::required(
            "memory",
            ToolInputValueSchema::string().description(
                "One concise, stable fact written in third person, e.g. \"Prefers Rust over \
                 Python\" or \"Is building a Discord bot called Chudbot\". For second-hand \
                 facts, name the source in the text, e.g. \"According to Alice, ...\". Do not \
                 store transient conversation details or sensitive personal information.",
            ),
        ),
    ])
}

/// JSON Schema for appending a `forget` event.
///
/// `memory` names what should stop being used, mirroring the remember tool's
/// field name. `reason`, when present, is folded into the event body as
/// explanatory text for later compaction.
fn forget_schema() -> ToolInputSchema {
    ToolInputSchema::object([
        ToolInputField::optional("target_user_id", target_user_id_schema()),
        ToolInputField::required(
            "memory",
            ToolInputValueSchema::string().description(
                "Description of the remembered fact that should be forgotten or no longer \
                 used.",
            ),
        ),
        ToolInputField::optional(
            "reason",
            ToolInputValueSchema::string().description(
                "Optional reason, e.g. the user asked for it or the fact turned out wrong.",
            ),
        ),
    ])
}

/// Read a required, trimmed, non-empty string field from provider JSON.
fn required_string(input: &serde_json::Value, field: &str) -> Result<String, MemoryToolError> {
    input
        .get(field)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| MemoryToolError::InvalidInput(format!("`{field}` is required")))
}

/// Read a required string that may arrive under a legacy alias field.
///
/// The error names only the advertised field so the model is steered toward
/// the current schema.
fn required_string_with_alias(
    input: &serde_json::Value,
    field: &str,
    alias: &str,
) -> Result<String, MemoryToolError> {
    required_string(input, field).or_else(|error| match required_string(input, alias) {
        Ok(value) => Ok(value),
        Err(_) => Err(error),
    })
}

/// Read an optional string-array field, rejecting empty or non-string members.
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

/// Read an optional confidence value in the inclusive `0..=1` range.
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

/// Normalize the optional target id accepted by model-facing schemas.
///
/// Discord `<@id>` and `<@!id>` mentions are unwrapped because models often copy
/// mention text from message context into tool arguments. Non-numeric targets
/// are rejected: platform user ids are numeric snowflakes, and accepting a
/// username would silently read or write a memory namespace no real lookup will
/// ever hit again.
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
    if !unwrapped.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(MemoryToolError::InvalidInput(format!(
            "`target_user_id` `{unwrapped}` is not a numeric platform user id; copy the id \
             from `author.id` or `mentioned_users[].id` in the message JSON (or use `<@id>` \
             mention syntax). Usernames and display names are not accepted."
        )));
    }
    Ok(unwrapped.to_string())
}

/// Serialize a memory event into the lookup payload and trace response shape.
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

/// Serialize a diary entry into the compact lookup payload shape.
///
/// The storage record includes provenance and usage fields that are intentionally
/// omitted here; the model only needs the window bounds, creation time, and
/// markdown summary.
fn memory_diary_entry_trace(entry: &UserMemoryDiaryEntry) -> serde_json::Value {
    serde_json::json!({
        "id": entry.id,
        "window_start": timestamp_rfc3339(entry.window_start),
        "window_end": timestamp_rfc3339(entry.window_end),
        "created_at": timestamp_rfc3339(entry.created_at),
        "markdown": entry.markdown,
    })
}

/// Format timestamps for stable JSON payloads and trace assertions.
fn timestamp_rfc3339(timestamp: OffsetDateTime) -> String {
    timestamp
        .format(&Rfc3339)
        .unwrap_or_else(|_| timestamp.to_string())
}

/// Stable string labels used in model-visible event JSON.
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

    #[test]
    fn rejects_non_numeric_target_user_ids() {
        let error = normalize_target_user_id("chud").unwrap_err();
        assert!(error.to_string().contains("numeric platform user id"));
        assert!(normalize_target_user_id("<@chud>").is_err());
        assert!(normalize_target_user_id("").is_err());
    }

    #[test]
    fn forget_input_accepts_memory_and_legacy_query_fields() {
        let memory = serde_json::json!({ "memory": "the pineapple incident" });
        assert_eq!(
            required_string_with_alias(&memory, "memory", "query").unwrap(),
            "the pineapple incident"
        );
        let query = serde_json::json!({ "query": "the pineapple incident" });
        assert_eq!(
            required_string_with_alias(&query, "memory", "query").unwrap(),
            "the pineapple incident"
        );
        let error = required_string_with_alias(&serde_json::json!({}), "memory", "query")
            .unwrap_err()
            .to_string();
        assert!(error.contains("`memory` is required"));
    }

    #[test]
    fn memory_schemas_advertise_numeric_target_and_no_event_metadata() {
        let remember = remember_schema().json_schema();
        assert!(remember["properties"]["tags"].is_null());
        assert!(remember["properties"]["confidence"].is_null());
        assert!(
            remember["properties"]["target_user_id"]["description"]
                .as_str()
                .unwrap()
                .contains("mentioned_users[].id")
        );

        let forget = forget_schema().json_schema();
        assert!(forget["properties"]["memory"].is_object());
        assert!(forget["properties"]["query"].is_null());
        assert_eq!(forget["required"], serde_json::json!(["memory"]));
    }
}
