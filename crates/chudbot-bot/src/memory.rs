//! User memory tools, prompts, and background compaction runtime.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Duration;

use chudbot_api::{
    AgentLimits, AgentOutcome, AgentRun, BotStorage, ClientTool, ClientToolCall, ClientToolOutput,
    ClientToolResultContent, ClientToolSpec, ContentBlock, ConversationId, ExternalId,
    MediaCategory, MediaStore, MemoryJobCompletion, MemoryJobKind, MemoryJobSchedule,
    MemoryTurnWindow, Model, ModelId, ModelSpec, NewUserMemoryDiaryEntry,
    NewUserMemoryDocumentRevision, NewUserMemoryEvent, ProviderName, ProviderOptions,
    SamplingOptions, ToolInputSchema, Transcript, TranscriptTurn, TurnId, TurnRole, UsageRecord,
    UserMemoryAudioTranscription, UserMemoryDiaryEntry, UserMemoryDocument, UserMemoryEvent,
    UserMemoryEventKind, UserMemoryImageContext, UserMemoryJob, UserMemoryKey, UserMemoryTurn,
    UserProfile, UserRef,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::config::{AgentConfig, SystemAgentConfig};
use crate::{LlmProviderRegistry, RoutedLlmBackend};

/// Tool name for current or target user memory lookup.
pub const LOOKUP_USER_MEMORY_TOOL: &str = "lookup_user_memory";
/// Tool name for appending a memory event.
pub const REMEMBER_USER_MEMORY_TOOL: &str = "remember_user_memory";
/// Tool name for appending a forget/tombstone event.
pub const FORGET_USER_MEMORY_TOOL: &str = "forget_user_memory";

const MEMORY_MODEL_ID: &str = "grok-4.3";
const MEMORY_REASONING_EFFORT: &str = "high";
pub const MEMORY_DIARY_AGENT: &str = "memory_diary";
pub const MEMORY_COMPACT_AGENT: &str = "memory_compact";
const LOOKUP_DIARY_ENTRY_LIMIT: u32 = 3;
const MEMORY_DIARY_IMAGE_MIME_TYPES: &[&str] = &["image/png", "image/jpeg", "image/webp"];

const DIARY_PROMPT: &str = "You write concise user-memory diary entries for Chudbot. \
Read the bounded transcript slice and optional current memory profile. Extract only \
stable, useful observations about the subject user. Include uncertainty when evidence \
is weak. Prefer factual bullets over prose. Consider relationships, preferences and \
dislikes, projects, work, hobbies, recurring topics, server lore, running jokes, \
good-natured roast material, corrections, stale facts, and visually meaningful \
image evidence. Do not invent facts.";

const COMPACTOR_PROMPT: &str = "You maintain a compact Markdown memory profile for one \
Chudbot user in one server/workspace. Produce a complete replacement profile, not a diff. \
Use explicit memory events, diary entries, corrections, and forget requests. Keep the \
profile short, normally 1-3 KB. Remove or rewrite forgotten and stale facts. Preserve \
useful uncertainty. Output Markdown only.";

const EMPTY_MEMORY: &str = "(no stored memory)";

/// User-memory runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Global memory switch.
    #[serde(default)]
    pub enabled: bool,
    /// Scheduler poll interval in seconds.
    #[serde(default = "default_poll_interval_seconds")]
    pub poll_interval_seconds: u64,
    /// Human-readable compaction interval such as `12h` or `24h`.
    #[serde(default = "default_compaction_interval")]
    pub compaction_interval: String,
    /// Human-readable maximum age for turns considered by diary backfill.
    #[serde(default = "default_diary_backfill_window")]
    pub diary_backfill_window: String,
    /// Human-readable source window summarized by one diary entry.
    #[serde(default = "default_diary_interval")]
    pub diary_interval: String,
    /// SQL lease duration in seconds.
    #[serde(default = "default_lease_seconds")]
    pub lease_seconds: u64,
    /// Maximum jobs to claim per scheduler tick.
    #[serde(default = "default_max_jobs_per_tick")]
    pub max_jobs_per_tick: u32,
    /// Maximum jobs to run concurrently inside this process.
    #[serde(default = "default_max_concurrent_jobs")]
    pub max_concurrent_jobs: u32,
    /// Maximum completed turns included in one diary job.
    #[serde(default = "default_max_transcript_turns_per_diary_job")]
    pub max_transcript_turns_per_diary_job: u32,
    /// Base retry backoff after a failed memory job.
    #[serde(default = "default_retry_backoff_seconds")]
    pub retry_backoff_seconds: u64,
    /// Attempts after which a job is marked failed instead of retried.
    #[serde(default = "default_max_job_attempts")]
    pub max_job_attempts: i32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            poll_interval_seconds: default_poll_interval_seconds(),
            compaction_interval: default_compaction_interval(),
            diary_backfill_window: default_diary_backfill_window(),
            diary_interval: default_diary_interval(),
            lease_seconds: default_lease_seconds(),
            max_jobs_per_tick: default_max_jobs_per_tick(),
            max_concurrent_jobs: default_max_concurrent_jobs(),
            max_transcript_turns_per_diary_job: default_max_transcript_turns_per_diary_job(),
            retry_backoff_seconds: default_retry_backoff_seconds(),
            max_job_attempts: default_max_job_attempts(),
        }
    }
}

impl MemoryConfig {
    /// Parse and validate the human-readable compaction interval.
    pub fn compaction_interval_seconds(&self) -> Result<u64, MemoryConfigError> {
        parse_duration_seconds(&self.compaction_interval)
    }

    /// Parse and validate the maximum diary backfill window.
    pub fn diary_backfill_window_seconds(&self) -> Result<u64, MemoryConfigError> {
        parse_duration_seconds(&self.diary_backfill_window)
    }

    /// Parse and validate the source window summarized by one diary entry.
    pub fn diary_interval_seconds(&self) -> Result<u64, MemoryConfigError> {
        parse_duration_seconds(&self.diary_interval)
    }

    /// Poll interval as a non-zero duration.
    pub fn poll_interval(&self) -> Duration {
        Duration::from_secs(self.poll_interval_seconds.max(1))
    }

    /// Resolve the configured memory agents, falling back to the built-in
    /// default specs for any job kind without a matching configured agent.
    /// Return the memory agents and providers that would be used at runtime.
    pub fn resolved_agent_providers(
        &self,
        agents: &BTreeMap<String, AgentConfig>,
        default_limits: AgentLimits,
    ) -> Vec<(String, ProviderName)> {
        self.resolve_agent_set(agents, default_limits)
            .iter()
            .map(|agent| (agent.name.clone(), agent.provider.clone()))
            .collect()
    }

    pub(crate) fn resolve_agent_set(
        &self,
        agents: &BTreeMap<String, AgentConfig>,
        default_limits: AgentLimits,
    ) -> MemoryAgentSet {
        MemoryAgentSet {
            diary: self.resolve_agent(
                MEMORY_DIARY_AGENT,
                DIARY_PROMPT,
                default_max_diary_output_tokens(),
                agents,
                default_limits,
            ),
            compact: self.resolve_agent(
                MEMORY_COMPACT_AGENT,
                COMPACTOR_PROMPT,
                default_max_profile_output_tokens(),
                agents,
                default_limits,
            ),
        }
    }

    fn resolve_agent(
        &self,
        default_name: &'static str,
        default_prompt: &'static str,
        fallback_max_output_tokens: u32,
        agents: &BTreeMap<String, AgentConfig>,
        default_limits: AgentLimits,
    ) -> SystemAgentConfig {
        if let Some(agent) = agents.get(default_name) {
            let resolved = SystemAgentConfig::from_agent_config(
                default_name.to_string(),
                agent,
                default_limits,
            );
            resolved.log_loaded_from_config();
            return resolved;
        }

        let resolved = SystemAgentConfig::from_parts(
            default_name,
            default_memory_provider(),
            default_prompt,
            ModelSpec {
                id: ModelId::new(MEMORY_MODEL_ID),
                server_tools: BTreeSet::new(),
                sampling: SamplingOptions {
                    max_output_tokens: Some(fallback_max_output_tokens),
                    temperature: Some(0.2),
                    top_p: Some(0.9),
                },
                provider_options: Some(ProviderOptions {
                    value: json!({ "reasoning_effort": MEMORY_REASONING_EFFORT }),
                }),
            },
            AgentLimits::default(),
        );
        resolved.log_using_default();
        resolved
    }

    fn lease_duration(&self) -> Duration {
        Duration::from_secs(self.lease_seconds.max(1))
    }

    fn retry_backoff(&self, attempts: i32) -> time::Duration {
        let attempts = attempts.max(1) as u64;
        let seconds = self
            .retry_backoff_seconds
            .max(1)
            .saturating_mul(attempts.min(12));
        time::Duration::seconds(i64::try_from(seconds).unwrap_or(i64::MAX))
    }
}

/// Resolved memory agents used by the background scheduler.
#[derive(Debug, Clone)]
pub(crate) struct MemoryAgentSet {
    /// Diary agent.
    pub(crate) diary: SystemAgentConfig,
    /// Profile compaction agent.
    pub(crate) compact: SystemAgentConfig,
}

impl MemoryAgentSet {
    /// Iterate all configured memory agents.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &SystemAgentConfig> {
        [&self.diary, &self.compact].into_iter()
    }
}

fn default_memory_provider() -> ProviderName {
    ProviderName::new("grok")
}

fn default_poll_interval_seconds() -> u64 {
    60
}

fn default_compaction_interval() -> String {
    "24h".to_string()
}

fn default_diary_backfill_window() -> String {
    "3d".to_string()
}

fn default_diary_interval() -> String {
    "24h".to_string()
}

fn default_lease_seconds() -> u64 {
    300
}

fn default_max_jobs_per_tick() -> u32 {
    4
}

fn default_max_concurrent_jobs() -> u32 {
    4
}

fn default_max_transcript_turns_per_diary_job() -> u32 {
    40
}

fn default_max_diary_output_tokens() -> u32 {
    1024
}

fn default_max_profile_output_tokens() -> u32 {
    2048
}

fn default_retry_backoff_seconds() -> u64 {
    300
}

fn default_max_job_attempts() -> i32 {
    5
}

/// Memory config validation errors.
#[derive(Debug, Error)]
pub enum MemoryConfigError {
    /// Duration string is invalid.
    #[error("invalid memory duration `{value}`; expected digits followed by s, m, h, or d")]
    InvalidDuration {
        /// Invalid value.
        value: String,
    },
}

/// Parse a duration with `s`, `m`, `h`, or `d` suffix.
pub fn parse_duration_seconds(value: &str) -> Result<u64, MemoryConfigError> {
    let value = value.trim();
    let Some(unit) = value.chars().last() else {
        return Err(MemoryConfigError::InvalidDuration {
            value: value.to_string(),
        });
    };
    let amount = &value[..value.len().saturating_sub(unit.len_utf8())];
    let amount = amount
        .parse::<u64>()
        .map_err(|_| MemoryConfigError::InvalidDuration {
            value: value.to_string(),
        })?;
    let multiplier = match unit {
        's' => 1,
        'm' => 60,
        'h' => 60 * 60,
        'd' => 24 * 60 * 60,
        _ => {
            return Err(MemoryConfigError::InvalidDuration {
                value: value.to_string(),
            });
        }
    };
    amount
        .checked_mul(multiplier)
        .filter(|seconds| *seconds > 0)
        .ok_or_else(|| MemoryConfigError::InvalidDuration {
            value: value.to_string(),
        })
}

/// Prompt guidance inserted into top-level memory-enabled agents.
pub fn prompt_guidance() -> &'static str {
    "CRITICAL: Memory System\n\
- CRITICAL: If a user is the `author` of a message, you MUST load memory about that user. Do not respond to a user if you do not load their memory document first. Use the `lookup_user_memory` any time you see a user for the first time.\n\
- CRITICAL: If a user's memory has not been loaded, then any **mention** of a user should trigger a `lookup_user_memory` call, even if they are not the author.\n\
- The `lookup_user_memory` tool gives you a memory document about a user, recent events, and recent diary entries. These recent events can be `remember` or `forget`.\n\
- Use the `remember_user_memory` tool to store facts about a user. If there's something you think would be useful in the future, you should use this tool to remember it.\n\
- There is a `forget_user_memory` which works like `remember_user_memory`, but instead stores a fact to forget about a user.\n\
- If a user asks you explicitly to remember or forget something about themselves, then you should absolutely use the tools to store the user's preference and respect their humanity!\n\
- If a user tells you a fact about another user, you are allowed to remember / forget it. Take memories from 3rd parties with a \"grain of salt\".\n\
- If the current message conflicts with stored memory, trust the current message and remember the correction when appropriate.\n\
- Avoid repeating or storing any memory which reveals sensitive personal information (credit card, physical address, legal name, SSN, etc)\n\n\n\
IT IS CRITICAL TO USE THE MEMORY SYSTEM PROACTIVELY! The tool calls are cheap, use the tools!\n\
VERY IMPORTANT: If a user is the `author` of a message, you MUST load memory about that user. Do not respond to a user if you do not load their memory document first. Use the `lookup_user_memory` any time you see a user for the first time.\n"
}

/// Build the neutral memory key for a platform user.
pub fn key_from_user_ref(user: &UserRef) -> UserMemoryKey {
    UserMemoryKey {
        platform: user.platform.clone(),
        scope_key: scope_key(user.guild_id.as_ref().map(chudbot_api::ExternalId::as_str)),
        user_key: user.user_id.as_str().to_string(),
    }
}

fn scope_key(guild_id: Option<&str>) -> String {
    guild_id
        .map(|guild| format!("guild:{guild}"))
        .unwrap_or_else(|| "global".to_string())
}

fn memory_scope_id(scope_key: &str) -> &str {
    scope_key.strip_prefix("guild:").unwrap_or(scope_key)
}

fn memory_guild_id(scope_key: &str) -> Option<&str> {
    scope_key.strip_prefix("guild:")
}

fn memory_user_ref(key: &UserMemoryKey) -> UserRef {
    UserRef {
        platform: key.platform.clone(),
        guild_id: memory_guild_id(&key.scope_key).map(ExternalId::new),
        user_id: ExternalId::new(key.user_key.clone()),
    }
}

fn memory_profile_display_name(profile: &UserProfile, user_key: &str) -> Option<String> {
    let name = profile
        .display_name
        .as_deref()
        .or(profile.name.as_deref())
        .unwrap_or(profile.username.as_str())
        .trim();
    (!name.is_empty() && name != user_key).then(|| name.to_string())
}

/// Runtime client tool kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryToolKind {
    /// Lookup profile/events.
    Lookup,
    /// Append remember event.
    Remember,
    /// Append forget event.
    Forget,
}

/// Client tool for user-memory lookup/write operations.
#[derive(Debug, Clone)]
pub struct MemoryClientTool<S> {
    storage: S,
    kind: MemoryToolKind,
    base_key: UserMemoryKey,
    actor_user_key: String,
    actor_display_name: String,
    conversation_id: ConversationId,
    turn_id: TurnId,
}

impl<S> MemoryClientTool<S> {
    /// Construct a memory client tool.
    pub fn new(
        storage: S,
        kind: MemoryToolKind,
        base_key: UserMemoryKey,
        actor_display_name: String,
        conversation_id: ConversationId,
        turn_id: TurnId,
    ) -> Self {
        Self {
            storage,
            kind,
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

impl<S> ClientTool for MemoryClientTool<S>
where
    S: BotStorage,
{
    type Error = MemoryToolError;

    fn spec(&self) -> ClientToolSpec {
        match self.kind {
            MemoryToolKind::Lookup => ClientToolSpec {
                description: "Look up the compact remembered profile and recent un-compacted memory events for the current user or another user id in this server.".to_string(),
                input_schema: lookup_schema(),
            },
            MemoryToolKind::Remember => ClientToolSpec {
                description: "Remember a stable preference, relationship, project, correction, recurring fact, or running joke for the current user or a target user id in this server.".to_string(),
                input_schema: remember_schema(),
            },
            MemoryToolKind::Forget => ClientToolSpec {
                description: "Record that a remembered fact should be forgotten or no longer used for the current user or a target user id in this server.".to_string(),
                input_schema: forget_schema(),
            },
        }
    }

    #[tracing::instrument(
        name = "tool.user_memory",
        skip_all,
        fields(kind = ?self.kind, tool_call = %call.id)
    )]
    async fn call(&self, call: ClientToolCall) -> Result<ClientToolOutput, Self::Error> {
        match self.kind {
            MemoryToolKind::Lookup => self.lookup(call.input).await,
            MemoryToolKind::Remember => self.remember(call.input).await,
            MemoryToolKind::Forget => self.forget(call.input).await,
        }
    }
}

impl<S> MemoryClientTool<S>
where
    S: BotStorage,
{
    async fn lookup(&self, input: serde_json::Value) -> Result<ClientToolOutput, MemoryToolError> {
        let key = self.target_key(&input)?;
        let document = self
            .storage
            .load_user_memory_document(key.clone())
            .await
            .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
        let since = document
            .as_ref()
            .and_then(|document| document.source_event_cutoff);
        let events = self
            .storage
            .list_pending_memory_events(key.clone(), since)
            .await
            .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
        let diary_entries = self
            .storage
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
        let value = json!({
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

    async fn remember(
        &self,
        input: serde_json::Value,
    ) -> Result<ClientToolOutput, MemoryToolError> {
        let key = self.target_key(&input)?;
        let memory = required_string(&input, "memory")?;
        let tags = optional_string_array(&input, "tags")?;
        let confidence = optional_f32(&input, "confidence")?;
        let event = self
            .storage
            .append_user_memory_event(NewUserMemoryEvent {
                key: key.clone(),
                actor_user_key: Some(self.actor_user_key.clone()),
                kind: UserMemoryEventKind::Remember,
                body: memory,
                tags,
                confidence,
                source_conversation_id: Some(self.conversation_id),
                source_turn_id: Some(self.turn_id),
                source_tool_trace_id: None,
                supersedes_event_id: None,
            })
            .await
            .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
        let text = if key.user_key == self.actor_user_key {
            format!("Remembered for {} in this server.", self.actor_display_name)
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

    async fn forget(&self, input: serde_json::Value) -> Result<ClientToolOutput, MemoryToolError> {
        let key = self.target_key(&input)?;
        let query = required_string(&input, "query")?;
        let reason = input
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let body = match reason {
            Some(reason) => format!("{query}\n\nReason: {reason}"),
            None => query,
        };
        let event = self
            .storage
            .append_user_memory_event(NewUserMemoryEvent {
                key: key.clone(),
                actor_user_key: Some(self.actor_user_key.clone()),
                kind: UserMemoryEventKind::Forget,
                body,
                tags: Vec::new(),
                confidence: None,
                source_conversation_id: Some(self.conversation_id),
                source_turn_id: Some(self.turn_id),
                source_tool_trace_id: None,
                supersedes_event_id: None,
            })
            .await
            .map_err(|error| MemoryToolError::Storage(error.to_string()))?;
        let text = if key.user_key == self.actor_user_key {
            format!(
                "Recorded a forget request for {} in this server.",
                self.actor_display_name
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
}

/// Errors from memory client tools.
#[derive(Debug, Error)]
pub enum MemoryToolError {
    /// Tool input was invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Storage operation failed.
    #[error("storage error: {0}")]
    Storage(String),
}

fn lookup_schema() -> ToolInputSchema {
    ToolInputSchema::new(json!({
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
    ToolInputSchema::new(json!({
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
    ToolInputSchema::new(json!({
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
    json!({
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
    json!({
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

/// In-process memory scheduler and worker.
#[derive(Debug, Clone)]
pub struct MemoryRuntime<S, L, M> {
    storage: S,
    llms: L,
    media_store: M,
    config: MemoryConfig,
    agents: MemoryAgentSet,
}

impl<S, L, M> MemoryRuntime<S, L, M> {
    /// Construct a memory runtime.
    pub(crate) fn new(
        storage: S,
        llms: L,
        media_store: M,
        config: MemoryConfig,
        agents: MemoryAgentSet,
    ) -> Self {
        Self {
            storage,
            llms,
            media_store,
            config,
            agents,
        }
    }
}

fn memory_job_span(
    job: &UserMemoryJob,
    agent: &SystemAgentConfig,
    target_user_name: Option<&str>,
) -> tracing::Span {
    let span = tracing::info_span!(
        "memory.job",
        job = %job.id,
        kind = ?job.kind,
        memory_agent = %agent.name,
        provider = %agent.provider,
        model = %agent.model.id,
        memory_key = %job.memory_key,
        message_provider = %job.key.platform,
        scope_key = %job.key.scope_key,
        scope_id = %memory_scope_id(&job.key.scope_key),
        guild_id = tracing::field::Empty,
        user_id = %job.key.user_key,
        target_user_id = %job.key.user_key,
        target_user_name = tracing::field::Empty,
        attempts = job.attempts,
    );
    if let Some(guild_id) = memory_guild_id(&job.key.scope_key) {
        span.record("guild_id", tracing::field::display(guild_id));
    }
    if let Some(name) = target_user_name {
        span.record("target_user_name", tracing::field::display(name));
    }
    span
}

impl<S, L, M> MemoryRuntime<S, L, M>
where
    S: BotStorage + Clone + Send + Sync + 'static,
    L: LlmProviderRegistry + Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
{
    /// Run the memory scheduler until shutdown.
    pub async fn run_until_shutdown(&self, shutdown: CancellationToken) -> Result<(), MemoryError> {
        if !self.config.enabled {
            tracing::debug!("memory runtime disabled");
            return Ok(());
        }
        self.config.compaction_interval_seconds()?;
        self.config.diary_backfill_window_seconds()?;
        self.config.diary_interval_seconds()?;
        tracing::info!(
            diary_agent = %self.agents.diary.name,
            diary_provider = %self.agents.diary.provider,
            diary_model = %self.agents.diary.model.id,
            compact_agent = %self.agents.compact.name,
            compact_provider = %self.agents.compact.provider,
            compact_model = %self.agents.compact.model.id,
            poll_interval_seconds = self.config.poll_interval_seconds,
            diary_backfill_window = %self.config.diary_backfill_window,
            diary_interval = %self.config.diary_interval,
            max_jobs_per_tick = self.config.max_jobs_per_tick,
            max_concurrent_jobs = self.config.max_concurrent_jobs,
            "memory runtime starting"
        );
        loop {
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break,
                result = self.run_tick() => {
                    if let Err(error) = result {
                        tracing::warn!(error = %error, "memory scheduler tick failed");
                    }
                }
            }
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break,
                () = tokio::time::sleep(self.config.poll_interval()) => {}
            }
        }
        tracing::info!("memory runtime stopped");
        Ok(())
    }

    fn agent_config(&self, kind: MemoryJobKind) -> &SystemAgentConfig {
        match kind {
            MemoryJobKind::Diary => &self.agents.diary,
            MemoryJobKind::Compact => &self.agents.compact,
        }
    }

    async fn run_tick(&self) -> Result<(), MemoryError> {
        let now = OffsetDateTime::now_utc();
        let compaction_interval = self.config.compaction_interval_seconds()?;
        let diary_backfill_window = self.config.diary_backfill_window_seconds()?;
        let diary_interval = self.config.diary_interval_seconds()?;
        let compact_due_before =
            now - time::Duration::seconds(i64::try_from(compaction_interval).unwrap_or(i64::MAX));
        let diary_cutoff =
            now - time::Duration::seconds(i64::try_from(diary_backfill_window).unwrap_or(i64::MAX));
        let diary_due_before =
            now - time::Duration::seconds(i64::try_from(diary_interval).unwrap_or(i64::MAX));
        let enqueued = self
            .storage
            .enqueue_due_memory_jobs(MemoryJobSchedule {
                now,
                diary_cutoff,
                diary_due_before,
                diary_window_seconds: diary_interval,
                compact_due_before,
            })
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let lease_until = now
            + time::Duration::seconds(
                i64::try_from(self.config.lease_duration().as_secs()).unwrap_or(i64::MAX),
            );
        let worker_id = format!(
            "memory:{}:{}",
            std::process::id(),
            now.unix_timestamp_nanos()
        );
        let jobs = self
            .storage
            .claim_memory_jobs(worker_id, self.config.max_jobs_per_tick.max(1), lease_until)
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        tracing::debug!(
            enqueued,
            claimed = jobs.len(),
            "memory scheduler tick claimed work"
        );
        self.run_claimed_jobs(jobs).await
    }

    async fn run_claimed_jobs(&self, jobs: Vec<UserMemoryJob>) -> Result<(), MemoryError> {
        let mut pending = VecDeque::from(jobs);
        let mut active_keys = BTreeSet::new();
        let mut running = JoinSet::new();
        let max_concurrent = self.config.max_concurrent_jobs.max(1) as usize;

        while !pending.is_empty() || !running.is_empty() {
            while running.len() < max_concurrent {
                let Some(index) = pending
                    .iter()
                    .position(|job| !active_keys.contains(&job.memory_key))
                else {
                    break;
                };
                let job = pending.remove(index).expect("pending index exists");
                active_keys.insert(job.memory_key.clone());
                let runtime = (*self).clone();
                running.spawn(async move {
                    let memory_key = job.memory_key.clone();
                    let target_user_name = runtime.load_memory_job_user_name(&job).await;
                    let agent = runtime.agent_config(job.kind);
                    let span = memory_job_span(&job, agent, target_user_name.as_deref());
                    let result = runtime.run_job_with_completion(job).instrument(span).await;
                    (memory_key, result)
                });
            }

            let Some(result) = running.join_next().await else {
                break;
            };
            match result {
                Ok((memory_key, result)) => {
                    active_keys.remove(&memory_key);
                    if let Err(error) = result {
                        tracing::warn!(memory_key, error = %error, "memory job failed");
                    }
                }
                Err(error) => {
                    tracing::warn!(error = %error, "memory job task join failed");
                }
            }
        }
        Ok(())
    }

    async fn load_memory_job_user_name(&self, job: &UserMemoryJob) -> Option<String> {
        let user = memory_user_ref(&job.key);
        let profiles = match self.storage.load_user_profiles(vec![user]).await {
            Ok(profiles) => profiles,
            Err(error) => {
                tracing::warn!(
                    job = %job.id,
                    memory_key = %job.memory_key,
                    message_provider = %job.key.platform,
                    scope_key = %job.key.scope_key,
                    target_user_id = %job.key.user_key,
                    error = %error,
                    "failed to load memory subject profile for tracing"
                );
                return None;
            }
        };
        profiles
            .first()
            .and_then(|profile| memory_profile_display_name(&profile.profile, &job.key.user_key))
    }

    async fn run_job_with_completion(&self, job: UserMemoryJob) -> Result<(), MemoryError> {
        let result = self.run_job(&job).await;
        let completion = match result {
            Ok(()) => MemoryJobCompletion::Completed { job_id: job.id },
            Err(error) if job.attempts >= self.config.max_job_attempts.max(1) => {
                MemoryJobCompletion::Failed {
                    job_id: job.id,
                    error: error.to_string(),
                }
            }
            Err(error) => MemoryJobCompletion::Retry {
                job_id: job.id,
                error: error.to_string(),
                next_run_at: OffsetDateTime::now_utc() + self.config.retry_backoff(job.attempts),
            },
        };
        self.storage
            .finish_memory_job(completion)
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))
    }

    async fn run_job(&self, job: &UserMemoryJob) -> Result<(), MemoryError> {
        tracing::debug!(
            job = %job.id,
            kind = ?job.kind,
            memory_key = %job.memory_key,
            attempts = job.attempts,
            "running memory job"
        );
        match job.kind {
            MemoryJobKind::Diary => self.run_diary_job(job).await,
            MemoryJobKind::Compact => self.run_compact_job(job).await,
        }
    }

    async fn run_diary_job(&self, job: &UserMemoryJob) -> Result<(), MemoryError> {
        let (Some(window_start), Some(window_end)) = (job.window_start, job.window_end) else {
            tracing::warn!(job = %job.id, "diary job has no window");
            return Ok(());
        };
        let turns = self
            .storage
            .load_memory_turn_window(MemoryTurnWindow {
                key: job.key.clone(),
                window_start,
                window_end,
                max_turns: self.config.max_transcript_turns_per_diary_job.max(1),
            })
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        if turns.is_empty() {
            tracing::debug!(job = %job.id, "diary job window had no turns");
            return Ok(());
        }
        let document = self
            .storage
            .load_user_memory_document(job.key.clone())
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let transcript =
            diary_transcript(&job.key, document.as_ref(), &turns, &self.media_store).await;
        let agent_config = self.agent_config(MemoryJobKind::Diary).clone();
        let output = self.run_memory_model(&agent_config, transcript).await?;
        self.storage
            .save_user_memory_diary_entry(NewUserMemoryDiaryEntry {
                key: job.key.clone(),
                window_start,
                window_end,
                source_turn_ids: turns.iter().map(|turn| turn.turn_id).collect(),
                markdown: output.text,
                agent_name: agent_config.name.clone(),
                llm_provider: agent_config.provider.clone(),
                llm_model: output.model_id,
                usage: output.usage,
            })
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        Ok(())
    }

    async fn run_compact_job(&self, job: &UserMemoryJob) -> Result<(), MemoryError> {
        let document = self
            .storage
            .load_user_memory_document(job.key.clone())
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let events = self
            .storage
            .list_pending_memory_events(
                job.key.clone(),
                document
                    .as_ref()
                    .and_then(|document| document.source_event_cutoff),
            )
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let diaries = self
            .storage
            .list_pending_memory_diary_entries(
                job.key.clone(),
                document
                    .as_ref()
                    .and_then(|document| document.source_diary_cutoff),
            )
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        if events.is_empty() && diaries.is_empty() {
            tracing::debug!(job = %job.id, "compact job had no source material");
            return Ok(());
        }

        let input = compact_input(&job.key, document.as_ref(), &events, &diaries);
        let agent_config = self.agent_config(MemoryJobKind::Compact).clone();
        let output = self
            .run_memory_model(&agent_config, Transcript::from_user_text(input))
            .await?;
        let MemoryModelOutput {
            text: markdown,
            model_id: llm_model,
            usage,
        } = output;
        let source_event_cutoff = events
            .iter()
            .map(|event| event.created_at)
            .max()
            .or_else(|| {
                document
                    .as_ref()
                    .and_then(|document| document.source_event_cutoff)
            });
        let source_diary_cutoff =
            diaries
                .iter()
                .map(|entry| entry.created_at)
                .max()
                .or_else(|| {
                    document
                        .as_ref()
                        .and_then(|document| document.source_diary_cutoff)
                });
        tracing::debug!(
            job = %job.id,
            model = %llm_model,
            events = events.len(),
            diaries = diaries.len(),
            markdown_chars = markdown.chars().count(),
            usage_records = usage.len(),
            "saving compact memory profile"
        );
        self.storage
            .save_user_memory_document_revision(NewUserMemoryDocumentRevision {
                key: job.key.clone(),
                markdown,
                source_event_ids: events.iter().map(|event| event.id).collect(),
                source_diary_entry_ids: diaries.iter().map(|entry| entry.id).collect(),
                source_event_cutoff,
                source_diary_cutoff,
                agent_name: agent_config.name.clone(),
                llm_provider: agent_config.provider.clone(),
                llm_model,
                usage,
            })
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        Ok(())
    }

    async fn run_memory_model(
        &self,
        agent_config: &SystemAgentConfig,
        transcript: Transcript,
    ) -> Result<MemoryModelOutput, MemoryError> {
        let agent = agent_config.spec.clone().into_agent(Model {
            backend: RoutedLlmBackend::new(self.llms.clone(), agent_config.provider.clone()),
            spec: agent_config.model.clone(),
        });
        let run = agent
            .run(transcript)
            .await
            .map_err(|error| MemoryError::Model(error.to_string()))?;
        memory_model_output(run, &agent_config.model.id)
    }
}

#[derive(Debug, Clone)]
struct MemoryModelOutput {
    text: String,
    model_id: ModelId,
    usage: Vec<UsageRecord>,
}

fn memory_model_output(
    run: AgentRun,
    fallback_model_id: &ModelId,
) -> Result<MemoryModelOutput, MemoryError> {
    let usage = run.all_usage();
    let model_id = run
        .last_model_id
        .unwrap_or_else(|| fallback_model_id.clone());
    match run.outcome {
        AgentOutcome::Completed { answer } => {
            let text = answer.text.trim().to_string();
            if text.is_empty() {
                return Err(MemoryError::Model(
                    "memory model returned empty text".to_string(),
                ));
            }
            Ok(MemoryModelOutput {
                text,
                model_id,
                usage,
            })
        }
        AgentOutcome::IterationLimit { max_iterations } => Err(MemoryError::Model(format!(
            "memory model hit iteration limit ({max_iterations})"
        ))),
        AgentOutcome::Failed { error, partial } => {
            let mut message = error.to_string();
            if let Some(partial) = partial
                && !partial.text.trim().is_empty()
            {
                message.push_str("\n\nPartial answer:\n");
                message.push_str(&partial.text);
            }
            Err(MemoryError::Model(message))
        }
        AgentOutcome::Cancelled { reason } => Err(MemoryError::Model(format!(
            "memory model cancelled: {reason}"
        ))),
    }
}

async fn diary_transcript<M>(
    key: &UserMemoryKey,
    document: Option<&UserMemoryDocument>,
    turns: &[UserMemoryTurn],
    media_store: &M,
) -> Transcript
where
    M: MediaStore,
{
    let mut blocks = Vec::new();
    blocks.push(ContentBlock::Text {
        text: diary_header_text(key, document),
    });
    for turn in turns {
        blocks.push(ContentBlock::Text {
            text: diary_turn_text(turn),
        });
        append_diary_image_blocks(&mut blocks, turn, media_store).await;
    }
    let mut transcript = Transcript::new();
    transcript.push(TranscriptTurn {
        role: TurnRole::User,
        blocks,
        metadata: serde_json::Value::Null,
    });
    transcript
}

async fn append_diary_image_blocks<M>(
    blocks: &mut Vec<ContentBlock>,
    turn: &UserMemoryTurn,
    media_store: &M,
) where
    M: MediaStore,
{
    for (index, image) in turn.image_context.iter().enumerate() {
        blocks.push(ContentBlock::Text {
            text: format!(
                "Visual content for turn {} image {} (source: {}, uri: {}).",
                turn.turn_id,
                index + 1,
                memory_image_source_label(&image.source),
                image.image_uri
            ),
        });
        match media_store.media_from_uri(&image.image_uri).await {
            Ok(media) if memory_diary_supports_media(media.as_ref()) => {
                blocks.push(ContentBlock::Media { media });
            }
            Ok(media) => tracing::debug!(
                turn = %turn.turn_id,
                source = %image.source,
                uri = %media.uri(),
                category = ?media.category(),
                mime_type = %media.mime_type(),
                "skipping unsupported diary image media"
            ),
            Err(error) => tracing::warn!(
                turn = %turn.turn_id,
                source = %image.source,
                uri = %image.image_uri,
                error = %error,
                "skipping diary image media"
            ),
        }
    }
}

#[cfg(test)]
fn diary_input(
    key: &UserMemoryKey,
    document: Option<&UserMemoryDocument>,
    turns: &[UserMemoryTurn],
) -> String {
    let mut out = diary_header_text(key, document);
    for turn in turns {
        out.push_str(&diary_turn_text(turn));
    }
    out
}

fn diary_header_text(key: &UserMemoryKey, document: Option<&UserMemoryDocument>) -> String {
    let mut out = String::new();
    out.push_str("# Subject\n");
    out.push_str(&format!(
        "platform: {}\nscope: {}\nuser: {}\n\n",
        key.platform, key.scope_key, key.user_key
    ));
    out.push_str("# Current Memory Profile\n");
    out.push_str(
        document
            .map(|document| document.markdown.trim())
            .filter(|markdown| !markdown.is_empty())
            .unwrap_or(EMPTY_MEMORY),
    );
    out.push_str("\n\n# Completed Turns\n");
    out
}

fn diary_turn_text(turn: &UserMemoryTurn) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "\n## Turn {} ({})\nUser [{}]: {}\n",
        turn.turn_id, turn.completed_at, turn.user_display_name, turn.user_content
    ));
    if let Some(answer) = &turn.assistant_content {
        out.push_str("Assistant: ");
        out.push_str(answer);
        out.push('\n');
    }
    append_image_context(&mut out, &turn.image_context);
    append_audio_transcriptions(&mut out, &turn.audio_transcriptions);
    out
}

fn append_image_context(out: &mut String, images: &[UserMemoryImageContext]) {
    if images.is_empty() {
        return;
    }
    out.push_str("Image content blocks:\n");
    for (index, image) in images.iter().enumerate() {
        let mut metadata = vec![
            format!("source: {}", memory_image_source_label(&image.source)),
            format!("uri: {}", image.image_uri),
        ];
        if let Some(mime_type) = image
            .mime_type
            .as_deref()
            .filter(|mime_type| !mime_type.is_empty())
        {
            metadata.push(format!("mime_type: {mime_type}"));
        }
        out.push_str(&format!(
            "- Image {} ({})\n",
            index + 1,
            metadata.join(", ")
        ));
    }
}

fn memory_image_source_label(source: &str) -> &str {
    if source.starts_with("platform:") {
        "user_or_quoted_message_attachment"
    } else if source == "generate_image" {
        "generated_image"
    } else {
        source
    }
}

fn memory_diary_supports_media(media: &dyn chudbot_api::MediaRef) -> bool {
    matches!(media.category(), MediaCategory::Image)
        && MEMORY_DIARY_IMAGE_MIME_TYPES
            .iter()
            .any(|supported| image_mime_type_eq(media.mime_type(), supported))
}

fn image_mime_type_eq(actual: &str, expected: &str) -> bool {
    let actual = actual.split(';').next().unwrap_or("").trim();
    actual.eq_ignore_ascii_case(expected)
}

fn append_audio_transcriptions(out: &mut String, transcriptions: &[UserMemoryAudioTranscription]) {
    let mut rendered_any = false;
    for (index, transcription) in transcriptions.iter().enumerate() {
        let text = transcription.text.trim();
        if text.is_empty() {
            continue;
        }
        if !rendered_any {
            out.push_str("Audio transcriptions:\n");
            rendered_any = true;
        }
        let mut metadata = Vec::new();
        if let Some(uri) = transcription
            .audio_uri
            .as_deref()
            .filter(|uri| !uri.is_empty())
        {
            metadata.push(format!("uri: {uri}"));
        }
        if let Some(language) = transcription
            .language
            .as_deref()
            .filter(|language| !language.is_empty())
        {
            metadata.push(format!("language: {language}"));
        }
        if let Some(duration) = transcription.duration_seconds {
            metadata.push(format!("duration_seconds: {duration:.2}"));
        }
        let metadata = if metadata.is_empty() {
            String::new()
        } else {
            format!(" ({})", metadata.join(", "))
        };
        out.push_str(&format!("- Audio {}{}: {}\n", index + 1, metadata, text));
    }
}

fn compact_input(
    key: &UserMemoryKey,
    document: Option<&UserMemoryDocument>,
    events: &[UserMemoryEvent],
    diaries: &[UserMemoryDiaryEntry],
) -> String {
    let mut out = String::new();
    out.push_str("# Subject\n");
    out.push_str(&format!(
        "platform: {}\nscope: {}\nuser: {}\n\n",
        key.platform, key.scope_key, key.user_key
    ));
    out.push_str("# Current Memory Profile\n");
    out.push_str(
        document
            .map(|document| document.markdown.trim())
            .filter(|markdown| !markdown.is_empty())
            .unwrap_or(EMPTY_MEMORY),
    );
    out.push_str("\n\n# New Raw Memory Events\n");
    if events.is_empty() {
        out.push_str(EMPTY_MEMORY);
        out.push('\n');
    } else {
        for event in events {
            out.push_str(&format!(
                "\n- id: {}\n  kind: {}\n  created_at: {}\n  body: {}\n",
                event.id,
                event_kind_label(event.kind),
                event.created_at,
                event.body.replace('\n', "\n    ")
            ));
        }
    }
    out.push_str("\n# New Diary Entries\n");
    if diaries.is_empty() {
        out.push_str(EMPTY_MEMORY);
        out.push('\n');
    } else {
        for diary in diaries {
            out.push_str(&format!(
                "\n## Diary {} ({} - {})\n{}\n",
                diary.id, diary.window_start, diary.window_end, diary.markdown
            ));
        }
    }
    out.push_str("\n# Required Profile Headings\n");
    out.push_str(
        "# User Memory\n\n## Identity And Names\n## Relationships\n## Preferences\n## Projects And Interests\n## Server Lore\n## Roast Material\n## Boundaries And Avoidances\n## Uncertain Or Low-Confidence Notes\n",
    );
    out
}

/// Errors from the memory runtime.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Configuration is invalid.
    #[error(transparent)]
    Config(#[from] MemoryConfigError),
    /// Storage operation failed.
    #[error("storage error: {0}")]
    Storage(String),
    /// Model operation failed.
    #[error("model error: {0}")]
    Model(String),
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chudbot_api::{
        AgentLimits, ConversationId, ExternalId, ModelSpec, PlatformName, SamplingOptions,
        UserMemoryEvent, UserMemoryEventKind, UserProfile,
    };
    use test_case::test_case;
    use time::macros::datetime;

    use super::*;

    #[test_case("1s", 1)]
    #[test_case("5m", 300)]
    #[test_case("2h", 7200)]
    #[test_case("3d", 259200)]
    fn parses_duration_suffixes(input: &str, expected: u64) {
        assert_eq!(parse_duration_seconds(input).unwrap(), expected);
    }

    #[test]
    fn rejects_invalid_duration() {
        assert!(parse_duration_seconds("24").is_err());
        assert!(parse_duration_seconds("0h").is_err());
        assert!(parse_duration_seconds("xh").is_err());
    }

    #[test]
    fn default_diary_backfill_window_is_three_days() {
        assert_eq!(
            MemoryConfig::default()
                .diary_backfill_window_seconds()
                .unwrap(),
            3 * 24 * 60 * 60
        );
    }

    #[test]
    fn default_diary_interval_is_one_day() {
        assert_eq!(
            MemoryConfig::default().diary_interval_seconds().unwrap(),
            24 * 60 * 60
        );
    }

    #[test]
    fn memory_agent_providers_use_named_agents_with_default_fallbacks() {
        let mut agents = BTreeMap::new();
        agents.insert(
            MEMORY_DIARY_AGENT.to_string(),
            test_agent_config("openai", "gpt-5.5"),
        );

        let providers = MemoryConfig::default()
            .resolved_agent_providers(&agents, AgentLimits { max_iterations: 4 });

        assert_eq!(
            providers,
            vec![
                (MEMORY_DIARY_AGENT.to_string(), ProviderName::new("openai")),
                (MEMORY_COMPACT_AGENT.to_string(), ProviderName::new("grok")),
            ]
        );
    }

    #[test]
    fn memory_config_rejects_removed_model_fields() {
        for field in [
            "provider",
            "max_diary_output_tokens",
            "max_profile_output_tokens",
        ] {
            let mut value = json!({ "enabled": true });
            value
                .as_object_mut()
                .expect("object")
                .insert(field.to_string(), json!("stale"));
            let error = serde_json::from_value::<MemoryConfig>(value).unwrap_err();

            assert!(
                error
                    .to_string()
                    .contains(&format!("unknown field `{field}`"))
            );
        }
    }

    fn test_agent_config(provider: &str, model: &str) -> AgentConfig {
        AgentConfig {
            provider: ProviderName::new(provider),
            system_prompt: "configured prompt".to_string(),
            model: ModelSpec {
                id: ModelId::new(model),
                server_tools: BTreeSet::new(),
                sampling: SamplingOptions::default(),
                provider_options: None,
            },
            server_tools: None,
            client_tools: None,
            limits: None,
            image_generation: None,
            video_generation: None,
            audio_transcription: None,
            memory: false,
            subagents: BTreeMap::new(),
        }
    }

    #[test]
    fn builds_guild_scoped_memory_key() {
        let key = key_from_user_ref(&UserRef {
            platform: PlatformName::new("discord"),
            guild_id: Some(ExternalId::new("guild-1")),
            user_id: ExternalId::new("user-1"),
        });

        assert_eq!(key.platform.as_str(), "discord");
        assert_eq!(key.scope_key, "guild:guild-1");
        assert_eq!(key.user_key, "user-1");
        assert_eq!(key.memory_key(), "discord:guild:guild-1:user-1");
    }

    #[test]
    fn memory_user_ref_extracts_guild_scope() {
        let user = memory_user_ref(&UserMemoryKey {
            platform: PlatformName::new("discord"),
            scope_key: "guild:guild-1".to_string(),
            user_key: "user-1".to_string(),
        });

        assert_eq!(user.platform.as_str(), "discord");
        assert_eq!(
            user.guild_id.as_ref().map(ExternalId::as_str),
            Some("guild-1")
        );
        assert_eq!(user.user_id.as_str(), "user-1");
    }

    #[test]
    fn memory_profile_display_name_prefers_readable_names() {
        let profile = UserProfile {
            id: UserRef {
                platform: PlatformName::new("discord"),
                guild_id: Some(ExternalId::new("guild-1")),
                user_id: ExternalId::new("user-1"),
            },
            username: "alice_global".to_string(),
            name: Some("Alice Global".to_string()),
            display_name: Some("Alice Guild".to_string()),
            avatar_url: None,
            is_bot: false,
        };

        assert_eq!(
            memory_profile_display_name(&profile, "user-1").as_deref(),
            Some("Alice Guild")
        );
    }

    #[test]
    fn memory_profile_display_name_omits_id_fallback() {
        let profile = UserProfile {
            id: UserRef {
                platform: PlatformName::new("discord"),
                guild_id: Some(ExternalId::new("guild-1")),
                user_id: ExternalId::new("user-1"),
            },
            username: "user-1".to_string(),
            name: None,
            display_name: None,
            avatar_url: None,
            is_bot: false,
        };

        assert_eq!(memory_profile_display_name(&profile, "user-1"), None);
    }

    #[test]
    fn prompt_guidance_names_tools_and_proactive_policy() {
        let guidance = prompt_guidance();

        assert!(guidance.contains(LOOKUP_USER_MEMORY_TOOL));
        assert!(guidance.contains(REMEMBER_USER_MEMORY_TOOL));
        assert!(guidance.contains(FORGET_USER_MEMORY_TOOL));
        assert!(guidance.contains("CRITICAL: Memory System"));
        assert!(guidance.contains("`author` of a message"));
        assert!(guidance.contains("MUST load memory about that user"));
        assert!(guidance.contains("Do not respond to a user"));
        assert!(guidance.contains("any time you see a user for the first time"));
        assert!(guidance.contains("any **mention** of a user"));
        assert!(guidance.contains("IT IS CRITICAL TO USE THE MEMORY SYSTEM PROACTIVELY"));
        assert!(guidance.contains("The tool calls are cheap"));
        assert!(guidance.contains("respect their humanity"));
        assert!(guidance.contains("grain of salt"));
        assert!(guidance.contains("trust the current message"));
        assert!(guidance.contains("sensitive personal information"));
    }

    #[test]
    fn diary_input_includes_audio_transcriptions() {
        let key = UserMemoryKey {
            platform: PlatformName::new("discord"),
            scope_key: "guild:guild-1".to_string(),
            user_key: "user-1".to_string(),
        };
        let turn = UserMemoryTurn {
            conversation_id: ConversationId::new(),
            turn_id: TurnId::new(),
            completed_at: datetime!(2026-06-03 22:27:01 UTC),
            user_display_name: "Chud".to_string(),
            user_content: "@Chudbot".to_string(),
            assistant_content: Some("Noted.".to_string()),
            image_context: Vec::new(),
            audio_transcriptions: vec![UserMemoryAudioTranscription {
                tool_trace_id: 42,
                audio_uri: Some("file://audio/voice.ogg".to_string()),
                text: "I am allergic to coconut.".to_string(),
                language: Some("en".to_string()),
                duration_seconds: Some(3.25),
            }],
        };

        let input = diary_input(&key, None, &[turn]);

        assert!(input.contains("Audio transcriptions:"));
        assert!(input.contains("file://audio/voice.ogg"));
        assert!(input.contains("language: en"));
        assert!(input.contains("duration_seconds: 3.25"));
        assert!(input.contains("I am allergic to coconut."));
    }

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
