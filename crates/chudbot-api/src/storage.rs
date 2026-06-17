//! Bot persistence contracts.
//!
//! These operations are shaped around how the bot uses persistence rather than
//! around how a SQL database happens to expose rows. A Postgres implementation
//! can still answer them with joins and indexes; a JSON-file implementation
//! could answer the same calls by loading one document.
//!
//! # Runtime Flow
//!
//! 1. Platform code resolves an incoming message to an existing conversation
//!    with [`ConversationLookup`] or opens one with [`OpenConversation`].
//! 2. The bot starts a [`Turn`] with [`BeginTurn`], captures the prompt and
//!    model-facing context with [`SaveTurnInput`], then appends tool and model
//!    step trace rows as the agent runs.
//! 3. Platform messages and threads are connected back to durable state through
//!    [`MessageLink`] and [`ChannelLink`].
//! 4. [`FinishTurn`] records the terminal outcome, usage, reply text, and the
//!    response ordering that future turns may replay.
//! 5. The trace viewer and future model requests load [`ConversationSnapshot`],
//!    while background memory jobs use the user-memory DTOs and job contracts
//!    near the bottom of this module.
//!
//! The storage backend owns atomicity for operations that allocate ordinals,
//! claim leases, or decide retry eligibility. The bot runtime owns provider
//! calls, platform IO, and policy decisions before it invokes this trait.

use std::future::Future;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::ids::{
    ChannelRef, ConversationId, ExternalId, MessageRef, ModelId, PlatformName, ProviderName,
    TurnId, UserRef,
};
use crate::media::MediaUri;
use crate::platform::UserProfile;
use crate::tool::ToolTrace;
use crate::transcript::{ProviderContinuation, Transcript};
use crate::usage::{UsageCostQuery, UsageCostRow, UsageRecord};

// Conversation lifecycle, trace replay, and runtime policy DTOs.

/// Privacy mode for context gathering before a model request.
///
/// The runtime interprets these modes when choosing how much platform history
/// to fetch. Storage only persists the selected policy and user opt-in state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PrivacyMode {
    /// Fetch history visible in the active channel.
    Open {
        /// Maximum platform messages to include.
        history_size: u32,
    },
    /// Fetch history only from a configured channel.
    ChannelOnly {
        /// Allowed channel.
        channel: ChannelRef,
        /// Maximum platform messages to include.
        history_size: u32,
    },
    /// Fetch history, but redact messages for users who have not opted in.
    OptIn,
    /// Rebuild context only from stored conversation turns.
    ConversationOnly,
}

/// Durable handles that can identify a conversation.
///
/// Message and channel lookups go through storage-owned link tables so platform
/// adapters do not need to know the internal conversation id.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConversationLookup {
    /// Load by internal conversation id.
    Id {
        /// Conversation id.
        id: ConversationId,
    },
    /// Load the conversation linked to a platform message.
    Message {
        /// Platform message.
        message: MessageRef,
    },
    /// Load the conversation linked to a platform channel. This covers
    /// messaging platforms where a reply thread has its own channel id.
    Channel {
        /// Platform channel.
        channel: ChannelRef,
    },
}

/// Read model for a conversation and all trace data needed by callers.
///
/// This is the main read shape for both prompt replay and `/c/<id>` rendering:
/// callers do not stitch separate row types together outside storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSnapshot {
    /// Conversation metadata.
    pub conversation: Conversation,
    /// Turns ordered by user-message ordinal.
    pub turns: Vec<TurnSnapshot>,
}

/// Conversation metadata shared by future turns and the trace viewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    /// Conversation id.
    pub id: ConversationId,
    /// Created timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Platform channel where the conversation started.
    pub channel: ChannelRef,
    /// User who started it.
    pub created_by: UserRef,
    /// Root platform message.
    pub root_message: MessageRef,
    /// Initial model recorded at conversation creation.
    pub initial_model: ModelId,
    /// Agent selected when this conversation was opened.
    pub agent_name: String,
    /// Provider selected when this conversation was opened.
    pub provider: ProviderName,
    /// Frozen system/developer instructions for this conversation.
    ///
    /// Existing conversations always load this from storage. Static app config
    /// changes only affect conversations opened after the change.
    pub system_instructions: String,
    /// Optional title.
    pub title: Option<String>,
    /// Stop timestamp.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub stopped_at: Option<OffsetDateTime>,
    /// User who stopped it.
    pub stopped_by: Option<UserRef>,
}

/// One turn plus the trace data needed to reconstruct model input and output.
///
/// The vectors are already ordered for replay/rendering. Storage backends should
/// do that ordering once instead of leaking table-specific sort rules to
/// callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnSnapshot {
    /// Turn metadata.
    pub turn: Turn,
    /// System/developer instructions used for this attempt/turn.
    pub system_instructions: Option<String>,
    /// Novel context items captured for this turn.
    pub context: Vec<ContextItem>,
    /// Tool/server/grounding trace events.
    pub tool_trace: Vec<ToolTrace>,
    /// Ordered provider model-step traces captured during this attempt.
    #[serde(default)]
    pub model_steps: Vec<ModelStepTrace>,
    /// Assets that should be replayed with this turn.
    pub replay_assets: Vec<TurnAsset>,
    /// Usage/cost accumulated by this turn.
    pub usage: Vec<UsageRecord>,
}

/// Provider model-step trace for replay and audit.
///
/// A single user turn may involve multiple provider calls when the model asks
/// for tools or continuation. Each step records the provider/model boundary and
/// the opaque continuation the provider needs for future replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStepTrace {
    /// Zero-based model step ordinal within the attempt.
    pub ordinal: i32,
    /// Why the step ended.
    pub kind: ModelStepKind,
    /// Provider that produced the step.
    pub provider: ProviderName,
    /// Model id reported by the provider for this step.
    pub model: ModelId,
    /// Opaque provider continuation/output to replay later.
    pub continuation: Option<ProviderContinuation>,
}

/// Model step terminal kind.
///
/// This describes why one provider call stopped, not the final turn status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStepKind {
    /// Provider returned final assistant content.
    Final,
    /// Provider requested another provider call without client tools.
    Continue,
    /// Provider requested client-side tool calls.
    ClientTools,
}

/// Turn metadata for one user message.
///
/// Turn ordering has two axes: [`Self::ordinal`] tracks user-message arrival,
/// while [`Self::response_ordinal`] tracks which assistant replies are eligible
/// for future replay. That split lets hot-thread turns run concurrently without
/// leaking incomplete or later replies into an earlier prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Turn id.
    pub id: TurnId,
    /// Zero-based user-message ordinal within the conversation.
    ///
    /// This is the stable viewer/order-of-arrival number. It is deliberately
    /// separate from [`Self::response_ordinal`], because concurrent hot-thread
    /// turns can complete in a different order than users sent them.
    pub ordinal: i64,
    /// Highest response ordinal visible to this turn's model input.
    ///
    /// Storage captures this when the turn begins, using the platform message
    /// timestamp. A turn only replays completed turns with
    /// `response_ordinal <= history_cutoff`, so pending concurrent turns do
    /// not leak into the prompt and retries rebuild the original context.
    pub history_cutoff: Option<i64>,
    /// Zero-based ordinal assigned when this turn's assistant response becomes
    /// part of future conversation history.
    ///
    /// Failed and cancelled turns stay `None`: they are visible in the trace
    /// but are not replayed into later model requests.
    pub response_ordinal: Option<i64>,
    /// Created timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Timestamp of the user's platform message.
    #[serde(with = "time::serde::rfc3339")]
    pub user_message_created_at: OffsetDateTime,
    /// Completed timestamp.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub completed_at: Option<OffsetDateTime>,
    /// User message.
    pub user_message: MessageRef,
    /// User.
    pub user: UserRef,
    /// User display name at turn time.
    pub user_display_name: String,
    /// User content.
    pub user_content: String,
    /// Assistant reply message.
    pub assistant_message: Option<MessageRef>,
    /// Assistant content.
    pub assistant_content: Option<String>,
    /// Status.
    pub status: TurnStatus,
    /// Error if failed/cancelled.
    pub error: Option<String>,
    /// Agent active for this turn.
    pub agent_name: Option<String>,
    /// Provider/model that answered this turn.
    pub provider: Option<ProviderName>,
    /// Model that answered this turn.
    pub model: Option<ModelId>,
    /// Build version (`app_versions.id`) active when this turn last ran.
    ///
    /// `None` for turns imported from storage that predate version tracking.
    #[serde(default)]
    pub app_version_id: Option<i32>,
}

/// Turn status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    /// Running or waiting.
    Pending,
    /// Completed successfully.
    Completed,
    /// Failed and eligible for retry if it is the latest failed turn.
    Failed,
    /// Cancelled by operator/runtime.
    Cancelled,
}

/// Context item persisted for trace/viewer replay.
///
/// These are normalized model-input records. They may point at platform
/// messages, stored media, memory recall, or tool-generated context; the source
/// string is intentionally runtime-defined so new context producers do not need
/// new storage API methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextItem {
    /// Position in the turn context.
    pub position: i32,
    /// Source label.
    pub source: String,
    /// Role string.
    pub role: String,
    /// Content. For assets this is usually the stable asset URI string.
    pub content: String,
    /// Platform message when applicable.
    pub message: Option<MessageRef>,
}

/// Asset associated with a turn.
///
/// Assets are stored separately from transcript text so traces can replay media
/// by stable [`MediaUri`] without embedding bytes in conversation rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnAsset {
    /// Stored asset URI.
    pub uri: MediaUri,
    /// Owning turn.
    pub turn_id: TurnId,
    /// Context source label or tool name that produced it.
    pub source: String,
    /// MIME type.
    pub mime_type: Option<String>,
}

/// Input for opening a new durable conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenConversation {
    /// Channel where it starts.
    pub channel: ChannelRef,
    /// User who starts it.
    pub created_by: UserRef,
    /// Root message.
    pub root_message: MessageRef,
    /// Initial model.
    pub initial_model: ModelId,
    /// Agent name.
    pub agent_name: String,
    /// Provider.
    pub provider: ProviderName,
    /// Frozen conversation instructions.
    pub system_instructions: String,
    /// Optional title.
    pub title: Option<String>,
}

/// Input for allocating a new turn in a conversation.
///
/// Storage uses this to allocate the user-message ordinal and capture the
/// history cutoff before model work starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeginTurn {
    /// Conversation id.
    pub conversation_id: ConversationId,
    /// User message.
    pub user_message: MessageRef,
    /// Timestamp of the user's platform message.
    #[serde(with = "time::serde::rfc3339")]
    pub user_message_created_at: OffsetDateTime,
    /// User.
    pub user: UserRef,
    /// Display name.
    pub user_display_name: String,
    /// Content.
    pub user_content: String,
}

/// Save the prompt/context metadata for a turn before the model runs.
///
/// This call marks the boundary between runtime prompt assembly and provider
/// execution. Implementations may persist both normalized context rows and the
/// optional transcript snapshot, but callers must not assume every backend
/// stores the transcript verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveTurnInput {
    /// Turn id.
    pub turn_id: TurnId,
    /// Agent name.
    pub agent_name: String,
    /// Provider.
    pub provider: ProviderName,
    /// Model.
    pub model: ModelId,
    /// System/developer instructions.
    pub system_instructions: String,
    /// Context items.
    pub context: Vec<ContextItem>,
    /// Initial transcript assembled from the context.
    ///
    /// This is skipped during serialization because it is a runtime convenience,
    /// not part of the portable storage DTO. Some stores persist only the
    /// normalized context rows above.
    #[serde(skip)]
    pub transcript: Option<Transcript>,
}

/// Terminal update for a turn.
///
/// A completed turn becomes replayable for later model requests. Failed and
/// cancelled turns remain visible in traces but do not receive a response
/// ordinal unless a later retry succeeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FinishTurn {
    /// Completed turn.
    Completed {
        /// Turn id.
        turn_id: TurnId,
        /// Assistant answer.
        assistant_content: String,
        /// Posted assistant message.
        assistant_message: MessageRef,
        /// Usage/cost accumulated by this turn.
        usage: Vec<UsageRecord>,
    },
    /// Failed turn.
    Failed {
        /// Turn id.
        turn_id: TurnId,
        /// Error.
        error: String,
        /// Salvaged assistant content.
        assistant_content: Option<String>,
        /// Posted failure message, if any.
        assistant_message: Option<MessageRef>,
        /// Usage/cost accumulated before failure.
        usage: Vec<UsageRecord>,
    },
    /// Cancelled turn.
    Cancelled {
        /// Turn id.
        turn_id: TurnId,
        /// Reason.
        reason: String,
        /// Usage/cost accumulated before cancellation.
        usage: Vec<UsageRecord>,
    },
}

/// Link from a platform message to a conversation/turn.
///
/// This is the storage boundary that lets platform replies, edits, and follow-up
/// messages find the internal conversation without embedding Chudbot ids in the
/// platform payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageLink {
    /// Platform message.
    pub message: MessageRef,
    /// Conversation id.
    pub conversation_id: ConversationId,
    /// Turn id.
    pub turn_id: TurnId,
    /// Link role, e.g. `user`, `assistant`, `assistant_status`.
    pub role: String,
}

/// Link from a platform channel/thread to a conversation/turn.
///
/// Thread-like platforms may expose the reply surface as its own channel id.
/// This link keeps that platform shape out of the conversation model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelLink {
    /// Platform channel.
    pub channel: ChannelRef,
    /// Conversation id.
    pub conversation_id: ConversationId,
    /// Turn id that created or claimed this channel.
    pub turn_id: TurnId,
    /// Link role, e.g. `thread`.
    pub role: String,
}

/// Prepared retry data.
///
/// The storage backend owns the atomic eligibility check. In the concurrent
/// turn model, a failed turn can be retried even when later turns have
/// completed, because the retry reuses the failed turn's original
/// `history_cutoff` and receives a fresh `response_ordinal` only if it
/// succeeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryTurn {
    /// Conversation with all turns needed to rebuild history.
    pub conversation: ConversationSnapshot,
    /// Turn being retried.
    pub turn_id: TurnId,
}

/// Conversation stop/resume request.
///
/// Stopping is durable state, not a platform-only affordance, so future events
/// can consistently reject or resume work across process restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConversationStop {
    /// Stop conversation.
    Stop {
        /// Conversation id.
        conversation_id: ConversationId,
        /// Admin user.
        stopped_by: UserRef,
    },
    /// Resume conversation.
    Resume {
        /// Conversation id.
        conversation_id: ConversationId,
    },
}

/// Inputs needed to resolve the effective agent selection for an event.
///
/// The runtime passes every scope it knows; storage applies the precedence used
/// by persisted selections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveAgent {
    /// Messaging platform/provider name.
    pub message_provider: PlatformName,
    /// Conversation id when known.
    pub conversation_id: Option<ConversationId>,
    /// Guild/workspace key when known.
    pub guild_key: Option<String>,
    /// Channel key.
    pub channel_key: String,
    /// User key.
    pub user_key: String,
}

/// Scoped agent selection target.
///
/// The variants are ordered from narrowest to broadest. Resolution checks can
/// walk the same hierarchy without making callers understand storage keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum AgentSelection {
    /// Conversation-scoped selection.
    Conversation {
        /// Conversation id.
        conversation_id: ConversationId,
    },
    /// User-in-guild scoped selection.
    User {
        /// Messaging platform/provider name.
        message_provider: PlatformName,
        /// Guild/workspace key.
        guild_key: String,
        /// User key.
        user_key: String,
    },
    /// Channel-scoped selection.
    Channel {
        /// Messaging platform/provider name.
        message_provider: PlatformName,
        /// Guild/workspace key when known.
        guild_key: Option<String>,
        /// Channel key.
        channel_key: String,
    },
    /// Guild/workspace-scoped selection.
    Guild {
        /// Messaging platform/provider name.
        message_provider: PlatformName,
        /// Guild/workspace key.
        guild_key: String,
    },
    /// Platform fallback selection.
    Platform {
        /// Messaging platform/provider name.
        message_provider: PlatformName,
    },
}

/// Settings needed to decide privacy/context for an incoming message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSettings {
    /// Effective privacy mode.
    pub privacy: PrivacyMode,
    /// Whether the acting user opted in.
    pub user_opted_in: bool,
}

// Media generation, viewer profile, and user-memory DTOs.

/// Video job status persisted by storage.
///
/// Async video providers return provider job ids first and assets later. This
/// row is the durable join point between a turn, provider polling, and the final
/// stored media URI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredVideoJob {
    /// Turn id.
    pub turn_id: TurnId,
    /// Provider.
    pub provider: ProviderName,
    /// Provider job id.
    pub provider_job_id: String,
    /// Prompt.
    pub prompt: String,
    /// Status string.
    pub status: String,
    /// Stored output asset URI.
    pub output_uri: Option<MediaUri>,
    /// Error.
    pub error: Option<String>,
}

/// Input for creating a video job row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateVideoJob {
    /// Turn id.
    pub turn_id: TurnId,
    /// Provider.
    pub provider: ProviderName,
    /// Provider job id.
    pub provider_job_id: String,
    /// Prompt.
    pub prompt: String,
}

/// Input for updating a video job row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateVideoJob {
    /// Provider.
    pub provider: ProviderName,
    /// Provider job id.
    pub provider_job_id: String,
    /// Status.
    pub status: String,
    /// Stored output asset URI.
    pub output_uri: Option<MediaUri>,
    /// Error.
    pub error: Option<String>,
}

/// Input for counting video generations that consume a rolling-window quota.
///
/// Storage counts jobs by platform scope so the runtime can enforce limits
/// without hard-coding provider-specific job states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountActiveVideoGenerations {
    /// Messaging platform whose video generations are counted.
    pub platform: PlatformName,
    /// Platform workspace/server scope id. `None` counts unscoped channels for
    /// the platform.
    pub scope_id: Option<ExternalId>,
    /// Rolling window length in seconds.
    pub interval_seconds: u64,
}

/// Stored user metadata for viewer read models.
///
/// This is deliberately a viewer/read shape: runtime identity comes from
/// [`UserRef`], while this DTO carries display and avatar data for trace pages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredUserProfile {
    /// Last platform profile seen for this user.
    pub profile: UserProfile,
    /// Cached local avatar media URI, when downloaded.
    pub avatar: Option<MediaUri>,
}

/// Platform-neutral key for one user's memory in one workspace/scope.
///
/// The `scope_key` is already normalized by the platform/runtime layer, usually
/// with a prefix such as `guild:`. Storage treats the three fields as opaque
/// equality keys and does not parse platform-specific ids.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UserMemoryKey {
    /// Messaging platform, e.g. `discord`.
    pub platform: PlatformName,
    /// Workspace/server scope, e.g. `guild:<guild_id>`.
    pub scope_key: String,
    /// Platform user id inside the scope.
    pub user_key: String,
}

impl UserMemoryKey {
    /// Stable string key used by durable memory job leases and dedupe records.
    pub fn memory_key(&self) -> String {
        // Keep this flat and deterministic: leases compare the rendered key
        // without needing to understand platform-specific scope syntax.
        format!("{}:{}:{}", self.platform, self.scope_key, self.user_key)
    }
}

/// Raw user-memory event kind for the append-only memory ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserMemoryEventKind {
    /// Explicit or inferred memory addition.
    Remember,
    /// Correction to older memory.
    Correction,
    /// Tombstone/forget request.
    Forget,
    /// Observation emitted by the diary pipeline.
    DiaryObservation,
    /// Operator-authored note.
    OperatorNote,
}

/// Raw user-memory ledger event.
///
/// Events are provenance-rich inputs to memory compaction. They are not the
/// current profile by themselves; [`UserMemoryDocument`] is the compact read
/// model used for lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMemoryEvent {
    /// Event id.
    pub id: uuid::Uuid,
    /// Subject user.
    pub key: UserMemoryKey,
    /// Acting user key, when known.
    pub actor_user_key: Option<String>,
    /// Event kind.
    pub kind: UserMemoryEventKind,
    /// Event body.
    pub body: String,
    /// Lightweight tags.
    pub tags: Vec<String>,
    /// Confidence score supplied by a tool or operator.
    pub confidence: Option<f32>,
    /// Source conversation.
    pub source_conversation_id: Option<ConversationId>,
    /// Source turn.
    pub source_turn_id: Option<TurnId>,
    /// Source tool trace row id.
    pub source_tool_trace_id: Option<i64>,
    /// Superseded event id, when this event replaces another.
    pub supersedes_event_id: Option<uuid::Uuid>,
    /// Created timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Updated timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

/// Input for appending a memory ledger event.
///
/// Appends may be produced by tools, operators, diary jobs, or future import
/// paths. Storage assigns ids and timestamps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewUserMemoryEvent {
    /// Subject user.
    pub key: UserMemoryKey,
    /// Acting user key, when known.
    pub actor_user_key: Option<String>,
    /// Event kind.
    pub kind: UserMemoryEventKind,
    /// Event body.
    pub body: String,
    /// Lightweight tags.
    pub tags: Vec<String>,
    /// Confidence score supplied by a tool or operator.
    pub confidence: Option<f32>,
    /// Source conversation.
    pub source_conversation_id: Option<ConversationId>,
    /// Source turn.
    pub source_turn_id: Option<TurnId>,
    /// Source tool trace row id.
    pub source_tool_trace_id: Option<i64>,
    /// Superseded event id, when this event replaces another.
    pub supersedes_event_id: Option<uuid::Uuid>,
}

/// Generated diary artifact for one user's recent turns.
///
/// A diary entry summarizes a bounded turn window and becomes source material
/// for later compaction. It is also exposed to memory lookup as recent context
/// without replacing the compact profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMemoryDiaryEntry {
    /// Entry id.
    pub id: uuid::Uuid,
    /// Subject user.
    pub key: UserMemoryKey,
    /// Window start.
    #[serde(with = "time::serde::rfc3339")]
    pub window_start: OffsetDateTime,
    /// Window end.
    #[serde(with = "time::serde::rfc3339")]
    pub window_end: OffsetDateTime,
    /// Source turns summarized.
    pub source_turn_ids: Vec<TurnId>,
    /// Generated Markdown.
    pub markdown: String,
    /// Memory agent name.
    pub agent_name: String,
    /// LLM provider registry key.
    pub llm_provider: ProviderName,
    /// LLM model id.
    pub llm_model: ModelId,
    /// Usage records for the diary generation.
    pub usage: Vec<UsageRecord>,
    /// Created timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Updated timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

/// Input for saving a generated diary artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewUserMemoryDiaryEntry {
    /// Subject user.
    pub key: UserMemoryKey,
    /// Window start.
    #[serde(with = "time::serde::rfc3339")]
    pub window_start: OffsetDateTime,
    /// Window end.
    #[serde(with = "time::serde::rfc3339")]
    pub window_end: OffsetDateTime,
    /// Source turns summarized.
    pub source_turn_ids: Vec<TurnId>,
    /// Generated Markdown.
    pub markdown: String,
    /// Memory agent name.
    pub agent_name: String,
    /// LLM provider registry key.
    pub llm_provider: ProviderName,
    /// LLM model id.
    pub llm_model: ModelId,
    /// Usage records for the diary generation.
    pub usage: Vec<UsageRecord>,
}

/// Current compact user memory profile.
///
/// This is the profile shape consumed by lookup tools. It records source
/// cutoffs so storage can ask for only newly pending ledger events and diary
/// entries during the next compaction pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMemoryDocument {
    /// Subject user.
    pub key: UserMemoryKey,
    /// Monotonic profile revision.
    pub revision: i64,
    /// Complete Markdown profile.
    pub markdown: String,
    /// Last compaction timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub last_compacted_at: OffsetDateTime,
    /// Highest event timestamp included in the profile.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub source_event_cutoff: Option<OffsetDateTime>,
    /// Highest diary timestamp included in the profile.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub source_diary_cutoff: Option<OffsetDateTime>,
    /// Created timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    /// Updated timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
}

/// Input for replacing the compact memory document.
///
/// Saving a revision should update the current document and preserve a
/// historical revision/source row atomically in backends that support it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewUserMemoryDocumentRevision {
    /// Subject user.
    pub key: UserMemoryKey,
    /// Complete replacement Markdown profile.
    pub markdown: String,
    /// Source memory event ids included in this revision.
    pub source_event_ids: Vec<uuid::Uuid>,
    /// Source diary entry ids included in this revision.
    pub source_diary_entry_ids: Vec<uuid::Uuid>,
    /// Highest event timestamp included in this revision.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub source_event_cutoff: Option<OffsetDateTime>,
    /// Highest diary timestamp included in this revision.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub source_diary_cutoff: Option<OffsetDateTime>,
    /// Memory agent name that generated this revision.
    pub agent_name: String,
    /// LLM provider registry key.
    pub llm_provider: ProviderName,
    /// LLM model id.
    pub llm_model: ModelId,
    /// Usage records for the compaction generation.
    pub usage: Vec<UsageRecord>,
}

/// Durable memory job kind.
///
/// Memory work is split so diary jobs summarize raw turns, then compact jobs
/// merge ledger and diary material into the current profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryJobKind {
    /// Generate a diary entry from recent turns.
    Diary,
    /// Compact events and diary entries into the current profile.
    Compact,
}

/// Durable memory job claimed for processing.
///
/// Jobs use leases rather than in-process queues so multiple bot processes can
/// share work and recover abandoned jobs after a crash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMemoryJob {
    /// Job id.
    pub id: uuid::Uuid,
    /// Job kind.
    pub kind: MemoryJobKind,
    /// Subject user.
    pub key: UserMemoryKey,
    /// Durable parallelism key.
    pub memory_key: String,
    /// Optional turn window start.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub window_start: Option<OffsetDateTime>,
    /// Optional turn window end.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub window_end: Option<OffsetDateTime>,
    /// Number of times this job has been claimed.
    pub attempts: i32,
    /// Current lease owner.
    pub leased_by: Option<String>,
    /// Current lease expiry.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub leased_until: Option<OffsetDateTime>,
    /// Active-job dedupe key.
    pub dedupe_key: String,
}

/// Scheduler inputs for enqueueing due memory work.
///
/// The scheduler passes policy timestamps; storage decides which users/windows
/// are due and deduplicates active jobs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemoryJobSchedule {
    /// Scheduler timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub now: OffsetDateTime,
    /// Completed turns before this timestamp are ignored for diary backfill.
    #[serde(with = "time::serde::rfc3339")]
    pub diary_cutoff: OffsetDateTime,
    /// The next pending diary window must start at or before this timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub diary_due_before: OffsetDateTime,
    /// Source window length for one diary entry, in seconds.
    pub diary_window_seconds: u64,
    /// Documents compacted before this timestamp are due for refresh.
    #[serde(with = "time::serde::rfc3339")]
    pub compact_due_before: OffsetDateTime,
}

/// Memory job completion status.
///
/// A worker reports the result of a claimed job with this enum. Storage then
/// clears the lease, schedules the next attempt, or marks the job terminal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum MemoryJobCompletion {
    /// Job completed successfully.
    Completed {
        /// Job id.
        job_id: uuid::Uuid,
    },
    /// Job failed but should be retried.
    Retry {
        /// Job id.
        job_id: uuid::Uuid,
        /// Error text.
        error: String,
        /// Next attempt timestamp.
        #[serde(with = "time::serde::rfc3339")]
        next_run_at: OffsetDateTime,
    },
    /// Job exhausted retries or should not be retried.
    Failed {
        /// Job id.
        job_id: uuid::Uuid,
        /// Error text.
        error: String,
    },
}

impl MemoryJobCompletion {
    /// Borrow the job id shared by every completion outcome.
    pub fn job_id(&self) -> uuid::Uuid {
        // Every variant is a state transition for exactly one leased job row.
        match self {
            Self::Completed { job_id }
            | Self::Retry { job_id, .. }
            | Self::Failed { job_id, .. } => *job_id,
        }
    }
}

/// Request for a bounded memory diary transcript window.
///
/// The memory worker uses this to load only completed turns for one subject and
/// one time window. Storage filters replay-only material before returning
/// [`UserMemoryTurn`] rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTurnWindow {
    /// Subject user.
    pub key: UserMemoryKey,
    /// Window start.
    #[serde(with = "time::serde::rfc3339")]
    pub window_start: OffsetDateTime,
    /// Window end.
    #[serde(with = "time::serde::rfc3339")]
    pub window_end: OffsetDateTime,
    /// Maximum completed turns to return.
    pub max_turns: u32,
}

/// One completed turn loaded for the memory pipeline.
///
/// This is narrower than [`TurnSnapshot`]: diary generation needs the user and
/// assistant text plus selected media/tool outputs, not every persisted trace
/// row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMemoryTurn {
    /// Conversation id.
    pub conversation_id: ConversationId,
    /// Turn id.
    pub turn_id: TurnId,
    /// Completed timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub completed_at: OffsetDateTime,
    /// User display name at turn time.
    pub user_display_name: String,
    /// User message text.
    pub user_content: String,
    /// Assistant reply text, if any.
    pub assistant_content: Option<String>,
    /// Image attachments and generated images associated with this turn.
    #[serde(default)]
    pub image_context: Vec<UserMemoryImageContext>,
    /// Audio transcriptions produced during this turn.
    #[serde(default)]
    pub audio_transcriptions: Vec<UserMemoryAudioTranscription>,
}

/// One image reference loaded for the memory diary pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMemoryImageContext {
    /// Stable stored image URI.
    pub image_uri: MediaUri,
    /// Context source label or tool name that produced it.
    pub source: String,
    /// MIME type.
    pub mime_type: Option<String>,
}

/// One audio transcription loaded for the memory diary pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMemoryAudioTranscription {
    /// Tool trace row id that produced the transcription.
    pub tool_trace_id: i64,
    /// Stable stored audio URI when available.
    pub audio_uri: Option<String>,
    /// Transcribed text.
    pub text: String,
    /// Language reported by the provider when available.
    pub language: Option<String>,
    /// Audio duration in seconds when available.
    pub duration_seconds: Option<f64>,
}

/// Bot persistence API.
///
/// The trait is organized in the order the runtime usually touches storage:
/// conversation/turn lifecycle, message/channel links, runtime policy,
/// generated media accounting, usage reports, and background user-memory work.
/// Implementations own transactional details for allocation, ordering, retry
/// eligibility, and job leasing; callers receive workflow-shaped results.
pub trait BotStorage: Send + Sync {
    /// Storage error type.
    type Error: std::error::Error + Send + Sync + 'static;

    // Conversation lifecycle and trace capture.

    /// Load a complete conversation snapshot by id or by linked platform
    /// message/channel.
    ///
    /// This is the primary history-loading operation for follow-up model
    /// requests and the web trace viewer.
    fn load_conversation(
        &self,
        lookup: ConversationLookup,
    ) -> impl Future<Output = Result<Option<ConversationSnapshot>, Self::Error>> + Send;

    /// Open a new conversation and return its empty snapshot.
    fn open_conversation(
        &self,
        input: OpenConversation,
    ) -> impl Future<Output = Result<ConversationSnapshot, Self::Error>> + Send;

    /// Begin a new turn and allocate its ordering metadata.
    fn begin_turn(
        &self,
        input: BeginTurn,
    ) -> impl Future<Output = Result<Turn, Self::Error>> + Send;

    /// Persist the turn's resolved agent, system prompt, and model-facing
    /// context before the agent runs.
    fn save_turn_input(
        &self,
        input: SaveTurnInput,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Append one tool/server/grounding trace event for a turn.
    fn append_tool_trace(
        &self,
        turn_id: TurnId,
        ordinal: i32,
        trace: ToolTrace,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Append one provider model-step trace for a turn.
    fn append_model_step_trace(
        &self,
        turn_id: TurnId,
        trace: ModelStepTrace,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Complete, fail, or cancel a turn.
    ///
    /// Implementations should persist usage with the terminal update. On
    /// success, they also assign the response ordinal that makes the assistant
    /// reply replayable for later turns.
    fn finish_turn(
        &self,
        input: FinishTurn,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Atomically prepare a retry for a failed turn. Returns `None` when the
    /// turn is not failed, cancelled, already running/completed, or unknown.
    fn prepare_retry(
        &self,
        turn_id: TurnId,
    ) -> impl Future<Output = Result<Option<RetryTurn>, Self::Error>> + Send;

    /// Stop or resume a conversation.
    fn set_conversation_stop(
        &self,
        input: ConversationStop,
    ) -> impl Future<Output = Result<bool, Self::Error>> + Send;

    // Platform link indexes.

    /// Link a platform message to a turn/conversation.
    fn link_message(
        &self,
        link: MessageLink,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Link a platform channel/thread to a turn/conversation.
    fn link_channel(
        &self,
        link: ChannelLink,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Load the exact message link, including its turn id and role.
    fn load_message_link(
        &self,
        message: MessageRef,
    ) -> impl Future<Output = Result<Option<MessageLink>, Self::Error>> + Send;

    /// Load all platform message links for a turn.
    fn load_message_links_for_turn(
        &self,
        turn_id: TurnId,
    ) -> impl Future<Output = Result<Vec<MessageLink>, Self::Error>> + Send;

    // Runtime policy and user/profile metadata.

    /// Resolve the effective agent name for a turn.
    fn resolve_agent(
        &self,
        input: ResolveAgent,
    ) -> impl Future<Output = Result<Option<String>, Self::Error>> + Send;

    /// Load effective runtime settings for a guild/user pair.
    fn runtime_settings(
        &self,
        message_provider: PlatformName,
        guild_key: Option<String>,
        user_key: String,
    ) -> impl Future<Output = Result<RuntimeSettings, Self::Error>> + Send;

    /// Set guild/workspace privacy mode.
    fn set_privacy_mode(
        &self,
        message_provider: PlatformName,
        guild_key: String,
        privacy: PrivacyMode,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Set a user's opt-in status for a guild/workspace.
    fn set_user_privacy(
        &self,
        message_provider: PlatformName,
        guild_key: String,
        user_key: String,
        opted_in: bool,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Load a user's opt-in status for a guild/workspace.
    fn user_privacy(
        &self,
        message_provider: PlatformName,
        guild_key: String,
        user_key: String,
    ) -> impl Future<Output = Result<Option<bool>, Self::Error>> + Send;

    /// Load one scoped agent selection.
    fn load_agent_selection(
        &self,
        selection: AgentSelection,
    ) -> impl Future<Output = Result<Option<String>, Self::Error>> + Send;

    /// Set one scoped agent selection.
    fn set_agent_selection(
        &self,
        selection: AgentSelection,
        agent_name: String,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Clear one scoped agent selection.
    fn clear_agent_selection(
        &self,
        selection: AgentSelection,
    ) -> impl Future<Output = Result<bool, Self::Error>> + Send;

    /// Upsert a platform user profile.
    fn upsert_user(
        &self,
        user: UserProfile,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Load a user's cached avatar media URI.
    fn load_user_avatar(
        &self,
        user: UserRef,
    ) -> impl Future<Output = Result<Option<MediaUri>, Self::Error>> + Send;

    /// Mark a user's cached avatar media URI.
    fn set_user_avatar(
        &self,
        user: UserRef,
        avatar: MediaUri,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Load platform user profiles and cached avatar metadata for viewer DTOs.
    fn load_user_profiles(
        &self,
        users: Vec<UserRef>,
    ) -> impl Future<Output = Result<Vec<StoredUserProfile>, Self::Error>> + Send;

    /// Set the generated conversation title.
    fn set_conversation_title(
        &self,
        conversation_id: ConversationId,
        title: String,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    // Generated media and usage accounting.

    /// Create a video job.
    fn create_video_job(
        &self,
        input: CreateVideoJob,
    ) -> impl Future<Output = Result<StoredVideoJob, Self::Error>> + Send;

    /// Update a video job.
    fn update_video_job(
        &self,
        input: UpdateVideoJob,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Count pending and successful video generations for one platform scope in
    /// a rolling window.
    fn count_active_video_generations(
        &self,
        input: CountActiveVideoGenerations,
    ) -> impl Future<Output = Result<u64, Self::Error>> + Send;

    /// Aggregate stored usage/cost records — turn usage plus background
    /// memory-job usage — for reporting, costliest groups first.
    fn usage_cost_report(
        &self,
        query: UsageCostQuery,
    ) -> impl Future<Output = Result<Vec<UsageCostRow>, Self::Error>> + Send;

    // User-memory ledger, compaction, and durable job queue.

    /// Load the current compact memory profile for one user.
    fn load_user_memory_document(
        &self,
        key: UserMemoryKey,
    ) -> impl Future<Output = Result<Option<UserMemoryDocument>, Self::Error>> + Send;

    /// Append one raw user-memory ledger event.
    fn append_user_memory_event(
        &self,
        event: NewUserMemoryEvent,
    ) -> impl Future<Output = Result<UserMemoryEvent, Self::Error>> + Send;

    /// List memory events that have not yet been compacted into the current
    /// profile according to the supplied cutoff.
    fn list_pending_memory_events(
        &self,
        key: UserMemoryKey,
        since: Option<OffsetDateTime>,
    ) -> impl Future<Output = Result<Vec<UserMemoryEvent>, Self::Error>> + Send;

    /// List diary entries that have not yet been compacted into the current
    /// profile according to the supplied cutoff.
    fn list_pending_memory_diary_entries(
        &self,
        key: UserMemoryKey,
        since: Option<OffsetDateTime>,
    ) -> impl Future<Output = Result<Vec<UserMemoryDiaryEntry>, Self::Error>> + Send;

    /// List the latest diary entries for one user, returned oldest-to-newest
    /// within the bounded result.
    fn list_recent_memory_diary_entries(
        &self,
        key: UserMemoryKey,
        limit: u32,
    ) -> impl Future<Output = Result<Vec<UserMemoryDiaryEntry>, Self::Error>> + Send;

    /// Save one generated diary entry.
    fn save_user_memory_diary_entry(
        &self,
        entry: NewUserMemoryDiaryEntry,
    ) -> impl Future<Output = Result<UserMemoryDiaryEntry, Self::Error>> + Send;

    /// Atomically save a new current memory document and document-version row.
    fn save_user_memory_document_revision(
        &self,
        document: NewUserMemoryDocumentRevision,
    ) -> impl Future<Output = Result<UserMemoryDocument, Self::Error>> + Send;

    /// Enqueue all memory jobs due at this scheduler tick.
    fn enqueue_due_memory_jobs(
        &self,
        schedule: MemoryJobSchedule,
    ) -> impl Future<Output = Result<u64, Self::Error>> + Send;

    /// Claim due memory jobs with SQL leases.
    fn claim_memory_jobs(
        &self,
        worker_id: String,
        limit: u32,
        lease_until: OffsetDateTime,
    ) -> impl Future<Output = Result<Vec<UserMemoryJob>, Self::Error>> + Send;

    /// Mark a memory job completed, retryable, or failed.
    fn finish_memory_job(
        &self,
        completion: MemoryJobCompletion,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Load a bounded completed-turn window for one user's diary job.
    fn load_memory_turn_window(
        &self,
        window: MemoryTurnWindow,
    ) -> impl Future<Output = Result<Vec<UserMemoryTurn>, Self::Error>> + Send;
}

#[cfg(test)]
mod tests {
    use time::macros::datetime;
    use uuid::Uuid;

    use super::*;
    use crate::PlatformName;

    #[test]
    fn memory_dto_timestamps_serialize_as_rfc3339_strings() {
        let timestamp = datetime!(2026-06-03 22:27:01.816929 UTC);
        let key = UserMemoryKey {
            platform: PlatformName::new("discord"),
            scope_key: "guild:guild-1".to_string(),
            user_key: "user-1".to_string(),
        };

        assert_timestamp_fields(
            "UserMemoryEvent",
            serde_json::to_value(UserMemoryEvent {
                id: Uuid::nil(),
                key: key.clone(),
                actor_user_key: Some("user-1".to_string()),
                kind: UserMemoryEventKind::Remember,
                body: "memory".to_string(),
                tags: Vec::new(),
                confidence: None,
                source_conversation_id: None,
                source_turn_id: None,
                source_tool_trace_id: None,
                supersedes_event_id: None,
                created_at: timestamp,
                updated_at: timestamp,
            })
            .unwrap(),
            &["created_at", "updated_at"],
        );
        assert_timestamp_fields(
            "UserMemoryDiaryEntry",
            serde_json::to_value(UserMemoryDiaryEntry {
                id: Uuid::nil(),
                key: key.clone(),
                window_start: timestamp,
                window_end: timestamp,
                source_turn_ids: Vec::new(),
                markdown: "diary".to_string(),
                agent_name: "memory_diary".to_string(),
                llm_provider: ProviderName::new("xai"),
                llm_model: ModelId::new("grok-4.3"),
                usage: Vec::new(),
                created_at: timestamp,
                updated_at: timestamp,
            })
            .unwrap(),
            &["window_start", "window_end", "created_at", "updated_at"],
        );
        assert_timestamp_fields(
            "NewUserMemoryDiaryEntry",
            serde_json::to_value(NewUserMemoryDiaryEntry {
                key: key.clone(),
                window_start: timestamp,
                window_end: timestamp,
                source_turn_ids: Vec::new(),
                markdown: "diary".to_string(),
                agent_name: "memory_diary".to_string(),
                llm_provider: ProviderName::new("xai"),
                llm_model: ModelId::new("grok-4.3"),
                usage: Vec::new(),
            })
            .unwrap(),
            &["window_start", "window_end"],
        );
        assert_timestamp_fields(
            "UserMemoryDocument",
            serde_json::to_value(UserMemoryDocument {
                key: key.clone(),
                revision: 1,
                markdown: "profile".to_string(),
                last_compacted_at: timestamp,
                source_event_cutoff: Some(timestamp),
                source_diary_cutoff: Some(timestamp),
                created_at: timestamp,
                updated_at: timestamp,
            })
            .unwrap(),
            &[
                "last_compacted_at",
                "source_event_cutoff",
                "source_diary_cutoff",
                "created_at",
                "updated_at",
            ],
        );
        assert_timestamp_fields(
            "NewUserMemoryDocumentRevision",
            serde_json::to_value(NewUserMemoryDocumentRevision {
                key: key.clone(),
                markdown: "profile".to_string(),
                source_event_ids: Vec::new(),
                source_diary_entry_ids: Vec::new(),
                source_event_cutoff: Some(timestamp),
                source_diary_cutoff: Some(timestamp),
                agent_name: "memory_compact".to_string(),
                llm_provider: ProviderName::new("xai"),
                llm_model: ModelId::new("grok-4.3"),
                usage: Vec::new(),
            })
            .unwrap(),
            &["source_event_cutoff", "source_diary_cutoff"],
        );
        assert_timestamp_fields(
            "UserMemoryJob",
            serde_json::to_value(UserMemoryJob {
                id: Uuid::nil(),
                kind: MemoryJobKind::Diary,
                key: key.clone(),
                memory_key: key.memory_key(),
                window_start: Some(timestamp),
                window_end: Some(timestamp),
                attempts: 1,
                leased_by: Some("worker".to_string()),
                leased_until: Some(timestamp),
                dedupe_key: "diary:key".to_string(),
            })
            .unwrap(),
            &["window_start", "window_end", "leased_until"],
        );
        assert_timestamp_fields(
            "MemoryJobSchedule",
            serde_json::to_value(MemoryJobSchedule {
                now: timestamp,
                diary_cutoff: timestamp,
                diary_due_before: timestamp,
                diary_window_seconds: 86_400,
                compact_due_before: timestamp,
            })
            .unwrap(),
            &[
                "now",
                "diary_cutoff",
                "diary_due_before",
                "compact_due_before",
            ],
        );
        assert_timestamp_fields(
            "MemoryJobCompletion",
            serde_json::to_value(MemoryJobCompletion::Retry {
                job_id: Uuid::nil(),
                error: "retry".to_string(),
                next_run_at: timestamp,
            })
            .unwrap(),
            &["next_run_at"],
        );
        assert_timestamp_fields(
            "MemoryTurnWindow",
            serde_json::to_value(MemoryTurnWindow {
                key: key.clone(),
                window_start: timestamp,
                window_end: timestamp,
                max_turns: 40,
            })
            .unwrap(),
            &["window_start", "window_end"],
        );
        assert_timestamp_fields(
            "UserMemoryTurn",
            serde_json::to_value(UserMemoryTurn {
                conversation_id: ConversationId(Uuid::nil()),
                turn_id: TurnId(Uuid::nil()),
                completed_at: timestamp,
                user_display_name: "Chud".to_string(),
                user_content: "hi".to_string(),
                assistant_content: Some("hello".to_string()),
                image_context: Vec::new(),
                audio_transcriptions: Vec::new(),
            })
            .unwrap(),
            &["completed_at"],
        );
    }

    fn assert_timestamp_fields(name: &str, value: serde_json::Value, fields: &[&str]) {
        for field in fields {
            assert_eq!(
                value
                    .get(*field)
                    .and_then(serde_json::Value::as_str)
                    .map(|timestamp| timestamp.starts_with("2026-06-03T22:27:01.816929")),
                Some(true),
                "{name}.{field} was not an RFC3339 string: {:?}",
                value.get(*field)
            );
        }
    }
}
