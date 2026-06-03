//! Bot persistence contracts.
//!
//! These operations are shaped around how the bot uses persistence rather than
//! around how a SQL database happens to expose rows. A Postgres implementation
//! can still answer them with joins and indexes; a JSON-file implementation
//! could answer the same calls by loading one document.

use std::future::Future;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::ids::{
    ChannelRef, ConversationId, MessageRef, ModelId, PlatformName, ProviderName, TurnId, UserRef,
};
use crate::media::MediaUri;
use crate::platform::UserProfile;
use crate::tool::ToolTrace;
use crate::transcript::{ProviderContinuation, Transcript};
use crate::usage::UsageRecord;

/// Privacy mode for context gathering.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PrivacyMode {
    /// Open channel-history access.
    Open {
        /// History size.
        history_size: u32,
    },
    /// Only operate in one channel.
    ChannelOnly {
        /// Allowed channel.
        channel: ChannelRef,
        /// History size.
        history_size: u32,
    },
    /// Per-user opt-in.
    OptIn,
    /// Only conversation history.
    ConversationOnly,
}

/// How to locate a conversation.
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

/// Full conversation snapshot needed to prepare later model requests and
/// render a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSnapshot {
    /// Conversation metadata.
    pub conversation: Conversation,
    /// Turns ordered by user-message ordinal.
    pub turns: Vec<TurnSnapshot>,
}

/// Conversation metadata.
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

/// One turn plus the trace data needed to reconstruct model input.
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
    /// Assets that should be replayed with this turn.
    pub replay_assets: Vec<TurnAsset>,
    /// Usage/cost accumulated by this turn.
    pub usage: Vec<UsageRecord>,
}

/// Turn metadata.
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
    /// Provider continuation from the final assistant step.
    ///
    /// Replay-only provider plumbing. This may contain encrypted reasoning
    /// state and must not cross the web API.
    #[serde(skip)]
    pub continuation: Option<ProviderContinuation>,
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

/// New conversation input.
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

/// Begin a turn in a conversation.
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
    /// Initial transcript assembled from the context. This is optional
    /// because some stores may only persist normalized context rows.
    #[serde(skip)]
    pub transcript: Option<Transcript>,
}

/// Finish a turn.
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
        /// Provider continuation to replay later.
        continuation: Option<ProviderContinuation>,
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

/// Agent resolution input.
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

/// Video job status persisted by storage.
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

/// Stored user metadata for viewer read models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredUserProfile {
    /// Last platform profile seen for this user.
    pub profile: UserProfile,
    /// Cached local avatar media URI, when downloaded.
    pub avatar: Option<MediaUri>,
}

/// Bot persistence API.
pub trait BotStorage: Send + Sync {
    /// Storage error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Load a complete conversation snapshot by id or by linked platform
    /// message. This is the primary history-loading operation for follow-up
    /// model requests.
    fn load_conversation(
        &self,
        lookup: ConversationLookup,
    ) -> impl Future<Output = Result<Option<ConversationSnapshot>, Self::Error>> + Send;

    /// Open a new conversation and return its empty snapshot.
    fn open_conversation(
        &self,
        input: OpenConversation,
    ) -> impl Future<Output = Result<ConversationSnapshot, Self::Error>> + Send;

    /// Begin a new turn.
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

    /// Complete, fail, or cancel a turn.
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
}

#[cfg(test)]
mod tests {
    use time::macros::datetime;
    use uuid::Uuid;

    use super::*;
    use crate::{ExternalId, PlatformName};

    #[test]
    fn turn_serialization_omits_provider_continuation() {
        let platform = PlatformName::new("discord");
        let channel = ExternalId::new("channel-1");
        let turn = Turn {
            id: TurnId(Uuid::nil()),
            ordinal: 0,
            history_cutoff: None,
            response_ordinal: Some(0),
            created_at: datetime!(2026-06-03 12:00 UTC),
            user_message_created_at: datetime!(2026-06-03 12:00 UTC),
            completed_at: Some(datetime!(2026-06-03 12:00 UTC)),
            user_message: MessageRef {
                platform: platform.clone(),
                guild_id: Some(ExternalId::new("guild-1")),
                channel_id: channel.clone(),
                message_id: ExternalId::new("user-message-1"),
            },
            user: UserRef {
                platform: platform.clone(),
                guild_id: Some(ExternalId::new("guild-1")),
                user_id: ExternalId::new("user-1"),
            },
            user_display_name: "Chud".to_string(),
            user_content: "hi".to_string(),
            assistant_message: Some(MessageRef {
                platform,
                guild_id: Some(ExternalId::new("guild-1")),
                channel_id: channel,
                message_id: ExternalId::new("assistant-message-1"),
            }),
            assistant_content: Some("hello".to_string()),
            status: TurnStatus::Completed,
            error: None,
            agent_name: Some("grok".to_string()),
            provider: Some(ProviderName::new("xai")),
            model: Some(ModelId::new("grok-4.3")),
            app_version_id: Some(7),
            continuation: Some(ProviderContinuation {
                provider: ProviderName::new("xai"),
                data: serde_json::json!([
                    { "type": "reasoning", "id": "rs_1", "encrypted_content": "BLOB" },
                ]),
            }),
        };

        let value = serde_json::to_value(turn).unwrap();

        assert!(
            value.get("continuation").is_none(),
            "provider continuation leaked into serialized turn"
        );
    }
}
