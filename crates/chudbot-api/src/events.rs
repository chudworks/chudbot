//! Live update events shared by bot and web services.

use serde::{Deserialize, Serialize};

use crate::ids::{ConversationId, UserRef};

/// Event published after persisted state changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum LiveEvent {
    /// A single conversation changed.
    Conversation {
        /// Conversation id.
        conversation_id: ConversationId,
        /// What changed.
        kind: ConversationEventKind,
    },
    /// A platform user profile changed. Viewers may refresh any visible user
    /// profile/avatar references.
    UserProfileUpdated {
        /// User whose profile changed.
        user: UserRef,
    },
}

impl LiveEvent {
    /// Whether this event should be forwarded to a viewer of `conversation_id`.
    pub fn applies_to_conversation(&self, conversation_id: ConversationId) -> bool {
        match self {
            Self::Conversation {
                conversation_id: id,
                ..
            } => *id == conversation_id,
            Self::UserProfileUpdated { .. } => true,
        }
    }

    /// SSE event name for web subscribers.
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::Conversation { kind, .. } => kind.event_name(),
            Self::UserProfileUpdated { .. } => "user_profile_updated",
        }
    }
}

/// Hint about what changed inside a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationEventKind {
    /// A conversation row was inserted.
    Created,
    /// A turn row was inserted.
    TurnStarted,
    /// A turn row was updated.
    TurnUpdated,
    /// A tool/server/grounding trace event was appended.
    ToolTraceRecorded,
    /// Context rows were appended for a turn.
    ContextRecorded,
    /// Conversation title changed.
    TitleUpdated,
    /// Conversation-level metadata changed.
    ConversationUpdated,
}

impl ConversationEventKind {
    /// SSE event name.
    pub fn event_name(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::TurnStarted => "turn_started",
            Self::TurnUpdated => "turn_updated",
            Self::ToolTraceRecorded => "tool_trace_recorded",
            Self::ContextRecorded => "context_recorded",
            Self::TitleUpdated => "title_updated",
            Self::ConversationUpdated => "conversation_updated",
        }
    }
}

/// Sink for live update events.
pub trait EventSink: Send + Sync {
    /// Publish one event. Implementations may drop events when no viewer is
    /// listening; subscribers refetch full state on every event.
    fn publish(&self, event: LiveEvent);
}

/// Event sink that drops every event.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopEventSink;

impl EventSink for NoopEventSink {
    fn publish(&self, _event: LiveEvent) {}
}
