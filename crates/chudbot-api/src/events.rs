//! Live update event contracts shared by bot, storage, and web services.
//!
//! These events are intentionally small invalidation hints. Producers publish
//! them after durable state changes, and web subscribers use them to decide
//! which trace-viewer state to refetch. The event stream is not the source of
//! truth and should not be treated as a complete change payload.
//!
//! The types stay provider- and platform-neutral: concrete crates decide how to
//! deliver events, while this crate owns the serialized vocabulary and routing
//! rules.

use serde::{Deserialize, Serialize};

use crate::ids::{ConversationId, UserRef};

/// Live invalidation event published after persisted state changes.
///
/// A [`LiveEvent`] describes the minimum information a subscriber needs to
/// decide whether a visible trace may be stale. It deliberately does not embed
/// updated rows, transcripts, tool traces, or user profiles; subscribers should
/// reload the relevant API resource after receiving an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum LiveEvent {
    /// A single conversation changed.
    ///
    /// This is the normal path for trace-viewer updates. The conversation id is
    /// used for routing, and [`ConversationEventKind`] supplies the stable SSE
    /// event name exposed to web clients.
    Conversation {
        /// Conversation whose persisted state changed.
        conversation_id: ConversationId,
        /// Coarse invalidation category for the changed conversation state.
        kind: ConversationEventKind,
    },
    /// A platform user profile changed.
    ///
    /// User references can appear in any visible conversation, so this event is
    /// broadcast to all conversation viewers rather than routed to one
    /// conversation id.
    UserProfileUpdated {
        /// User whose display profile or avatar may have changed.
        user: UserRef,
    },
}

impl LiveEvent {
    /// Return whether this event should be forwarded to a conversation viewer.
    ///
    /// Conversation-scoped events only apply to their own conversation. Profile
    /// changes are global invalidations because a single user can be rendered
    /// in many traces.
    pub fn applies_to_conversation(&self, conversation_id: ConversationId) -> bool {
        match self {
            // Route direct conversation changes only to the matching viewer.
            Self::Conversation {
                conversation_id: id,
                ..
            } => *id == conversation_id,
            // User display data is shared across traces, so every viewer may
            // need to refresh visible profile/avatar references.
            Self::UserProfileUpdated { .. } => true,
        }
    }

    /// Return the stable Server-Sent Events event name for web subscribers.
    ///
    /// Conversation events use the same vocabulary as their
    /// [`ConversationEventKind`]. Global events define their own names here.
    pub fn event_name(&self) -> &'static str {
        match self {
            // Keep conversation routing and naming in one place so adding a new
            // conversation kind only requires extending ConversationEventKind.
            Self::Conversation { kind, .. } => kind.event_name(),
            // This event is not conversation-scoped, so it has no
            // ConversationEventKind counterpart.
            Self::UserProfileUpdated { .. } => "user_profile_updated",
        }
    }
}

/// Coarse hint about what changed inside a conversation.
///
/// These variants form the public event-name vocabulary for conversation trace
/// viewers. They should stay broad enough that clients can refetch full state
/// instead of depending on provider-, platform-, or storage-specific payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConversationEventKind {
    /// A conversation was opened and its metadata was inserted.
    Created,
    /// A turn was inserted and should appear in the trace.
    TurnStarted,
    /// A turn's status, assistant content, usage, or related metadata changed.
    TurnUpdated,
    /// A client tool, server tool, or grounding trace entry was appended.
    ToolTraceRecorded,
    /// Context rows were recorded for a turn.
    ContextRecorded,
    /// The conversation title changed.
    TitleUpdated,
    /// Conversation-level metadata changed outside the more specific variants.
    ConversationUpdated,
}

impl ConversationEventKind {
    /// Return the stable Server-Sent Events event name for this conversation hint.
    ///
    /// These names are part of the web contract. Keep them lowercase snake_case
    /// and append new names rather than renaming existing ones.
    pub fn event_name(self) -> &'static str {
        match self {
            // Keep the serialized kind and SSE name aligned for predictable
            // client-side subscription handling.
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

/// Fire-and-forget sink for live update events.
///
/// Implementations bridge durable state changes to an in-process bus, SSE hub,
/// or another notification mechanism. Publishing is best-effort by design:
/// callers should commit state before calling [`Self::publish`], and subscribers
/// should treat every event as a prompt to refetch authoritative state.
pub trait EventSink: Send + Sync {
    /// Publish one event.
    ///
    /// Implementations may drop events when no viewer is listening, and they do
    /// not need to preserve a replay log. Event loss should only delay a viewer
    /// refresh; it must not make persisted bot state incorrect.
    fn publish(&self, event: LiveEvent);
}

/// Event sink that deliberately drops every event.
///
/// Use this in tests, offline tools, or deployments that do not expose live
/// trace updates. It preserves the same call sites as a real sink without
/// starting a delivery mechanism.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopEventSink;

impl EventSink for NoopEventSink {
    /// Drop the event without side effects.
    fn publish(&self, _event: LiveEvent) {}
}
