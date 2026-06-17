//! Outcome markers returned by platform event handling.
//!
//! Event handlers do the durable work before returning one of these values:
//! storage is updated, live events are published, and platform replies or
//! reactions have already been attempted. `BotAction` is the small public
//! summary that lets the runtime log the event, decide whether to add status
//! reactions, and expose the affected turn or conversation ids when relevant.

use crate::prelude::*;

/// High-level result of handling one platform event.
///
/// The enum deliberately describes the bot-visible outcome rather than the
/// source platform event. That keeps Discord-specific details out of the
/// orchestration boundary while still carrying enough ids for callers to
/// correlate a completed, failed, or cancelled turn with stored trace data.
pub enum BotAction {
    /// The event was recognized but did not require bot work.
    Ignored,
    /// The platform event stream asked the bot process to stop.
    Shutdown,
    /// A turn reached a successful terminal state.
    CompletedTurn {
        /// Conversation that owns the completed turn.
        conversation_id: ConversationId,
        /// Stored turn that was marked completed.
        turn_id: TurnId,
    },
    /// A turn reached a failed terminal state.
    FailedTurn {
        /// Conversation that owns the failed turn.
        conversation_id: ConversationId,
        /// Stored turn that was marked failed.
        turn_id: TurnId,
    },
    /// A turn was cancelled after it had already been created.
    CancelledTurn {
        /// Conversation that owns the cancelled turn.
        conversation_id: ConversationId,
        /// Stored turn that was marked cancelled.
        turn_id: TurnId,
    },
    /// A reaction or command stopped an active conversation.
    StoppedConversation {
        /// Conversation whose runtime state was stopped.
        conversation_id: ConversationId,
    },
    /// A reaction or command resumed a stopped conversation.
    ResumedConversation {
        /// Conversation whose runtime state was resumed.
        conversation_id: ConversationId,
    },
    /// A message was refused before a durable turn was created.
    RefusedMessage,
    /// A platform command was handled and acknowledged.
    HandledCommand,
}

/// Stable snake_case label for tracing and event-task diagnostics.
pub(crate) fn bot_action_kind(action: &BotAction) -> &'static str {
    // Keep these strings decoupled from Rust variant spelling so log filters
    // and dashboards do not churn if the enum names ever change.
    match action {
        BotAction::Ignored => "ignored",
        BotAction::Shutdown => "shutdown",
        BotAction::CompletedTurn { .. } => "completed_turn",
        BotAction::FailedTurn { .. } => "failed_turn",
        BotAction::CancelledTurn { .. } => "cancelled_turn",
        BotAction::StoppedConversation { .. } => "stopped_conversation",
        BotAction::ResumedConversation { .. } => "resumed_conversation",
        BotAction::RefusedMessage => "refused_message",
        BotAction::HandledCommand => "handled_command",
    }
}
