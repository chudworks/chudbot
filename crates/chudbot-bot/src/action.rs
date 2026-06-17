//! Actions returned by platform event handling.

use crate::prelude::*;

pub enum BotAction {
    /// Event did not require work.
    Ignored,
    /// Event stream asked the bot to stop.
    Shutdown,
    /// Turn completed.
    CompletedTurn {
        /// Conversation id.
        conversation_id: ConversationId,
        /// Turn id.
        turn_id: TurnId,
    },
    /// Turn failed.
    FailedTurn {
        /// Conversation id.
        conversation_id: ConversationId,
        /// Turn id.
        turn_id: TurnId,
    },
    /// Turn was cancelled.
    CancelledTurn {
        /// Conversation id.
        conversation_id: ConversationId,
        /// Turn id.
        turn_id: TurnId,
    },
    /// Conversation was stopped.
    StoppedConversation {
        /// Conversation id.
        conversation_id: ConversationId,
    },
    /// Conversation was resumed.
    ResumedConversation {
        /// Conversation id.
        conversation_id: ConversationId,
    },
    /// Message was refused before turn creation.
    RefusedMessage,
    /// Platform command was handled.
    HandledCommand,
}

pub(crate) fn bot_action_kind(action: &BotAction) -> &'static str {
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
