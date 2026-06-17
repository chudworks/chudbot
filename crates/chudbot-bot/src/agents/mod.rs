//! Agent construction for runtime conversations and reserved system agents.
//!
//! The child modules keep the normal conversation-agent builder separate from
//! the bot-owned agents used for moderation preflight and conversation titles.
//! This wrapper keeps those implementation modules private while re-exporting
//! the crate-internal helpers used by `BotRuntime`.

mod conversation;
mod system;
mod title;
mod tos;

pub(crate) use system::*;
pub(crate) use title::*;
pub(crate) use tos::*;
