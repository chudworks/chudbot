//! Core library for the grok-discord-bot project.
//!
//! Provides the Grok API client (behind a mockable trait), the domain
//! types for conversations / turns / context items / tool calls, and the
//! Postgres data layer. Both the Discord bot and the Axum web viewer
//! depend on this crate.

#![warn(missing_docs)]

pub mod domain;
pub mod grok;

pub use domain::{Conversation, Turn};
pub use grok::{GrokClient, GrokError};
