//! Core library for the grok-discord-bot project.
//!
//! Provides the LLM provider abstraction (mockable trait with xAI and
//! Anthropic implementations, both with server-side web search support),
//! the conversation domain types, the Postgres data layer, and the TOML
//! configuration loader. Both the Discord bot and the Axum web viewer
//! depend on this crate.

#![warn(missing_docs)]

pub mod config;
pub mod db;
pub mod domain;
pub mod llm;

pub use config::{Config, ConfigError};
pub use db::{Db, DbError};
pub use domain::{Conversation, ContextItem, ConversationView, Turn, TurnView};
pub use llm::{
    AnyProvider, ChatMessage, CompletionRequest, CompletionResponse, LlmError, LlmProvider,
    MessageRole, ToolCallRecord,
};
