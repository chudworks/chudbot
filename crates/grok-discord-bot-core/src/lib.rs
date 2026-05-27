//! Core library for the grok-discord-bot project.
//!
//! Provides:
//!   - the LLM provider abstraction (`LlmProvider::step`, with xAI and
//!     Anthropic impls; both support server-side web search and
//!     client-side tool calls);
//!   - the agentic harness in [`agent`] that loops the provider through
//!     tool calls until it produces a final answer;
//!   - conversation domain types ([`Conversation`], [`Turn`], …) and the
//!     Postgres data layer ([`Db`]);
//!   - the TOML configuration loader.
//!
//! Both the Discord bot and the Axum web viewer depend on this crate.

#![warn(missing_docs)]

pub mod agent;
pub mod config;
pub mod db;
pub mod domain;
pub mod llm;

pub use agent::{AgentRun, run as run_agent};
pub use config::{BotConfig, Config, ConfigError, PrivacyMode};
pub use db::{Db, DbError};
pub use domain::{Conversation, ContextItem, ConversationView, Turn, TurnView};
pub use llm::{
    AnyProvider, ChatTurn, LlmError, LlmProvider, MessageRole, StepRequest, StepResponse,
    ToolCallRecord, ToolDefinition, ToolError, ToolExecutor, ToolUseRequest, TurnBlock,
};
