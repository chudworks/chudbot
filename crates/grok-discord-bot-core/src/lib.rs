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
pub mod imagegen;
pub mod llm;
pub mod storage;
pub mod videogen;

pub use agent::{AgentObserver, AgentRun, NoopObserver, run as run_agent};
pub use config::{
    Config, ConfigError, ImageProviderKind, LlmProviderKind, Persona, PrivacyMode, StorageConfig,
    VideoProviderKind,
};
pub use db::{Db, DbError};
pub use domain::{
    ContextItem, Conversation, ConversationView, DiscordUser, Turn, TurnView, VideoJob,
};
pub use imagegen::{AnyImageProvider, ImageProvider};
pub use llm::{
    AnthropicOptions, AnyProvider, ChatTurn, LlmError, LlmProvider, MessageRole, ProviderOptions,
    StepRequest, StepResponse, ToolCallRecord, ToolDefinition, ToolError, ToolExecutor,
    ToolUseRequest, TurnBlock, XaiOptions,
};
pub use videogen::{AnyVideoProvider, VideoProvider};
