//! LLM provider abstraction.
//!
//! The bot doesn't care which model it's talking to — it submits a
//! [`CompletionRequest`] and gets back a [`CompletionResponse`] that
//! captures both the answer and every server-side tool call the provider
//! made on our behalf (e.g. web search). Each tool call is recorded in
//! the DB so the web viewer can show the full trace per turn.
//!
//! Two real implementations live in [`xai`] and [`anthropic`]. A
//! [`mock::MockProvider`] is exposed for tests.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod anthropic;
pub mod mock;
pub mod xai;

use crate::config::{AnthropicConfig, LlmConfig, LlmProviderKind, XaiConfig};

/// Role of a message in a chat conversation. Mirrors the OpenAI/Anthropic
/// conventions; providers translate as needed (Anthropic lifts `System`
/// out of the messages list into a top-level field).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System / developer prompt.
    System,
    /// End user.
    User,
    /// Model.
    Assistant,
}

impl MessageRole {
    /// Lowercase string form for HTTP payloads and DB storage.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    /// Parse the lowercase string form. Unknown values fall back to `User`.
    pub fn from_str_lossy(s: &str) -> Self {
        match s {
            "system" => Self::System,
            "assistant" => Self::Assistant,
            _ => Self::User,
        }
    }
}

/// A single message in the chat sequence sent to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the speaker.
    pub role: MessageRole,
    /// Text content. We don't currently send multi-modal blocks.
    pub content: String,
}

/// Input to [`LlmProvider::complete`].
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Chat history, ordered oldest-to-newest. The last entry is the
    /// current user prompt.
    pub messages: Vec<ChatMessage>,
    /// If true, the provider should enable its server-side web search
    /// tool. Each provider has its own way of expressing this (xAI:
    /// `search_parameters`; Anthropic: `tools[web_search_*]`). Tool calls
    /// the model makes during the turn are returned in
    /// [`CompletionResponse::tool_calls`].
    pub enable_web_search: bool,
    /// Soft cap on output tokens (Anthropic requires it; xAI ignores it).
    pub max_tokens: u32,
}

/// Output from [`LlmProvider::complete`].
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    /// Final answer text from the model.
    pub content: String,
    /// Server-side tool calls performed during this turn (e.g. web
    /// searches). One entry per invocation.
    pub tool_calls: Vec<ToolCallRecord>,
    /// Model id reported back by the provider.
    pub model_id: String,
}

/// A record of one server-side tool call. We store the raw JSON of both
/// the request and response so the viewer can render the exact data the
/// model saw without inventing a normalized schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Logical name of the tool (`web_search`, `x_search`, etc.).
    pub tool_name: String,
    /// Tool input (e.g. the search query). Provider-specific shape.
    pub request: serde_json::Value,
    /// Tool output (e.g. search results / citations). Provider-specific shape.
    pub response: serde_json::Value,
}

/// Errors that any [`LlmProvider`] implementation can return.
#[derive(Debug, Error)]
pub enum LlmError {
    /// Network/transport failure (DNS, TCP, TLS, timeout).
    #[error("transport error: {0}")]
    Transport(String),
    /// API returned a non-success HTTP status.
    #[error("api error {status}: {body}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Response body (may be JSON or plain text).
        body: String,
    },
    /// Response could not be decoded into the expected shape.
    #[error("decode error: {0}")]
    Decode(String),
    /// Returned by a config-driven constructor when the relevant
    /// `[llm.<provider>]` block is missing. Should be caught earlier by
    /// [`Config::load`] validation; this is a defense-in-depth.
    #[error("missing config for provider `{0}`")]
    MissingConfig(&'static str),
}

/// Shared interface for both Grok and Claude. Implementations are
/// stateless except for the HTTP client and credentials, so they're
/// cheap to clone and safe to share across tasks.
pub trait LlmProvider: Send + Sync {
    /// Short, stable identifier (e.g. `xai/grok-4.1-fast`,
    /// `anthropic/claude-sonnet-4-6`). Stored on each conversation row.
    fn name(&self) -> &str;

    /// Issue a chat completion. Returns the model's final answer plus
    /// the trace of every server-side tool call performed during the turn.
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse, LlmError>> + Send;
}

/// Static-dispatch union of every supported provider. Built once at
/// startup from [`LlmConfig`] and held inside the bot/web state. Using an
/// enum (rather than `Box<dyn LlmProvider>`) keeps the call sites monomorphic.
#[derive(Debug, Clone)]
pub enum AnyProvider {
    /// xAI Grok.
    Xai(xai::XaiProvider),
    /// Anthropic Claude.
    Anthropic(anthropic::AnthropicProvider),
}

impl AnyProvider {
    /// Construct an [`AnyProvider`] from validated [`LlmConfig`].
    pub fn from_config(config: &LlmConfig) -> Result<Self, LlmError> {
        match config.provider {
            LlmProviderKind::Xai => {
                let cfg = config.xai.as_ref().ok_or(LlmError::MissingConfig("xai"))?;
                Ok(Self::Xai(xai::XaiProvider::new(cfg.clone())))
            }
            LlmProviderKind::Anthropic => {
                let cfg = config
                    .anthropic
                    .as_ref()
                    .ok_or(LlmError::MissingConfig("anthropic"))?;
                Ok(Self::Anthropic(anthropic::AnthropicProvider::new(cfg.clone())))
            }
        }
    }
}

impl LlmProvider for AnyProvider {
    fn name(&self) -> &str {
        match self {
            Self::Xai(p) => p.name(),
            Self::Anthropic(p) => p.name(),
        }
    }

    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError> {
        match self {
            Self::Xai(p) => p.complete(request).await,
            Self::Anthropic(p) => p.complete(request).await,
        }
    }
}

// Allow constructing the AnyProvider directly from references to
// per-provider config sections (used by tests + non-config builders).
impl From<XaiConfig> for AnyProvider {
    fn from(c: XaiConfig) -> Self {
        Self::Xai(xai::XaiProvider::new(c))
    }
}

impl From<AnthropicConfig> for AnyProvider {
    fn from(c: AnthropicConfig) -> Self {
        Self::Anthropic(anthropic::AnthropicProvider::new(c))
    }
}
