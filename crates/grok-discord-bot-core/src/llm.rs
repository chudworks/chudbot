//! LLM provider abstraction.
//!
//! Models a real agentic harness:
//!
//! - The caller constructs a chat history as [`ChatTurn`]s (text plus
//!   optional `ToolUse` / `ToolResult` blocks — both providers support
//!   this round-trip natively).
//! - The caller hands the provider its [`step`](LlmProvider::step) one
//!   round-trip at a time; the provider returns either the final answer
//!   ([`StepResponse::Final`]) or a list of client-side tool invocations
//!   the model is asking us to run ([`StepResponse::UseTools`]).
//! - The driver in [`crate::agent::run`] handles the loop: it executes
//!   tool calls via a caller-supplied [`ToolExecutor`], appends the
//!   results back into history, and re-runs the step until the model is
//!   done — or an iteration cap is hit.
//!
//! Each provider also surfaces *server-side* tool calls (xAI's
//! `web_search`, Anthropic's `web_search_20250305`) as
//! [`ToolCallRecord`]s in the same trace, so the web viewer can render
//! the full mixed timeline of what the model did.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod anthropic;
pub mod mock;
pub mod xai;

use crate::config::{AnthropicConfig, LlmProviderKind, XaiConfig};

/// Role of a message in a chat conversation.
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

/// One turn of the chat history sent to or received from the model.
/// Both providers natively support mixing text with tool_use /
/// tool_result blocks within a turn, so we model that explicitly rather
/// than flattening to a string.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    /// Speaker.
    pub role: MessageRole,
    /// Ordered content blocks. Almost every turn is one [`TurnBlock::Text`].
    /// Assistant turns may interleave text with [`TurnBlock::ToolUse`];
    /// user turns may carry one or more [`TurnBlock::ToolResult`]s
    /// answering the model's prior tool uses.
    pub blocks: Vec<TurnBlock>,
}

impl ChatTurn {
    /// Convenience: a turn made of a single text block.
    pub fn text(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            blocks: vec![TurnBlock::Text(content.into())],
        }
    }
}

/// A block of content inside a [`ChatTurn`].
#[derive(Debug, Clone)]
pub enum TurnBlock {
    /// Plain text — most common.
    Text(String),
    /// User-supplied image referenced by URL. Providers fetch the URL
    /// themselves (xAI's Responses API and Anthropic's Messages API
    /// both accept URL-typed images directly).
    Image {
        /// Image URL (typically Discord's CDN).
        url: String,
        /// Optional MIME type hint, e.g. `image/png`.
        mime_type: Option<String>,
    },
    /// Assistant asked us to invoke a client-side tool with this input.
    ToolUse {
        /// Provider-supplied id, opaque to us; must be echoed back in
        /// the matching `ToolResult`.
        id: String,
        /// Tool name as declared in the [`ToolDefinition`] we sent.
        name: String,
        /// Input arguments (provider-supplied JSON).
        input: serde_json::Value,
    },
    /// Result for one of the model's prior tool uses.
    ToolResult {
        /// Must match the `id` of the [`TurnBlock::ToolUse`] we're
        /// answering.
        tool_use_id: String,
        /// Result content (JSON-encoded payload as a string, or an error
        /// message when `is_error` is true).
        content: String,
        /// Whether the call failed; signals the model to retry or back off.
        is_error: bool,
    },
}

/// One client-side tool the model is allowed to invoke. Declared on
/// every [`StepRequest`]; both providers honor this list to constrain
/// what the model can call.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    /// Stable name used in tool_use blocks and in our local dispatch.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON Schema describing the `input` object the model must produce.
    pub input_schema: serde_json::Value,
}

/// Provider-specific request knobs that don't fit the shared sampling
/// vocabulary (temperature, top_p, max_tokens). Each provider reads its
/// own field and ignores the others. Fields are optional so personas
/// only have to declare the knobs they actually want to override.
#[derive(Debug, Clone, Default)]
pub struct ProviderOptions {
    /// xAI-specific knobs (e.g. `reasoning_effort` on the Responses API).
    pub xai: Option<XaiOptions>,
    /// Anthropic-specific knobs. Placeholder today; reserved for
    /// fields like extended-thinking budgets when we add them.
    pub anthropic: Option<AnthropicOptions>,
}

/// xAI-specific per-request knobs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct XaiOptions {
    /// Reasoning effort hint forwarded as `reasoning: { effort: ... }`
    /// on the Responses API. Valid values per xAI: `low` | `medium` |
    /// `high`. Only meaningful for reasoning-capable models (grok-4
    /// family); non-reasoning models silently ignore it.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
}

/// Anthropic-specific per-request knobs. Empty for now; kept as a
/// named slot so future fields (e.g. extended-thinking budget) drop
/// in without churning Persona / StepRequest signatures.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnthropicOptions {}

/// Input to [`LlmProvider::step`].
#[derive(Debug, Clone)]
pub struct StepRequest {
    /// Model id to route this call to (e.g. `grok-4.3`,
    /// `claude-sonnet-4-6`). Per-request so a single provider instance
    /// can serve multiple personas using the same provider.
    pub model: String,
    /// Full history through the current point, including prior tool
    /// uses/results from earlier iterations.
    pub messages: Vec<ChatTurn>,
    /// Client-side tools the model may invoke.
    pub tools: Vec<ToolDefinition>,
    /// If true, enable the provider's *server-side* web search tool
    /// (orthogonal to the client-side `tools` list).
    pub enable_web_search: bool,
    /// Soft cap on output tokens. Anthropic requires it; xAI tolerates it.
    pub max_tokens: u32,
    /// Sampling temperature (0.0-2.0). `None` lets the provider's
    /// default apply. Higher = more random; lower = more focused.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass (0.0-1.0). `None` defers to
    /// the provider's default.
    pub top_p: Option<f32>,
    /// Per-provider knobs that don't fit the shared sampling fields.
    /// Each provider implementation reads only its own slot.
    pub provider_options: ProviderOptions,
}

/// One round-trip result from a provider.
#[derive(Debug, Clone)]
pub enum StepResponse {
    /// The model produced a final answer. Stop the loop.
    Final {
        /// Final answer text.
        content: String,
        /// Server-side tool calls performed during this step (e.g.
        /// `web_search`). Each entry is fully resolved (request + response).
        server_tool_calls: Vec<ToolCallRecord>,
        /// Model id reported by the provider for this call.
        model_id: String,
    },
    /// The model is asking us to invoke one or more client-side tools.
    /// The caller must execute them and feed the results back via the
    /// next [`StepRequest`].
    UseTools {
        /// Any text the model emitted alongside the tool uses (some
        /// providers — Anthropic — return preceding text).
        partial_text: Option<String>,
        /// Tool invocations to execute, in declared order.
        tool_uses: Vec<ToolUseRequest>,
        /// Server-side tool calls performed during this step.
        server_tool_calls: Vec<ToolCallRecord>,
        /// Model id reported by the provider for this call.
        model_id: String,
    },
}

/// One client-side tool the model wants us to run.
#[derive(Debug, Clone)]
pub struct ToolUseRequest {
    /// Provider-supplied id; echo back in the matching `ToolResult`.
    pub id: String,
    /// Tool name (must match one in the request's [`ToolDefinition`] list).
    pub name: String,
    /// Provider-parsed input. Already JSON; pass through to the tool impl.
    pub input: serde_json::Value,
}

/// One server-side tool call (e.g. web search) the provider executed
/// on our behalf during a step. Recorded into the DB so the viewer can
/// render the full trace; not consumed by the agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    /// Logical name of the tool (`web_search`, `x_search`,
    /// `fetch_messages`, …). Same column in the `tool_calls` table.
    pub tool_name: String,
    /// Tool input (provider-specific shape for server-side calls; our
    /// declared schema for client-side calls).
    pub request: serde_json::Value,
    /// Tool output (search results / citations / fetched messages /
    /// error payload).
    pub response: serde_json::Value,
}

/// Caller-supplied executor for client-side tools. The agent loop calls
/// this for every [`ToolUseRequest`] the model emits.
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool by name, returning either the JSON result or an
    /// error string that will be sent back to the model as a failed
    /// tool result.
    fn execute(
        &self,
        name: &str,
        input: serde_json::Value,
    ) -> impl std::future::Future<Output = Result<serde_json::Value, ToolError>> + Send;
}

/// Error returned by a [`ToolExecutor`].
#[derive(Debug, Error)]
pub enum ToolError {
    /// The named tool is not registered.
    #[error("unknown tool `{0}`")]
    Unknown(String),
    /// Input was missing a required field or had the wrong shape.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Tool ran but failed.
    #[error("execution failed: {0}")]
    Execution(String),
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
        /// Response body.
        body: String,
    },
    /// Response could not be decoded into the expected shape.
    #[error("decode error: {0}")]
    Decode(String),
    /// Returned by a config-driven constructor when the relevant
    /// `[llm.<provider>]` block is missing.
    #[error("missing config for provider `{0}`")]
    MissingConfig(&'static str),
    /// Hit the agent-loop iteration cap.
    #[error("too many iterations ({0})")]
    TooManyIterations(u32),
    /// Model emitted a tool name we don't know how to translate.
    #[error("malformed tool call: {0}")]
    MalformedToolCall(String),
}

/// Shared interface for one round-trip to the model. Drive the
/// iteration via [`crate::agent::run`].
pub trait LlmProvider: Send + Sync {
    /// Short, stable identifier — just the provider name (e.g. `xai`,
    /// `anthropic`). The specific model used per call comes from
    /// [`StepRequest::model`].
    fn name(&self) -> &str;

    /// One round-trip. The caller must include every prior turn
    /// (including tool uses/results from earlier iterations) in
    /// [`StepRequest::messages`].
    fn step(
        &self,
        request: StepRequest,
    ) -> impl std::future::Future<Output = Result<StepResponse, LlmError>> + Send;
}

/// Static-dispatch union of every supported provider. Each variant is
/// model-agnostic — pick the model per request via
/// [`StepRequest::model`].
#[derive(Debug, Clone)]
pub enum AnyProvider {
    /// xAI Grok.
    Xai(xai::XaiProvider),
    /// Anthropic Claude.
    Anthropic(anthropic::AnthropicProvider),
}

impl AnyProvider {
    /// Provider kind discriminator, useful for looking up which
    /// provider a persona maps to.
    pub fn kind(&self) -> LlmProviderKind {
        match self {
            Self::Xai(_) => LlmProviderKind::Xai,
            Self::Anthropic(_) => LlmProviderKind::Anthropic,
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

    async fn step(&self, request: StepRequest) -> Result<StepResponse, LlmError> {
        match self {
            Self::Xai(p) => p.step(request).await,
            Self::Anthropic(p) => p.step(request).await,
        }
    }
}

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
