//! Model-facing tool protocol contracts.
//!
//! The API crate keeps tool traffic in provider-neutral shapes so runtime code,
//! provider adapters, storage, and the trace viewer agree on the same contract
//! without importing each other's transport types.
//!
//! There are three intentionally separate tool surfaces:
//!
//! 1. Client tools are Chudbot-owned functions. A [`ClientToolExecutor`] exposes
//!    [`ClientToolDefinition`] values to the model, receives [`ClientToolCall`]
//!    values back from a provider step, and returns a [`ClientToolOutput`].
//! 2. Server tools are provider-owned functions such as hosted search. Providers
//!    report [`ServerToolUse`] records after they have already run; Chudbot never
//!    sends a matching [`ClientToolResult`] for them.
//! 3. Grounding metadata is provider-wide citation or grounding data. It is
//!    stored as [`GroundingMetadata`] so the trace can show it without treating
//!    it as either a client tool call or a provider tool result.
//!
//! The high-level client-tool flow is:
//!
//! 1. Build the model-visible tool list from [`ClientToolExecutor::tools`].
//! 2. Send those specs through the selected provider adapter.
//! 3. Execute returned [`ClientToolCall`] values with
//!    [`ClientToolExecutor::execute`].
//! 4. Append [`ClientToolResult`] blocks, plus any ephemeral native media, to
//!    the next transcript step.
//! 5. Persist [`ToolTrace::Client`] records for auditing and replay.

use std::convert::Infallible;
use std::future::Future;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ids::{ProviderName, ToolName, ToolUseId};
use crate::media::BoxedMediaRef;
use crate::usage::UsageRecord;

/// One named client-side tool exposed by a [`ClientToolExecutor`].
///
/// Definitions are the advertisement half of the client-tool protocol: they
/// describe names and input schemas that provider crates serialize into their
/// native "tools" request shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolDefinition {
    /// Stable model-visible tool name.
    pub name: ToolName,
    /// Model-facing description and input schema.
    pub spec: ClientToolSpec,
}

impl ClientToolDefinition {
    /// Construct a named tool definition.
    pub fn new(name: impl Into<ToolName>, spec: ClientToolSpec) -> Self {
        Self {
            name: name.into(),
            spec,
        }
    }
}

/// Error produced by a client-side tool executor.
///
/// Unknown tools are separated from execution failures because the agent loop
/// uses the sentinel to diagnose disabled or unregistered tool names. Both
/// cases are ultimately converted into model-visible error results so the model
/// can recover in a later step.
#[derive(Debug, Error)]
pub enum ClientToolExecutorError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// The executor does not own the requested tool name.
    #[error("unknown tool `{name}`")]
    Unknown {
        /// Unknown or currently disabled model-visible tool name.
        name: ToolName,
    },
    /// The executor owns the tool but execution failed.
    #[error("execution failed: {source}")]
    Execution {
        /// Tool-specific source error.
        #[source]
        source: E,
    },
}

impl<E> ClientToolExecutorError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    /// Build the unknown-tool sentinel used for name mismatches.
    pub fn unknown(name: impl Into<ToolName>) -> Self {
        Self::Unknown { name: name.into() }
    }

    /// Wrap a tool-owned execution failure.
    pub fn execution(source: E) -> Self {
        Self::Execution { source }
    }

    /// Return true when this is the unknown-tool sentinel.
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown { .. })
    }
}

/// A client-side executor that owns the entire model-visible tool surface for
/// one agent.
///
/// Implementations usually live in runtime crates that have access to storage,
/// media, platform adapters, and configured subagents. This trait stays in
/// `chudbot-api` so the agent loop can statically dispatch calls without taking
/// dependencies on those concrete services.
pub trait ClientToolExecutor: Send + Sync {
    /// Tool execution error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Tool specifications shown to the model for one agent run.
    ///
    /// The agent may filter this list through static agent config before it is
    /// passed to a provider adapter.
    fn tools(&self) -> Vec<ClientToolDefinition>;

    /// Execute one model-requested tool call.
    ///
    /// The agent loop is allowed to call this concurrently for independent
    /// model-emitted calls, so implementations must keep per-call state local or
    /// protect shared state internally.
    fn execute(
        &self,
        call: ClientToolCall,
    ) -> impl Future<Output = Result<ClientToolOutput, ClientToolExecutorError<Self::Error>>> + Send;
}

/// Executor with no client tools.
///
/// This is the default tool executor for agents that should only call the model
/// and any provider-side server tools configured on that model.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoClientTools;

impl ClientToolExecutor for NoClientTools {
    type Error = Infallible;

    fn tools(&self) -> Vec<ClientToolDefinition> {
        Vec::new()
    }

    async fn execute(
        &self,
        call: ClientToolCall,
    ) -> Result<ClientToolOutput, ClientToolExecutorError<Self::Error>> {
        // With an empty advertised surface, every request is a protocol mismatch
        // that the agent loop can turn into a model-visible tool error.
        Err(ClientToolExecutorError::unknown(call.name))
    }
}

/// A client-side tool the model may invoke.
///
/// This is only the model-facing shape. The Rust implementation remains behind
/// [`ClientToolExecutor::execute`], and concrete runtime crates decide how to
/// validate or coerce the JSON input after providers parse it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolSpec {
    /// Natural-language description shown to the model.
    pub description: String,
    /// JSON Schema for the tool input object.
    pub input_schema: ToolInputSchema,
}

/// JSON Schema describing a client-side tool's input object.
///
/// Providers use different envelope field names for this data. OpenAI and xAI
/// call it `parameters`; Anthropic calls it `input_schema`. The schema document
/// itself is JSON Schema, so the API crate keeps one provider-neutral wrapper
/// and lets provider crates place it into their native request shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolInputSchema {
    /// Provider-neutral JSON Schema document.
    json_schema: serde_json::Value,
}

impl ToolInputSchema {
    /// Wrap a JSON Schema document.
    ///
    /// Callers are responsible for passing a schema valid for the target
    /// provider. This type preserves the document without normalizing provider
    /// extensions or schema draft details.
    pub fn new(json_schema: serde_json::Value) -> Self {
        Self { json_schema }
    }

    /// A strict empty object schema for no-argument tools.
    pub fn empty_object() -> Self {
        // Keep no-arg tools explicit and strict so providers do not infer that
        // arbitrary input keys are accepted.
        Self::new(serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false,
        }))
    }

    /// Borrow the raw JSON Schema document for provider conversion.
    pub fn as_json_schema(&self) -> &serde_json::Value {
        &self.json_schema
    }

    /// Consume the wrapper and return the raw JSON Schema document.
    pub fn into_json_schema(self) -> serde_json::Value {
        self.json_schema
    }
}

impl From<serde_json::Value> for ToolInputSchema {
    fn from(json_schema: serde_json::Value) -> Self {
        Self::new(json_schema)
    }
}

/// A client-side tool invocation emitted by a model and evaluated by Chudbot.
///
/// Provider adapters parse their native call envelopes into this shape before
/// the agent loop validates the name against the enabled client-tool set and
/// dispatches it to a [`ClientToolExecutor`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolCall {
    /// Provider-supplied id to echo in the matching [`ClientToolResult`].
    pub id: ToolUseId,
    /// Requested model-visible tool name.
    pub name: ToolName,
    /// Provider-parsed JSON input for the tool implementation.
    pub input: serde_json::Value,
}

/// Model-facing client-tool result content.
///
/// Tool implementations choose JSON when the next model step should inspect
/// structured fields and text when a plain diagnostic or prose result is enough.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClientToolResultContent {
    /// Structured JSON result.
    Json {
        /// Result value returned to the model.
        value: serde_json::Value,
    },
    /// Plain-text result.
    Text {
        /// Result text returned to the model.
        text: String,
    },
}

/// A result for one model-requested client tool call.
///
/// The `tool_use_id` must match the call id supplied by the provider so the
/// next provider request can associate this result with the correct model tool
/// call. The agent loop preserves provider call order when appending multiple
/// results to the transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolResult {
    /// Id of the call being answered.
    pub tool_use_id: ToolUseId,
    /// Model-facing result content.
    pub content: ClientToolResultContent,
    /// Whether the model should treat this result as a tool error.
    pub is_error: bool,
}

/// Persistable client-side tool trace.
///
/// This is the auditable record of a Chudbot-owned tool call: what the model
/// requested, what result was returned, what richer response payload should be
/// shown in the trace viewer, and what usage/cost nested work incurred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolTrace {
    /// Tool call requested by the model.
    pub call: ClientToolCall,
    /// Tool result furnished back to the model.
    pub result: ClientToolResult,
    /// Full response JSON stored for trace/debugging.
    ///
    /// This may contain richer data than [`ClientToolResult::content`], such as
    /// stored media URIs or implementation metadata used by bot-side delivery
    /// logic. It is not automatically sent back to the model.
    pub trace_response: serde_json::Value,
    /// Usage/cost incurred by this client tool, including nested agents or
    /// media generation.
    pub usage: Vec<UsageRecord>,
}

/// Server-side tool use run inside the provider.
///
/// These are not evaluated by our code and do not have tool results to feed
/// into a later model call. The provider may surface raw metadata such as
/// call ids, statuses, or citations, but that is trace/grounding data, not a
/// client-supplied result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerToolUse {
    /// Provider that ran the tool.
    pub provider: ProviderName,
    /// Logical tool name, e.g. `web_search` or `x_search`.
    pub name: ToolName,
    /// Provider call id when present.
    pub id: Option<String>,
    /// Provider status when present.
    pub status: Option<String>,
    /// Provider-native raw event or response fragment.
    pub raw: serde_json::Value,
    /// Usage/cost for this server-side tool when the provider reports it.
    pub usage: Vec<UsageRecord>,
}

/// Grounding metadata returned outside a server tool's result channel.
///
/// xAI and OpenAI can return response-wide citations. Keep those separate
/// from [`ServerToolUse`] so we do not pretend the bot evaluated a tool and
/// produced a result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingMetadata {
    /// Provider that produced the metadata.
    pub provider: ProviderName,
    /// Raw provider-native grounding or citation metadata.
    pub raw: serde_json::Value,
}

/// Persistable tool event.
///
/// Turn traces store client tools, provider-side server tools, and grounding
/// metadata in a single ordered stream because all three can affect what the
/// model saw or what the viewer should explain. Consumers should still branch
/// on the variant instead of assuming every trace has a client call/result pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolTrace {
    /// Chudbot-owned client-side tool call/result.
    Client {
        /// Trace record for a tool executed by [`ClientToolExecutor`].
        trace: ClientToolTrace,
    },
    /// Provider-side tool use, with no client-furnished result.
    Server {
        /// Provider-reported server tool use.
        tool: ServerToolUse,
    },
    /// Provider grounding/citation metadata.
    Grounding {
        /// Grounding metadata.
        metadata: GroundingMetadata,
    },
}

/// Output from a client-side tool executor.
///
/// This is the runtime-only shape returned by tool implementations before the
/// agent loop splits it into transcript content and a persistable
/// [`ClientToolTrace`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolOutput {
    /// Result content sent back to the model.
    pub result: ClientToolResultContent,
    /// Additional media made visible to the next model step.
    ///
    /// This is intentionally not persisted in tool traces. The tool's JSON/text
    /// result remains the auditable protocol output, while these handles let
    /// tools such as `read` expose a stored image as native model media.
    #[serde(skip)]
    pub media: Vec<BoxedMediaRef>,
    /// Whether the tool result should be marked as an error when it is
    /// furnished back to the model.
    pub is_error: bool,
    /// Response stored in the trace viewer and database.
    ///
    /// This can be more complete than `result`, but should still be safe for an
    /// unauthenticated trace viewer because conversation UUIDs are the only web
    /// access control.
    pub trace_response: serde_json::Value,
    /// Usage/cost incurred by the tool, including nested agents or generators.
    pub usage: Vec<UsageRecord>,
}
