//! Model-facing tool protocol contracts.

use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::ids::{ProviderName, ToolName, ToolUseId};
use crate::media::BoxedMediaRef;
use crate::usage::UsageRecord;

/// One client-side tool.
pub trait ClientTool: Send + Sync {
    /// Tool error type.
    type Error: std::error::Error + Send + Sync;

    /// Tool specification shown to the model.
    fn spec(&self) -> ClientToolSpec;

    /// Execute one call to this tool.
    fn call(
        &self,
        call: ClientToolCall,
    ) -> impl Future<Output = Result<ClientToolOutput, Self::Error>> + Send;
}

/// A client-side tool the model may invoke.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolSpec {
    /// Description shown to the model.
    pub description: String,
    /// JSON Schema for the input object.
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
    json_schema: serde_json::Value,
}

impl ToolInputSchema {
    /// Wrap a JSON Schema document.
    pub fn new(json_schema: serde_json::Value) -> Self {
        Self { json_schema }
    }

    /// An empty object schema.
    pub fn empty_object() -> Self {
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

/// A client-side tool invocation emitted by a model and evaluated by our code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolCall {
    /// Provider-supplied id to echo in the matching result.
    pub id: ToolUseId,
    /// Tool name.
    pub name: ToolName,
    /// Provider-parsed JSON input.
    pub input: serde_json::Value,
}

/// Model-facing client tool result content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClientToolResultContent {
    /// JSON result.
    Json {
        /// Result value.
        value: serde_json::Value,
    },
    /// Plain-text result.
    Text {
        /// Result text.
        text: String,
    },
}

/// A result for one model-requested client tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolResult {
    /// Id of the call being answered.
    pub tool_use_id: ToolUseId,
    /// Model-facing result content.
    pub content: ClientToolResultContent,
    /// Whether the tool failed.
    pub is_error: bool,
}

/// Persistable client-side tool trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolTrace {
    /// Tool call requested by the model.
    pub call: ClientToolCall,
    /// Tool result furnished back to the model.
    pub result: ClientToolResult,
    /// Full response JSON stored for trace/debugging.
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
    /// Provider-native raw event.
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
    /// Raw provider-native metadata.
    pub raw: serde_json::Value,
}

/// Persistable tool event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolTrace {
    /// Client-side tool call/result.
    Client {
        /// Trace record.
        trace: ClientToolTrace,
    },
    /// Provider-side tool use, with no client-furnished result.
    Server {
        /// Server tool use.
        tool: ServerToolUse,
    },
    /// Provider grounding/citation metadata.
    Grounding {
        /// Grounding metadata.
        metadata: GroundingMetadata,
    },
}

/// Output from a client-side tool executor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolOutput {
    /// Result sent back to the model.
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
    /// Response stored in the trace.
    pub trace_response: serde_json::Value,
    /// Usage/cost incurred by the tool.
    pub usage: Vec<UsageRecord>,
}
