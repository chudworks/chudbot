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

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::future::Future;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::ids::{ProviderName, ToolName, ToolUseId};
use crate::media::BoxedMediaRef;
use crate::usage::UsageRecord;

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

/// Provider-neutral contract describing a client-side tool's input object.
///
/// Built-in tools should use the typed object contract instead of constructing
/// JSON by hand. Provider crates still receive provider-shaped schema values
/// because OpenAI/xAI/OpenAI-compatible, Anthropic, and Gemini use different
/// tool envelopes and schema dialects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInputSchema {
    #[serde(rename = "type")]
    schema_type: ToolInputObjectType,
    /// Object properties keyed by model-visible input name.
    pub properties: BTreeMap<String, ToolInputValueSchema>,
    /// Required input property names.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
    /// Whether undeclared input fields are accepted.
    #[serde(rename = "additionalProperties")]
    pub additional_properties: bool,
}

impl ToolInputSchema {
    /// Build a strict object input contract.
    pub fn object(fields: impl IntoIterator<Item = ToolInputField>) -> Self {
        let mut properties = BTreeMap::new();
        let mut required = Vec::new();
        for field in fields {
            if field.required {
                required.push(field.name.clone());
            }
            properties.insert(field.name, field.schema);
        }
        Self {
            schema_type: ToolInputObjectType::Object,
            properties,
            required,
            additional_properties: false,
        }
    }

    /// A strict empty object schema for no-argument tools.
    pub fn empty_object() -> Self {
        Self::object([])
    }

    /// Build the provider-neutral JSON Schema representation.
    pub fn json_schema(&self) -> Value {
        serde_json::to_value(self).expect("tool input schema serializes")
    }

    /// Consume the wrapper and return the generic JSON Schema representation.
    pub fn into_json_schema(self) -> Value {
        self.json_schema()
    }
}

impl Default for ToolInputSchema {
    fn default() -> Self {
        Self::empty_object()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ToolInputObjectType {
    Object,
}

/// One named property in a tool input object.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolInputField {
    name: String,
    schema: ToolInputValueSchema,
    required: bool,
}

impl ToolInputField {
    /// Define a required property.
    pub fn required(name: impl Into<String>, schema: ToolInputValueSchema) -> Self {
        Self {
            name: name.into(),
            schema,
            required: true,
        }
    }

    /// Define an optional property.
    pub fn optional(name: impl Into<String>, schema: ToolInputValueSchema) -> Self {
        Self {
            name: name.into(),
            schema,
            required: false,
        }
    }
}

/// Provider-neutral schema for one tool input value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInputValueSchema {
    #[serde(flatten)]
    kind: ToolInputValueKind,
    /// Model-visible field description.
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    /// Allowed string values.
    #[serde(default, rename = "enum", skip_serializing_if = "Vec::is_empty")]
    enum_values: Vec<String>,
    /// Default value hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    default: Option<Value>,
    /// Inclusive minimum.
    #[serde(skip_serializing_if = "Option::is_none")]
    minimum: Option<Value>,
    /// Inclusive maximum.
    #[serde(skip_serializing_if = "Option::is_none")]
    maximum: Option<Value>,
    /// Exclusive minimum.
    #[serde(rename = "exclusiveMinimum", skip_serializing_if = "Option::is_none")]
    exclusive_minimum: Option<Value>,
    /// Minimum string length.
    #[serde(rename = "minLength", skip_serializing_if = "Option::is_none")]
    min_length: Option<usize>,
    /// Maximum string length.
    #[serde(rename = "maxLength", skip_serializing_if = "Option::is_none")]
    max_length: Option<usize>,
    /// Minimum array length.
    #[serde(rename = "minItems", skip_serializing_if = "Option::is_none")]
    min_items: Option<usize>,
    /// Maximum array length.
    #[serde(rename = "maxItems", skip_serializing_if = "Option::is_none")]
    max_items: Option<usize>,
}

impl ToolInputValueSchema {
    /// String value.
    pub fn string() -> Self {
        Self::new(ToolInputValueKind::Typed {
            schema_type: ToolInputValueType::String,
            items: None,
        })
    }

    /// Integer value.
    pub fn integer() -> Self {
        Self::new(ToolInputValueKind::Typed {
            schema_type: ToolInputValueType::Integer,
            items: None,
        })
    }

    /// Number value.
    pub fn number() -> Self {
        Self::new(ToolInputValueKind::Typed {
            schema_type: ToolInputValueType::Number,
            items: None,
        })
    }

    /// Boolean value.
    pub fn boolean() -> Self {
        Self::new(ToolInputValueKind::Typed {
            schema_type: ToolInputValueType::Boolean,
            items: None,
        })
    }

    /// Array value.
    pub fn array(items: ToolInputValueSchema) -> Self {
        Self::new(ToolInputValueKind::Typed {
            schema_type: ToolInputValueType::Array,
            items: Some(Box::new(items)),
        })
    }

    /// Union value emitted as `anyOf` in the provider-neutral JSON Schema.
    pub fn any_of(schemas: impl IntoIterator<Item = ToolInputValueSchema>) -> Self {
        Self::new(ToolInputValueKind::AnyOf {
            any_of: schemas.into_iter().collect(),
        })
    }

    /// Add a model-visible description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Restrict a string field to a fixed set of values.
    pub fn enum_values<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.enum_values = values.into_iter().map(Into::into).collect();
        self
    }

    /// Add a default value hint.
    pub fn default(mut self, value: impl Into<Value>) -> Self {
        self.default = Some(value.into());
        self
    }

    /// Add an inclusive minimum.
    pub fn minimum(mut self, value: impl Into<Value>) -> Self {
        self.minimum = Some(value.into());
        self
    }

    /// Add an inclusive maximum.
    pub fn maximum(mut self, value: impl Into<Value>) -> Self {
        self.maximum = Some(value.into());
        self
    }

    /// Add an exclusive minimum.
    pub fn exclusive_minimum(mut self, value: impl Into<Value>) -> Self {
        self.exclusive_minimum = Some(value.into());
        self
    }

    /// Add a minimum string length.
    pub fn min_length(mut self, value: usize) -> Self {
        self.min_length = Some(value);
        self
    }

    /// Add a maximum string length.
    pub fn max_length(mut self, value: usize) -> Self {
        self.max_length = Some(value);
        self
    }

    /// Add a minimum array length.
    pub fn min_items(mut self, value: usize) -> Self {
        self.min_items = Some(value);
        self
    }

    /// Add a maximum array length.
    pub fn max_items(mut self, value: usize) -> Self {
        self.max_items = Some(value);
        self
    }

    fn new(kind: ToolInputValueKind) -> Self {
        Self {
            kind,
            description: None,
            enum_values: Vec::new(),
            default: None,
            minimum: None,
            maximum: None,
            exclusive_minimum: None,
            min_length: None,
            max_length: None,
            min_items: None,
            max_items: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum ToolInputValueKind {
    Typed {
        #[serde(rename = "type")]
        schema_type: ToolInputValueType,
        #[serde(skip_serializing_if = "Option::is_none")]
        items: Option<Box<ToolInputValueSchema>>,
    },
    AnyOf {
        #[serde(rename = "anyOf")]
        any_of: Vec<ToolInputValueSchema>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ToolInputValueType {
    String,
    Integer,
    Number,
    Boolean,
    Array,
}

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

impl ClientToolResultContent {
    /// Stable kind label for logging and diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Json { .. } => "json",
            Self::Text { .. } => "text",
        }
    }
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn default_input_schema_is_empty_object() {
        let schema = ToolInputSchema::default();

        assert_eq!(schema, ToolInputSchema::empty_object());
        assert_eq!(
            schema.json_schema(),
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            })
        );
    }

    #[test]
    fn typed_input_schema_emits_strict_json_schema_object() {
        let schema = ToolInputSchema::object([
            ToolInputField::required(
                "prompt",
                ToolInputValueSchema::string()
                    .description("Prompt text.")
                    .min_length(1),
            ),
            ToolInputField::optional(
                "references",
                ToolInputValueSchema::array(ToolInputValueSchema::string()).max_items(3),
            ),
            ToolInputField::optional(
                "mode",
                ToolInputValueSchema::string()
                    .enum_values(["fast", "quality"])
                    .default("fast"),
            ),
        ]);

        assert_eq!(
            schema.json_schema(),
            json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Prompt text.",
                        "minLength": 1
                    },
                    "references": {
                        "type": "array",
                        "items": { "type": "string" },
                        "maxItems": 3
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "quality"],
                        "default": "fast"
                    }
                },
                "required": ["prompt"],
                "additionalProperties": false
            })
        );
        let round_trip: ToolInputSchema =
            serde_json::from_value(schema.json_schema()).expect("schema deserializes");
        assert_eq!(round_trip, schema);
    }

    #[test]
    fn typed_input_schema_supports_unions_without_raw_json() {
        let schema = ToolInputSchema::object([ToolInputField::optional(
            "value",
            ToolInputValueSchema::any_of([
                ToolInputValueSchema::string(),
                ToolInputValueSchema::array(ToolInputValueSchema::string()),
            ]),
        )]);

        assert_eq!(
            schema.json_schema()["properties"]["value"],
            json!({
                "anyOf": [
                    { "type": "string" },
                    { "type": "array", "items": { "type": "string" } }
                ]
            })
        );
    }
}
