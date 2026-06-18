//! Language-model backend and model-step contracts.
//!
//! This module defines the narrow boundary between the agent loop and concrete
//! provider crates. It does not route provider names, execute client tools, or
//! validate TOML; it only describes the request a routed backend receives and
//! the control-flow outcome it returns.
//!
//! # Step Flow
//!
//! 1. Configuration resolves a [`ModelSpec`] and pairs it with an already-routed
//!    backend in a [`Model`].
//! 2. The agent loop builds a [`ModelStepRequest`] from the current
//!    [`Transcript`], enabled client tools, enabled server tools, sampling, and
//!    opaque provider options.
//! 3. The [`LlmBackend`] implementation serializes that request into its native
//!    API shape and performs one provider round trip.
//! 4. The backend returns a [`ModelStep`] telling the agent loop whether to
//!    finish, execute client tools, or continue the provider conversation.
//!
//! # Contract Boundaries
//!
//! - Provider routing lives outside this module, typically behind
//!   [`crate::registries::LlmProviderRegistry`]. A [`ModelStepRequest`] is
//!   already addressed to one backend and intentionally carries no provider key.
//! - Provider-specific request knobs travel through [`ProviderOptions`].
//!   Provider-specific replay state travels back through
//!   [`ProviderContinuation`]. The API crate treats both as opaque JSON.
//! - Client tools are executed by Chudbot code after a model step with
//!   [`ModelStepKind::ClientTools`]. Server tools and grounding metadata
//!   are provider-owned trace data; they do not imply any client-furnished tool
//!   result.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;

pub(crate) use crate::collector::ModelStepCollector;
pub use crate::collector::{ModelStepCollectionError, ModelStepReducerError, collect_model_step};

use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::ids::{ModelId, ProviderName, ToolName};
use crate::reasoning::ReasoningItem;
use crate::storage::ModelStepKind;
use crate::tool::{ClientToolCall, ClientToolSpec, GroundingMetadata, ServerToolUse};
use crate::transcript::{ContentBlock, ProviderContinuation, Transcript};
use crate::usage::UsageRecord;

/// Provider-side/server-side tool names normalized by the agent loop.
///
/// Static config can deserialize this as a list of strings. The agent runner
/// lowercases names before sending a request to a backend, so providers receive a
/// normalized set and decide what each name means for that backend.
pub type ServerToolSet = BTreeSet<String>;

/// Runtime backend capable of one language-model provider's API.
///
/// Implementations live in provider crates or in thin routing adapters. By the
/// time a caller holds an `LlmBackend`, the provider has already been selected;
/// `step` receives only model-level request data and must not need global config
/// or registry state beyond what the backend value already owns.
pub trait LlmBackend: Send + Sync {
    /// Backend error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Provider name used for tracing, continuation ownership, and audit data.
    fn backend_name(&self) -> &ProviderName;

    /// Execute one model round trip and stream normalized provider events.
    ///
    /// A provider adapter is responsible for translating transcripts, client
    /// tool specs, server-tool names, sampling, and provider options into the
    /// provider's native request format. The returned stream must finish with
    /// one [`ModelStepEvent::Finished`] event so the agent loop can reduce the
    /// event stream into a [`ModelStep`] for persistence and compatibility
    /// callers.
    fn step(
        &self,
        request: ModelStepRequest,
    ) -> impl Stream<Item = Result<ModelStepEvent, Self::Error>> + Send + '_;

    /// Fetch provider-reported metadata for one model, when supported.
    ///
    /// Metadata discovery is a control-plane convenience for validation and UI
    /// surfaces. Generation must still work for providers that cannot discover
    /// model limits or only expose static configuration.
    fn fetch_model_info(
        &self,
        _request: ModelInfoRequest,
    ) -> impl Future<Output = Result<Option<ModelInfo>, Self::Error>> + Send {
        // Metadata discovery is optional; absence of a provider models endpoint
        // is represented as a cacheable miss, not a backend failure.
        async { Ok(None) }
    }
}

/// Callable language model: routed backend code plus static model config.
///
/// This is the value an agent runs. The backend owns the provider route and
/// credentials; the spec owns the model id and model-level request policy.
#[derive(Debug, Clone)]
pub struct Model<B> {
    /// Backend implementation or registry route selected for this model.
    pub backend: B,
    /// Static model config applied to every step for this model.
    pub spec: ModelSpec,
}

/// Static model configuration.
///
/// This is the TOML-shaped part: model id, sampling, provider-specific static
/// options, and provider-side/server-side tools. It does not know how to call
/// any API, and it does not carry the provider route; routing is represented by
/// the backend half of [`Model`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Provider model id, e.g. the string sent in a vendor request's `model`.
    pub id: ModelId,
    /// Provider-side/server-side tools this model config allows.
    ///
    /// The agent combines this allowlist with the agent-level allowlist before
    /// building a [`ModelStepRequest`].
    #[serde(default)]
    pub server_tools: ServerToolSet,
    /// Provider-neutral sampling options applied to every step.
    #[serde(default)]
    pub sampling: SamplingOptions,
    /// Opaque provider-specific options applied to every step.
    #[serde(default)]
    pub provider_options: Option<ProviderOptions>,
}

/// Fully shaped request for one provider round trip.
///
/// The agent loop builds this after applying system instructions, tool
/// allowlists, and model policy. Provider adapters should treat it as immutable
/// input, translating the fields into native wire format without re-resolving
/// config.
#[derive(Debug, Clone)]
pub struct ModelStepRequest {
    /// Provider model id to request for this step.
    pub model: ModelId,
    /// Full model-facing transcript so far, including current instructions.
    pub transcript: Transcript,
    /// Client-side tools available for the model to request this step.
    ///
    /// Providers serialize these into their function/tool declaration format.
    /// Actual execution stays outside provider crates and only happens if the
    /// response has [`ModelStepKind::ClientTools`].
    pub client_tools: BTreeMap<ToolName, ClientToolSpec>,
    /// Provider-side/server-side tools available for the provider to run.
    ///
    /// These names are already normalized by the agent loop; each provider maps
    /// the logical names it understands to provider-native request fields.
    pub server_tools: ServerToolSet,
    /// Provider-neutral sampling options for this step.
    pub sampling: SamplingOptions,
    /// Opaque provider-specific options for this already-routed backend.
    pub provider_options: Option<ProviderOptions>,
}

/// Provider metadata lookup for one model.
///
/// This is intentionally smaller than [`ModelStepRequest`]: model-info queries
/// should not depend on transcript state or tool exposure, but may still need
/// provider-specific options such as deployment or project settings.
#[derive(Debug, Clone)]
pub struct ModelInfoRequest {
    /// Provider model id.
    pub model: ModelId,
    /// Provider-specific options for the already-routed backend.
    pub provider_options: Option<ProviderOptions>,
}

/// Provider-reported model metadata normalized for config and UI consumers.
///
/// Providers rarely agree on naming, completeness, or whether output limits are
/// reported separately. Keep known shared limits in typed fields and retain raw
/// metadata for audit/debugging without expanding the common contract each time
/// one provider adds a field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Provider model id.
    pub id: ModelId,
    /// Maximum input/context tokens accepted by the model.
    pub context_window_tokens: Option<u64>,
    /// Maximum output tokens the model can produce, when reported separately.
    pub max_output_tokens: Option<u64>,
    /// Raw provider model metadata for auditing and future extraction.
    pub raw: Option<serde_json::Value>,
}

/// Agent-loop decision collected from one provider round trip.
///
/// A provider adapter must finish each streamed step with exactly one terminal
/// kind. The shared [`ModelStepOutput`] payload carries the ordered provider
/// items and usage emitted during that round trip.
#[derive(Debug, Clone)]
pub struct ModelStep {
    /// Terminal kind for this model step.
    pub kind: ModelStepKind,
    /// Collected provider output.
    pub output: ModelStepOutput,
}

impl ModelStep {
    /// Build a collected step from its terminal kind and output.
    pub fn new(kind: ModelStepKind, output: ModelStepOutput) -> Self {
        Self { kind, output }
    }

    /// Terminal kind for this model step.
    pub fn kind(&self) -> ModelStepKind {
        self.kind
    }

    /// Borrow the collected output.
    pub fn output(&self) -> &ModelStepOutput {
        &self.output
    }

    /// Consume the step and return its collected output.
    pub fn into_output(self) -> ModelStepOutput {
        self.output
    }
}

/// Ordered provider response data shared by all [`ModelStep`] outcomes.
///
/// Provider adapters normalize native responses into ordered items before the
/// agent loop decides what to do next. Transcript blocks are the only items the
/// agent appends to [`Transcript`]; reasoning, server tools, grounding, and
/// usage are trace metadata.
#[derive(Debug, Clone)]
pub struct ModelStepOutput {
    /// Actual model id reported by the provider for this step.
    ///
    /// Providers may return aliases, dated variants, or deployment-specific
    /// identifiers that differ from the requested [`ModelStepRequest::model`].
    pub model_id: ModelId,
    /// Ordered provider output items.
    pub items: Vec<ModelStepItem>,
    /// Usage/cost reported for this model step.
    pub usage: Vec<UsageRecord>,
}

impl ModelStepOutput {
    /// Build an empty collected output for a concrete model id.
    pub fn new(model_id: ModelId) -> Self {
        Self {
            model_id,
            items: Vec::new(),
            usage: Vec::new(),
        }
    }

    /// Model-facing output blocks emitted by this step.
    fn output_blocks(&self) -> impl Iterator<Item = &ModelOutputBlock> {
        self.items.iter().filter_map(|item| match item {
            ModelStepItem::OutputBlock(block) => Some(block),
            ModelStepItem::Reasoning(_)
            | ModelStepItem::ServerToolUse(_)
            | ModelStepItem::Grounding(_) => None,
        })
    }

    /// Model-facing transcript blocks emitted by this step.
    pub fn transcript_blocks(&self) -> impl Iterator<Item = ContentBlock> + '_ {
        self.output_blocks()
            .cloned()
            .map(ModelOutputBlock::into_content_block)
    }

    /// Final answer blocks, excluding tool calls and provider continuations.
    pub fn answer_blocks(&self) -> Vec<ContentBlock> {
        self.output_blocks()
            .filter_map(|block| match block {
                ModelOutputBlock::Text { .. } => Some(block.clone().into_content_block()),
                ModelOutputBlock::ClientToolCall(_) | ModelOutputBlock::Continuation(_) => None,
            })
            .collect()
    }

    /// Client-side tool calls requested by the model.
    pub fn client_tool_calls(&self) -> impl Iterator<Item = &ClientToolCall> {
        self.output_blocks().filter_map(|block| match block {
            ModelOutputBlock::ClientToolCall(call) => Some(call),
            ModelOutputBlock::Text { .. } | ModelOutputBlock::Continuation(_) => None,
        })
    }

    /// Viewer-safe reasoning summary metadata.
    pub fn reasoning(&self) -> impl Iterator<Item = &ReasoningItem> {
        self.items.iter().filter_map(|item| match item {
            ModelStepItem::Reasoning(reasoning) => Some(reasoning),
            ModelStepItem::OutputBlock(_)
            | ModelStepItem::ServerToolUse(_)
            | ModelStepItem::Grounding(_) => None,
        })
    }

    /// Provider-owned server-side tool activity.
    pub fn server_tool_uses(&self) -> impl Iterator<Item = &ServerToolUse> {
        self.items.iter().filter_map(|item| match item {
            ModelStepItem::ServerToolUse(tool) => Some(tool),
            ModelStepItem::OutputBlock(_)
            | ModelStepItem::Reasoning(_)
            | ModelStepItem::Grounding(_) => None,
        })
    }

    /// Provider grounding/citation metadata.
    pub fn grounding(&self) -> impl Iterator<Item = &GroundingMetadata> {
        self.items.iter().filter_map(|item| match item {
            ModelStepItem::Grounding(metadata) => Some(metadata),
            ModelStepItem::OutputBlock(_)
            | ModelStepItem::Reasoning(_)
            | ModelStepItem::ServerToolUse(_) => None,
        })
    }

    /// Last provider continuation emitted in transcript-visible output.
    pub fn continuation(&self) -> Option<&ProviderContinuation> {
        self.output_blocks()
            .filter_map(|block| match block {
                ModelOutputBlock::Continuation(continuation) => Some(continuation),
                ModelOutputBlock::Text { .. } | ModelOutputBlock::ClientToolCall(_) => None,
            })
            .last()
    }

    /// Concatenate text transcript blocks for user-facing answer text.
    pub fn answer_text(&self) -> String {
        let mut text = String::new();
        for block in self.output_blocks() {
            if let ModelOutputBlock::Text { text: block_text } = block {
                text.push_str(block_text);
            }
        }
        text
    }
}

/// One model-facing output block emitted by a provider/model step.
///
/// This is intentionally narrower than [`ContentBlock`]: providers may emit
/// assistant text, client-tool-call intents, and provider continuations, but
/// only the agent runtime can append client-tool results or tool-result media.
#[derive(Debug, Clone)]
pub enum ModelOutputBlock {
    /// Plain UTF-8 assistant text.
    Text {
        /// Text fragment or completed text block.
        text: String,
    },
    /// Assistant-requested client tool invocation.
    ClientToolCall(ClientToolCall),
    /// Opaque provider continuation state.
    Continuation(ProviderContinuation),
}

impl ModelOutputBlock {
    /// Convert this model-output block to the transcript block replayed later.
    pub fn into_content_block(self) -> ContentBlock {
        match self {
            Self::Text { text } => ContentBlock::Text { text },
            Self::ClientToolCall(call) => ContentBlock::ClientToolCall(call),
            Self::Continuation(continuation) => ContentBlock::Continuation(continuation),
        }
    }
}

/// One ordered item emitted by a model step.
#[derive(Debug, Clone)]
pub enum ModelStepItem {
    /// Model-facing output that can be appended to the assistant transcript turn.
    OutputBlock(ModelOutputBlock),
    /// Viewer-safe reasoning summary metadata.
    Reasoning(ReasoningItem),
    /// Provider-owned hosted/server-side tool activity.
    ServerToolUse(ServerToolUse),
    /// Provider-owned citation or grounding metadata.
    Grounding(GroundingMetadata),
}

/// Stream event emitted by one provider/model round trip.
#[derive(Debug, Clone)]
pub enum ModelStepEvent {
    /// Delta for one ordered output item.
    Delta(ModelStepDelta),
    /// Opaque provider continuation state.
    Continuation(ProviderContinuation),
    /// Provider-owned hosted/server-side tool activity.
    ServerToolUse(ServerToolUse),
    /// Provider-owned citation or grounding metadata.
    Grounding(GroundingMetadata),
    /// Usage/cost reported by the provider for this step.
    Usage(UsageRecord),
    /// Terminal control-flow classification for this provider step.
    Finished {
        /// Terminal kind.
        kind: ModelStepKind,
        /// Actual model id reported by the provider for this step.
        model_id: ModelId,
    },
}

impl ModelStepEvent {
    /// Stable kind label for logging and diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Delta(delta) => delta.kind(),
            Self::Continuation(_) => "continuation",
            Self::ServerToolUse(_) => "server_tool_use",
            Self::Grounding(_) => "grounding",
            Self::Usage(_) => "usage",
            Self::Finished { .. } => "finished",
        }
    }
}

/// Streaming delta for one model-step item.
#[derive(Debug, Clone)]
pub enum ModelStepDelta {
    /// Text delta for one assistant transcript block.
    Text {
        /// Provider or adapter item id for ordering and accumulation.
        item_id: String,
        /// Text fragment.
        delta: String,
    },
    /// Reasoning-summary delta.
    ReasoningSummary {
        /// Provider or adapter item id for ordering and accumulation.
        item_id: String,
        /// Provider that emitted this reasoning item.
        provider: ProviderName,
        /// Provider-specific summary kind, if any.
        kind: Option<String>,
        /// Summary text fragment.
        delta: String,
    },
    /// Client-tool-call delta.
    ClientToolCall {
        /// Provider or adapter item id for ordering and accumulation.
        item_id: String,
        /// Stable tool-use id.
        id: crate::ids::ToolUseId,
        /// Tool name, when known for this delta.
        name: Option<ToolName>,
        /// JSON argument fragment.
        arguments_delta: String,
    },
}

impl ModelStepDelta {
    /// Stable kind label for logging and diagnostics.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Text { .. } => "delta.text",
            Self::ReasoningSummary { .. } => "delta.reasoning_summary",
            Self::ClientToolCall { .. } => "delta.client_tool_call",
        }
    }
}

/// Convert completed reasoning items into delta events for providers that only
/// expose reasoning after the terminal response is known.
pub fn reasoning_items_to_delta_events(
    items: impl IntoIterator<Item = ReasoningItem>,
    fallback_prefix: &str,
) -> Vec<ModelStepEvent> {
    let mut events = Vec::new();
    for (item_index, item) in items.into_iter().enumerate() {
        let item_id = item
            .id
            .clone()
            .unwrap_or_else(|| format!("{fallback_prefix}:{item_index}"));
        for summary in item.summary {
            if summary.text.is_empty() {
                continue;
            }
            events.push(ModelStepEvent::Delta(ModelStepDelta::ReasoningSummary {
                item_id: item_id.clone(),
                provider: item.provider.clone(),
                kind: summary.kind,
                delta: summary.text,
            }));
        }
    }
    events
}

/// Provider-neutral sampling knobs shared by model providers.
///
/// Every field is optional because providers differ in defaults and support.
/// Provider adapters should omit unsupported knobs rather than inventing a
/// cross-provider default in this contract.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SamplingOptions {
    /// Max output tokens for one model step.
    pub max_output_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass.
    pub top_p: Option<f32>,
}

/// Provider-specific options for the already-routed backend.
///
/// The API crate intentionally keeps this as raw JSON so provider crates can
/// evolve their own typed option structs without forcing shared schema changes.
/// Config validation should still catch stale wrapper keys around this value;
/// the payload itself belongs to the provider adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderOptions {
    /// Provider-owned serialized value, usually decoded by the provider crate.
    pub value: serde_json::Value,
}
