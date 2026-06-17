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
//! - Client tools are executed by Chudbot code after a
//!   [`ModelStep::UseClientTools`] response. Server tools and grounding metadata
//!   are provider-owned trace data; they do not imply any client-furnished tool
//!   result.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::ids::{ModelId, ProviderName, ToolName};
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

    /// Execute one model round trip and return the agent-loop control decision.
    ///
    /// A provider adapter is responsible for translating transcripts, client
    /// tool specs, server-tool names, sampling, and provider options into the
    /// provider's native request format. The returned [`ModelStep`] must contain
    /// normalized tool, grounding, continuation, and usage data for the rest of
    /// the bot to persist without understanding provider wire formats.
    fn step(
        &self,
        request: ModelStepRequest,
    ) -> impl Future<Output = Result<ModelStep, Self::Error>> + Send;

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
    /// response is [`ModelStep::UseClientTools`].
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

/// Agent-loop decision returned by one provider round trip.
///
/// A provider adapter should return exactly one variant per request. The shared
/// [`AssistantStep`] payload carries any assistant content, client tool calls,
/// server-tool traces, grounding metadata, continuation state, and usage emitted
/// by that round trip.
#[derive(Debug, Clone)]
pub enum ModelStep {
    /// Final assistant answer for the current agent run.
    ///
    /// The agent persists trace and usage data, appends the assistant content to
    /// the transcript, and stops iterating.
    Final {
        /// Assistant step data. `client_tool_calls` should be empty.
        step: AssistantStep,
    },
    /// Assistant requested client-side tools that Chudbot must execute.
    ///
    /// The agent appends this assistant step, executes the requested client
    /// tools through its tool executor, appends tool results as a user turn, and
    /// asks the backend for another step.
    UseClientTools {
        /// Assistant step data. `client_tool_calls` should be non-empty.
        step: AssistantStep,
    },
    /// Provider returned useful continuation/usage/server-tool state but no
    /// client tools and no user-visible answer yet. The agent loop should
    /// append the continuation and call the provider again, bounded by its
    /// iteration limits.
    Continue {
        /// Assistant step data.
        step: AssistantStep,
    },
}

/// Provider response data shared by all [`ModelStep`] outcomes.
///
/// Provider adapters normalize native responses into this struct before the
/// agent loop decides what to do next. Fields here are also the source for
/// persisted model-step traces, tool traces, continuation replay, and usage
/// accounting.
#[derive(Debug, Clone)]
pub struct AssistantStep {
    /// Assistant content blocks emitted before any client tool calls.
    ///
    /// For final steps these blocks become the user-visible answer. For
    /// non-final steps they are preserved in the transcript so the provider can
    /// continue from the exact assistant state it emitted.
    pub content: Vec<ContentBlock>,
    /// Client-side tool calls requested by the model for Chudbot to execute.
    pub client_tool_calls: Vec<ClientToolCall>,
    /// Server-side provider tools already run during this step. These do not
    /// produce client-furnished results.
    pub server_tool_uses: Vec<ServerToolUse>,
    /// Provider grounding/citation metadata not tied to a client tool result.
    pub grounding: Vec<GroundingMetadata>,
    /// Actual model id reported by the provider for this step.
    ///
    /// Providers may return aliases, dated variants, or deployment-specific
    /// identifiers that differ from the requested [`ModelStepRequest::model`].
    pub model_id: ModelId,
    /// Opaque continuation to replay only to the provider that emitted it.
    pub continuation: Option<ProviderContinuation>,
    /// Usage/cost reported for this model step.
    pub usage: Vec<UsageRecord>,
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
