//! Language-model backend and model-step contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::ids::{ModelId, ProviderName, ToolName};
use crate::tool::{ClientToolCall, ClientToolSpec, GroundingMetadata, ServerToolUse};
use crate::transcript::{ContentBlock, ProviderContinuation, Transcript};
use crate::usage::UsageRecord;

/// Case-insensitive provider-side/server-side tool names.
///
/// Static config can deserialize this as a list of strings. The agent runner
/// lowercases names before sending a request to a backend, so providers receive a
/// normalized set and decide what each name means for that backend.
pub type ServerToolSet = BTreeSet<String>;

/// Runtime backend capable of calling language-model APIs.
pub trait LlmBackend: Send + Sync {
    /// Backend error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Short backend name.
    fn backend_name(&self) -> &ProviderName;

    /// Execute one model round trip.
    fn step(
        &self,
        request: ModelStepRequest,
    ) -> impl Future<Output = Result<ModelStep, Self::Error>> + Send;

    /// Fetch provider-reported metadata for one model, when this backend can
    /// discover it.
    fn fetch_model_info(
        &self,
        _request: ModelInfoRequest,
    ) -> impl Future<Output = Result<Option<ModelInfo>, Self::Error>> + Send {
        async { Ok(None) }
    }
}

/// A callable language model: backend code plus static model config.
#[derive(Debug, Clone)]
pub struct Model<B> {
    /// Backend implementation, e.g. xAI or OpenAI client code.
    pub backend: B,
    /// Static model config.
    pub spec: ModelSpec,
}

/// Static model configuration.
///
/// This is the TOML-shaped part: model id, sampling, provider-specific static
/// options, and provider-side/server-side tools. It does not know how to call
/// any API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Provider model id.
    pub id: ModelId,
    /// Provider-side/server-side tools this model config allows.
    #[serde(default)]
    pub server_tools: ServerToolSet,
    /// Sampling options.
    #[serde(default)]
    pub sampling: SamplingOptions,
    /// Provider-specific options.
    #[serde(default)]
    pub provider_options: Option<ProviderOptions>,
}

/// One model step request.
#[derive(Debug, Clone)]
pub struct ModelStepRequest {
    /// Model id.
    pub model: ModelId,
    /// Full transcript so far.
    pub transcript: Transcript,
    /// Client-side tools available this step.
    pub client_tools: BTreeMap<ToolName, ClientToolSpec>,
    /// Provider-side/server-side tools available this step.
    pub server_tools: ServerToolSet,
    /// Sampling options.
    pub sampling: SamplingOptions,
    /// Provider-specific options.
    pub provider_options: Option<ProviderOptions>,
}

/// One model metadata request.
#[derive(Debug, Clone)]
pub struct ModelInfoRequest {
    /// Provider model id.
    pub model: ModelId,
    /// Provider-specific options for the already-routed backend.
    pub provider_options: Option<ProviderOptions>,
}

/// Provider-reported model metadata.
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

/// One model step response.
#[derive(Debug, Clone)]
pub enum ModelStep {
    /// Final assistant answer.
    Final {
        /// Assistant step data. `client_tool_calls` should be empty.
        step: AssistantStep,
    },
    /// Assistant requested client-side tools.
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

/// Shared data returned by both final and client-tool model steps.
#[derive(Debug, Clone)]
pub struct AssistantStep {
    /// Assistant content blocks emitted before any tool calls.
    pub content: Vec<ContentBlock>,
    /// Client-side tool calls requested by the model.
    pub client_tool_calls: Vec<ClientToolCall>,
    /// Server-side provider tools already run during this step. These do not
    /// produce client-furnished results.
    pub server_tool_uses: Vec<ServerToolUse>,
    /// Provider grounding/citation metadata.
    pub grounding: Vec<GroundingMetadata>,
    /// Actual model id reported by the provider.
    pub model_id: ModelId,
    /// Opaque continuation to replay to this provider.
    pub continuation: Option<ProviderContinuation>,
    /// Usage/cost reported for this model step.
    pub usage: Vec<UsageRecord>,
}

/// Shared sampling knobs.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderOptions {
    /// Provider-owned serialized value.
    pub value: serde_json::Value,
}
