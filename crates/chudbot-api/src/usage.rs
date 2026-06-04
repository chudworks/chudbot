//! Usage and cost accounting contracts.

use serde::{Deserialize, Serialize};

use crate::ids::{ModelId, ProviderName, ToolName};

/// What produced a usage/cost record.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UsageSubject {
    /// One language-model step.
    ModelStep,
    /// Provider-side/server-side tool.
    ServerTool {
        /// Tool name.
        name: ToolName,
    },
    /// Client-side tool.
    ClientTool {
        /// Tool name.
        name: ToolName,
    },
    /// Nested agent called as a tool.
    SubAgent {
        /// Tool/agent name.
        name: ToolName,
    },
    /// Image generation.
    ImageGeneration,
    /// Video generation.
    VideoGeneration,
    /// Audio transcription.
    AudioTranscription,
}

/// Native provider cost amount.
///
/// Use strings so providers can preserve exact API-native units without
/// forcing every backend through floating point dollars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAmount {
    /// Amount in the provider-native unit.
    pub amount: String,
    /// Unit name, e.g. `usd`, `usd_ticks`, `credits`, or `requests`.
    pub unit: String,
    /// Whether the value is estimated rather than directly reported.
    pub estimated: bool,
}

/// Usage/cost record for model, server-tool, client-tool, media, or sub-agent
/// work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Provider that produced this usage.
    pub provider: ProviderName,
    /// Model id when applicable.
    pub model: Option<ModelId>,
    /// Subject that incurred the usage.
    pub subject: UsageSubject,
    /// Input tokens.
    pub input_tokens: Option<u64>,
    /// Cached input tokens.
    pub cached_input_tokens: Option<u64>,
    /// Output tokens.
    pub output_tokens: Option<u64>,
    /// Reasoning tokens.
    pub reasoning_tokens: Option<u64>,
    /// Total tokens.
    pub total_tokens: Option<u64>,
    /// Native cost amount.
    pub cost: Option<CostAmount>,
    /// Raw provider usage object for auditing.
    pub raw: Option<serde_json::Value>,
}

impl UsageRecord {
    /// Start a usage record for a provider and subject.
    pub fn new(provider: ProviderName, subject: UsageSubject) -> Self {
        Self {
            provider,
            model: None,
            subject,
            input_tokens: None,
            cached_input_tokens: None,
            output_tokens: None,
            reasoning_tokens: None,
            total_tokens: None,
            cost: None,
            raw: None,
        }
    }
}
