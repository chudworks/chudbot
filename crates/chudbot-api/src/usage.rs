//! Usage and cost accounting contracts.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::ids::{ModelId, PlatformName, ProviderName, ToolName};

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

/// Scope filter for one usage cost report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UsageCostScope {
    /// Every conversation on the platform.
    All,
    /// One guild/workspace/server.
    Guild {
        /// Platform guild id.
        guild_id: String,
    },
    /// One channel or thread, optionally inside a guild.
    Channel {
        /// Platform guild id when the channel belongs to one.
        guild_id: Option<String>,
        /// Platform channel id.
        channel_id: String,
    },
}

/// Grouping dimension for one usage cost report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageCostGrouping {
    /// One row aggregating everything in scope.
    Total,
    /// Per guild/workspace; guild-less usage aggregates under `direct`.
    Guild,
    /// Per channel/thread.
    Channel,
    /// Per platform user who drove the turn.
    User,
    /// Per agent.
    Agent,
    /// Per provider registry key.
    Provider,
    /// Per provider/model pair.
    Model,
    /// Per usage subject kind, e.g. `model_step` or `image_generation`.
    Kind,
}

/// Filter and grouping for aggregating stored usage/cost records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageCostQuery {
    /// Messaging platform whose usage is reported.
    pub platform: PlatformName,
    /// Scope filter.
    pub scope: UsageCostScope,
    /// Include only usage recorded at or after this time. `None` = lifetime.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub since: Option<OffsetDateTime>,
    /// Grouping dimension.
    pub group_by: UsageCostGrouping,
    /// Maximum number of rows returned, costliest groups first.
    pub limit: u32,
}

/// One aggregated usage/cost row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageCostRow {
    /// Group key for the grouping dimension: guild id, channel key, user key,
    /// agent name, provider, `provider/model`, or subject kind. `None` for
    /// [`UsageCostGrouping::Total`].
    pub key: Option<String>,
    /// Human-friendly label when storage knows one, e.g. a user display name.
    pub label: Option<String>,
    /// Usage records aggregated into this row.
    pub records: u64,
    /// Distinct conversations contributing usage.
    pub conversations: u64,
    /// Distinct turns contributing usage.
    pub turns: u64,
    /// Summed input tokens.
    pub input_tokens: u64,
    /// Summed cached input tokens.
    pub cached_input_tokens: u64,
    /// Summed output tokens.
    pub output_tokens: u64,
    /// Summed reasoning tokens.
    pub reasoning_tokens: u64,
    /// Summed total tokens.
    pub total_tokens: u64,
    /// Summed cost in USD as a decimal string, when any record carried a
    /// USD-convertible cost.
    pub cost_usd: Option<String>,
    /// Whether any summed cost was estimated rather than provider-reported.
    pub cost_estimated: bool,
    /// Records with no USD-convertible cost amount.
    pub unpriced_records: u64,
}
