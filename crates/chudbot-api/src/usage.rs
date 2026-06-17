//! Usage and cost accounting contracts.
//!
//! This module defines the provider-neutral shape that moves through the
//! runtime:
//!
//! 1. Provider and tool adapters emit [`UsageRecord`] values after model,
//!    media, audio, server-tool, client-tool, or sub-agent work.
//! 2. Agent and bot orchestration keep those records attached to the trace
//!    objects that produced them.
//! 3. Storage persists both normalized fields and the raw provider payload,
//!    then reports aggregate rows through [`UsageCostQuery`] and
//!    [`UsageCostRow`].
//!
//! The API boundary is intentionally lossy only where providers differ. Token
//! counts are optional, costs keep provider-native string units, and `raw`
//! preserves the original usage object for audit or future extraction.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::ids::{ModelId, PlatformName, ProviderName, ToolName};

/// Work unit that produced a usage/cost record.
///
/// Storage splits this into a stable subject kind plus an optional name. Keep
/// the enum broad enough for reporting, while storing backend-specific detail
/// in [`UsageRecord::raw`] instead of adding provider-only variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UsageSubject {
    /// One language-model step in an agent run.
    ModelStep,
    /// Provider-side/server-side tool executed inside a model step.
    ServerTool {
        /// Provider-normalized tool name.
        name: ToolName,
    },
    /// Runtime client-side tool executed by the bot.
    ClientTool {
        /// Runtime tool registry name.
        name: ToolName,
    },
    /// Nested agent called through the client-tool interface.
    SubAgent {
        /// Tool/agent name exposed to the parent agent.
        name: ToolName,
    },
    /// Image generation request.
    ImageGeneration,
    /// Video generation request or polling result.
    VideoGeneration,
    /// Audio transcription request.
    AudioTranscription,
}

/// Native provider cost amount.
///
/// Use strings so providers can preserve exact API-native units without
/// forcing every backend through floating point dollars. Reporting currently
/// treats `usd` and `usd_ticks` as USD-convertible and counts other units as
/// unpriced records instead of guessing an exchange rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAmount {
    /// Amount in the provider-native unit, serialized as decimal text.
    pub amount: String,
    /// Unit name, e.g. `usd`, `usd_ticks`, `credits`, or `requests`.
    pub unit: String,
    /// Whether the value is estimated locally rather than directly reported by
    /// the provider.
    pub estimated: bool,
}

/// Usage/cost record for model, server-tool, client-tool, media, or sub-agent
/// work.
///
/// A record has required routing dimensions (`provider` and `subject`) and
/// optional measurements because providers differ in which token and cost
/// counters they expose. `None` means not reported or not applicable; aggregate
/// reports sum missing token fields as zero but preserve unpriced cost rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Provider registry key that produced this usage.
    pub provider: ProviderName,
    /// Provider model id when the usage belongs to a model-backed operation.
    pub model: Option<ModelId>,
    /// Subject that incurred the usage.
    pub subject: UsageSubject,
    /// Input tokens reported by the provider.
    pub input_tokens: Option<u64>,
    /// Cached input tokens included in [`Self::input_tokens`], when reported.
    pub cached_input_tokens: Option<u64>,
    /// Output tokens reported by the provider.
    pub output_tokens: Option<u64>,
    /// Reasoning tokens included in [`Self::output_tokens`] or
    /// provider-specific totals, when reported.
    pub reasoning_tokens: Option<u64>,
    /// Total tokens reported by the provider, when it reports a total.
    pub total_tokens: Option<u64>,
    /// Provider-native cost amount, direct or locally estimated.
    pub cost: Option<CostAmount>,
    /// Raw provider usage object for auditing and later parser improvements.
    pub raw: Option<serde_json::Value>,
}

impl UsageRecord {
    /// Start a usage record for a provider and subject.
    ///
    /// The constructor records the dimensions common to every backend and
    /// leaves all measurements absent. Provider adapters should then fill only
    /// the counters they can defend from the provider response or a documented
    /// local estimator.
    pub fn new(provider: ProviderName, subject: UsageSubject) -> Self {
        Self {
            // Required reporting dimensions.
            provider,
            subject,
            // Optional provider measurements.
            model: None,
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
///
/// These ids are platform-native strings at the API boundary. Storage maps them
/// to its internal conversation/channel keys before running aggregate queries.
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
///
/// The grouping controls the meaning of [`UsageCostRow::key`]. Rows are
/// returned in cost order by the storage implementation, with unpriced groups
/// after groups that have USD-convertible cost.
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
///
/// Reports include both normal turn usage and background memory-job usage when
/// the selected platform/scope/time window matches those rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageCostQuery {
    /// Messaging platform whose usage is reported.
    pub platform: PlatformName,
    /// Platform scope to include.
    pub scope: UsageCostScope,
    /// Include only usage recorded at or after this time. `None` means
    /// lifetime usage.
    #[serde(with = "time::serde::rfc3339::option", default)]
    pub since: Option<OffsetDateTime>,
    /// Grouping dimension for the returned rows.
    pub group_by: UsageCostGrouping,
    /// Maximum number of rows returned, costliest groups first.
    pub limit: u32,
}

/// One aggregated usage/cost row.
///
/// Numeric counters are already summed by storage. `cost_usd` is present only
/// when at least one record in the group has a USD-convertible cost unit; use
/// `unpriced_records` to tell whether the row is missing costs for part or all
/// of the underlying usage.
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
