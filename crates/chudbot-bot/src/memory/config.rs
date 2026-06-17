//! Configuration and default agent resolution for user memory.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use chudbot_api::{
    AgentLimits, ModelId, ModelSpec, ProviderName, ProviderOptions, SamplingOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

use crate::config::{AgentConfig, SystemAgentConfig};

use super::{compact, diary};

const MEMORY_MODEL_ID: &str = "grok-4.3";
const MEMORY_REASONING_EFFORT: &str = "high";

/// User-memory runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Global memory switch.
    #[serde(default)]
    pub enabled: bool,
    /// Scheduler poll interval in seconds.
    #[serde(default = "default_poll_interval_seconds")]
    pub poll_interval_seconds: u64,
    /// Human-readable compaction interval such as `12h` or `24h`.
    #[serde(default = "default_compaction_interval")]
    pub compaction_interval: String,
    /// Human-readable maximum age for turns considered by diary backfill.
    #[serde(default = "default_diary_backfill_window")]
    pub diary_backfill_window: String,
    /// Human-readable source window summarized by one diary entry.
    #[serde(default = "default_diary_interval")]
    pub diary_interval: String,
    /// SQL lease duration in seconds.
    #[serde(default = "default_lease_seconds")]
    pub lease_seconds: u64,
    /// Maximum jobs to claim per scheduler tick.
    #[serde(default = "default_max_jobs_per_tick")]
    pub max_jobs_per_tick: u32,
    /// Maximum jobs to run concurrently inside this process.
    #[serde(default = "default_max_concurrent_jobs")]
    pub max_concurrent_jobs: u32,
    /// Maximum completed turns included in one diary job.
    #[serde(default = "default_max_transcript_turns_per_diary_job")]
    pub max_transcript_turns_per_diary_job: u32,
    /// Base retry backoff after a failed memory job.
    #[serde(default = "default_retry_backoff_seconds")]
    pub retry_backoff_seconds: u64,
    /// Attempts after which a job is marked failed instead of retried.
    #[serde(default = "default_max_job_attempts")]
    pub max_job_attempts: i32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            poll_interval_seconds: default_poll_interval_seconds(),
            compaction_interval: default_compaction_interval(),
            diary_backfill_window: default_diary_backfill_window(),
            diary_interval: default_diary_interval(),
            lease_seconds: default_lease_seconds(),
            max_jobs_per_tick: default_max_jobs_per_tick(),
            max_concurrent_jobs: default_max_concurrent_jobs(),
            max_transcript_turns_per_diary_job: default_max_transcript_turns_per_diary_job(),
            retry_backoff_seconds: default_retry_backoff_seconds(),
            max_job_attempts: default_max_job_attempts(),
        }
    }
}

impl MemoryConfig {
    /// Parse and validate the human-readable compaction interval.
    pub fn compaction_interval_seconds(&self) -> Result<u64, MemoryConfigError> {
        parse_duration_seconds(&self.compaction_interval)
    }

    /// Parse and validate the maximum diary backfill window.
    pub fn diary_backfill_window_seconds(&self) -> Result<u64, MemoryConfigError> {
        parse_duration_seconds(&self.diary_backfill_window)
    }

    /// Parse and validate the source window summarized by one diary entry.
    pub fn diary_interval_seconds(&self) -> Result<u64, MemoryConfigError> {
        parse_duration_seconds(&self.diary_interval)
    }

    /// Poll interval as a non-zero duration.
    pub fn poll_interval(&self) -> Duration {
        Duration::from_secs(self.poll_interval_seconds.max(1))
    }

    /// Resolve the configured memory agents, falling back to the built-in
    /// default specs for any job kind without a matching configured agent.
    /// Return the memory agents and providers that would be used at runtime.
    pub fn resolved_agent_providers(
        &self,
        agents: &BTreeMap<String, AgentConfig>,
        default_limits: AgentLimits,
    ) -> Vec<(String, ProviderName)> {
        self.resolve_agent_set(agents, default_limits)
            .iter()
            .map(|agent| (agent.name.clone(), agent.provider.clone()))
            .collect()
    }

    pub(crate) fn resolve_agent_set(
        &self,
        agents: &BTreeMap<String, AgentConfig>,
        default_limits: AgentLimits,
    ) -> MemoryAgentSet {
        MemoryAgentSet {
            diary: diary::resolve_agent(agents, default_limits),
            compact: compact::resolve_agent(agents, default_limits),
        }
    }

    pub(crate) fn lease_duration(&self) -> Duration {
        Duration::from_secs(self.lease_seconds.max(1))
    }

    pub(crate) fn retry_backoff(&self, attempts: i32) -> time::Duration {
        let attempts = attempts.max(1) as u64;
        let seconds = self
            .retry_backoff_seconds
            .max(1)
            .saturating_mul(attempts.min(12));
        time::Duration::seconds(i64::try_from(seconds).unwrap_or(i64::MAX))
    }
}

/// Resolved memory agents used by the background scheduler.
#[derive(Debug, Clone)]
pub(crate) struct MemoryAgentSet {
    /// Diary agent.
    pub(crate) diary: SystemAgentConfig,
    /// Profile compaction agent.
    pub(crate) compact: SystemAgentConfig,
}

impl MemoryAgentSet {
    /// Iterate all configured memory agents.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &SystemAgentConfig> {
        [&self.diary, &self.compact].into_iter()
    }
}

pub(crate) fn resolve_memory_agent(
    default_name: &'static str,
    default_prompt: &'static str,
    fallback_max_output_tokens: u32,
    agents: &BTreeMap<String, AgentConfig>,
    default_limits: AgentLimits,
) -> SystemAgentConfig {
    if let Some(agent) = agents.get(default_name) {
        let resolved =
            SystemAgentConfig::from_agent_config(default_name.to_string(), agent, default_limits);
        resolved.log_loaded_from_config();
        return resolved;
    }

    let resolved = SystemAgentConfig::from_parts(
        default_name,
        default_memory_provider(),
        default_prompt,
        ModelSpec {
            id: ModelId::new(MEMORY_MODEL_ID),
            server_tools: BTreeSet::new(),
            sampling: SamplingOptions {
                max_output_tokens: Some(fallback_max_output_tokens),
                temperature: Some(0.2),
                top_p: Some(0.9),
            },
            provider_options: Some(ProviderOptions {
                value: json!({ "reasoning_effort": MEMORY_REASONING_EFFORT }),
            }),
        },
        AgentLimits::default(),
    );
    resolved.log_using_default();
    resolved
}

fn default_memory_provider() -> ProviderName {
    ProviderName::new("grok")
}

fn default_poll_interval_seconds() -> u64 {
    60
}

fn default_compaction_interval() -> String {
    "24h".to_string()
}

fn default_diary_backfill_window() -> String {
    "3d".to_string()
}

fn default_diary_interval() -> String {
    "24h".to_string()
}

fn default_lease_seconds() -> u64 {
    300
}

fn default_max_jobs_per_tick() -> u32 {
    4
}

fn default_max_concurrent_jobs() -> u32 {
    4
}

fn default_max_transcript_turns_per_diary_job() -> u32 {
    40
}

fn default_retry_backoff_seconds() -> u64 {
    300
}

fn default_max_job_attempts() -> i32 {
    5
}

/// Memory config validation errors.
#[derive(Debug, Error)]
pub enum MemoryConfigError {
    /// Duration string is invalid.
    #[error("invalid memory duration `{value}`; expected digits followed by s, m, h, or d")]
    InvalidDuration {
        /// Invalid value.
        value: String,
    },
}

/// Parse a duration with `s`, `m`, `h`, or `d` suffix.
pub fn parse_duration_seconds(value: &str) -> Result<u64, MemoryConfigError> {
    let value = value.trim();
    let Some(unit) = value.chars().last() else {
        return Err(MemoryConfigError::InvalidDuration {
            value: value.to_string(),
        });
    };
    let amount = &value[..value.len().saturating_sub(unit.len_utf8())];
    let amount = amount
        .parse::<u64>()
        .map_err(|_| MemoryConfigError::InvalidDuration {
            value: value.to_string(),
        })?;
    let multiplier = match unit {
        's' => 1,
        'm' => 60,
        'h' => 60 * 60,
        'd' => 24 * 60 * 60,
        _ => {
            return Err(MemoryConfigError::InvalidDuration {
                value: value.to_string(),
            });
        }
    };
    amount
        .checked_mul(multiplier)
        .filter(|seconds| *seconds > 0)
        .ok_or_else(|| MemoryConfigError::InvalidDuration {
            value: value.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use chudbot_api::{AgentLimits, ModelId, ModelSpec, SamplingOptions};
    use test_case::test_case;

    use super::*;
    use crate::config::AgentConfig;
    use crate::memory::{MEMORY_COMPACT_AGENT, MEMORY_DIARY_AGENT};

    #[test_case("1s", 1)]
    #[test_case("5m", 300)]
    #[test_case("2h", 7200)]
    #[test_case("3d", 259200)]
    fn parses_duration_suffixes(input: &str, expected: u64) {
        assert_eq!(parse_duration_seconds(input).unwrap(), expected);
    }

    #[test]
    fn rejects_invalid_duration() {
        assert!(parse_duration_seconds("24").is_err());
        assert!(parse_duration_seconds("0h").is_err());
        assert!(parse_duration_seconds("xh").is_err());
    }

    #[test]
    fn default_diary_backfill_window_is_three_days() {
        assert_eq!(
            MemoryConfig::default()
                .diary_backfill_window_seconds()
                .unwrap(),
            3 * 24 * 60 * 60
        );
    }

    #[test]
    fn default_diary_interval_is_one_day() {
        assert_eq!(
            MemoryConfig::default().diary_interval_seconds().unwrap(),
            24 * 60 * 60
        );
    }

    #[test]
    fn memory_agent_providers_use_named_agents_with_default_fallbacks() {
        let mut agents = BTreeMap::new();
        agents.insert(
            MEMORY_DIARY_AGENT.to_string(),
            test_agent_config("openai", "gpt-5.5"),
        );

        let providers = MemoryConfig::default()
            .resolved_agent_providers(&agents, AgentLimits { max_iterations: 4 });

        assert_eq!(
            providers,
            vec![
                (MEMORY_DIARY_AGENT.to_string(), ProviderName::new("openai")),
                (MEMORY_COMPACT_AGENT.to_string(), ProviderName::new("grok")),
            ]
        );
    }

    fn test_agent_config(provider: &str, model: &str) -> AgentConfig {
        AgentConfig {
            provider: ProviderName::new(provider),
            system_prompt: "configured prompt".to_string(),
            model: ModelSpec {
                id: ModelId::new(model),
                server_tools: BTreeSet::new(),
                sampling: SamplingOptions::default(),
                provider_options: None,
            },
            server_tools: None,
            client_tools: None,
            limits: None,
            image_generation: None,
            video_generation: None,
            audio_transcription: None,
            memory: false,
            subagents: BTreeMap::new(),
        }
    }
}
