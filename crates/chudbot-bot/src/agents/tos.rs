//! ToS and moderation preflight resolution.
//!
//! The preflight runs before the normal conversation turn. It asks a reserved
//! system agent to classify the incoming platform message as allowed or refused,
//! while still letting deployments override that agent through normal config.
//! Operational failures fail open so the bot does not go silent because a
//! moderation model is unavailable; provider-level safety refusals fail closed
//! because they indicate the submitted content tripped the provider's policy.

use crate::prelude::*;
use crate::*;

/// Cached resolver for the ToS preflight system agent.
///
/// A configured `tos_preflight` agent wins exactly. When it is absent, startup
/// resolution synthesizes moderation agents from the configured default agent
/// and any platform-specific default agents so each platform inherits the model
/// family it would otherwise talk to.
#[derive(Debug)]
pub(crate) struct TosPreflightSystemAgents {
    /// Exact `tos_preflight` entry loaded from normal agent config.
    pub(crate) configured: Option<SystemAgentConfig>,
    /// Inherited defaults keyed by platform for non-global platform agents.
    pub(crate) platform_defaults: BTreeMap<PlatformName, SystemAgentConfig>,
    /// Inherited default built from the global default agent.
    pub(crate) default: Option<SystemAgentConfig>,
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Return the cached preflight agent for a platform.
    pub(crate) fn tos_preflight_agent(
        &self,
        platform: &PlatformName,
    ) -> Result<&SystemAgentConfig, BotError> {
        self.system_agents
            .tos_preflight
            .get(platform, &self.config.default_agent)
    }

    /// Run the ToS/moderation preflight for an inbound platform message.
    ///
    /// Missing providers and ordinary model/runtime failures fail open. Errors
    /// that look like provider safety refusals fail closed, matching completed
    /// model output that begins with or contains a `REFUSE` verdict.
    pub(crate) async fn moderation_allows(
        &self,
        message: &PlatformMessage,
        display_name: &str,
    ) -> Result<bool, BotError> {
        let agent_config = self.tos_preflight_agent(&message.id.platform)?;
        if !self.llms.contains_provider(&agent_config.provider) {
            tracing::warn!(
                agent = %agent_config.name,
                provider = %agent_config.provider,
                "moderation provider is missing; failing open"
            );
            return Ok(true);
        }

        let mut transcript = Transcript::new();
        // Include the display name in the classified text so the moderation
        // model sees the same speaker attribution the chat agent will see.
        transcript.push(TranscriptTurn::text(
            TurnRole::User,
            format!(
                "Message to classify:\n<<<\n[{display_name}]: {}\n>>>",
                message.content
            ),
        ));
        let agent = self.system_agent(agent_config);
        let run = match collect_agent_run(agent.run(transcript)).await {
            Ok(run) => run,
            Err(error) => {
                let message = error.to_string();
                if error_indicates_safety_refusal(&message) {
                    tracing::info!(
                        error = %error,
                        "moderation provider refusal detected; treating as refused"
                    );
                    return Ok(false);
                }
                tracing::warn!(error = %error, "moderation errored; failing open");
                return Ok(true);
            }
        };
        match run.outcome {
            AgentOutcome::Completed { answer } => {
                // The prompt asks for a compact verdict, but providers may add
                // casing or surrounding prose. Treat `REFUSE` at the start or
                // after whitespace as a refusal without parsing arbitrary text.
                let verdict = answer.text.trim().to_ascii_uppercase();
                let allowed = !verdict.starts_with("REFUSE")
                    && !verdict.contains(" REFUSE")
                    && verdict != "REFUSE";
                tracing::info!(verdict = %verdict, allowed, "moderation classified message");
                Ok(allowed)
            }
            AgentOutcome::IterationLimit { max_iterations } => {
                tracing::warn!(
                    max_iterations,
                    "moderation hit iteration limit; failing open"
                );
                Ok(true)
            }
            AgentOutcome::Failed { error, .. } => {
                if error_indicates_safety_refusal(&error.to_string()) {
                    tracing::info!(
                        error = %error,
                        "moderation provider refusal detected; treating as refused"
                    );
                    return Ok(false);
                }
                tracing::warn!(error = %error, "moderation failed; failing open");
                Ok(true)
            }
            AgentOutcome::Cancelled { reason } => {
                tracing::warn!(reason = %reason, "moderation was cancelled; failing open");
                Ok(true)
            }
        }
    }
}

impl TosPreflightSystemAgents {
    /// Resolve the preflight agent set once from bot config.
    ///
    /// The result is stored on `BotRuntime` and borrowed for each message, so
    /// per-message moderation does not re-run config inheritance or logging.
    pub(crate) fn from_config(config: &BotConfig) -> Self {
        if let Some(configured) = configured_system_agent(config, TOS_PREFLIGHT_AGENT) {
            return Self {
                configured: Some(configured),
                platform_defaults: BTreeMap::new(),
                default: None,
            };
        }

        let default = config.agents.get(&config.default_agent).map(|source| {
            let resolved = default_tos_preflight_agent(source, config.limits);
            resolved.log_using_default_inherited(&config.default_agent, None);
            resolved
        });
        let mut platform_defaults = BTreeMap::new();
        for (platform, binding) in &config.platforms {
            if binding.agent == config.default_agent {
                continue;
            }
            let Some(source) = config.agents.get(&binding.agent) else {
                tracing::warn!(
                    system_agent = TOS_PREFLIGHT_AGENT,
                    platform = %platform,
                    inherited_agent = %binding.agent,
                    "platform default agent is missing while resolving system agent"
                );
                continue;
            };
            let resolved = default_tos_preflight_agent(source, config.limits);
            resolved.log_using_default_inherited(&binding.agent, Some(platform));
            platform_defaults.insert(platform.clone(), resolved);
        }

        Self {
            configured: None,
            platform_defaults,
            default,
        }
    }

    pub(crate) fn get(
        &self,
        platform: &PlatformName,
        fallback_agent: &str,
    ) -> Result<&SystemAgentConfig, BotError> {
        if let Some(configured) = &self.configured {
            return Ok(configured);
        }
        self.platform_defaults
            .get(platform)
            .or(self.default.as_ref())
            .ok_or_else(|| BotError::MissingAgent {
                name: fallback_agent.to_string(),
            })
    }
}

/// Build the implicit preflight agent inherited from a normal chat agent.
///
/// The moderation prompt is fixed, but provider, model, and loop limits follow
/// the source agent. Sampling is deterministic and capped tightly because the
/// caller only needs a short `ALLOW` or `REFUSE` verdict.
pub(crate) fn default_tos_preflight_agent(
    source: &AgentConfig,
    default_limits: AgentLimits,
) -> SystemAgentConfig {
    SystemAgentConfig::from_parts(
        TOS_PREFLIGHT_AGENT,
        source.provider.clone(),
        MODERATION_PROMPT,
        ModelSpec {
            id: source.model.id.clone(),
            server_tools: Default::default(),
            sampling: SamplingOptions {
                max_output_tokens: Some(8),
                temperature: Some(SamplingNumber::from_static("0.0")),
                top_p: None,
            },
            provider_options: None,
        },
        source.limits.unwrap_or(default_limits),
    )
}

/// Detect provider safety refusals returned as model or tool errors.
///
/// These strings come from provider/tool surfaces that do not produce a normal
/// moderation answer. Matching them keeps policy refusals fail-closed while
/// preserving fail-open behavior for ordinary outages.
pub(crate) fn error_indicates_safety_refusal(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("safety_check") || lower.contains("violates usage guidelines")
}

/// Return whether any client-tool trace contains a provider safety refusal.
pub(crate) fn safety_refusal_in_tool_trace(trace: &[ToolTrace]) -> bool {
    trace.iter().any(|trace| {
        let ToolTrace::Client { trace } = trace else {
            return false;
        };
        if !trace.result.is_error {
            return false;
        }
        match &trace.result.content {
            ClientToolResultContent::Text { text } => error_indicates_safety_refusal(text),
            ClientToolResultContent::Json { value } => error_indicates_safety_refusal(
                // Tool errors are inconsistently shaped: some wrap a string in
                // `error`, while others are themselves a JSON string.
                value
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_else(|| value.as_str().unwrap_or("")),
            ),
        }
    })
}
