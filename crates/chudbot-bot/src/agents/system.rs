//! Shared reserved-agent resolution helpers.
//!
//! Reserved system agents are internal jobs backed by ordinary agent-shaped
//! configuration. If a reserved name exists in `[bot.agents]`, that entry wins
//! outright. Otherwise the job-specific resolver synthesizes a small default
//! agent by inheriting the provider, model, and limits from the relevant
//! user-facing agent.
//!
//! This module owns the common cache and runtime construction path. The
//! job-specific fallback tables live in [`TosPreflightSystemAgents`] and
//! [`ConversationTitleSystemAgents`].

use crate::prelude::*;
use crate::*;

/// Startup-resolved view of reserved system agents.
///
/// Resolution is cached when [`BotRuntime`] is constructed so hot paths do not
/// rebuild effective system-agent configs for every message or title job. The
/// child resolvers intentionally preserve different inheritance rules:
///
/// - `tos_preflight`: configured reserved agent, then platform default agent,
///   then global default agent.
/// - `conversation_title`: configured reserved agent, then source conversation
///   agent, then platform default agent, then global default agent.
#[derive(Debug, Clone)]
pub(crate) struct RuntimeSystemAgents {
    /// Safety preflight resolver/cache.
    pub(crate) tos_preflight: TosPreflightSystemAgents,
    /// Conversation title-generation resolver/cache.
    pub(crate) conversation_title: ConversationTitleSystemAgents,
}

impl RuntimeSystemAgents {
    /// Resolve all cacheable system-agent configs from bot configuration.
    pub(crate) fn from_config(config: &BotConfig) -> Self {
        // Keep this at the runtime boundary: config validation does not build a
        // runtime, and request paths should reuse the already-resolved view.
        Self {
            tos_preflight: TosPreflightSystemAgents::from_config(config),
            conversation_title: ConversationTitleSystemAgents::from_config(config),
        }
    }
}

/// Load a reserved system agent that was explicitly configured by name.
///
/// A configured reserved agent is a hard override for inherited defaults, so
/// callers should skip building fallback tables when this returns `Some`.
pub(crate) fn configured_system_agent(config: &BotConfig, name: &str) -> Option<SystemAgentConfig> {
    config.agents.get(name).map(|agent| {
        // Reuse the same resolved shape as defaults so logging and runtime
        // construction see one representation regardless of provenance.
        let resolved = SystemAgentConfig::from_agent_config(name.to_string(), agent, config.limits);
        resolved.log_loaded_from_config();
        resolved
    })
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Build an executable agent from a resolved system-agent config.
    ///
    /// System agents never receive client tools; their tool/model surface is
    /// whatever was resolved into [`SystemAgentConfig`] before runtime use.
    pub(crate) fn system_agent(
        &self,
        agent_config: &SystemAgentConfig,
    ) -> Agent<RoutedLlmBackend<R::Llms>> {
        Agent::new(
            Model {
                backend: RoutedLlmBackend::new(self.llms.clone(), agent_config.provider.clone()),
                spec: agent_config.model.clone(),
            },
            agent_config.spec.clone(),
            NoClientTools,
        )
    }
}
