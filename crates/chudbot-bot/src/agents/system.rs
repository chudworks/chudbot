//! Shared reserved-agent resolution helpers.

use crate::prelude::*;
use crate::*;

/// Cached runtime view of reserved agents that can be inherited from configured agents.
#[derive(Debug, Clone)]
pub(crate) struct RuntimeSystemAgents {
    pub(crate) tos_preflight: TosPreflightSystemAgents,
    pub(crate) conversation_title: ConversationTitleSystemAgents,
}

impl RuntimeSystemAgents {
    pub(crate) fn from_config(config: &BotConfig) -> Self {
        Self {
            tos_preflight: TosPreflightSystemAgents::from_config(config),
            conversation_title: ConversationTitleSystemAgents::from_config(config),
        }
    }
}

pub(crate) fn configured_system_agent(config: &BotConfig, name: &str) -> Option<SystemAgentConfig> {
    config.agents.get(name).map(|agent| {
        let resolved = SystemAgentConfig::from_agent_config(name.to_string(), agent, config.limits);
        resolved.log_loaded_from_config();
        resolved
    })
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
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
