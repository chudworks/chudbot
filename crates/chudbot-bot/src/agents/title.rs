//! Conversation-title system-agent resolution and execution.
//!
//! Titles are generated after the first completed user-visible turn, but the
//! title model call is kept out of that turn's reply path. The background job
//! reloads the stored conversation, builds a tiny one-turn transcript from the
//! first completed exchange, and stores a cleaned display title for the trace
//! viewer.

use crate::prelude::*;
use crate::*;

/// Startup-cached resolver for the reserved conversation-title agent.
///
/// A configured `[bot.agents.conversation_title]` entry is used exactly. When
/// that reserved entry is absent, the title agent inherits from normal agent
/// config so titles are generated with the same provider family as the agent
/// that answered the conversation.
#[derive(Debug, Clone)]
pub(crate) struct ConversationTitleSystemAgents {
    /// Exact reserved-agent override loaded from config.
    pub(crate) configured: Option<SystemAgentConfig>,
    /// Synthesized title-agent defaults keyed by the source agent name.
    pub(crate) agent_defaults: BTreeMap<String, SystemAgentConfig>,
    /// Precomputed fallback for each configured platform's default agent.
    pub(crate) platform_defaults: BTreeMap<PlatformName, SystemAgentConfig>,
    /// Fallback inherited from the deployment's default agent.
    pub(crate) default: Option<SystemAgentConfig>,
}

impl ConversationTitleSystemAgents {
    /// Resolve all title-agent variants once from runtime config.
    ///
    /// The cached shape keeps per-turn title spawning cheap while preserving the
    /// inheritance order used before startup caching: explicit reserved agent,
    /// source agent, platform default agent, then deployment default agent.
    pub(crate) fn from_config(config: &BotConfig) -> Self {
        if let Some(configured) = configured_system_agent(config, CONVERSATION_TITLE_AGENT) {
            // Exact config wins globally; inherited defaults are unreachable.
            return Self {
                configured: Some(configured),
                agent_defaults: BTreeMap::new(),
                platform_defaults: BTreeMap::new(),
                default: None,
            };
        }

        let mut agent_defaults = BTreeMap::new();
        for (agent_name, source) in &config.agents {
            let resolved = default_conversation_title_agent(source, config.limits);
            resolved.log_using_default_inherited(agent_name, None);
            agent_defaults.insert(agent_name.clone(), resolved);
        }
        let default = agent_defaults.get(&config.default_agent).cloned();
        let mut platform_defaults = BTreeMap::new();
        for (platform, binding) in &config.platforms {
            let Some(resolved) = agent_defaults.get(&binding.agent) else {
                tracing::warn!(
                    system_agent = CONVERSATION_TITLE_AGENT,
                    platform = %platform,
                    inherited_agent = %binding.agent,
                    "platform default agent is missing while resolving system agent"
                );
                continue;
            };
            platform_defaults.insert(platform.clone(), resolved.clone());
        }

        Self {
            configured: None,
            agent_defaults,
            platform_defaults,
            default,
        }
    }

    /// Return the title agent for a conversation answered by `source_agent_name`.
    pub(crate) fn get(
        &self,
        source_agent_name: &str,
        platform: &PlatformName,
        fallback_agent: &str,
    ) -> Result<&SystemAgentConfig, BotError> {
        if let Some(configured) = &self.configured {
            return Ok(configured);
        }
        self.agent_defaults
            .get(source_agent_name)
            .or_else(|| self.platform_defaults.get(platform))
            .or(self.default.as_ref())
            .ok_or_else(|| BotError::MissingAgent {
                name: fallback_agent.to_string(),
            })
    }
}

/// Build the inherited default title agent for a normal configured agent.
///
/// Defaults intentionally copy provider, model id, provider options, and loop
/// limits from the source agent, but replace the prompt, sampling, and server
/// tools with title-specific settings.
pub(crate) fn default_conversation_title_agent(
    source: &AgentConfig,
    default_limits: AgentLimits,
) -> SystemAgentConfig {
    SystemAgentConfig::from_parts(
        CONVERSATION_TITLE_AGENT,
        source.provider.clone(),
        TITLE_SYSTEM_PROMPT,
        ModelSpec {
            id: source.model.id.clone(),
            server_tools: Default::default(),
            sampling: SamplingOptions {
                max_output_tokens: Some(TITLE_MAX_TOKENS),
                temperature: Some(0.3),
                top_p: None,
            },
            provider_options: source.model.provider_options.clone(),
        },
        source.limits.unwrap_or(default_limits),
    )
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Start fire-and-log title generation for a newly answered conversation.
    ///
    /// This runs after the first turn has already been posted and stored, so a
    /// title failure must not fail the user's turn. The background task logs and
    /// exits; a later caller can still set a title if the conversation remains
    /// untitled.
    pub(crate) fn spawn_title_generation(
        &self,
        conversation_id: ConversationId,
        agent_name: String,
    ) {
        let runtime = (*self).clone();
        spawn_background_task(&self.background, "title generation", async move {
            if let Err(error) = runtime.generate_title(conversation_id, &agent_name).await {
                tracing::warn!(
                    conversation = %conversation_id,
                    agent = %agent_name,
                    error = %error,
                    "title generation failed"
                );
            }
        });
    }

    /// Generate and persist a title when the stored conversation is still untitled.
    ///
    /// The model sees only a synthetic user turn containing the first completed
    /// user message and stored assistant reply. It does not see the full stored
    /// trace, prior platform history, or any live client tools, and its run is
    /// not appended as another conversation turn. The trace viewer learns about
    /// the result only through the stored title and a `TitleUpdated` event.
    pub(crate) async fn generate_title(
        &self,
        conversation_id: ConversationId,
        agent_name: &str,
    ) -> Result<(), BotError> {
        let Some(snapshot) = self
            .storage
            .load_conversation(ConversationLookup::Id {
                id: conversation_id,
            })
            .await
            .map_err(storage_error)?
        else {
            return Err(BotError::MissingConversation { conversation_id });
        };
        // Re-check after loading; background jobs can race with another title
        // writer or with a manually titled conversation.
        if snapshot.conversation.title.is_some() {
            tracing::debug!("conversation title already exists; skipping");
            return Ok(());
        }
        let Some(first) = snapshot
            .turns
            .iter()
            .find(|turn| matches!(turn.turn.status, chudbot_api::TurnStatus::Completed))
        else {
            tracing::debug!("no completed turns available for title generation");
            return Ok(());
        };
        let agent =
            self.conversation_title_agent(agent_name, &snapshot.conversation.channel.platform)?;
        // Use the stored assistant text, not the raw model answer. On the first
        // reply this can include the Discord full-trace footer, so the title
        // prompt must be strong enough to ignore reply chrome.
        let user_text = format!(
            "User said:\n{}\n\nAssistant replied:\n{}",
            first.turn.user_content,
            first.turn.assistant_content.as_deref().unwrap_or("")
        );
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn::text(TurnRole::User, user_text));
        let agent_runtime = self.system_agent(agent);
        let run = agent_runtime
            .run(transcript)
            .await
            .map_err(|error| BotError::Model {
                message: error.to_string(),
            })?;
        let raw = match run.outcome {
            AgentOutcome::Completed { answer } => answer.text,
            AgentOutcome::IterationLimit { max_iterations } => {
                return Err(BotError::Model {
                    message: format!("title generation hit iteration limit ({max_iterations})"),
                });
            }
            AgentOutcome::Failed { error, partial } => {
                let mut message = error.to_string();
                if let Some(partial) = partial
                    && !partial.text.trim().is_empty()
                {
                    message.push_str("\n\nPartial answer:\n");
                    message.push_str(&partial.text);
                }
                return Err(BotError::Model { message });
            }
            AgentOutcome::Cancelled { reason } => {
                return Err(BotError::Model {
                    message: format!("title generation cancelled: {reason}"),
                });
            }
        };
        let title = clean_title(&raw);
        if title.is_empty() {
            // Empty titles are a model-quality issue, not a conversation failure.
            tracing::warn!(raw = %raw, "title generation returned empty title");
            return Ok(());
        }
        self.storage
            .set_conversation_title(conversation_id, title.clone())
            .await
            .map_err(storage_error)?;
        self.publish_conversation(conversation_id, ConversationEventKind::TitleUpdated);
        tracing::info!(title = %title, "conversation title set");
        Ok(())
    }

    /// Resolve the cached title-agent config for the active conversation.
    pub(crate) fn conversation_title_agent(
        &self,
        source_agent_name: &str,
        platform: &PlatformName,
    ) -> Result<&SystemAgentConfig, BotError> {
        self.system_agents.conversation_title.get(
            source_agent_name,
            platform,
            &self.config.default_agent,
        )
    }
}

/// Normalize title-model output into the stored display title.
///
/// Providers sometimes echo label text or quote the title despite the prompt.
/// Cleanup is deliberately small: strip the common wrappers and enforce the
/// display-length cap without trying to reinterpret the model's words.
pub(crate) fn clean_title(raw: &str) -> String {
    let trimmed = raw.trim();
    let trimmed = trimmed
        .strip_prefix("Title:")
        .or_else(|| trimmed.strip_prefix("title:"))
        .or_else(|| trimmed.strip_prefix("Conversation:"))
        .unwrap_or(trimmed)
        .trim();
    let trimmed = trimmed
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            trimmed
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })
        .unwrap_or(trimmed)
        .trim();
    if trimmed.chars().count() <= TITLE_MAX_CHARS {
        return trimmed.to_string();
    }
    trimmed.chars().take(TITLE_MAX_CHARS).collect::<String>()
}
