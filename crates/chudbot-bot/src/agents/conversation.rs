//! Conversation-facing agent construction.
//!
//! This module bridges configured agents and per-turn runtime state into the
//! executable agent used by a conversation turn. It owns the last-mile choices
//! about runtime-only tool exposure, nested subagent wiring, and the operational
//! guidance embedded in system prompts.

use crate::prelude::*;
use crate::*;

/// Concrete agent assembled for conversation turns and recursive subagent calls.
type ConversationAgent<R> =
    Agent<RoutedLlmBackend<<R as BotRuntimeTypes>::Llms>, RuntimeToolExecutor<R>>;

/// Resolved agent plus rendered instructions used to assemble one executable agent.
pub(crate) struct ConversationAgentAssembly<'a> {
    /// Config map key used for recursion checks and diagnostics.
    pub(crate) agent_name: &'a str,
    /// Static deployment config for the resolved agent.
    pub(crate) agent_config: &'a AgentConfig,
    /// Fully rendered system instructions for this top-level turn or subagent.
    pub(crate) rendered_system_instructions: String,
    /// Whether this assembled agent can deliver final reply artifacts and write memory.
    pub(crate) top_level: bool,
}

/// Runtime turn context shared by the top-level agent and configured subagents.
pub(crate) struct ConversationAgentContext<'a> {
    pub(crate) reply_to: &'a MessageRef,
    pub(crate) turn_user: &'a UserRef,
    pub(crate) turn_user_display_name: &'a str,
    pub(crate) conversation_id: ConversationId,
    pub(crate) turn_id: TurnId,
}

/// Role-specific client-tool surface for a conversation agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConversationToolPolicy {
    top_level: bool,
    memory_enabled: bool,
}

impl ConversationToolPolicy {
    fn new(top_level: bool, memory_enabled: bool) -> Self {
        Self {
            top_level,
            memory_enabled,
        }
    }

    fn fetch_messages(self) -> bool {
        true
    }

    fn memory_lookup(self) -> bool {
        self.memory_enabled
    }

    fn memory_writes(self) -> bool {
        self.top_level && self.memory_enabled
    }

    fn generated_media_tools(self) -> bool {
        self.top_level
    }

    fn final_reply_attach(self) -> bool {
        self.top_level
    }
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Assemble the executable agent used for one turn or subagent call.
    ///
    /// The assembly carries the resolved configured agent plus the already
    /// rendered system instructions that should be used for this run. `stack`
    /// tracks the active agent expansion
    /// chain so configured subagent cycles fail during construction instead of
    /// becoming recursive tool calls at runtime.
    pub(crate) fn build_conversation_agent(
        &self,
        assembly: ConversationAgentAssembly<'_>,
        context: &ConversationAgentContext<'_>,
        stack: &mut Vec<String>,
    ) -> Result<ConversationAgent<R>, BotError> {
        let ConversationAgentAssembly {
            agent_name,
            agent_config,
            rendered_system_instructions,
            top_level,
        } = assembly;
        let policy =
            ConversationToolPolicy::new(top_level, self.agent_memory_enabled(agent_config));
        self.ensure_conversation_agent_services_exist(agent_name, agent_config, policy)?;

        // Only the active expansion path is recursive. Sibling subagents may
        // legitimately point at the same configured agent.
        if stack.iter().any(|name| name == agent_name) {
            tracing::warn!("recursive agent reference detected");
            return Err(BotError::RecursiveAgent {
                name: agent_name.to_string(),
            });
        }
        stack.push(agent_name.to_string());

        // Start from the static agent config, replace its base prompt with the
        // rendered instructions for this run, then layer in runtime-only tool
        // exposure.
        let mut spec = agent_config.agent_spec(self.config.limits);
        spec.system_prompt = rendered_system_instructions;
        if policy.memory_lookup() {
            ensure_client_tool_enabled(&mut spec.client_tools, memory::LOOKUP_USER_MEMORY_TOOL);
        }
        if policy.memory_writes() {
            ensure_client_tool_enabled(&mut spec.client_tools, memory::REMEMBER_USER_MEMORY_TOOL);
            ensure_client_tool_enabled(&mut spec.client_tools, memory::FORGET_USER_MEMORY_TOOL);
        }

        // Seed the executor with per-turn handles. Later blocks only enable
        // feature bits or attach configured bindings; they do not change turn
        // identity.
        let mut tool_executor = RuntimeToolExecutor::new(
            RuntimeToolDeps {
                platforms: self.platforms.clone(),
                storage: self.storage.clone(),
                media_store: self.media_store.clone(),
                images: self.images.clone(),
                videos: self.videos.clone(),
                audio: self.audio.clone(),
                video_rate_limit_locks: self.video_rate_limit_locks.clone(),
            },
            RuntimeToolContext::new(
                context.reply_to.clone(),
                context.conversation_id,
                context.turn_id,
                context.turn_user.clone(),
            ),
        );
        // Conversation helpers are available to both top-level agents and
        // subagents. They operate on the same turn context and do not create
        // final reply artifacts.
        if policy.fetch_messages() {
            tracing::debug!(tool = FETCH_MESSAGES_TOOL, "attaching runtime tool");
            tool_executor.enable_tools(RuntimeToolFlags::FETCH_MESSAGES);
        }
        if policy.memory_lookup() {
            let base_key = memory::key_from_user_ref(context.turn_user);
            let memory_context = MemoryToolContext::new(
                base_key,
                context.turn_user_display_name.to_string(),
                context.conversation_id,
                context.turn_id,
            );
            if policy.memory_writes() {
                tracing::debug!("attaching user memory tools");
                tool_executor.enable_memory(memory_context);
            } else {
                tracing::debug!("attaching read-only user memory tool");
                tool_executor.enable_memory_lookup(memory_context);
            }
        }
        tracing::debug!(tool = POST_STATUS_TOOL, "attaching runtime tool");
        tracing::debug!(tool = ADD_REACTION_TOOL, "attaching runtime tool");
        tracing::debug!(tool = USAGE_REPORT_TOOL, "attaching runtime tool");
        tool_executor.enable_tools(RuntimeToolFlags::CONVERSATION_HELPERS);

        // Generated media is only delivered from the top-level trace. Subagents
        // return text to their parent, so exposing generation there would make
        // the tool's final-reply delivery contract false.
        if policy.generated_media_tools() {
            if let Some(binding) = &agent_config.image_generation {
                tracing::debug!(
                    tool = GENERATE_IMAGE_TOOL,
                    provider = %binding.provider,
                    model = %binding.model,
                    "attaching image generation tool"
                );
                tool_executor.enable_image_generation(binding.clone());
            }

            if let Some(binding) = &agent_config.video_generation {
                tracing::debug!(
                    tool = GENERATE_VIDEO_TOOL,
                    provider = %binding.provider,
                    model = %binding.model,
                    "attaching video generation tool"
                );
                tool_executor.enable_video_generation(binding.clone());
            }
        }

        if let Some(binding) = &agent_config.audio_transcription {
            tracing::debug!(
                tool = TRANSCRIBE_AUDIO_TOOL,
                provider = %binding.provider,
                model = ?binding.model.as_ref(),
                "attaching audio transcription tool"
            );
            tool_executor.enable_audio_transcription(binding.clone());
        }

        // Stored-media inspection is always wired in for conversation agents.
        // `attach` is top-level only because it queues final reply artifacts.
        tracing::debug!(tool = READ_ASSET_TOOL, "attaching media access tool");
        tracing::debug!(tool = STAT_ASSET_TOOL, "attaching media access tool");
        tracing::debug!(tool = PUBLIC_URL_ASSET_TOOL, "attaching media access tool");
        let mut media_tools = RuntimeToolFlags::MEDIA_INSPECT;
        if policy.final_reply_attach() {
            tracing::debug!(tool = ATTACH_ASSET_TOOL, "attaching media access tool");
            media_tools |= RuntimeToolFlags::MEDIA_ATTACH;
        }
        tool_executor.enable_tools(media_tools);

        for (tool_name, binding) in &agent_config.subagents {
            let (subagent_name, subagent_config) = self
                .config
                .agent_or_platform_default(Some(&binding.agent), &context.reply_to.platform)?;
            tracing::debug!(
                tool = %tool_name,
                subagent = %subagent_name,
                provider = %subagent_config.provider,
                model = %subagent_config.model.id,
                "attaching subagent tool"
            );
            let prompt = self.compose_subagent_system_prompt(subagent_config);
            let nested = self.build_conversation_agent(
                ConversationAgentAssembly {
                    agent_name: &subagent_name,
                    agent_config: subagent_config,
                    rendered_system_instructions: prompt,
                    top_level: false,
                },
                context,
                stack,
            )?;
            tool_executor.add_subagent(tool_name.clone(), binding.description.clone(), nested);
        }

        // Successful expansion removes this node from the active path before
        // building siblings or returning to the caller.
        stack.pop();
        let model = Model {
            backend: RoutedLlmBackend::new(self.llms.clone(), agent_config.provider.clone()),
            spec: agent_config.model.clone(),
        };
        let tool_count = tool_executor.tools().len();
        tracing::debug!(client_tools = tool_count, "built agent with client tools");
        Ok(Agent::new(model, spec, tool_executor))
    }

    /// Compose the system prompt for the top-level agent answering a turn.
    ///
    /// Top-level prompts may include user-memory guidance and a concrete trace
    /// URL for the current conversation.
    pub(crate) fn compose_system_prompt(
        &self,
        agent: &AgentConfig,
        conversation_id: Option<ConversationId>,
    ) -> String {
        self.compose_system_prompt_inner(
            agent,
            ConversationToolPolicy::new(true, self.agent_memory_enabled(agent)),
            conversation_id,
        )
    }

    /// Compose the narrower system prompt used by subagent tools.
    ///
    /// Subagents receive read-only memory lookup guidance, but not final media
    /// delivery guidance, memory-write instructions, or a conversation trace
    /// URL.
    pub(crate) fn compose_subagent_system_prompt(&self, agent: &AgentConfig) -> String {
        self.compose_system_prompt_inner(
            agent,
            ConversationToolPolicy::new(false, self.agent_memory_enabled(agent)),
            None,
        )
    }

    /// Shared system-prompt builder for top-level agents and subagents.
    ///
    /// The tool policy and `conversation_id` are explicit knobs because prompt
    /// text is advisory; the runtime tool executor remains the enforcement
    /// boundary for which calls are actually accepted.
    fn compose_system_prompt_inner(
        &self,
        agent: &AgentConfig,
        policy: ConversationToolPolicy,
        conversation_id: Option<ConversationId>,
    ) -> String {
        let mut out = String::new();
        // Deployment-wide policy leads so it frames all later operational and
        // persona instructions.
        if let Some(extra) = self
            .config
            .extra_system_prompt
            .as_deref()
            .map(str::trim)
            .filter(|extra| !extra.is_empty())
        {
            out.push_str("Operator policy:\n");
            out.push_str(extra);
            out.push_str("\n\n");
        }

        // Runtime identity helps trace readers and model operators understand
        // which configured provider/model produced the turn.
        out.push_str("Operational context:\n");
        out.push_str(&format!(
            "Bot build: {}. You are answering as model `{}` via `{}`.\n",
            self.config.version, agent.model.id, agent.provider
        ));
        if let Some(conversation_id) = conversation_id {
            out.push_str(&trace_link_prompt_guidance(
                &self.config.web_base_url,
                conversation_id,
            ));
        }

        // Capability guidance is assembled from config. The executor built
        // above still decides whether a tool call is permitted.
        out.push_str("Capabilities this turn:\n");
        if !agent.model.server_tools.is_empty() {
            out.push_str("- Provider-side tools configured on this model.\n");
        }
        if policy.fetch_messages() {
            out.push_str("- Recent platform messages are available through fetch_messages.\n");
        }
        if policy.generated_media_tools() {
            if let Some(binding) = &agent.image_generation {
                out.push_str(&format!(
                    concat!(
                        "- Image generation and image editing are available through generate_image ",
                        "using provider `{}` and model `{}`. When the user asks to edit, restyle, ",
                        "transform, or make a variation of an existing image, pass the exact ",
                        "available URI in reference_images.\n"
                    ),
                    binding.provider, binding.model
                ));
            }
            if let Some(binding) = &agent.video_generation {
                out.push_str(&format!(
                    "- Video generation is available through generate_video using provider `{}` and model `{}`.\n",
                    binding.provider, binding.model
                ));
                if let Some(limit) = &binding.rate_limit {
                    out.push_str(&format!(
                        "- Each non-bypassed platform scope is limited to {} active video generation{} per {}.\n",
                        limit.limit,
                        if limit.limit == 1 { "" } else { "s" },
                        limit.interval
                    ));
                }
            }
        }
        if let Some(binding) = &agent.audio_transcription {
            out.push_str(&format!(
                "- Audio transcription is available through transcribe_audio using provider `{}`{}.\n",
                binding.provider,
                binding
                    .model
                    .as_ref()
                    .map(|model| format!(" and model `{model}`"))
                    .unwrap_or_default()
            ));
            out.push_str("- Platform message JSON may include `audio_attachments` or attachment `audio_uri` fields. Use transcribe_audio with those media://audio/... URIs when the user's audio is relevant.\n");
        }
        if !agent.subagents.is_empty() {
            out.push_str("- Specialist subagents are available as tools.\n");
        }
        if policy.memory_writes() {
            out.push_str("- User memory is available through lookup_user_memory, remember_user_memory, and forget_user_memory.\n");
        } else if policy.memory_lookup() {
            out.push_str("- User memory lookup is available through lookup_user_memory. Subagents can read memory but cannot remember or forget facts.\n");
        }
        if policy.final_reply_attach() {
            out.push_str("- Stored media assets can be checked with stat, resolved to a configured public URL with public_url, visually inspected with read, and explicitly attached to the final platform reply with attach. read and attach only accept verified stored image assets, never return file bytes, and reject videos, audio, PDFs, unknown MIME types, public URLs, and local filesystem paths. attach deduplicates with generated media already queued for the final reply.\n");
            out.push_str("- Generated image and video media are attached to the final platform reply automatically; do not paste media URLs, media:// URIs, filenames, or markdown media links in user-facing text.\n");
            out.push_str("- Slow work (video generation, subagent calls, research) SHOULD be narrated with calls to the post_status_message tool.\n");
        } else {
            out.push_str("- Stored media assets can be checked with stat, resolved to a configured public URL with public_url, and visually inspected with read. read only accepts verified stored image assets, never returns file bytes, and rejects videos, audio, PDFs, unknown MIME types, public URLs, and local filesystem paths.\n");
            out.push_str("- Slow work (subagent calls or research) SHOULD be narrated with calls to the post_status_message tool.\n");
        }
        out.push_str("- A subtle Unicode emoji reaction can be added to the user's current message with add_reaction when a compact nonverbal acknowledgement, mood, or topic cue is helpful; use it sparingly and never instead of answering.\n");
        if policy.memory_writes() {
            out.push_str(memory::PROMPT_GUIDANCE);
        }
        out.push_str("Agent Persona Prompt:\n");
        out.push_str(agent.system_prompt.trim());
        out
    }

    /// Return whether user-memory behavior should be exposed for this agent.
    ///
    /// Memory is gated by both deployment config and the individual agent flag.
    pub(crate) fn agent_memory_enabled(&self, agent: &AgentConfig) -> bool {
        self.memory_config.enabled && agent.memory
    }

    /// Apply final reply formatting using this runtime's configured web base URL.
    pub(crate) fn format_reply(
        &self,
        text: &str,
        is_new: bool,
        conversation_id: ConversationId,
    ) -> String {
        format_reply_content(text, is_new, conversation_id, &self.config.web_base_url)
    }

    /// Verify that an agent's LLM provider exists in the runtime registry.
    pub(crate) fn ensure_provider_exists(
        &self,
        agent_name: &str,
        agent: &AgentConfig,
    ) -> Result<(), BotError> {
        if self.llms.contains_provider(&agent.provider) {
            tracing::trace!(
                agent = %agent_name,
                provider = %agent.provider,
                "provider is available"
            );
            return Ok(());
        }
        tracing::warn!(
            agent = %agent_name,
            provider = %agent.provider,
            "agent provider is not configured"
        );
        Err(BotError::MissingProvider {
            agent: agent_name.to_string(),
            provider: agent.provider.clone(),
        })
    }

    /// Verify every provider-backed service referenced by an agent.
    ///
    /// Construction checks all configured media helpers up front so the model is
    /// never built with prompt text or tool bindings for missing services.
    pub(crate) fn ensure_agent_services_exist(
        &self,
        agent_name: &str,
        agent: &AgentConfig,
    ) -> Result<(), BotError> {
        // Every agent needs an LLM provider; media services are conditional on
        // that agent exposing the corresponding generation/transcription tool.
        self.ensure_provider_exists(agent_name, agent)?;
        if let Some(binding) = &agent.image_generation
            && !self.images.contains_generator(&binding.provider)
        {
            tracing::warn!(
                agent = %agent_name,
                provider = %binding.provider,
                model = %binding.model,
                "agent image generation provider is not configured"
            );
            return Err(BotError::MissingImageGenerator {
                agent: agent_name.to_string(),
                provider: binding.provider.clone(),
            });
        }
        if let Some(binding) = &agent.video_generation
            && !self.videos.contains_generator(&binding.provider)
        {
            tracing::warn!(
                agent = %agent_name,
                provider = %binding.provider,
                model = %binding.model,
                "agent video generation provider is not configured"
            );
            return Err(BotError::MissingVideoGenerator {
                agent: agent_name.to_string(),
                provider: binding.provider.clone(),
            });
        }
        if let Some(binding) = &agent.audio_transcription
            && !self.audio.contains_transcriber(&binding.provider)
        {
            tracing::warn!(
                agent = %agent_name,
                provider = %binding.provider,
                model = ?binding.model.as_ref(),
                "agent audio transcription provider is not configured"
            );
            return Err(BotError::MissingAudioTranscriber {
                agent: agent_name.to_string(),
                provider: binding.provider.clone(),
            });
        }
        Ok(())
    }

    /// Verify services for the role-specific conversation tool surface.
    fn ensure_conversation_agent_services_exist(
        &self,
        agent_name: &str,
        agent: &AgentConfig,
        policy: ConversationToolPolicy,
    ) -> Result<(), BotError> {
        self.ensure_provider_exists(agent_name, agent)?;
        if policy.generated_media_tools() {
            if let Some(binding) = &agent.image_generation
                && !self.images.contains_generator(&binding.provider)
            {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %binding.provider,
                    model = %binding.model,
                    "agent image generation provider is not configured"
                );
                return Err(BotError::MissingImageGenerator {
                    agent: agent_name.to_string(),
                    provider: binding.provider.clone(),
                });
            }
            if let Some(binding) = &agent.video_generation
                && !self.videos.contains_generator(&binding.provider)
            {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %binding.provider,
                    model = %binding.model,
                    "agent video generation provider is not configured"
                );
                return Err(BotError::MissingVideoGenerator {
                    agent: agent_name.to_string(),
                    provider: binding.provider.clone(),
                });
            }
        }
        if let Some(binding) = &agent.audio_transcription
            && !self.audio.contains_transcriber(&binding.provider)
        {
            tracing::warn!(
                agent = %agent_name,
                provider = %binding.provider,
                model = ?binding.model.as_ref(),
                "agent audio transcription provider is not configured"
            );
            return Err(BotError::MissingAudioTranscriber {
                agent: agent_name.to_string(),
                provider: binding.provider.clone(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_tool_policy_keeps_delivery_and_memory_writes_top_level() {
        let top_level = ConversationToolPolicy::new(true, true);
        assert!(top_level.fetch_messages());
        assert!(top_level.memory_lookup());
        assert!(top_level.memory_writes());
        assert!(top_level.generated_media_tools());
        assert!(top_level.final_reply_attach());

        let subagent = ConversationToolPolicy::new(false, true);
        assert!(subagent.fetch_messages());
        assert!(subagent.memory_lookup());
        assert!(!subagent.memory_writes());
        assert!(!subagent.generated_media_tools());
        assert!(!subagent.final_reply_attach());
    }
}
