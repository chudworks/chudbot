//! Top-level and nested conversation agent construction.

use crate::prelude::*;
use crate::*;

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    pub(crate) fn build_agent(
        &self,
        agent_name: &str,
        agent_config: &AgentConfig,
        system_prompt: String,
        settings: &RuntimeSettings,
        reply_to: &MessageRef,
        turn_user: &UserRef,
        turn_user_display_name: &str,
        conversation_id: ConversationId,
        turn_id: TurnId,
        top_level: bool,
        stack: &mut Vec<String>,
    ) -> Result<RuntimeAgent<R>, BotError> {
        self.ensure_agent_services_exist(agent_name, agent_config)?;
        if stack.iter().any(|name| name == agent_name) {
            tracing::warn!("recursive agent reference detected");
            return Err(BotError::RecursiveAgent {
                name: agent_name.to_string(),
            });
        }
        stack.push(agent_name.to_string());

        // Start from the static agent config, then layer in runtime-only tool exposure.
        let mut spec = agent_config.agent_spec(self.config.limits);
        spec.system_prompt = system_prompt;
        if top_level && self.agent_memory_enabled(agent_config) {
            ensure_client_tool_enabled(&mut spec.client_tools, memory::LOOKUP_USER_MEMORY_TOOL);
            ensure_client_tool_enabled(&mut spec.client_tools, memory::REMEMBER_USER_MEMORY_TOOL);
            ensure_client_tool_enabled(&mut spec.client_tools, memory::FORGET_USER_MEMORY_TOOL);
        }

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
            RuntimeToolContext {
                default_channel: channel_from_message(reply_to),
                reply_to: reply_to.clone(),
                conversation_id,
                turn_id,
                turn_user: turn_user.clone(),
                privacy: settings.privacy.clone(),
            },
        );
        // Top-level agents get conversation-management tools. Subagents keep a narrower
        // surface and only receive explicitly configured generation/media/subagent tools.
        if top_level {
            if !matches!(settings.privacy, PrivacyMode::ConversationOnly) {
                tracing::debug!(tool = FETCH_MESSAGES_TOOL, "attaching runtime tool");
                tool_executor.enabled.fetch_messages = true;
            }
            if self.agent_memory_enabled(agent_config) {
                let base_key = memory::key_from_user_ref(turn_user);
                tracing::debug!("attaching user memory tools");
                tool_executor.enable_memory(MemoryToolContext::new(
                    base_key,
                    turn_user_display_name.to_string(),
                    conversation_id,
                    turn_id,
                ));
            }
            tracing::debug!(tool = POST_STATUS_TOOL, "attaching runtime tool");
            tool_executor.enabled.post_status = true;
            tracing::debug!(tool = ADD_REACTION_TOOL, "attaching runtime tool");
            tool_executor.enabled.add_reaction = true;
            tracing::debug!(tool = USAGE_REPORT_TOOL, "attaching runtime tool");
            tool_executor.enabled.usage_report = true;
        }

        if let Some(binding) = &agent_config.image_generation {
            tracing::debug!(
                tool = GENERATE_IMAGE_TOOL,
                provider = %binding.provider,
                model = %binding.model,
                "attaching image generation tool"
            );
            tool_executor.image_generation = Some(binding.clone());
        }

        if let Some(binding) = &agent_config.video_generation {
            tracing::debug!(
                tool = GENERATE_VIDEO_TOOL,
                provider = %binding.provider,
                model = %binding.model,
                "attaching video generation tool"
            );
            tool_executor.video_generation = Some(binding.clone());
        }

        if let Some(binding) = &agent_config.audio_transcription {
            tracing::debug!(
                tool = TRANSCRIBE_AUDIO_TOOL,
                provider = %binding.provider,
                model = ?binding.model.as_ref(),
                "attaching audio transcription tool"
            );
            tool_executor.audio_transcription = Some(binding.clone());
        }

        tracing::debug!(tool = READ_ASSET_TOOL, "attaching media access tool");
        tool_executor.enabled.media_access.read = true;
        tracing::debug!(tool = STAT_ASSET_TOOL, "attaching media access tool");
        tool_executor.enabled.media_access.stat = true;
        tracing::debug!(tool = PUBLIC_URL_ASSET_TOOL, "attaching media access tool");
        tool_executor.enabled.media_access.public_url = true;
        tracing::debug!(tool = ATTACH_ASSET_TOOL, "attaching media access tool");
        tool_executor.enabled.media_access.attach = true;

        for (tool_name, binding) in &agent_config.subagents {
            // Subagents are recursively built with their own executor. Boxing only hides
            // the nested agent/tool-executor type; tool dispatch remains static inside it.
            let (subagent_name, subagent_config) = self
                .config
                .agent_or_platform_default(Some(&binding.agent), &reply_to.platform)?;
            tracing::debug!(
                tool = %tool_name,
                subagent = %subagent_name,
                provider = %subagent_config.provider,
                model = %subagent_config.model.id,
                "attaching subagent tool"
            );
            let prompt = self
                .compose_subagent_system_prompt(subagent_config, &PrivacyMode::ConversationOnly);
            let nested = self.build_agent(
                &subagent_name,
                subagent_config,
                prompt,
                settings,
                reply_to,
                turn_user,
                turn_user_display_name,
                conversation_id,
                turn_id,
                false,
                stack,
            )?;
            tool_executor.subagents.push(RuntimeSubagent {
                name: tool_name.clone(),
                tool: Box::new(nested.into_subagent(binding.description.clone())),
            });
        }

        stack.pop();
        let model = Model {
            backend: RoutedLlmBackend::new(self.llms.clone(), agent_config.provider.clone()),
            spec: agent_config.model.clone(),
        };
        let tool_count = tool_executor.tools().len();
        tracing::debug!(client_tools = tool_count, "built agent with client tools");
        Ok(Agent::new(model, spec, tool_executor))
    }

    pub(crate) fn compose_system_prompt(
        &self,
        agent: &AgentConfig,
        privacy: &PrivacyMode,
        conversation_id: Option<ConversationId>,
    ) -> String {
        self.compose_system_prompt_inner(
            agent,
            privacy,
            self.agent_memory_enabled(agent),
            conversation_id,
        )
    }

    pub(crate) fn compose_subagent_system_prompt(
        &self,
        agent: &AgentConfig,
        privacy: &PrivacyMode,
    ) -> String {
        self.compose_system_prompt_inner(agent, privacy, false, None)
    }

    pub(crate) fn compose_system_prompt_inner(
        &self,
        agent: &AgentConfig,
        privacy: &PrivacyMode,
        include_memory: bool,
        conversation_id: Option<ConversationId>,
    ) -> String {
        let mut out = String::new();
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
        out.push_str("Capabilities this turn:\n");
        if !agent.model.server_tools.is_empty() {
            out.push_str("- Provider-side tools configured on this model.\n");
        }
        if !matches!(privacy, PrivacyMode::ConversationOnly) {
            out.push_str("- Recent platform messages are available through fetch_messages.\n");
        }
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
            out.push_str("- Platform message JSON may include `audio_attachments` or attachment `audio_uri` fields. Use transcribe_audio with those file://audio/... URIs when the user's audio is relevant.\n");
        }
        if !agent.subagents.is_empty() {
            out.push_str("- Specialist subagents are available as tools.\n");
        }
        if include_memory {
            out.push_str("- User memory is available through lookup_user_memory, remember_user_memory, and forget_user_memory.\n");
        }
        out.push_str("- Stored media assets can be checked with stat, resolved to a configured public URL with public_url, visually inspected with read, and explicitly attached to the final platform reply with attach. read and attach only accept verified stored image assets, never return file bytes, and reject videos, audio, PDFs, unknown MIME types, public URLs, and local filesystem paths. attach deduplicates with generated media already queued for the final reply.\n");
        out.push_str("- Generated image and video media are attached to the final platform reply automatically; do not paste media URLs, file:// URIs, filenames, or markdown media links in user-facing text.\n");
        out.push_str("- Slow work (video generation, subagent calls, research) SHOULD be narrated with calls to the post_status_message tool.\n");
        out.push_str("- A subtle Unicode emoji reaction can be added to the user's current message with add_reaction when a compact nonverbal acknowledgement, mood, or topic cue is helpful; use it sparingly and never instead of answering.\n");
        if include_memory {
            out.push_str(memory::PROMPT_GUIDANCE);
        }
        out.push_str("Agent Persona Prompt:\n");
        out.push_str(agent.system_prompt.trim());
        out
    }

    pub(crate) fn agent_memory_enabled(&self, agent: &AgentConfig) -> bool {
        self.memory_config.enabled && agent.memory
    }

    pub(crate) fn format_reply(
        &self,
        text: &str,
        is_new: bool,
        conversation_id: ConversationId,
    ) -> String {
        format_reply_content(text, is_new, conversation_id, &self.config.web_base_url)
    }

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

    pub(crate) fn ensure_agent_services_exist(
        &self,
        agent_name: &str,
        agent: &AgentConfig,
    ) -> Result<(), BotError> {
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
}
