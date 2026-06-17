//! Agent construction, reserved system-agent resolution, and prompt assembly.

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

/// Resolver for the safety preflight agent.
#[derive(Debug, Clone)]
pub(crate) struct TosPreflightSystemAgents {
    pub(crate) configured: Option<SystemAgentConfig>,
    pub(crate) platform_defaults: BTreeMap<PlatformName, SystemAgentConfig>,
    pub(crate) default: Option<SystemAgentConfig>,
}

impl TosPreflightSystemAgents {
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

/// Resolver for conversation title generation agents.
#[derive(Debug, Clone)]
pub(crate) struct ConversationTitleSystemAgents {
    pub(crate) configured: Option<SystemAgentConfig>,
    pub(crate) agent_defaults: BTreeMap<String, SystemAgentConfig>,
    pub(crate) platform_defaults: BTreeMap<PlatformName, SystemAgentConfig>,
    pub(crate) default: Option<SystemAgentConfig>,
}

impl ConversationTitleSystemAgents {
    pub(crate) fn from_config(config: &BotConfig) -> Self {
        if let Some(configured) = configured_system_agent(config, CONVERSATION_TITLE_AGENT) {
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

pub(crate) fn configured_system_agent(config: &BotConfig, name: &str) -> Option<SystemAgentConfig> {
    config.agents.get(name).map(|agent| {
        let resolved = SystemAgentConfig::from_agent_config(name.to_string(), agent, config.limits);
        resolved.log_loaded_from_config();
        resolved
    })
}

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
                temperature: Some(0.0),
                top_p: None,
            },
            provider_options: None,
        },
        source.limits.unwrap_or(default_limits),
    )
}

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
                tool_executor.enable_memory(memory::MemoryToolContext::new(
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

    pub(crate) fn tos_preflight_agent(
        &self,
        platform: &PlatformName,
    ) -> Result<&SystemAgentConfig, BotError> {
        self.system_agents
            .tos_preflight
            .get(platform, &self.config.default_agent)
    }

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
        transcript.push(TranscriptTurn::text(
            TurnRole::User,
            format!(
                "Message to classify:\n<<<\n[{display_name}]: {}\n>>>",
                message.content
            ),
        ));
        let agent = self.system_agent(&agent_config);
        let run = match agent.run(transcript).await {
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

    pub(crate) fn spawn_memory_runtime(&self, shutdown: CancellationToken) {
        if !self.memory_config.enabled {
            return;
        }
        let memory_agents = self
            .memory_config
            .resolve_agent_set(&self.config.agents, self.config.limits);
        let runtime = memory::MemoryRuntime::new(
            self.storage.clone(),
            self.llms.clone(),
            self.media_store.clone(),
            self.memory_config.clone(),
            memory_agents,
        );
        spawn_background_task(&self.background, "memory runtime", async move {
            if let Err(error) = runtime.run_until_shutdown(shutdown).await {
                tracing::warn!(error = %error, "memory runtime stopped with error");
            }
        });
    }

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
        let user_text = format!(
            "User said:\n{}\n\nAssistant replied:\n{}",
            first.turn.user_content,
            first.turn.assistant_content.as_deref().unwrap_or("")
        );
        let mut transcript = Transcript::new();
        transcript.push(TranscriptTurn::text(TurnRole::User, user_text));
        let agent_runtime = self.system_agent(&agent);
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

    pub(crate) fn spawn_avatar_download(&self, user: UserProfile) {
        let Some(url) = user
            .avatar_url
            .as_deref()
            .filter(|url| !url.trim().is_empty())
            .map(str::to_string)
        else {
            return;
        };
        let runtime = (*self).clone();
        spawn_background_task(&self.background, "avatar download", async move {
            if let Err(error) = runtime.download_avatar(user, url).await {
                tracing::warn!(error = %error, "avatar download failed");
            }
        });
    }

    pub(crate) async fn download_avatar(
        &self,
        user: UserProfile,
        url: String,
    ) -> Result<(), BotError> {
        let name = avatar_media_name(&user, &url);
        let expected_uri = MediaUri::new(format!("file://avatars/{name}"));
        if self
            .storage
            .load_user_avatar(user.id.clone())
            .await
            .map_err(storage_error)?
            .as_ref()
            .is_some_and(|uri| uri == &expected_uri)
        {
            tracing::trace!(uri = %expected_uri, "avatar already cached");
            return Ok(());
        }

        let response = reqwest::Client::new()
            .get(&url)
            .send()
            .await
            .map_err(|error| BotError::AvatarDownload(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(BotError::AvatarDownload(format!("http {status}")));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| BotError::AvatarDownload(error.to_string()))?
            .to_vec();
        let media = self
            .media_store
            .create_media(CreateMedia {
                category: MediaCategory::Avatar,
                bytes,
                mime_type: Some("image/png".to_string()),
                name: Some(name),
                extension: Some("png".to_string()),
            })
            .await
            .map_err(|error| BotError::AvatarDownload(error.to_string()))?;
        self.storage
            .set_user_avatar(user.id.clone(), media.uri().clone())
            .await
            .map_err(storage_error)?;
        self.publish_user(user.id);
        tracing::info!(uri = %media.uri(), "avatar cached");
        Ok(())
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
            out.push_str(memory::prompt_guidance());
        }
        out.push_str("Agent Persona Prompt:\n");
        out.push_str(agent.system_prompt.trim());
        out
    }

    pub(crate) fn agent_memory_enabled(&self, agent: &AgentConfig) -> bool {
        self.memory_config.enabled && agent.memory
    }

    pub(crate) fn publish_conversation(
        &self,
        conversation_id: ConversationId,
        kind: ConversationEventKind,
    ) {
        tracing::trace!(
            conversation = %conversation_id,
            event = conversation_event_kind(kind),
            "publishing conversation event"
        );
        self.events.publish(LiveEvent::Conversation {
            conversation_id,
            kind,
        });
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

/// Normalize the title model output into a short display title.
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

pub(crate) fn error_indicates_safety_refusal(error: &str) -> bool {
    let lower = error.to_ascii_lowercase();
    lower.contains("safety_check") || lower.contains("violates usage guidelines")
}

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
                value
                    .get("error")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_else(|| value.as_str().unwrap_or("")),
            ),
        }
    })
}
