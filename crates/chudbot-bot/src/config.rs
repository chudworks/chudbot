//! Bot-facing configuration types and runtime conversion helpers.
//!
//! `chudbot-bin` owns TOML loading and rich diagnostics; this module owns the
//! provider-neutral bot view after deserialization. It validates references
//! between named agents, turns agent config into `AgentSpec` values, and keeps
//! tool-facing descriptions close to the config that enables those tools.

use std::collections::BTreeMap;
use std::time::Duration;

use chudbot_api::{
    AgentLimits, AgentSpec, ExternalId, ModelId, ModelSpec, PlatformName, ProviderName, ToolName,
    UserRef,
};
use serde::{Deserialize, Serialize};

use crate::memory;
use crate::{
    BotError, DEFAULT_SHUTDOWN_DRAIN_TIMEOUT, DEFAULT_THREAD_THRESHOLD_CHARS,
    DEFAULT_THREAD_THRESHOLD_LINES,
};

/// Bot-level configuration consumed by the platform-neutral runtime.
///
/// This is the agent-first portion of the deployment config: named agents,
/// platform-to-agent defaults, global operator policy, and runtime limits. It
/// deliberately stores provider and platform references by registry key so this
/// crate does not depend on concrete provider or Discord implementations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotConfig {
    /// Public viewer base URL used in the first reply for a conversation.
    pub web_base_url: String,
    /// Default top-level agent when a platform has no explicit binding.
    pub default_agent: String,
    /// Named agents. An agent may be top-level, subagent-only, or both.
    pub agents: BTreeMap<String, AgentConfig>,
    /// Operator users allowed to stop/resume conversations with the stop
    /// reaction. A missing `guild_id` applies across the platform.
    #[serde(default)]
    pub admins: Vec<chudbot_api::UserRef>,
    /// Platform default bindings, e.g. `discord -> chudbot`.
    #[serde(default)]
    pub platforms: BTreeMap<PlatformName, PlatformBinding>,
    /// Optional operator-wide policy text.
    #[serde(default)]
    pub extra_system_prompt: Option<String>,
    /// Build/version label included in the operational system prompt.
    #[serde(default)]
    pub version: String,
    /// Default model/tool loop limits for agents that do not override them.
    #[serde(default)]
    pub limits: AgentLimits,
    /// Reply length above which a new conversation asks the platform to open a
    /// thread when supported.
    #[serde(default = "default_thread_threshold_chars")]
    pub thread_threshold_chars: usize,
    /// Approximate visible reply rows above which a new conversation asks the
    /// platform to open a thread when supported.
    #[serde(default = "default_thread_threshold_lines")]
    pub thread_threshold_lines: usize,
}

impl BotConfig {
    /// Validate references and local invariants that can be checked before the
    /// runtime builds providers, tools, or platform adapters.
    #[tracing::instrument(
        name = "bot.config.validate",
        skip_all,
        fields(
            agents = self.agents.len(),
            admins = self.admins.len(),
            platforms = self.platforms.len(),
            default_agent = %self.default_agent,
        )
    )]
    pub fn validate(&self) -> Result<(), BotError> {
        tracing::debug!("validating bot config");

        // Establish the global fallback first; several later paths assume it
        // exists when a platform or request does not name a valid agent.
        if !self.agents.contains_key(&self.default_agent) {
            tracing::warn!(
                missing_agent = %self.default_agent,
                "default agent is not configured"
            );
            return Err(BotError::MissingAgent {
                name: self.default_agent.clone(),
            });
        }

        // Platform defaults are optional, but every configured override must
        // still resolve to one of the named agents in this map.
        for binding in self.platforms.values() {
            if !self.agents.contains_key(&binding.agent) {
                tracing::warn!(
                    missing_agent = %binding.agent,
                    "platform binding references missing agent"
                );
                return Err(BotError::MissingAgent {
                    name: binding.agent.clone(),
                });
            }
        }

        // Agent-local bindings are turned into runtime tools. Validate them
        // here so missing names or malformed media/audio bindings fail before
        // the bot starts accepting events.
        for (agent_name, agent) in &self.agents {
            if let Some(binding) = &agent.image_generation {
                validate_generation_binding(agent_name, "image_generation", binding)?;
            }
            if let Some(binding) = &agent.video_generation {
                validate_generation_binding(agent_name, "video_generation", binding)?;
            }
            if let Some(binding) = &agent.audio_transcription {
                validate_transcription_binding(agent_name, "audio_transcription", binding)?;
            }
            for binding in agent.subagents.values() {
                if !self.agents.contains_key(&binding.agent) {
                    tracing::warn!(
                        agent = %agent_name,
                        missing_subagent = %binding.agent,
                        "subagent binding references missing agent"
                    );
                    return Err(BotError::MissingSubagent {
                        agent: agent_name.clone(),
                        subagent: binding.agent.clone(),
                    });
                }
            }
        }
        tracing::info!("bot config validated");
        Ok(())
    }

    /// Resolve the agent for an incoming turn.
    ///
    /// A valid explicit request wins. Unknown requests are treated like "no
    /// request" so platform/default routing remains usable when a user-provided
    /// agent name is stale or unavailable.
    pub fn agent_or_platform_default(
        &self,
        requested: Option<&str>,
        platform: &PlatformName,
    ) -> Result<(String, &AgentConfig), BotError> {
        // Prefer the user or platform command selection only when it names a
        // configured agent.
        if let Some(name) = requested
            && let Some(agent) = self.agents.get(name)
        {
            tracing::debug!(
                requested_agent = %name,
                platform = %platform,
                provider = %agent.provider,
                model = %agent.model.id,
                "resolved requested agent"
            );
            return Ok((name.to_string(), agent));
        }

        // Then fall back through the platform binding and finally the global
        // default. `validate` normally guarantees this lookup succeeds.
        let platform_default = self
            .platforms
            .get(platform)
            .map(|binding| binding.agent.as_str())
            .unwrap_or(self.default_agent.as_str());
        let resolved = self
            .agents
            .get(platform_default)
            .map(|agent| (platform_default.to_string(), agent))
            .ok_or_else(|| BotError::MissingAgent {
                name: platform_default.to_string(),
            })?;
        tracing::debug!(
            requested_agent = ?requested,
            platform = %platform,
            resolved_agent = %resolved.0,
            provider = %resolved.1.provider,
            model = %resolved.1.model.id,
            "resolved platform/default agent"
        );
        Ok(resolved)
    }
}

fn default_thread_threshold_chars() -> usize {
    DEFAULT_THREAD_THRESHOLD_CHARS
}

fn default_thread_threshold_lines() -> usize {
    DEFAULT_THREAD_THRESHOLD_LINES
}

/// Build the model-facing description for an agent's image generation tool.
///
/// The text explains the media reference contract because generated and
/// uploaded images are addressed by internal `file://images/...` URIs, not by
/// public URLs or guessed filenames.
pub(crate) fn image_generation_tool_description(
    provider: &ProviderName,
    model: &ModelId,
) -> String {
    // Keep the reference-image rules in the tool description itself; the model
    // sees this on every tool call decision, while config docs are not in-band.
    format!(
        concat!(
            "Generate an image with the configured `{}` image provider and `{}` model, ",
            "save it to media storage, and return its media URI.\n\n",
            "Use this whenever the user asks for an image, picture, drawing, illustration, ",
            "infographic, or other visual.\n\n",
            "To edit, restyle, transform, make a variation of, or combine images already ",
            "visible in the conversation, pass their exact `file://images/...` URI(s) in ",
            "`reference_images`. This is the expected path for requests like \"turn this ",
            "image into...\", \"make the image...\", \"use the previous image\", or ",
            "\"here's a different version\". User-uploaded images are listed in image ",
            "attachment reference notes; generated images are listed in prior tool ",
            "results and generated-media reference notes. Never invent or guess paths. ",
            "For two or three references, refer to them in the prompt as <IMAGE_0>, ",
            "<IMAGE_1>, etc. in the same order. If no real URI applies, omit ",
            "`reference_images` and generate from text alone.\n\n",
            "Generated media is attached to the final platform reply automatically. ",
            "Do not paste media URIs, filenames, public URLs, or markdown media links ",
            "in user-facing text."
        ),
        provider, model
    )
}

/// Build the model-facing description for an agent's video generation tool.
pub(crate) fn video_generation_tool_description(binding: &GenerationBinding) -> String {
    let mut description = format!(
        "Generate a video with the configured `{}` video provider and `{}` model, save it to media storage, and return its media URI.",
        binding.provider, binding.model
    );
    if let Some(limit) = &binding.rate_limit {
        // The runtime enforces the limit; this hint helps the model avoid
        // surprising users with retries that cannot run yet.
        description.push_str(&format!(
            "\n\nThis tool is limited to {} active video generation{} per {} for each non-bypassed platform scope.",
            limit.limit,
            if limit.limit == 1 { "" } else { "s" },
            limit.interval
        ));
    }
    description
}

/// Validate a media generation binding shared by image and video tools.
///
/// Image and video bindings intentionally share the same config shape, but only
/// video generation accepts the optional active-job rate limit.
pub(crate) fn validate_generation_binding(
    agent_name: &str,
    field: &'static str,
    binding: &GenerationBinding,
) -> Result<(), BotError> {
    // Registry lookups happen later, so catch empty registry keys here while the
    // error can still name the owning agent and config field.
    if binding.provider.as_str().trim().is_empty() {
        tracing::warn!(agent = %agent_name, field, "media generation provider is empty");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "provider is empty".to_string(),
        });
    }
    if binding.model.as_str().trim().is_empty() {
        tracing::warn!(agent = %agent_name, field, "media generation model is empty");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "model is empty".to_string(),
        });
    }
    if let Some(rate_limit) = &binding.rate_limit {
        // Keep the shared config struct narrow at runtime: image bindings can
        // deserialize the field for diagnostics, but validation rejects it.
        if field != "video_generation" {
            tracing::warn!(agent = %agent_name, field, "rate limit configured on non-video generation binding");
            return Err(BotError::InvalidGenerationBinding {
                agent: agent_name.to_string(),
                field,
                message: "rate_limit is only supported on video_generation".to_string(),
            });
        }
        validate_video_generation_rate_limit(agent_name, field, rate_limit)?;
    }
    Ok(())
}

/// Validate the video-specific portion of a generation binding.
fn validate_video_generation_rate_limit(
    agent_name: &str,
    field: &'static str,
    rate_limit: &VideoGenerationRateLimit,
) -> Result<(), BotError> {
    // A zero limit would permanently disable the tool while still advertising
    // it to the model, so fail configuration instead.
    if rate_limit.limit == 0 {
        tracing::warn!(agent = %agent_name, field, "video generation rate limit is zero");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "rate_limit.limit must be greater than zero".to_string(),
        });
    }
    if let Err(message) = rate_limit.interval_seconds() {
        tracing::warn!(
            agent = %agent_name,
            field,
            interval = %rate_limit.interval,
            "video generation rate limit interval is invalid"
        );
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message,
        });
    }

    // Bypass scopes are matched at runtime against platform and guild/scope id;
    // empty components would otherwise silently never match.
    for scope in &rate_limit.bypass_scopes {
        if scope.platform.as_str().trim().is_empty() {
            tracing::warn!(agent = %agent_name, field, "video generation rate limit bypass platform is empty");
            return Err(BotError::InvalidGenerationBinding {
                agent: agent_name.to_string(),
                field,
                message: "rate_limit.bypass_scopes platform must not be empty".to_string(),
            });
        }
        if scope.scope_id.as_str().trim().is_empty() {
            tracing::warn!(
                agent = %agent_name,
                field,
                platform = %scope.platform,
                "video generation rate limit bypass scope id is empty"
            );
            return Err(BotError::InvalidGenerationBinding {
                agent: agent_name.to_string(),
                field,
                message: "rate_limit.bypass_scopes scope_id must not be empty".to_string(),
            });
        }
    }
    Ok(())
}

/// Validate the audio transcription binding that enables the transcription tool.
fn validate_transcription_binding(
    agent_name: &str,
    field: &'static str,
    binding: &TranscriptionBinding,
) -> Result<(), BotError> {
    if binding.provider.as_str().trim().is_empty() {
        tracing::warn!(agent = %agent_name, field, "audio transcription provider is empty");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "provider is empty".to_string(),
        });
    }
    if let Some(model) = &binding.model
        && model.as_str().trim().is_empty()
    {
        tracing::warn!(agent = %agent_name, field, "audio transcription model is empty");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "model is empty".to_string(),
        });
    }
    if let Some(wake_word) = &binding.wake_word
        && wake_word.trim().is_empty()
    {
        tracing::warn!(agent = %agent_name, field, "audio transcription wake word is empty");
        return Err(BotError::InvalidGenerationBinding {
            agent: agent_name.to_string(),
            field,
            message: "wake_word is empty".to_string(),
        });
    }
    Ok(())
}

/// One named agent: prompt, provider/model, tool exposure, and subagents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// LLM provider registry key.
    pub provider: ProviderName,
    /// System prompt / agent instructions.
    pub system_prompt: String,
    /// Model config used for this agent.
    pub model: ModelSpec,
    /// Optional server-tool restriction for this agent. `None` means all
    /// server tools allowed by the model config are exposed.
    #[serde(default)]
    pub server_tools: Option<chudbot_api::ServerToolSet>,
    /// Optional client-tool allowlist. `None` means all runtime tools assembled
    /// for this agent are exposed.
    #[serde(default)]
    pub client_tools: Option<Vec<ToolName>>,
    /// Optional per-agent loop limits.
    #[serde(default)]
    pub limits: Option<AgentLimits>,
    /// Optional image generation binding exposed through `generate_image`.
    #[serde(default)]
    pub image_generation: Option<GenerationBinding>,
    /// Optional video generation binding exposed through `generate_video`.
    #[serde(default)]
    pub video_generation: Option<GenerationBinding>,
    /// Optional audio transcription binding exposed through `transcribe_audio`.
    #[serde(default)]
    pub audio_transcription: Option<TranscriptionBinding>,
    /// Whether top-level runs for this agent receive user-memory tools.
    #[serde(default)]
    pub memory: bool,
    /// Subagents exposed as named client-side tools.
    #[serde(default)]
    pub subagents: BTreeMap<ToolName, SubagentBinding>,
}

impl AgentConfig {
    /// Convert this configured agent into the provider-neutral runtime spec.
    ///
    /// Provider/model selection stays outside `AgentSpec`; the spec contains the
    /// instructions, loop limits, and tool exposure that are common to all LLM
    /// backends.
    pub(crate) fn agent_spec(&self, default_limits: AgentLimits) -> AgentSpec {
        let mut spec = AgentSpec::new(self.system_prompt.clone())
            .with_limits(self.limits.unwrap_or(default_limits));
        spec.server_tools = self.server_tools.clone();
        spec.client_tools = self.client_tools.clone();
        spec
    }
}

/// Effective configuration for a reserved system agent.
///
/// Reserved agents such as title generation and TOS preflight can be configured
/// directly or inherited from normal agents. This snapshot keeps the resolved
/// provider, model, and common `AgentSpec` together so runtime code can build a
/// routed agent without re-reading the full bot config.
#[derive(Debug, Clone)]
pub(crate) struct SystemAgentConfig {
    /// Reserved agent name used in logs and config lookup.
    pub(crate) name: String,
    /// LLM provider registry key.
    pub(crate) provider: ProviderName,
    /// Provider-neutral prompt, limits, and tool exposure.
    pub(crate) spec: AgentSpec,
    /// Model config paired with the provider.
    pub(crate) model: ModelSpec,
}

impl SystemAgentConfig {
    /// Build a reserved-agent snapshot from an ordinary configured agent.
    pub(crate) fn from_agent_config(
        name: String,
        agent: &AgentConfig,
        default_limits: AgentLimits,
    ) -> Self {
        Self {
            name,
            provider: agent.provider.clone(),
            spec: agent.agent_spec(default_limits),
            model: agent.model.clone(),
        }
    }

    /// Build a reserved-agent snapshot from built-in defaults.
    pub(crate) fn from_parts(
        name: impl Into<String>,
        provider: ProviderName,
        system_prompt: impl Into<String>,
        model: ModelSpec,
        limits: AgentLimits,
    ) -> Self {
        Self {
            name: name.into(),
            provider,
            spec: AgentSpec::new(system_prompt).with_limits(limits),
            model,
        }
    }

    /// Log that this reserved agent came from an explicit config entry.
    pub(crate) fn log_loaded_from_config(&self) {
        self.log_effective_config("config", "loaded system agent from config");
    }

    /// Log that this reserved agent used its built-in default settings.
    pub(crate) fn log_using_default(&self) {
        self.log_effective_config("default", "using default system agent");
    }

    /// Log that a reserved agent inherited provider/model settings from a
    /// configured normal agent.
    pub(crate) fn log_using_default_inherited(
        &self,
        inherited_agent: &str,
        inherited_platform: Option<&PlatformName>,
    ) {
        tracing::debug!(
            system_agent = %self.name,
            source = "default",
            inherited_agent,
            inherited_platform = ?inherited_platform,
            provider = %self.provider,
            model = %self.model.id,
            model_server_tools = ?self.model.server_tools,
            agent_server_tools = ?self.spec.server_tools,
            agent_client_tools = ?self.spec.client_tools,
            max_iterations = self.spec.limits.max_iterations,
            max_output_tokens = ?self.model.sampling.max_output_tokens,
            temperature = ?self.model.sampling.temperature,
            top_p = ?self.model.sampling.top_p,
            provider_options = ?self.model.provider_options.as_ref().map(|options| &options.value),
            system_prompt_chars = self.spec.system_prompt.chars().count(),
            system_prompt = %self.spec.system_prompt,
            "using default system agent inherited from agent"
        );
    }

    /// Emit a consistent structured view of the effective reserved-agent config.
    fn log_effective_config(&self, source: &'static str, message: &'static str) {
        tracing::debug!(
            system_agent = %self.name,
            source,
            provider = %self.provider,
            model = %self.model.id,
            model_server_tools = ?self.model.server_tools,
            agent_server_tools = ?self.spec.server_tools,
            agent_client_tools = ?self.spec.client_tools,
            max_iterations = self.spec.limits.max_iterations,
            max_output_tokens = ?self.model.sampling.max_output_tokens,
            temperature = ?self.model.sampling.temperature,
            top_p = ?self.model.sampling.top_p,
            provider_options = ?self.model.provider_options.as_ref().map(|options| &options.value),
            system_prompt_chars = self.spec.system_prompt.chars().count(),
            system_prompt = %self.spec.system_prompt,
            "{message}"
        );
    }
}

/// Binding from an agent to a media-generation provider and default model.
///
/// The same binding type is reused for image and video generation. Validation
/// keeps video-only options, such as rate limits, out of image bindings while
/// preserving one config shape for diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationBinding {
    /// Media-generation provider registry key.
    pub provider: ProviderName,
    /// Provider-specific image/video model id or tier.
    pub model: ModelId,
    /// Optional active-video rate limit for this video-generation binding.
    #[serde(default)]
    pub rate_limit: Option<VideoGenerationRateLimit>,
}

/// Active-video rate limit for a video-generation binding.
///
/// Limits are scoped by platform workspace/server, with explicit bypasses for
/// trusted or operational scopes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerationRateLimit {
    /// Maximum pending plus successful video generations per interval.
    pub limit: u32,
    /// Rolling interval, e.g. `4h`, `30m`, or `1d`.
    #[serde(default = "default_video_generation_rate_limit_interval")]
    pub interval: String,
    /// Platform scopes that are exempt from this limit.
    #[serde(default)]
    pub bypass_scopes: Vec<PlatformScopeBypass>,
}

/// One platform scope exempt from a video-generation rate limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformScopeBypass {
    /// Messaging platform, e.g. `discord`.
    pub platform: PlatformName,
    /// Platform workspace/server/guild scope id.
    pub scope_id: ExternalId,
}

impl VideoGenerationRateLimit {
    /// Parse the configured rolling interval into seconds.
    pub fn interval_seconds(&self) -> Result<u64, String> {
        memory::parse_duration_seconds(&self.interval)
            .map_err(|_| format!("rate_limit.interval `{}` is invalid", self.interval))
    }

    /// Return whether this user's platform scope is exempt from the limit.
    pub(crate) fn bypasses(&self, user: &UserRef) -> bool {
        let Some(scope_id) = &user.guild_id else {
            return false;
        };
        self.bypass_scopes
            .iter()
            .any(|scope| scope.platform == user.platform && scope.scope_id == *scope_id)
    }
}

fn default_video_generation_rate_limit_interval() -> String {
    "4h".to_string()
}

/// Binding from an agent to an audio transcription provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionBinding {
    /// Audio transcription provider registry key.
    pub provider: ProviderName,
    /// Provider-specific transcription model id when applicable.
    #[serde(default)]
    pub model: Option<ModelId>,
    /// Optional spoken wake word for no-mention audio outside an existing
    /// conversation.
    #[serde(default)]
    pub wake_word: Option<String>,
}

impl TranscriptionBinding {
    /// Return the normalized wake word when one is configured.
    pub(crate) fn wake_word(&self) -> Option<&str> {
        self.wake_word
            .as_deref()
            .map(str::trim)
            .filter(|wake_word| !wake_word.is_empty())
    }
}

/// Derive transcription keyterms that should always include the wake word.
pub(crate) fn audio_transcription_default_keyterms(binding: &TranscriptionBinding) -> Vec<String> {
    binding
        .wake_word()
        .map(|wake_word| vec![wake_word.to_string()])
        .unwrap_or_default()
}

/// Append default transcription keyterms without duplicating existing entries.
pub(crate) fn append_default_audio_keyterms(keyterms: &mut Vec<String>, defaults: &[String]) {
    for default in defaults {
        let default = default.trim();
        if default.is_empty() {
            continue;
        }

        // User-supplied provider keyterms keep their original spelling, but
        // duplicate detection ignores case and surrounding whitespace.
        let already_present = keyterms
            .iter()
            .any(|keyterm| keyterm.trim().eq_ignore_ascii_case(default));
        if !already_present {
            keyterms.push(default.to_string());
        }
    }
}

/// Platform default binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformBinding {
    /// Agent name used for this platform by default.
    pub agent: String,
}

/// Runtime controls for the bot event loop.
#[derive(Debug, Clone, Copy)]
pub struct BotRunOptions {
    /// How long graceful shutdown waits for in-flight event tasks.
    pub drain_timeout: Duration,
}

impl Default for BotRunOptions {
    fn default() -> Self {
        Self {
            drain_timeout: DEFAULT_SHUTDOWN_DRAIN_TIMEOUT,
        }
    }
}

/// A tool binding from one agent to another.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubagentBinding {
    /// Target agent name.
    pub agent: String,
    /// Tool description shown to the parent model.
    pub description: String,
}
