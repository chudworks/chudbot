//! TOML-backed configuration. Loaded once at startup, then passed by
//! reference (or per-section by value) into the subcommand entry points.
//!
//! The `[personas.*]` table is the source of truth for *which* model to
//! call and *with what system prompt*. Each persona names a provider
//! (`xai` or `anthropic`), a specific model id, and the prompt /
//! sampling parameters to use. The `[llm.<provider>]` blocks supply the
//! provider-level credentials shared across personas. Selection at
//! runtime is per-guild / per-channel / per-user / per-conversation,
//! resolved against `persona_selections` in the database; the
//! `default_persona` here is the floor fallback when nothing more
//! specific applies.
//!
//! Image and video generation are modular in the same way. Personas can
//! opt into a specific image / video backend via `image_provider =` and
//! `video_provider =`; backend credentials live under `[image.<kind>]`
//! and `[video.<kind>]`. A persona that doesn't name a backend simply
//! doesn't expose the `generate_image` / `generate_video` tools.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::llm::{AnthropicOptions, XaiOptions};

/// Errors returned by [`Config::load`].
#[derive(Debug, Error)]
pub enum ConfigError {
    /// File could not be opened or read.
    #[error("could not read config file at {}", path.display())]
    Read {
        /// Path that was attempted.
        path: PathBuf,
        /// Underlying io error.
        #[source]
        source: std::io::Error,
    },
    /// Contents could not be parsed as TOML or did not match the schema.
    /// The underlying [`toml::de::Error`] is the source so its
    /// line/column-aware Display gets surfaced by the chain walker in the
    /// binary's error reporter.
    #[error("could not parse config file {}", path.display())]
    Parse {
        /// Path that was being parsed.
        path: PathBuf,
        /// Underlying parse error from the `toml` crate.
        #[source]
        source: toml::de::Error,
    },
    /// `default_persona` doesn't name a persona in `[personas.*]`.
    #[error("default_persona = `{0}` is not defined in [personas.*]")]
    UnknownDefaultPersona(String),
    /// No personas at all.
    #[error("at least one persona must be defined in [personas.*]")]
    NoPersonas,
    /// A persona references a provider with no `[llm.<provider>]` block.
    #[error(
        "persona `{persona}` uses provider `{provider}` but no `[llm.{provider}]` section was found"
    )]
    MissingProviderForPersona {
        /// Persona name that triggered the failure.
        persona: String,
        /// Provider name referenced by that persona.
        provider: String,
    },
    /// A persona names an image provider with no matching credential block.
    #[error(
        "persona `{persona}` uses image_provider `{provider}` but no `[image.{provider}]` section was found"
    )]
    MissingImageProviderForPersona {
        /// Persona name that triggered the failure.
        persona: String,
        /// Image provider kind referenced by that persona.
        provider: String,
    },
    /// A persona names a video provider with no matching credential block.
    #[error(
        "persona `{persona}` uses video_provider `{provider}` but no `[video.{provider}]` section was found"
    )]
    MissingVideoProviderForPersona {
        /// Persona name that triggered the failure.
        persona: String,
        /// Video provider kind referenced by that persona.
        provider: String,
    },
}

/// Top-level configuration object.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Discord bot connection.
    pub discord: DiscordConfig,
    /// Postgres connection.
    pub postgres: PostgresConfig,
    /// Web viewer.
    pub web: WebConfig,
    /// Per-provider credentials. Either `[llm.xai]` or `[llm.anthropic]`
    /// (or both) must be present, depending on which providers the
    /// configured personas use.
    pub llm: LlmConfig,
    /// Image generation backends. Optional; include only the blocks for
    /// the backends any persona's `image_provider` field references.
    #[serde(default)]
    pub image: ImageConfig,
    /// Video generation backends. Optional; same shape as `image`.
    #[serde(default)]
    pub video: VideoConfig,
    /// Default [`PrivacyMode`] applied to guilds that don't have an
    /// explicit row in `guild_settings` yet. Optional — defaults to
    /// [`PrivacyMode::opt_in_default`]. Server admins can override per
    /// guild at runtime via the `/grok-mode set` slash command.
    #[serde(default = "PrivacyMode::opt_in_default")]
    pub default_privacy: PrivacyMode,
    /// Persona name used as the floor fallback when no more-specific
    /// selection is recorded in `persona_selections`. Must be a key in
    /// [`Self::personas`].
    pub default_persona: String,
    /// Named personas. Each ties together a model, a system prompt, and
    /// optional sampling knobs. Runtime selection picks one of these by
    /// name; see `persona_selections` in the DB for scope.
    pub personas: HashMap<String, Persona>,
    /// Media storage (image attachments today). Optional; defaults
    /// reasonably for a local single-host deploy.
    #[serde(default)]
    pub storage: StorageConfig,
    /// Operator-supplied text appended to the dynamically-built
    /// operational guidance on EVERY persona's system prompt — a global,
    /// non-persona slot for deployment-wide rules (e.g. the Discord
    /// developer ToS, content policy). Distinct from `personas.*.
    /// system_prompt`, which is the per-persona voice. Optional.
    #[serde(default)]
    pub extra_system_prompt: Option<String>,
}

/// Media storage settings. Local-only today; the URI scheme in the DB
/// (`file://images/<name>`, `file://videos/<name>`) is the seam for
/// adding more backends later (`s3://…`, etc.).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StorageConfig {
    /// Directory where images live (Discord attachments + generated
    /// images). The web viewer mounts `/images/*` at this directory.
    #[serde(default = "default_images_dir")]
    pub images_dir: PathBuf,
    /// Directory where generated videos live. The web viewer mounts
    /// `/videos/*` at this directory.
    #[serde(default = "default_videos_dir")]
    pub videos_dir: PathBuf,
    /// Directory where cached Discord avatars live. The web viewer
    /// mounts `/avatars/*` at this directory. Files are named
    /// `<user_id>_<avatar_hash>.<ext>` so a hash change cleanly
    /// supersedes the old file without manual eviction.
    #[serde(default = "default_avatars_dir")]
    pub avatars_dir: PathBuf,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            images_dir: default_images_dir(),
            videos_dir: default_videos_dir(),
            avatars_dir: default_avatars_dir(),
        }
    }
}

fn default_images_dir() -> PathBuf {
    PathBuf::from("images")
}

fn default_videos_dir() -> PathBuf {
    PathBuf::from("videos")
}

fn default_avatars_dir() -> PathBuf {
    PathBuf::from("avatars")
}

/// One named persona. The bot consults this on every turn to decide
/// which provider+model to call and what system prompt + sampling knobs
/// to use. Personas can mix providers freely; each one names its own.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Persona {
    /// Which provider to route this persona's calls through.
    pub provider: LlmProviderKind,
    /// Model id for that provider (e.g. `grok-4.3`, `claude-sonnet-4-6`).
    pub model: String,
    /// Top-level instruction sent to the model on every turn. Wired
    /// into the xAI Responses API's `instructions` field and lifted out
    /// of the Anthropic Messages API's chat history into its top-level
    /// `system` field.
    pub system_prompt: String,
    /// Sampling temperature (0.0-2.0). `None` lets the provider pick its
    /// default. Higher = more random; lower = more focused.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass (0.0-1.0). `None` lets the
    /// provider pick its default.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Provider-specific knobs for when this persona's `provider = "xai"`.
    /// Today: `reasoning_effort` (`"low"` | `"medium"` | `"high"`).
    /// Ignored when the persona doesn't route to xAI.
    #[serde(default)]
    pub xai: Option<XaiOptions>,
    /// Provider-specific knobs for when this persona's `provider =
    /// "anthropic"`. Placeholder today; reserved for future Anthropic-
    /// specific options. Ignored when the persona doesn't route to
    /// Anthropic.
    #[serde(default)]
    pub anthropic: Option<AnthropicOptions>,
    /// Optional image generation backend for this persona. When set,
    /// the `generate_image` tool is exposed and routed through the
    /// matching `[image.<kind>]` credentials block. When unset, the
    /// persona simply has no image generation tool.
    #[serde(default)]
    pub image_provider: Option<ImageProviderKind>,
    /// Optional video generation backend for this persona. Same
    /// semantics as [`Self::image_provider`].
    #[serde(default)]
    pub video_provider: Option<VideoProviderKind>,
}

/// Privacy / context-gathering policy. The four variants correspond to
/// the four designs discussed by the group; default is `opt_in`
/// (Design 3, the "privacy-maxxing opt-in" approach).
///
/// In all modes, the bot still sees:
///   - the user's own `@<bot>` mention (they're addressing it directly);
///   - prior turns of the same conversation reconstructed from the DB.
///
/// What varies is the *extra* context pulled from the surrounding
/// channel:
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum PrivacyMode {
    /// **Design 1** — open. The bot bulk-fetches recent messages from
    /// the channel each time it's mentioned, regardless of who sent
    /// them. Best answers, least privacy.
    Open {
        /// Number of recent channel messages to include. Defaults to 20.
        #[serde(default = "default_history_size")]
        history_size: u32,
    },
    /// **Design 2** — channel-scoped. The bot only responds to mentions
    /// inside `channel_id` and bulk-fetches history there. Mentions in
    /// other channels are silently ignored.
    ChannelOnly {
        /// Discord channel id the bot is confined to.
        channel_id: u64,
        /// Number of recent channel messages to include. Defaults to 20.
        #[serde(default = "default_history_size")]
        history_size: u32,
    },
    /// **Design 3** — opt-in (default). Quoted messages (Discord
    /// replies) are only included if their author has opted in via
    /// `/grok-privacy-in`, or if the message lives in a Grok-owned
    /// thread. No bulk channel history.
    OptIn,
    /// **Design 4** — conversation-only / privacy-maxxing. The bot only
    /// sees the user's `@`-mention and prior turns of the same
    /// conversation. Even Discord-reply-quoted messages are excluded.
    ConversationOnly,
}

fn default_history_size() -> u32 {
    20
}

impl PrivacyMode {
    /// Default factory used by serde when `[privacy]` is omitted.
    pub fn opt_in_default() -> Self {
        Self::OptIn
    }
}

/// Discord bot connection settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DiscordConfig {
    /// Bot token from the Discord Developer Portal.
    pub token: String,
    /// If set, register slash commands to this guild only. Guild
    /// commands appear instantly; global commands take up to an hour to
    /// propagate, which is painful during development. Set to your
    /// server's id for fast iteration; omit (or set to None) to register
    /// globally once the bot is in multiple servers.
    #[serde(default)]
    pub dev_guild_id: Option<u64>,
}

/// Postgres connection settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PostgresConfig {
    /// Standard `postgres://user:pass@host/db` URL.
    pub url: String,
}

/// Web viewer settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WebConfig {
    /// Public base URL of the viewer; used to build links posted into
    /// Discord (e.g. `https://grok.example.com`).
    pub base_url: String,
    /// `host:port` the Axum server listens on. Defaults to
    /// `127.0.0.1:1860` so the server is reachable only via loopback —
    /// matches the production deploy behind a Cloudflare tunnel.
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Directory containing the built React frontend (Vite's `dist/`).
    /// The Axum server serves any file that matches here as static
    /// content and falls back to `index.html` for SPA routes (e.g.
    /// `/c/<uuid>`). Defaults to `./frontend-build`, which is what
    /// `serve.sh deploy` writes to in production.
    #[serde(default = "default_frontend_dir")]
    pub frontend_dir: PathBuf,
    /// Prefix prepended to the browser tab title on every viewer page.
    /// The page-specific part (conversation title, "Viewer", "Not
    /// found", …) is appended by the frontend, so a value like
    /// `"Chudbot · "` yields tab titles such as `Chudbot · My chat`.
    #[serde(default = "default_title_prefix")]
    pub title_prefix: String,
    /// Optional path to a favicon file served verbatim at
    /// `/favicon.ico` (the URL browsers request automatically). Keep it
    /// OUTSIDE `frontend_dir` — `serve.sh deploy` atomically replaces
    /// that whole directory, so anything inside it is wiped each
    /// deploy. When unset, `/favicon.ico` 404s and browsers fall back
    /// to their default icon. A multi-resolution `.ico` is ideal, but
    /// any image format the browser accepts works.
    #[serde(default)]
    pub favicon_path: Option<PathBuf>,
}

fn default_listen() -> String {
    "127.0.0.1:1860".to_string()
}

fn default_frontend_dir() -> PathBuf {
    PathBuf::from("frontend-build")
}

fn default_title_prefix() -> String {
    "grok · ".to_string()
}

/// Per-provider credentials. The model is no longer part of these
/// blocks — personas pick that. A provider block only needs to exist
/// if at least one persona references it.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct LlmConfig {
    /// xAI provider credentials.
    pub xai: Option<XaiConfig>,
    /// Anthropic provider credentials.
    pub anthropic: Option<AnthropicConfig>,
}

/// Discriminator for which LLM provider a persona uses.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum LlmProviderKind {
    /// xAI / Grok.
    Xai,
    /// Anthropic / Claude.
    Anthropic,
}

impl LlmProviderKind {
    /// Lowercase string form, suitable for log fields and config lookups.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Xai => "xai",
            Self::Anthropic => "anthropic",
        }
    }
}

/// xAI Grok credentials.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct XaiConfig {
    /// API key issued at console.x.ai.
    pub api_key: String,
}

/// Anthropic Claude credentials.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicConfig {
    /// API key.
    pub api_key: String,
}

/// Image generation backend credentials. Each field is a separate
/// `[image.<kind>]` block; presence gates whether a persona referencing
/// that kind can actually serve image requests.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ImageConfig {
    /// xAI Grok Imagine credentials.
    pub xai: Option<XaiImageConfig>,
}

/// xAI Grok Imagine credentials. Typically the same key as
/// `[llm.xai]`, but stored separately so deployments can mix-and-match
/// (e.g. swap to a different image backend without dropping LLM access).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct XaiImageConfig {
    /// API key issued at console.x.ai.
    pub api_key: String,
}

/// Video generation backend credentials. Symmetric with [`ImageConfig`].
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct VideoConfig {
    /// xAI Grok Imagine Video credentials.
    pub xai: Option<XaiVideoConfig>,
}

/// xAI Grok Imagine Video credentials.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct XaiVideoConfig {
    /// API key issued at console.x.ai.
    pub api_key: String,
}

/// Discriminator for which image generation backend a persona uses.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ImageProviderKind {
    /// xAI Grok Imagine (`grok-imagine-image` family).
    Xai,
}

impl ImageProviderKind {
    /// Lowercase string form, suitable for log fields and config lookups.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Xai => "xai",
        }
    }
}

/// Discriminator for which video generation backend a persona uses.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum VideoProviderKind {
    /// xAI Grok Imagine Video.
    Xai,
}

impl VideoProviderKind {
    /// Lowercase string form.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Xai => "xai",
        }
    }
}

impl Config {
    /// Load and validate a config from `path`.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let config: Config = toml::from_str(&contents).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.personas.is_empty() {
            return Err(ConfigError::NoPersonas);
        }
        if !self.personas.contains_key(&self.default_persona) {
            return Err(ConfigError::UnknownDefaultPersona(
                self.default_persona.clone(),
            ));
        }
        for (name, persona) in &self.personas {
            let present = match persona.provider {
                LlmProviderKind::Xai => self.llm.xai.is_some(),
                LlmProviderKind::Anthropic => self.llm.anthropic.is_some(),
            };
            if !present {
                return Err(ConfigError::MissingProviderForPersona {
                    persona: name.clone(),
                    provider: persona.provider.as_str().to_string(),
                });
            }
            if let Some(kind) = persona.image_provider {
                let configured = match kind {
                    ImageProviderKind::Xai => self.image.xai.is_some(),
                };
                if !configured {
                    return Err(ConfigError::MissingImageProviderForPersona {
                        persona: name.clone(),
                        provider: kind.as_str().to_string(),
                    });
                }
            }
            if let Some(kind) = persona.video_provider {
                let configured = match kind {
                    VideoProviderKind::Xai => self.video.xai.is_some(),
                };
                if !configured {
                    return Err(ConfigError::MissingVideoProviderForPersona {
                        persona: name.clone(),
                        provider: kind.as_str().to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Look up a persona by name, falling back to `default_persona`
    /// when missing. Panics only if the config has no default persona
    /// at all — which `validate` already guarantees can't happen.
    pub fn persona_or_default(&self, name: Option<&str>) -> &Persona {
        if let Some(n) = name
            && let Some(p) = self.personas.get(n)
        {
            return p;
        }
        self.personas
            .get(&self.default_persona)
            .expect("validate() guarantees default_persona is present")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: `default_persona` is a top-level scalar so it must appear
    // BEFORE any `[section]` headers — otherwise the TOML parser
    // attaches it to whichever section was opened last.
    const MINIMAL_CONFIG: &str = r#"
        default_persona = "default"

        [discord]
        token = "abc"

        [postgres]
        url = "postgres://localhost/grok"

        [web]
        base_url = "http://localhost:8080"

        [llm.xai]
        api_key = "xai-key"

        [personas.default]
        provider = "xai"
        model = "grok-4.3"
        system_prompt = "You are a helpful AI."
    "#;

    #[test]
    fn parse_minimal_xai_config() {
        let config: Config = toml::from_str(MINIMAL_CONFIG).unwrap();
        config.validate().unwrap();
        assert_eq!(config.default_persona, "default");
        let persona = &config.personas["default"];
        assert_eq!(persona.provider, LlmProviderKind::Xai);
        assert_eq!(persona.model, "grok-4.3");
        assert_eq!(config.web.listen, "127.0.0.1:1860");
        assert_eq!(config.web.frontend_dir, PathBuf::from("frontend-build"));
        assert!(matches!(config.default_privacy, PrivacyMode::OptIn));
    }

    #[test]
    fn parse_multiple_personas() {
        let toml = r#"
            default_persona = "snark"

            [discord]
            token = "abc"

            [postgres]
            url = "postgres://localhost/grok"

            [web]
            base_url = "http://localhost:8080"

            [llm.xai]
            api_key = "xai-key"

            [llm.anthropic]
            api_key = "anth-key"

            [personas.default]
            provider = "xai"
            model = "grok-4.3"
            system_prompt = "Be helpful."

            [personas.snark]
            provider = "anthropic"
            model = "claude-sonnet-4-6"
            system_prompt = "Be sardonic."
            temperature = 1.2
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        config.validate().unwrap();
        assert_eq!(config.default_persona, "snark");
        assert_eq!(config.personas.len(), 2);
        assert_eq!(
            config.personas["snark"].provider,
            LlmProviderKind::Anthropic
        );
        assert_eq!(config.personas["snark"].temperature, Some(1.2));
    }

    #[test]
    fn parse_channel_only_privacy() {
        let toml = r#"
            default_persona = "default"

            [discord]
            token = "abc"

            [postgres]
            url = "postgres://localhost/grok"

            [web]
            base_url = "http://localhost:8080"

            [llm.xai]
            api_key = "xai-key"

            [personas.default]
            provider = "xai"
            model = "grok-4.3"
            system_prompt = "help"

            [default_privacy]
            mode = "channel_only"
            channel_id = 123456789
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        match config.default_privacy {
            PrivacyMode::ChannelOnly {
                channel_id,
                history_size,
            } => {
                assert_eq!(channel_id, 123_456_789);
                assert_eq!(history_size, 20);
            }
            _ => panic!("expected ChannelOnly"),
        }
    }

    #[test]
    fn unknown_default_persona_is_an_error() {
        let toml = r#"
            default_persona = "missing"

            [discord]
            token = "abc"

            [postgres]
            url = "postgres://localhost/grok"

            [web]
            base_url = "http://localhost:8080"

            [llm.xai]
            api_key = "xai-key"

            [personas.default]
            provider = "xai"
            model = "grok-4.3"
            system_prompt = "x"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(matches!(err, ConfigError::UnknownDefaultPersona(_)));
    }

    #[test]
    fn persona_without_provider_block_is_an_error() {
        let toml = r#"
            default_persona = "default"

            [discord]
            token = "abc"

            [postgres]
            url = "postgres://localhost/grok"

            [web]
            base_url = "http://localhost:8080"

            [llm]

            [personas.default]
            provider = "anthropic"
            model = "claude-sonnet-4-6"
            system_prompt = "x"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(matches!(err, ConfigError::MissingProviderForPersona { .. }));
    }

    #[test]
    fn persona_naming_image_provider_without_block_is_an_error() {
        // Persona names `image_provider = "xai"` but no `[image.xai]`
        // credentials block is configured, so validation rejects it.
        let toml = r#"
            default_persona = "default"

            [discord]
            token = "abc"

            [postgres]
            url = "postgres://localhost/grok"

            [web]
            base_url = "http://localhost:8080"

            [llm.xai]
            api_key = "xai-key"

            [personas.default]
            provider = "xai"
            model = "grok-4.3"
            system_prompt = "x"
            image_provider = "xai"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(matches!(
            err,
            ConfigError::MissingImageProviderForPersona { ref provider, .. }
                if provider == "xai"
        ));
    }

    #[test]
    fn persona_with_matching_image_block_validates() {
        let toml = r#"
            default_persona = "default"

            [discord]
            token = "abc"

            [postgres]
            url = "postgres://localhost/grok"

            [web]
            base_url = "http://localhost:8080"

            [llm.xai]
            api_key = "xai-key"

            [image.xai]
            api_key = "xai-image-key"

            [video.xai]
            api_key = "xai-video-key"

            [personas.default]
            provider = "xai"
            model = "grok-4.3"
            system_prompt = "x"
            image_provider = "xai"
            video_provider = "xai"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        config.validate().unwrap();
        assert_eq!(
            config.personas["default"].image_provider,
            Some(ImageProviderKind::Xai)
        );
        assert_eq!(
            config.personas["default"].video_provider,
            Some(VideoProviderKind::Xai)
        );
    }
}
