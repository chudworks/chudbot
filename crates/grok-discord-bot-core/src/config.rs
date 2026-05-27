//! TOML-backed configuration. Loaded once at startup, then passed by
//! reference (or per-section by value) into the subcommand entry points.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors returned by [`Config::load`].
#[derive(Debug, Error)]
pub enum ConfigError {
    /// File could not be opened or read.
    #[error("could not read config file at {path}: {source}")]
    Read {
        /// Path that was attempted.
        path: PathBuf,
        /// Underlying io error.
        #[source]
        source: std::io::Error,
    },
    /// Contents could not be parsed as TOML or did not match the schema.
    #[error("could not parse config file: {0}")]
    Parse(#[from] toml::de::Error),
    /// The selected LLM provider has no matching `[llm.<provider>]` section.
    #[error("config selects llm provider `{provider}` but no `[llm.{provider}]` section was found")]
    MissingProviderSection {
        /// Provider name from `llm.provider`.
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
    /// LLM provider selection and per-provider credentials.
    pub llm: LlmConfig,
    /// Default [`PrivacyMode`] applied to guilds that don't have an
    /// explicit row in `guild_settings` yet. Optional — defaults to
    /// [`PrivacyMode::opt_in_default`]. Server admins can override per
    /// guild at runtime via the `/grok-mode set` slash command.
    #[serde(default = "PrivacyMode::opt_in_default")]
    pub default_privacy: PrivacyMode,
    /// Bot persona — system prompt and sampling knobs the agent runs with.
    /// Optional; an omitted `[bot]` block applies the default persona.
    #[serde(default)]
    pub bot: BotConfig,
    /// Media storage (image attachments today). Optional; defaults
    /// reasonably for a local single-host deploy.
    #[serde(default)]
    pub storage: StorageConfig,
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
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            images_dir: default_images_dir(),
            videos_dir: default_videos_dir(),
        }
    }
}

fn default_images_dir() -> PathBuf {
    PathBuf::from("images")
}

fn default_videos_dir() -> PathBuf {
    PathBuf::from("videos")
}

/// Persona / sampling settings for the agent loop. Edit `system_prompt`
/// to give the bot a personality (sarcastic, terse, role-played, etc.);
/// raise `temperature` to make replies more chaotic or lower it for
/// more focused answers.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BotConfig {
    /// Top-level instruction sent to the model on every turn. Wired
    /// into the xAI Responses API's `instructions` field and lifted out
    /// of the Anthropic Messages API's chat history into its top-level
    /// `system` field.
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    /// Sampling temperature (0.0-2.0). `None` lets the provider pick its
    /// default. Higher = more random; lower = more focused.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability mass (0.0-1.0). `None` lets the
    /// provider pick its default.
    #[serde(default)]
    pub top_p: Option<f32>,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            system_prompt: default_system_prompt(),
            temperature: None,
            top_p: None,
        }
    }
}

fn default_system_prompt() -> String {
    "You are a helpful AI assistant in a private Discord server. Be direct \
and concise. When asked to verify a claim, use the web_search and x_search \
tools to ground your answer in current sources and cite URLs. When you need \
more context about an ongoing conversation in this channel (for example: \
\"what did they decide?\", \"what's the discussion been about?\"), call the \
`fetch_messages` tool to pull recent messages from the channel. Don't fetch \
speculatively — only when you actually need extra context to answer."
        .to_string()
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
    /// `host:port` the Axum server listens on.
    #[serde(default = "default_listen")]
    pub listen: String,
}

fn default_listen() -> String {
    "0.0.0.0:8080".to_string()
}

/// LLM provider selection plus per-provider config blocks.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlmConfig {
    /// Which provider to route requests through.
    pub provider: LlmProviderKind,
    /// xAI provider settings; required when `provider = "xai"`.
    pub xai: Option<XaiConfig>,
    /// Anthropic provider settings; required when `provider = "anthropic"`.
    pub anthropic: Option<AnthropicConfig>,
}

/// Discriminator for which LLM provider to use at runtime.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
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

/// xAI Grok configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct XaiConfig {
    /// API key issued at console.x.ai.
    pub api_key: String,
    /// Model id, e.g. `grok-4.1-fast` or `grok-4.3`.
    #[serde(default = "default_xai_model")]
    pub model: String,
}

fn default_xai_model() -> String {
    // Use the documented flagship; older `grok-4.1-fast` is no longer
    // listed in xAI's current model catalog.
    "grok-4.3".to_string()
}

/// Anthropic Claude configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicConfig {
    /// API key.
    pub api_key: String,
    /// Model id, e.g. `claude-sonnet-4-6`.
    #[serde(default = "default_anthropic_model")]
    pub model: String,
}

fn default_anthropic_model() -> String {
    "claude-sonnet-4-6".to_string()
}

impl Config {
    /// Load and validate a config from `path`.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let contents =
            std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        let config: Config = toml::from_str(&contents)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        match self.llm.provider {
            LlmProviderKind::Xai if self.llm.xai.is_none() => {
                Err(ConfigError::MissingProviderSection {
                    provider: "xai".into(),
                })
            }
            LlmProviderKind::Anthropic if self.llm.anthropic.is_none() => {
                Err(ConfigError::MissingProviderSection {
                    provider: "anthropic".into(),
                })
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_xai_config() {
        let toml = r#"
            [discord]
            token = "abc"

            [postgres]
            url = "postgres://localhost/grok"

            [web]
            base_url = "http://localhost:8080"

            [llm]
            provider = "xai"

            [llm.xai]
            api_key = "xai-key"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        config.validate().unwrap();
        assert_eq!(config.llm.provider, LlmProviderKind::Xai);
        assert_eq!(config.llm.xai.unwrap().model, "grok-4.3");
        assert_eq!(config.web.listen, "0.0.0.0:8080");
        assert!(matches!(config.default_privacy, PrivacyMode::OptIn));
    }

    #[test]
    fn parse_channel_only_privacy() {
        let toml = r#"
            [discord]
            token = "abc"

            [postgres]
            url = "postgres://localhost/grok"

            [web]
            base_url = "http://localhost:8080"

            [llm]
            provider = "xai"

            [llm.xai]
            api_key = "xai-key"

            [default_privacy]
            mode = "channel_only"
            channel_id = 123456789
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        match config.default_privacy {
            PrivacyMode::ChannelOnly { channel_id, history_size } => {
                assert_eq!(channel_id, 123_456_789);
                assert_eq!(history_size, 20);
            }
            _ => panic!("expected ChannelOnly"),
        }
    }

    #[test]
    fn provider_section_missing_is_an_error() {
        let toml = r#"
            [discord]
            token = "abc"

            [postgres]
            url = "postgres://localhost/grok"

            [web]
            base_url = "http://localhost:8080"

            [llm]
            provider = "anthropic"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        let err = config.validate().unwrap_err();
        assert!(matches!(err, ConfigError::MissingProviderSection { .. }));
    }
}
