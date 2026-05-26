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
}

/// Discord bot connection settings.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DiscordConfig {
    /// Bot token from the Discord Developer Portal.
    pub token: String,
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
    "grok-4.1-fast".to_string()
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
        assert_eq!(config.llm.xai.unwrap().model, "grok-4.1-fast");
        assert_eq!(config.web.listen, "0.0.0.0:8080");
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
