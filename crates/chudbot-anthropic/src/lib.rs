//! Anthropic provider crate for the chudbot 2.0 API.

mod llm;

use chudbot_api::ProviderName;
use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use serde::Deserialize;
use thiserror::Error;

pub use llm::AnthropicOptions;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const API_VERSION: &str = "2023-06-01";

/// Anthropic API client.
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    provider_name: ProviderName,
}

impl AnthropicClient {
    /// Construct from an Anthropic API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            provider_name: ProviderName::new("anthropic"),
        }
    }

    /// Override the base URL. Useful for local tests.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Borrow the underlying HTTP client.
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Borrow the API key.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Borrow the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub(crate) fn provider_name(&self) -> &ProviderName {
        &self.provider_name
    }

    pub(crate) async fn post_json<T>(
        &self,
        endpoint: &str,
        body: &serde_json::Value,
        label: &str,
    ) -> Result<T, AnthropicError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            "sending Anthropic JSON request"
        );
        with_retry(RetryPolicy::default(), label, || {
            let request = self
                .http
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .json(body);
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "Anthropic request transport error"
                    );
                    AnthropicError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received Anthropic response"
                );
                decode_response(resp).await
            }
        })
        .await
    }
}

pub(crate) async fn decode_response<T>(resp: reqwest::Response) -> Result<T, AnthropicError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = resp.status();
    if !status.is_success() {
        let body = truncate_body(resp.text().await.unwrap_or_default(), 600);
        tracing::warn!(
            status = status.as_u16(),
            body_chars = body.chars().count(),
            "Anthropic API returned non-success status"
        );
        return Err(AnthropicError::Api {
            status: status.as_u16(),
            body,
        });
    }
    resp.json().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode Anthropic response"
        );
        AnthropicError::Decode(e.to_string())
    })
}

pub(crate) fn truncate_body(mut body: String, max: usize) -> String {
    if body.len() > max {
        body.truncate(max);
    }
    body
}

/// Anthropic provider error.
#[derive(Debug, Error)]
pub enum AnthropicError {
    /// Network/transport failure.
    #[error("transport error: {0}")]
    Transport(String),
    /// API returned a non-success status.
    #[error("api error {status}: {body}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },
    /// Response decode failed.
    #[error("decode error: {0}")]
    Decode(String),
    /// Media reference could not be sent to Anthropic.
    #[error("media reference error: {0}")]
    Reference(String),
}

impl ClassifyError for AnthropicError {
    fn error_class(&self) -> ErrorClass {
        match self {
            Self::Api { status, .. } if *status == 429 || (500..=599).contains(status) => {
                ErrorClass::ServerTransient
            }
            Self::Transport(_) => ErrorClass::Network,
            Self::Api { .. } | Self::Decode(_) | Self::Reference(_) => ErrorClass::Permanent,
        }
    }
}
