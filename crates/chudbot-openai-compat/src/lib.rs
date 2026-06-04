//! OpenAI-compatible Chat Completions provider crate.
//!
//! This targets local/model-gateway hosts such as vLLM that expose the older
//! `POST /v1/chat/completions` protocol. The first-class `chudbot-openai`
//! crate uses OpenAI's Responses API; keep this crate separate because local
//! compat hosts generally standardize on Chat Completions message and tool
//! envelopes.

mod llm;

use chudbot_api::ProviderName;
use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

pub use llm::OpenAiCompatOptions;

/// OpenAI-compatible API client.
#[derive(Debug, Clone)]
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    api_key: Option<String>,
    base_url: String,
    provider_name: ProviderName,
}

impl OpenAiCompatClient {
    /// Construct from a base URL such as `http://127.0.0.1:8000/v1`.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: None,
            base_url: base_url.into(),
            provider_name: ProviderName::new("openai_compat"),
        }
    }

    /// Set an optional bearer token. Many local servers accept none or any
    /// placeholder token, but vLLM can enforce one when launched with
    /// `--api-key`.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Override the base URL. Useful for tests and gateway deployments.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
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
        body: &Value,
        label: &str,
    ) -> Result<T, OpenAiCompatError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = endpoint_url(&self.base_url, endpoint);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            body_keys = body.as_object().map_or(0, serde_json::Map::len),
            "sending OpenAI-compatible JSON request"
        );
        with_retry(RetryPolicy::default(), label, || {
            let mut request = self.http.post(&url).json(body);
            if let Some(api_key) = &self.api_key {
                request = request.bearer_auth(api_key);
            }
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "OpenAI-compatible request transport error"
                    );
                    OpenAiCompatError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received OpenAI-compatible response"
                );
                decode_response(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }
}

fn endpoint_url(base_url: &str, endpoint: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        endpoint.trim_start_matches('/')
    )
}

async fn decode_response<T>(
    resp: reqwest::Response,
    provider: &ProviderName,
    endpoint: &str,
) -> Result<T, OpenAiCompatError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = resp.status();
    let body = resp.text().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to read OpenAI-compatible response body"
        );
        OpenAiCompatError::Decode(e.to_string())
    })?;
    if !status.is_success() {
        let body = truncate_body(body, 600);
        tracing::warn!(
            provider = %provider,
            endpoint,
            status = status.as_u16(),
            body_chars = body.chars().count(),
            "OpenAI-compatible API returned non-success status"
        );
        return Err(OpenAiCompatError::Api {
            status: status.as_u16(),
            body,
        });
    }

    let value = serde_json::from_str::<Value>(&body).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode OpenAI-compatible response"
        );
        OpenAiCompatError::Decode(e.to_string())
    })?;
    serde_json::from_value(value).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode OpenAI-compatible response shape"
        );
        OpenAiCompatError::Decode(e.to_string())
    })
}

fn truncate_body(mut body: String, max: usize) -> String {
    if body.len() > max {
        body.truncate(max);
    }
    body
}

/// OpenAI-compatible provider error.
#[derive(Debug, Error)]
pub enum OpenAiCompatError {
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
    /// Media reference could not be sent to the host.
    #[error("media reference error: {0}")]
    Reference(String),
}

impl ClassifyError for OpenAiCompatError {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joins_endpoint_url_with_or_without_slashes() {
        assert_eq!(
            endpoint_url("http://127.0.0.1:8000/v1", "/chat/completions"),
            "http://127.0.0.1:8000/v1/chat/completions"
        );
        assert_eq!(
            endpoint_url("http://127.0.0.1:8000/v1/", "chat/completions"),
            "http://127.0.0.1:8000/v1/chat/completions"
        );
    }
}
