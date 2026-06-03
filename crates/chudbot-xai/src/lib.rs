//! xAI provider crate for the chudbot 2.0 API.

mod imagine;
mod llm;

use chudbot_api::ProviderName;
use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

pub use llm::XaiOptions;

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

/// xAI API client.
#[derive(Debug, Clone)]
pub struct XaiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    provider_name: ProviderName,
}

impl XaiClient {
    /// Construct from an xAI API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            provider_name: ProviderName::new("xai"),
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
        body: &Value,
        label: &str,
    ) -> Result<T, XaiError>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.post_json_with_policy(endpoint, body, RetryPolicy::default(), label)
            .await
    }

    pub(crate) async fn post_json_with_policy<T>(
        &self,
        endpoint: &str,
        body: &Value,
        policy: RetryPolicy,
        label: &str,
    ) -> Result<T, XaiError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            "sending xAI JSON request"
        );
        with_retry(policy, label, || {
            let request = self.http.post(&url).bearer_auth(&self.api_key).json(body);
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "xAI request transport error"
                    );
                    XaiError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received xAI response"
                );
                decode_response(resp).await
            }
        })
        .await
    }

    pub(crate) async fn get_json<T>(&self, endpoint: &str, label: &str) -> Result<T, XaiError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            "sending xAI JSON GET request"
        );
        with_retry(RetryPolicy::default(), label, || {
            let request = self.http.get(&url).bearer_auth(&self.api_key);
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "xAI GET request transport error"
                    );
                    XaiError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received xAI GET response"
                );
                decode_response(resp).await
            }
        })
        .await
    }
}

pub(crate) fn json_strip_nulls(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        map.retain(|_, v| !v.is_null());
    }
    value
}

pub(crate) async fn decode_response<T>(resp: reqwest::Response) -> Result<T, XaiError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = resp.status();
    if !status.is_success() {
        let body = truncate_body(resp.text().await.unwrap_or_default(), 600);
        tracing::warn!(
            status = status.as_u16(),
            body_chars = body.chars().count(),
            "xAI API returned non-success status"
        );
        return Err(XaiError::Api {
            status: status.as_u16(),
            body,
        });
    }
    resp.json().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode xAI response"
        );
        XaiError::Decode(e.to_string())
    })
}

pub(crate) fn truncate_body(mut body: String, max: usize) -> String {
    if body.len() > max {
        body.truncate(max);
    }
    body
}

/// xAI provider error.
#[derive(Debug, Error)]
pub enum XaiError {
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
    /// Media reference could not be sent to xAI.
    #[error("media reference error: {0}")]
    Reference(String),
}

impl ClassifyError for XaiError {
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
