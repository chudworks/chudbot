//! OpenAI-compatible Chat Completions provider crate.
//!
//! This targets local/model-gateway hosts such as vLLM that expose the older
//! `POST /v1/chat/completions` protocol. The first-class `chudbot-openai`
//! crate uses OpenAI's Responses API; keep this crate separate because local
//! compat hosts generally standardize on Chat Completions message and tool
//! envelopes.
//!
//! The crate root owns the shared HTTP transport: endpoint joining, bearer
//! authentication, retry policy, response decoding, and provider-neutral error
//! classification. The `llm` module builds Chat Completions payloads from
//! Chudbot transcripts and routes all network calls through
//! [`OpenAiCompatClient`] so local backends get consistent diagnostics.

mod llm;

use chudbot_api::ProviderName;
use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use serde::Deserialize;
use serde_json::Value;
use std::error::Error as StdError;
use thiserror::Error;

pub use llm::OpenAiCompatOptions;

/// OpenAI-compatible API client for Chat Completions-style local backends.
///
/// The client is intentionally small. Provider-specific request shaping lives
/// in the backend modules, while this type centralizes the parts that should
/// behave the same for `/chat/completions`, `/models`, and future compatibility
/// endpoints.
#[derive(Debug)]
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    api_key: Option<String>,
    base_url: String,
    provider_name: ProviderName,
}

impl OpenAiCompatClient {
    /// Construct from a configured provider name and base URL such as
    /// `http://127.0.0.1:8000/v1`.
    ///
    /// Pass the API root, not a method endpoint. Calls append paths like
    /// `/chat/completions` with exactly one slash between the two parts.
    pub fn new(provider_name: ProviderName, base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: None,
            base_url: base_url.into(),
            provider_name,
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

    /// Send a JSON POST through the shared retry and decode path.
    ///
    /// `label` identifies the operation in retry logs; `endpoint` remains the
    /// wire path so diagnostics can point at the exact failing compatibility
    /// method.
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
            let request_url = url.clone();
            let mut request = self.http.post(&request_url).json(body);
            // Local backends often ignore auth, but gateways and vLLM can use
            // the same configured bearer token path as hosted providers.
            if let Some(api_key) = &self.api_key {
                request = request.bearer_auth(api_key);
            }
            async move {
                let resp = request.send().await.map_err(|e| {
                    let error_chain = format_error_chain(&e);
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        url = %request_url,
                        error = %e,
                        error_chain = %error_chain,
                        error_debug = ?e,
                        "OpenAI-compatible request transport error"
                    );
                    OpenAiCompatError::Transport(error_chain)
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

    /// Fetch a JSON resource through the shared retry and decode path.
    ///
    /// The current LLM backend uses this for `/models`; keeping it beside
    /// [`Self::post_json`] makes transport logging and auth behavior identical
    /// across read and generation requests.
    pub(crate) async fn get_json<T>(
        &self,
        endpoint: &str,
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
            "sending OpenAI-compatible JSON GET request"
        );
        with_retry(RetryPolicy::default(), label, || {
            let request_url = url.clone();
            let mut request = self.http.get(&request_url);
            // Mirror POST auth behavior so model discovery works through the
            // same gateways as completions.
            if let Some(api_key) = &self.api_key {
                request = request.bearer_auth(api_key);
            }
            async move {
                let resp = request.send().await.map_err(|e| {
                    let error_chain = format_error_chain(&e);
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        url = %request_url,
                        error = %e,
                        error_chain = %error_chain,
                        error_debug = ?e,
                        "OpenAI-compatible GET request transport error"
                    );
                    OpenAiCompatError::Transport(error_chain)
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received OpenAI-compatible GET response"
                );
                decode_response(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }
}

/// Join a configured API root and method path without requiring config authors
/// to remember whether either side owns the slash.
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
    // Read the body before branching on status so API errors can include the
    // provider's payload instead of just an HTTP code.
    let body = resp.text().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to read OpenAI-compatible response body"
        );
        OpenAiCompatError::Decode(e.to_string())
    })?;
    if !status.is_success() {
        // Error bodies from local servers can include full prompts or traces;
        // keep enough for diagnosis while preventing oversized log records.
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

    // Decode via `Value` first so logs can distinguish invalid JSON from a
    // valid but incompatible response envelope.
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

fn format_error_chain(error: &dyn StdError) -> String {
    // Reqwest's display message can hide the useful source error, especially
    // for DNS, TLS, and proxy failures. Preserve the chain in the user-facing
    // provider error so config issues are debuggable from logs alone.
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        message.push_str(": ");
        message.push_str(&error.to_string());
        source = error.source();
    }
    message
}

/// OpenAI-compatible provider error.
///
/// These variants describe the transport boundary rather than model semantics:
/// request shaping failures are reported as decode/reference errors, while
/// backend HTTP statuses keep their response body for operator diagnosis.
#[derive(Debug, Error)]
pub enum OpenAiCompatError {
    /// Network, DNS, TLS, timeout, or other request transport failure.
    #[error("transport error: {0}")]
    Transport(String),
    /// API returned a non-success HTTP status.
    #[error("api error {status}: {body}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Truncated response body for diagnostics.
        body: String,
    },
    /// Response JSON was invalid or did not match the expected envelope.
    #[error("decode error: {0}")]
    Decode(String),
    /// Media reference could not be converted into a public URL or data URI.
    #[error("media reference error: {0}")]
    Reference(String),
}

impl ClassifyError for OpenAiCompatError {
    fn error_class(&self) -> ErrorClass {
        // Retry only failures that can plausibly clear without changing the
        // request. Bad payloads, unsupported media, and most HTTP statuses need
        // config or code changes rather than another attempt.
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

    #[derive(Debug, Error)]
    #[error("outer")]
    struct OuterError(#[source] InnerError);

    #[derive(Debug, Error)]
    #[error("inner")]
    struct InnerError;

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

    #[test]
    fn formats_error_source_chain() {
        assert_eq!(format_error_chain(&OuterError(InnerError)), "outer: inner");
    }
}
