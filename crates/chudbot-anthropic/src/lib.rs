//! Anthropic provider integration for Chudbot.
//!
//! This crate owns the Anthropic-specific HTTP client, Messages API language
//! model adapter, and local token-pricing estimates. The public surface is
//! intentionally small: callers construct an [`AnthropicClient`], optionally
//! override endpoint or pricing configuration, and then use the backend traits
//! implemented in the private modules.
//!
//! The shared client helpers in this module centralize retries, response
//! decoding, provider-scoped tracing, and payload redaction so the Messages API
//! implementation can focus on translating Chudbot transcripts into Anthropic
//! request and response shapes.

mod llm;
mod pricing;

use std::collections::BTreeMap;

use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use chudbot_api::{ModelId, ProviderName};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

pub use llm::AnthropicOptions;
pub use pricing::AnthropicTokenPricing;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
// Anthropic requires every request to pin an API version through this header.
const API_VERSION: &str = "2023-06-01";

/// Anthropic API client used by the provider implementations in this crate.
///
/// The client carries deployment-local provider identity as well as Anthropic
/// credentials. That provider name is what shows up in Chudbot traces, usage
/// records, and retry labels when a deployment registers more than one
/// Anthropic-compatible service.
#[derive(Debug)]
pub struct AnthropicClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    provider_name: ProviderName,
    pricing: pricing::AnthropicPricing,
}

impl AnthropicClient {
    /// Construct a client from a configured provider name and Anthropic API key.
    ///
    /// The default endpoint targets Anthropic's public API and token pricing is
    /// initialized with the built-in estimate table. Use the builder methods to
    /// point tests at a mock server or to override local cost estimates.
    pub fn new(provider_name: ProviderName, api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            provider_name,
            pricing: pricing::AnthropicPricing::default(),
        }
    }

    /// Override the base URL.
    ///
    /// This is mainly used by tests and Anthropic-compatible gateways. Endpoint
    /// arguments passed to request helpers are appended directly, so callers
    /// should provide the versioned root URL without a trailing endpoint path.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Override or add text-token pricing entries used for local cost estimates.
    ///
    /// Anthropic returns usage counts but not billable cost. These entries let
    /// the runtime turn usage into estimated `usd_ticks` without making another
    /// provider-specific API call.
    pub fn with_token_pricing(mut self, pricing: BTreeMap<ModelId, AnthropicTokenPricing>) -> Self {
        self.pricing.apply_token_overrides(pricing);
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

    pub(crate) fn pricing(&self) -> &pricing::AnthropicPricing {
        &self.pricing
    }

    /// POST a JSON body to an Anthropic endpoint and deserialize the JSON result.
    ///
    /// This helper is the single write path for the Messages API adapter. It
    /// attaches Anthropic's required authentication/version headers, logs a
    /// redacted payload at debug level, retries errors classified as transient,
    /// and delegates status/body handling to [`decode_response`].
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
        log_json_request(&self.provider_name, endpoint, body);
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
                decode_response(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }

    /// GET a JSON Anthropic endpoint and deserialize the result.
    ///
    /// Model metadata uses the same retry and decode path as generation calls
    /// but has no request body to redact.
    pub(crate) async fn get_json<T>(&self, endpoint: &str, label: &str) -> Result<T, AnthropicError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            "sending Anthropic JSON GET request"
        );
        with_retry(RetryPolicy::default(), label, || {
            let request = self
                .http
                .get(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION);
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "Anthropic GET request transport error"
                    );
                    AnthropicError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received Anthropic GET response"
                );
                decode_response(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }
}

/// Decode a provider response into the caller's expected shape.
///
/// The function reads the body once, preserves compact API-error bodies for
/// user-visible diagnostics, logs full successful JSON payloads only after
/// redaction, and then performs the typed deserialization expected by the
/// provider-specific caller.
pub(crate) async fn decode_response<T>(
    resp: reqwest::Response,
    provider: &ProviderName,
    endpoint: &str,
) -> Result<T, AnthropicError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = resp.status();
    // Keep response-body read failures separate from JSON-shape failures; both
    // are permanent for retry purposes, but the log text points at different
    // provider or transport defects.
    let body = resp.text().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to read Anthropic response body"
        );
        AnthropicError::Decode(e.to_string())
    })?;
    if !status.is_success() {
        // Preserve enough provider text to explain the failure without letting
        // large HTML/proxy errors dominate traces or Discord-visible messages.
        log_text_response(provider, endpoint, status.as_u16(), &body);
        let body = truncate_body(body, 600);
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

    let value = serde_json::from_str::<Value>(&body).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode Anthropic response"
        );
        AnthropicError::Decode(e.to_string())
    })?;
    log_json_response(provider, endpoint, status.as_u16(), &value);
    serde_json::from_value(value).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode Anthropic response shape"
        );
        AnthropicError::Decode(e.to_string())
    })
}

/// Truncate a response body to the byte budget used in diagnostics.
pub(crate) fn truncate_body(mut body: String, max: usize) -> String {
    if body.len() > max {
        body.truncate(max);
    }
    body
}

/// Log a JSON request after removing payload fields that can carry media bytes.
fn log_json_request(provider: &ProviderName, endpoint: &str, body: &Value) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = stringify_redacted_json(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            request = %body,
            "Anthropic JSON request payload",
        );
    }
}

/// Log a JSON response after removing payload fields that can carry media bytes.
fn log_json_response(provider: &ProviderName, endpoint: &str, status: u16, body: &Value) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = stringify_redacted_json(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            status,
            response = %body,
            "Anthropic JSON response payload",
        );
    }
}

/// Log a non-success text response with base64-like bodies redacted.
fn log_text_response(provider: &ProviderName, endpoint: &str, status: u16, body: &str) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = redact_text_body(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            status,
            response = %body,
            "Anthropic non-JSON response payload",
        );
    }
}

/// Serialize a JSON value for debug logs after recursively redacting blobs.
fn stringify_redacted_json(value: &Value) -> String {
    serde_json::to_string(&redact_json(value, None))
        .unwrap_or_else(|_| "[unserializable JSON payload]".to_string())
}

/// Recursively redact strings that are likely to contain media or thinking data.
fn redact_json(value: &Value, key: Option<&str>) -> Value {
    match value {
        Value::Array(items) => {
            Value::Array(items.iter().map(|item| redact_json(item, None)).collect())
        }
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), redact_json(value, Some(key))))
                .collect(),
        ),
        Value::String(text) => Value::String(redact_string(key, text)),
        other => other.clone(),
    }
}

/// Redact a string when the surrounding key or payload shape makes it sensitive.
fn redact_string(key: Option<&str>, text: &str) -> String {
    if let Some(redacted) = redact_data_uri(text) {
        return redacted;
    }

    // `encrypted_content` can contain Anthropic thinking continuations; `data`
    // is only redacted when it looks like an encoded media payload because it is
    // otherwise too generic to treat as sensitive by name alone.
    let known_payload_key = matches!(key, Some("b64_json" | "encrypted_content"));
    let possible_payload_key = matches!(key, Some("data"));
    if known_payload_key || (possible_payload_key && looks_like_base64(text)) {
        return format!(
            "[redacted base64-like string; chars={}]",
            text.chars().count()
        );
    }

    text.to_string()
}

/// Redact inline data URIs while preserving their MIME-type prefix.
fn redact_data_uri(text: &str) -> Option<String> {
    let (prefix, payload) = text.split_once(";base64,")?;
    if !prefix.starts_with("data:") {
        return None;
    }
    Some(format!(
        "{prefix};base64,[redacted base64 data; chars={}]",
        payload.chars().count()
    ))
}

/// Redact or cap text bodies from failed responses before logging them.
fn redact_text_body(body: &str) -> String {
    if looks_like_base64(body) {
        return format!(
            "[redacted base64-like body; chars={}]",
            body.chars().count()
        );
    }
    truncate_body(body.to_string(), 4_000)
}

/// Heuristic for provider payloads that are probably encoded binary blobs.
fn looks_like_base64(text: &str) -> bool {
    text.len() >= 256
        && text.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(b, b'+' | b'/' | b'=' | b'-' | b'_' | b'\r' | b'\n')
        })
}

/// Error type shared by Anthropic language-model and metadata calls.
///
/// The variants are deliberately coarse because retry decisions only need to
/// distinguish network/transient API failures from permanent request, decode,
/// or media-reference problems.
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
            // Anthropic rate limits and server failures are retryable through
            // the shared provider retry policy. Other HTTP failures usually
            // indicate invalid credentials, request shape, model, or media.
            Self::Api { status, .. } if *status == 429 || (500..=599).contains(status) => {
                ErrorClass::ServerTransient
            }
            Self::Transport(_) => ErrorClass::Network,
            Self::Api { .. } | Self::Decode(_) | Self::Reference(_) => ErrorClass::Permanent,
        }
    }
}
