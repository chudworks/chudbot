//! OpenAI provider crate for chudbot.
//!
//! This crate adapts OpenAI's public APIs to the provider-neutral contracts in
//! `chudbot-api`. The language-model implementation uses the Responses API,
//! the image implementation uses the OpenAI image endpoints, and the shared
//! client in this module owns transport, retry classification, request/response
//! logging, and local cost-estimation tables.

mod image;
mod llm;
mod pricing;

use std::collections::BTreeMap;

use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use chudbot_api::{ModelId, ProviderName};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

/// Provider-specific Responses API options decoded from an agent model spec.
pub use llm::OpenAiOptions;
/// Cost-estimation override types accepted by the OpenAI runtime service.
pub use pricing::{OpenAiImagePricing, OpenAiTokenPricing};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Shared OpenAI API client used by the LLM and image providers.
///
/// The client is intentionally small: it keeps the configured provider name
/// for trace attribution, the base URL for gateway/test deployments, and the
/// pricing tables needed to attach local cost estimates to usage records.
#[derive(Debug)]
pub struct OpenAiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    provider_name: ProviderName,
    pricing: pricing::OpenAiPricing,
}

impl OpenAiClient {
    /// Construct a client for a configured provider name and OpenAI API key.
    ///
    /// The provider name is the runtime service key from config, not the model
    /// id. It is carried through logs, continuations, and usage records.
    pub fn new(provider_name: ProviderName, api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            provider_name,
            pricing: pricing::OpenAiPricing::default(),
        }
    }

    /// Override the API base URL.
    ///
    /// The URL should include the API version prefix, for example
    /// `https://api.openai.com/v1`. Tests and compatible gateways use this to
    /// route the same request builders to non-production endpoints.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Override or add text-token pricing entries used for local cost estimates.
    pub fn with_token_pricing(mut self, pricing: BTreeMap<ModelId, OpenAiTokenPricing>) -> Self {
        self.pricing.apply_token_overrides(pricing);
        self
    }

    /// Override or add image-token pricing entries used for local cost estimates.
    pub fn with_image_pricing(mut self, pricing: BTreeMap<ModelId, OpenAiImagePricing>) -> Self {
        self.pricing.apply_image_overrides(pricing);
        self
    }

    /// Borrow the underlying HTTP client for endpoint-specific request builders.
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Borrow the API key for request builders that cannot use `post_json`.
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Borrow the configured API base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub(crate) fn provider_name(&self) -> &ProviderName {
        &self.provider_name
    }

    pub(crate) fn pricing(&self) -> &pricing::OpenAiPricing {
        &self.pricing
    }

    pub(crate) async fn post_json<T>(
        &self,
        endpoint: &str,
        body: &Value,
        label: &str,
    ) -> Result<T, OpenAiError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        log_json_request(&self.provider_name, endpoint, body);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            "sending OpenAI JSON request"
        );
        // Request builders are consumed by `send`, so construct a fresh one for
        // every retry attempt while keeping response decoding in one shared path.
        with_retry(RetryPolicy::default(), label, || {
            let request = self.http.post(&url).bearer_auth(&self.api_key).json(body);
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "OpenAI request transport error"
                    );
                    OpenAiError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received OpenAI response"
                );
                decode_response(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }

    pub(crate) async fn get_json<T>(&self, endpoint: &str, label: &str) -> Result<T, OpenAiError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            "sending OpenAI JSON GET request"
        );
        // Model metadata fetches use the same retry and decode behavior as JSON
        // writes so transport failures classify consistently across endpoints.
        with_retry(RetryPolicy::default(), label, || {
            let request = self.http.get(&url).bearer_auth(&self.api_key);
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "OpenAI GET request transport error"
                    );
                    OpenAiError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received OpenAI GET response"
                );
                decode_response(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }
}

/// Remove top-level null fields before sending JSON to OpenAI.
///
/// Request builders use `then_some(...).flatten()` to express optional API
/// fields. Stripping only the top-level object preserves nested payloads while
/// keeping provider requests free of explicit JSON nulls.
pub(crate) fn json_strip_nulls(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        map.retain(|_, v| !v.is_null());
    }
    value
}

/// Decode a JSON response after status handling and redacted debug logging.
pub(crate) async fn decode_response<T>(
    resp: reqwest::Response,
    provider: &ProviderName,
    endpoint: &str,
) -> Result<T, OpenAiError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = resp.status();
    // Read the body once so the error path, debug logs, and typed decode all
    // operate on the same bytes from reqwest.
    let body = resp.text().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to read OpenAI response body"
        );
        OpenAiError::Decode(e.to_string())
    })?;
    if !status.is_success() {
        // Keep detailed payloads available at DEBUG, but return a short error
        // body because API errors are stored and surfaced outside this crate.
        log_text_response(provider, endpoint, status.as_u16(), &body);
        let body = truncate_body(body, 600);
        tracing::warn!(
            status = status.as_u16(),
            body_chars = body.chars().count(),
            "OpenAI API returned non-success status"
        );
        return Err(OpenAiError::Api {
            status: status.as_u16(),
            body,
        });
    }

    // Decode through `Value` first so debug logs can redact the raw provider
    // shape before the endpoint-specific response type consumes it.
    let value = serde_json::from_str::<Value>(&body).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode OpenAI response"
        );
        OpenAiError::Decode(e.to_string())
    })?;
    log_json_response(provider, endpoint, status.as_u16(), &value);
    serde_json::from_value(value).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode OpenAI response shape"
        );
        OpenAiError::Decode(e.to_string())
    })
}

pub(crate) fn truncate_body(mut body: String, max: usize) -> String {
    if body.len() > max {
        body.truncate(max);
    }
    body
}

/// Log an outbound JSON request with media and encrypted payloads redacted.
pub(crate) fn log_json_request(provider: &ProviderName, endpoint: &str, body: &Value) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = stringify_redacted_json(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            request = %body,
            "OpenAI JSON request payload",
        );
    }
}

fn log_json_response(provider: &ProviderName, endpoint: &str, status: u16, body: &Value) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = stringify_redacted_json(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            status,
            response = %body,
            "OpenAI JSON response payload",
        );
    }
}

fn log_text_response(provider: &ProviderName, endpoint: &str, status: u16, body: &str) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = redact_text_body(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            status,
            response = %body,
            "OpenAI non-JSON response payload",
        );
    }
}

fn stringify_redacted_json(value: &Value) -> String {
    serde_json::to_string(&redact_json(value, None))
        .unwrap_or_else(|_| "[unserializable JSON payload]".to_string())
}

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

fn redact_string(key: Option<&str>, text: &str) -> String {
    if let Some(redacted) = redact_data_uri(text) {
        return redacted;
    }

    // OpenAI image and reasoning payloads can be very large. The key-aware path
    // catches known response fields, while the `data` heuristic covers API
    // variants without hiding ordinary text fields.
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

fn redact_text_body(body: &str) -> String {
    if looks_like_base64(body) {
        return format!(
            "[redacted base64-like body; chars={}]",
            body.chars().count()
        );
    }
    truncate_body(body.to_string(), 4_000)
}

fn looks_like_base64(text: &str) -> bool {
    text.len() >= 256
        && text.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(b, b'+' | b'/' | b'=' | b'-' | b'_' | b'\r' | b'\n')
        })
}

/// OpenAI provider error.
#[derive(Debug, Error)]
pub enum OpenAiError {
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
    /// Media reference could not be sent to OpenAI.
    #[error("media reference error: {0}")]
    Reference(String),
}

impl ClassifyError for OpenAiError {
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
