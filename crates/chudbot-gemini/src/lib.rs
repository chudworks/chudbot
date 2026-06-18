//! Google AI provider crate for Chudbot.
//!
//! The crate exposes one [`GeminiClient`] that implements Chudbot's language,
//! image, and video generation traits in the modality-specific submodules:
//! `llm`, `image`, and `video`. This root module owns the shared Google
//! transport concerns: API-key authentication, retry classification, response
//! decoding, inline media encoding, and redacted debug logging.
//!
//! The implementation talks directly to the Gemini Developer API shape under
//! `generativelanguage.googleapis.com`. Callers configure the runtime provider
//! name outside this crate; the client keeps it so traces and usage records can
//! identify the configured Chudbot provider instead of only the vendor.

mod image;
mod llm;
mod video;

use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use chudbot_api::{MediaRef, ProviderName};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

/// Gemini-specific language-model options accepted through agent config.
pub use llm::GeminiOptions;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Shared Google AI API client.
///
/// `GeminiClient` is intentionally small: it carries the HTTP client, endpoint
/// base, API key, and Chudbot provider name, then each trait implementation
/// builds the vendor-specific request body it needs.
#[derive(Debug)]
pub struct GeminiClient {
    /// Reused across LLM/image/video calls so connection pooling is shared.
    http: reqwest::Client,
    api_key: String,
    /// Base URL without the per-method path, usually the Google v1beta root.
    base_url: String,
    /// Configured provider identity used in traces, usage, and continuation data.
    provider_name: ProviderName,
}

impl GeminiClient {
    /// Construct a client for the configured provider name and Gemini API key.
    ///
    /// The default base URL targets Google's public Gemini Developer API. Tests
    /// and gateway deployments can replace it with [`Self::with_base_url`].
    pub fn new(provider_name: ProviderName, api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            provider_name,
        }
    }

    /// Override the API base URL.
    ///
    /// This is primarily for local tests and API-compatible gateways; callers
    /// should pass the root prefix, not a full method endpoint.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Borrow the underlying HTTP client for modality-specific operations.
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Borrow the API key for raw endpoints not handled by the shared JSON helpers.
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

    pub(crate) async fn post_json<T>(
        &self,
        endpoint: &str,
        body: &Value,
        label: &str,
    ) -> Result<T, GeminiError>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.post_json_with_policy(endpoint, body, RetryPolicy::default(), label)
            .await
    }

    /// POST a JSON request, applying Chudbot retry policy and response decoding.
    ///
    /// `endpoint` is the path suffix under [`Self::base_url`]. `label` is used
    /// only for retry diagnostics, so callers keep labels stable and compact.
    pub(crate) async fn post_json_with_policy<T>(
        &self,
        endpoint: &str,
        body: &Value,
        policy: RetryPolicy,
        label: &str,
    ) -> Result<T, GeminiError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        log_json_request(&self.provider_name, endpoint, body);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            "sending Gemini JSON request"
        );
        with_retry(policy, label, || {
            // Request builders are single-use, so each retry attempt must build
            // a fresh request inside the retry closure.
            let request = self
                .http
                .post(&url)
                .header("x-goog-api-key", &self.api_key)
                .json(body);
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "Gemini request transport error"
                    );
                    GeminiError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received Gemini response"
                );
                decode_response(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }

    pub(crate) async fn post_json_stream(
        &self,
        endpoint: &str,
        body: &Value,
        label: &str,
    ) -> Result<reqwest::Response, GeminiError> {
        let url = format!("{}{}", self.base_url, endpoint);
        log_json_request(&self.provider_name, endpoint, body);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            "sending Gemini streaming JSON request"
        );
        with_retry(RetryPolicy::default(), label, || {
            let request = self
                .http
                .post(&url)
                .header("x-goog-api-key", &self.api_key)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .json(body);
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "Gemini streaming request transport error"
                    );
                    GeminiError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received Gemini streaming response"
                );
                ensure_stream_success(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }

    /// GET a JSON endpoint, applying the default retry policy and shared decode path.
    pub(crate) async fn get_json<T>(&self, endpoint: &str, label: &str) -> Result<T, GeminiError>
    where
        T: for<'de> Deserialize<'de>,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            "sending Gemini JSON GET request"
        );
        with_retry(RetryPolicy::default(), label, || {
            // Keep request construction here for the same reason as POST:
            // retries need a new builder for each attempt.
            let request = self.http.get(&url).header("x-goog-api-key", &self.api_key);
            async move {
                let resp = request.send().await.map_err(|e| {
                    tracing::warn!(
                        provider = %self.provider_name,
                        endpoint = %endpoint,
                        error = %e,
                        "Gemini GET request transport error"
                    );
                    GeminiError::Transport(e.to_string())
                })?;
                tracing::debug!(
                    provider = %self.provider_name,
                    endpoint = %endpoint,
                    status = %resp.status(),
                    "received Gemini GET response"
                );
                decode_response(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }
}

/// Decode a Google JSON response into the caller's expected shape.
///
/// The response body is first parsed as [`serde_json::Value`] so debug logging
/// can redact inline media before the final typed deserialization step.
pub(crate) async fn decode_response<T>(
    resp: reqwest::Response,
    provider: &ProviderName,
    endpoint: &str,
) -> Result<T, GeminiError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = resp.status();
    let body = resp.text().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to read Gemini response body"
        );
        GeminiError::Decode(e.to_string())
    })?;
    if !status.is_success() {
        // Error bodies are not guaranteed to match the success schema, so keep
        // them as bounded text while still logging the redacted payload at DEBUG.
        log_text_response(provider, endpoint, status.as_u16(), &body);
        let body = truncate_body(body, 600);
        tracing::warn!(
            status = status.as_u16(),
            body_chars = body.chars().count(),
            "Gemini API returned non-success status"
        );
        return Err(GeminiError::Api {
            status: status.as_u16(),
            body,
        });
    }

    // Parse once to Value for logging, then deserialize from that same value so
    // the debug payload and typed response describe the same body.
    let value = serde_json::from_str::<Value>(&body).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode Gemini response"
        );
        GeminiError::Decode(e.to_string())
    })?;
    log_json_response(provider, endpoint, status.as_u16(), &value);
    serde_json::from_value(value).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode Gemini response shape"
        );
        GeminiError::Decode(e.to_string())
    })
}

async fn ensure_stream_success(
    resp: reqwest::Response,
    provider: &ProviderName,
    endpoint: &str,
) -> Result<reqwest::Response, GeminiError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }

    let body = resp.text().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to read Gemini streaming error body"
        );
        GeminiError::Decode(e.to_string())
    })?;
    log_text_response(provider, endpoint, status.as_u16(), &body);
    let body = truncate_body(body, 600);
    tracing::warn!(
        status = status.as_u16(),
        body_chars = body.chars().count(),
        "Gemini streaming API returned non-success status"
    );
    Err(GeminiError::Api {
        status: status.as_u16(),
        body,
    })
}

/// Remove null-valued top-level object fields before sending provider JSON.
///
/// The provider request builders use `json!` with optional values for clarity.
/// This helper keeps omitted config knobs out of the wire payload without
/// recursively altering nested raw provider options.
pub(crate) fn json_strip_nulls(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        map.retain(|_, v| !v.is_null());
    }
    value
}

/// Load a Chudbot media reference and encode it as Gemini `inlineData`.
///
/// Modality-specific callers validate whether the MIME type is acceptable for
/// their endpoint. This helper only performs the shared byte loading and base64
/// conversion needed by Gemini request parts.
pub(crate) async fn inline_media(media: &dyn MediaRef) -> Result<Value, GeminiError> {
    let mime_type = media.mime_type();
    let loaded = media.load().await.map_err(|load_error| {
        tracing::warn!(
            uri = %media.uri(),
            category = ?media.category(),
            error = %load_error,
            "failed to load media for Gemini inline input"
        );
        GeminiError::Reference(format!(
            "media `{}` could not be loaded for Gemini inline input ({load_error})",
            media.uri()
        ))
    })?;
    Ok(serde_json::json!({
        "inlineData": {
            "mimeType": mime_type,
            "data": base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                &loaded.bytes,
            ),
        }
    }))
}

/// Read either casing for Google fields observed in raw JSON payloads.
pub(crate) fn get_field<'a>(value: &'a Value, camel: &str, snake: &str) -> Option<&'a Value> {
    value.get(camel).or_else(|| value.get(snake))
}

/// Bound provider error/debug bodies before storing or logging them.
pub(crate) fn truncate_body(mut body: String, max: usize) -> String {
    if body.len() > max {
        body.truncate(max);
    }
    body
}

/// Log request JSON only when DEBUG is enabled so redaction work stays cold.
fn log_json_request(provider: &ProviderName, endpoint: &str, body: &Value) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = stringify_redacted_json(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            request = %body,
            "Gemini JSON request payload",
        );
    }
}

/// Log successful JSON response bodies with inline data redacted.
fn log_json_response(provider: &ProviderName, endpoint: &str, status: u16, body: &Value) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = stringify_redacted_json(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            status,
            response = %body,
            "Gemini JSON response payload",
        );
    }
}

/// Log non-JSON or error response text with a bounded body.
fn log_text_response(provider: &ProviderName, endpoint: &str, status: u16, body: &str) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = redact_text_body(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            status,
            response = %body,
            "Gemini non-JSON response payload",
        );
    }
}

/// Serialize a JSON value after replacing large inline payloads.
fn stringify_redacted_json(value: &Value) -> String {
    serde_json::to_string(&redact_json(value, None))
        .unwrap_or_else(|_| "[unserializable JSON payload]".to_string())
}

/// Recursively redact fields that are likely to contain binary media.
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

/// Redact Gemini inline `data` strings while preserving ordinary text parts.
fn redact_string(key: Option<&str>, text: &str) -> String {
    let possible_payload_key = matches!(key, Some("data"));
    if possible_payload_key && looks_like_base64(text) {
        return format!(
            "[redacted base64-like string; chars={}]",
            text.chars().count()
        );
    }

    text.to_string()
}

/// Bound plain-text debug bodies.
fn redact_text_body(body: &str) -> String {
    truncate_body(body.to_string(), 600)
}

/// Heuristic used only for log redaction; false positives are safer than leaks.
fn looks_like_base64(text: &str) -> bool {
    text.len() >= 128
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=' | b'-' | b'_'))
}

/// Errors produced by the Google AI provider.
///
/// The variants double as retry-classification inputs. Transport errors and
/// transient API statuses can be retried by the shared policy; malformed
/// requests, media failures, and response-shape mismatches are treated as
/// permanent.
#[derive(Debug, Error)]
pub enum GeminiError {
    /// Transport-level HTTP failure before a response body was available.
    #[error("Gemini transport error: {0}")]
    Transport(String),
    /// Gemini returned a non-success HTTP status.
    #[error("Gemini API error {status}: {body}")]
    Api {
        /// HTTP status code returned by the provider.
        status: u16,
        /// Bounded response body for diagnostics.
        body: String,
    },
    /// Response decoding or response-shape extraction failed.
    #[error("Gemini decode error: {0}")]
    Decode(String),
    /// Media reference could not be loaded or used for the requested modality.
    #[error("Gemini media reference error: {0}")]
    Reference(String),
}

impl ClassifyError for GeminiError {
    fn error_class(&self) -> ErrorClass {
        match self {
            GeminiError::Transport(_) => ErrorClass::Network,
            GeminiError::Api { status, .. } if *status == 429 || *status >= 500 => {
                ErrorClass::ServerTransient
            }
            GeminiError::Api { .. } | GeminiError::Decode(_) | GeminiError::Reference(_) => {
                ErrorClass::Permanent
            }
        }
    }
}
