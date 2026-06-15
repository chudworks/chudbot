//! Google Gemini API provider crate for the chudbot 2.0 API.

mod image;
mod llm;
mod video;

use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use chudbot_api::{MediaRef, ProviderName};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

pub use llm::GeminiOptions;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Google Gemini API client.
#[derive(Debug, Clone)]
pub struct GeminiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    provider_name: ProviderName,
}

impl GeminiClient {
    /// Construct from a Gemini API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            provider_name: ProviderName::new("gemini"),
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
    ) -> Result<T, GeminiError>
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

pub(crate) fn json_strip_nulls(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        map.retain(|_, v| !v.is_null());
    }
    value
}

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

pub(crate) fn get_field<'a>(value: &'a Value, camel: &str, snake: &str) -> Option<&'a Value> {
    value.get(camel).or_else(|| value.get(snake))
}

pub(crate) fn truncate_body(mut body: String, max: usize) -> String {
    if body.len() > max {
        body.truncate(max);
    }
    body
}

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
    let possible_payload_key = matches!(key, Some("data"));
    if possible_payload_key && looks_like_base64(text) {
        return format!(
            "[redacted base64-like string; chars={}]",
            text.chars().count()
        );
    }

    text.to_string()
}

fn redact_text_body(body: &str) -> String {
    truncate_body(body.to_string(), 600)
}

fn looks_like_base64(text: &str) -> bool {
    text.len() >= 128
        && text
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=' | b'-' | b'_'))
}

/// Gemini provider errors.
#[derive(Debug, Error)]
pub enum GeminiError {
    /// Transport-level HTTP failure.
    #[error("Gemini transport error: {0}")]
    Transport(String),
    /// Gemini API returned a non-success status.
    #[error("Gemini API error {status}: {body}")]
    Api {
        /// HTTP status.
        status: u16,
        /// Truncated response body.
        body: String,
    },
    /// Response/request decoding failed.
    #[error("Gemini decode error: {0}")]
    Decode(String),
    /// Media reference could not be resolved for a provider request.
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
