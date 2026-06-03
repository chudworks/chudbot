//! OpenAI provider crate for the chudbot 2.0 API.

mod image;
mod llm;

use chudbot_api::ProviderName;
use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

pub use llm::OpenAiOptions;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI API client.
#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    provider_name: ProviderName,
}

impl OpenAiClient {
    /// Construct from an OpenAI API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            provider_name: ProviderName::new("openai"),
        }
    }

    /// Override the base URL. Useful for local tests or gateway deployments.
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
}

pub(crate) fn json_strip_nulls(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        map.retain(|_, v| !v.is_null());
    }
    value
}

pub(crate) async fn decode_response<T>(
    resp: reqwest::Response,
    provider: &ProviderName,
    endpoint: &str,
) -> Result<T, OpenAiError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = resp.status();
    let body = resp.text().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to read OpenAI response body"
        );
        OpenAiError::Decode(e.to_string())
    })?;
    if !status.is_success() {
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
