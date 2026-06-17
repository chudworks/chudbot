//! xAI provider crate for chudbot.

mod audio;
mod imagine;
mod llm;

use chudbot_api::ProviderName;
use chudbot_api::retry::{ClassifyError, ErrorClass, RetryPolicy, with_retry};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

pub use llm::XaiOptions;

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";
const X_GROK_CONV_ID_HEADER: &str = "x-grok-conv-id";

/// xAI API client.
#[derive(Debug, Clone)]
pub struct XaiClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    provider_name: ProviderName,
}

impl XaiClient {
    /// Construct from a configured provider name and xAI API key.
    pub fn new(provider_name: ProviderName, api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            provider_name,
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
        let grok_conv_id = prompt_cache_key_header_value(body).map(str::to_string);
        log_json_request(&self.provider_name, endpoint, body);
        tracing::debug!(
            provider = %self.provider_name,
            endpoint = %endpoint,
            base_url = %self.base_url,
            x_grok_conv_id = grok_conv_id.is_some(),
            "sending xAI JSON request"
        );
        with_retry(policy, label, || {
            let mut request = self.http.post(&url).bearer_auth(&self.api_key).json(body);
            if let Some(grok_conv_id) = &grok_conv_id {
                request = request.header(X_GROK_CONV_ID_HEADER, grok_conv_id);
            }
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
                decode_response(resp, &self.provider_name, endpoint).await
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
                decode_response(resp, &self.provider_name, endpoint).await
            }
        })
        .await
    }
}

fn prompt_cache_key_header_value(body: &Value) -> Option<&str> {
    body.get("prompt_cache_key")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
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
) -> Result<T, XaiError>
where
    T: for<'de> Deserialize<'de>,
{
    let status = resp.status();
    let body = resp.text().await.map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to read xAI response body"
        );
        XaiError::Decode(e.to_string())
    })?;
    if !status.is_success() {
        log_text_response(provider, endpoint, status.as_u16(), &body);
        let body = truncate_body(body, 600);
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

    let value = serde_json::from_str::<Value>(&body).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode xAI response"
        );
        XaiError::Decode(e.to_string())
    })?;
    log_json_response(provider, endpoint, status.as_u16(), &value);
    serde_json::from_value(value).map_err(|e| {
        tracing::warn!(
            status = status.as_u16(),
            error = %e,
            "failed to decode xAI response shape"
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

fn log_json_request(provider: &ProviderName, endpoint: &str, body: &Value) {
    if tracing::enabled!(tracing::Level::DEBUG) {
        let body = stringify_redacted_json(body);
        tracing::debug!(
            provider = %provider,
            endpoint = %endpoint,
            request = %body,
            "xAI JSON request payload",
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
            "xAI JSON response payload",
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
            "xAI non-JSON response payload",
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::prompt_cache_key_header_value;

    #[test]
    fn prompt_cache_key_header_value_uses_non_empty_request_key() {
        let body = json!({ "prompt_cache_key": "conv-123" });

        assert_eq!(prompt_cache_key_header_value(&body), Some("conv-123"));
    }

    #[test]
    fn prompt_cache_key_header_value_ignores_missing_or_empty_key() {
        assert_eq!(prompt_cache_key_header_value(&json!({})), None);
        assert_eq!(
            prompt_cache_key_header_value(&json!({ "prompt_cache_key": "" })),
            None
        );
    }
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
