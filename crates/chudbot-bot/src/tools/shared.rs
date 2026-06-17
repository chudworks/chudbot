//! Shared tool input parsing and media-reference resolution.

use super::*;

/// Error type for concrete bot tool implementations before executor wrapping.
#[derive(Debug, Error)]
pub(crate) enum BotToolError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("platform error: {0}")]
    Platform(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("rate limit: {0}")]
    RateLimit(String),
    #[error("generator error: {0}")]
    Generator(String),
    #[error("media error: {0}")]
    Media(String),
}

pub(crate) async fn resolve_tool_media_arg<M>(
    media_store: &M,
    category: MediaCategory,
    value: &serde_json::Value,
) -> Result<chudbot_api::BoxedMediaRef, BotToolError>
where
    M: MediaStore,
{
    let text = value.as_str().ok_or_else(|| {
        BotToolError::InvalidInput("media references must be strings".to_string())
    })?;
    if text.starts_with("http://") || text.starts_with("https://") {
        return Ok(UrlMediaRef::new(category, text, "application/octet-stream").boxed());
    }
    media_store
        .media_from_uri(&MediaUri::new(text))
        .await
        .map_err(|error| BotToolError::Media(error.to_string()))
}

pub(crate) fn tool_required_string(
    input: &serde_json::Value,
    field: &str,
) -> Result<String, BotToolError> {
    input
        .get(field)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| BotToolError::InvalidInput(format!("`{field}` is required")))
}

pub(crate) fn tool_optional_string(
    input: &serde_json::Value,
    field: &str,
) -> Result<Option<String>, BotToolError> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    value
        .as_str()
        .map(str::to_string)
        .map(Some)
        .ok_or_else(|| BotToolError::InvalidInput(format!("`{field}` must be a string")))
}

pub(crate) fn tool_optional_string_list(
    input: &serde_json::Value,
    field: &str,
) -> Result<Option<Vec<String>>, BotToolError> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    if let Some(text) = value.as_str() {
        return Ok(Some(vec![text.to_string()]));
    }
    let Some(values) = value.as_array() else {
        return Err(BotToolError::InvalidInput(format!(
            "`{field}` must be a string or array of strings"
        )));
    };
    values
        .iter()
        .map(|value| {
            value.as_str().map(str::to_string).ok_or_else(|| {
                BotToolError::InvalidInput(format!("`{field}` must only contain strings"))
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

pub(crate) fn tool_optional_u8(
    input: &serde_json::Value,
    field: &str,
) -> Result<Option<u8>, BotToolError> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    let Some(value) = value.as_u64() else {
        return Err(BotToolError::InvalidInput(format!(
            "`{field}` must be an integer"
        )));
    };
    u8::try_from(value)
        .map(Some)
        .map_err(|_| BotToolError::InvalidInput(format!("`{field}` is too large")))
}

pub(crate) fn tool_optional_u8_bounded(
    input: &serde_json::Value,
    field: &str,
    max: u8,
) -> Result<Option<u8>, BotToolError> {
    let value = tool_optional_u8(input, field)?;
    if let Some(value) = value
        && value > max
    {
        return Err(BotToolError::InvalidInput(format!(
            "`{field}` must be at most {max}"
        )));
    }
    Ok(value)
}
