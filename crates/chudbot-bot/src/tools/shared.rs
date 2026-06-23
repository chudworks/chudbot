//! Shared tool input parsing and media-reference resolution.
//!
//! Concrete tools keep their JSON schemas and domain validation in their own
//! modules, but use these helpers for common argument decoding and error
//! shaping. The helpers are intentionally small: they convert model-supplied
//! JSON into typed Rust values, preserve accepted strings exactly, and return
//! [`BotToolError`] messages that the runtime executor can expose as ordinary
//! tool execution failures.
//!
//! Tool outputs follow a separate convention: `ClientToolOutput::result` is
//! the model-visible result, `trace_response` is the auditable value stored in
//! the turn trace, and `media` is only an ephemeral list of native media handles
//! for the next model step. Final platform reply attachments are queued from
//! successful delivery-producing tool traces elsewhere, not from this module.

use super::*;

/// Error type for concrete bot tool implementations before executor wrapping.
///
/// The runtime executor stringifies these errors into a
/// `ClientToolExecutorError::execution`, so messages should be concise,
/// actionable, and safe to show in model/tool traces.
#[derive(Debug, Error)]
pub(crate) enum BotToolError {
    /// The model supplied missing, malformed, or unsupported tool input.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// A platform adapter operation failed, such as sending or fetching a message.
    #[error("platform error: {0}")]
    Platform(String),
    /// Persistent bot storage failed while servicing the tool.
    #[error("storage error: {0}")]
    Storage(String),
    /// A tool-specific rate limiter rejected the request.
    #[error("rate limit: {0}")]
    RateLimit(String),
    /// A media/text generation provider failed.
    #[error("generator error: {0}")]
    Generator(String),
    /// Media storage, loading, or URI resolution failed.
    #[error("media error: {0}")]
    Media(String),
}

/// Resolve a model-supplied media argument for generator/transcriber tools.
///
/// HTTP(S) strings are treated as direct provider-visible media URLs and are
/// wrapped as [`UrlMediaRef`] with the caller's category. All other strings are
/// interpreted as media-store URIs and resolved through [`MediaStore`], which is
/// the trust boundary for stored assets. This permissive split is for provider
/// input references only; user-facing media access tools apply stricter
/// stored-media validation before calling the store.
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
    // Direct URLs do not carry store metadata, so preserve the requested media
    // category and use a neutral MIME type until the provider fetches them.
    if text.starts_with("http://") || text.starts_with("https://") {
        return Ok(UrlMediaRef::new(category, text, "application/octet-stream").boxed());
    }
    // Stored references must resolve through the media store; callers should not
    // guess at filesystem paths, MIME types, or byte availability here.
    media_store
        .media_from_uri(&MediaUri::new(text))
        .await
        .map_err(|error| BotToolError::Media(error.to_string()))
}

/// Read a required non-empty string field from tool input.
///
/// Whitespace-only values are rejected, but otherwise the original string is
/// returned unchanged so downstream tools keep prompts, model names, and other
/// user-authored text exactly as supplied.
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

/// Read an optional string field from tool input.
///
/// Missing fields become `None`. Present fields must be JSON strings, including
/// empty strings when a particular tool wants to interpret them itself.
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

/// Read an optional string-or-string-array field from tool input.
///
/// Several model-facing schemas accept both a scalar convenience form and an
/// array form. This helper normalizes both into `Vec<String>` while preserving
/// order and rejecting mixed-type arrays.
pub(crate) fn tool_optional_string_list(
    input: &serde_json::Value,
    field: &str,
) -> Result<Option<Vec<String>>, BotToolError> {
    let Some(value) = input.get(field) else {
        return Ok(None);
    };
    // Accept a single string as shorthand for a one-item list; this keeps
    // aliases such as `keyterm` ergonomic without adding tool-specific parsing.
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

/// Read an optional unsigned 8-bit integer from tool input.
///
/// JSON numbers must be non-negative integers. Floats, negative numbers, and
/// integers larger than `u8::MAX` are rejected before the tool sees them.
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

/// Read an optional `u8` and enforce an inclusive upper bound.
///
/// The lower bound remains the JSON unsigned-integer rule from
/// [`tool_optional_u8`]; callers that need a positive value should add their own
/// minimum check so the error can name the domain-specific constraint.
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
