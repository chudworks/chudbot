//! Stored-media access tools: `read`, `stat`, `public_url`, and `attach`.
//!
//! These tools accept model-facing `file://...` media URIs, not filesystem
//! paths or arbitrary network URLs. The prefix check keeps the input shape
//! scoped to stored media, and `MediaStore::media_from_uri` remains the
//! authority for validating supported storage prefixes, existence, metadata,
//! and access handles.

use super::*;

/// Describe the `read` tool exposed to model providers.
///
/// `read` is intentionally narrower than a general media fetcher: it accepts
/// only stored images that can be represented in a model transcript. The tool
/// returns JSON metadata plus a media handle for the next provider request; it
/// never serializes file bytes into the tool result.
pub(crate) fn read_asset_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Read a stored Chudbot image asset by file:// URI. Only verified image assets already in media storage are accepted; videos, audio, PDFs, unknown MIME types, and arbitrary filesystem paths are rejected. The tool returns metadata and makes the image visible to the next model step, but never returns raw bytes.".to_string(),
        input_schema: asset_uri_tool_schema(),
    }
}

/// Describe the `stat` tool exposed to model providers.
///
/// `stat` checks whether a stored media URI resolves and reports metadata when
/// it does. It does not require an image MIME type because callers use it to
/// inspect any stored media category without loading bytes.
pub(crate) fn stat_asset_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Validate a stored Chudbot media URI and return whether it exists with MIME type and size metadata. This only checks media storage; it does not read or return file bytes.".to_string(),
        input_schema: asset_uri_tool_schema(),
    }
}

/// Describe the `public_url` tool exposed to model providers.
///
/// `public_url` resolves the same stored-media handle into a configured public
/// URL for media categories that are safe to expose outside the bot runtime. A
/// missing public URL is shaped as an unavailable JSON result, not as a raw file
/// read or attachment.
pub(crate) fn public_url_asset_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Resolve a supported stored Chudbot media URI to its configured public URL when one is available. Images, videos, audio, and avatars are supported; unknown/non-media MIME types are rejected. This only returns metadata and a URL; it does not read or return file bytes.".to_string(),
        input_schema: asset_uri_tool_schema(),
    }
}

/// Describe the `attach` tool exposed to model providers.
///
/// `attach` validates an existing stored image and marks its URI for final
/// platform delivery. It does not expose the image to the next model step; the
/// send path later resolves the URI, deduplicates it with generated media, and
/// loads bytes only when preparing the platform reply.
pub(crate) fn attach_asset_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Attach an existing stored Chudbot image asset to the final platform reply. Only verified image assets already in media storage are accepted; videos, audio, PDFs, unknown MIME types, public URLs, and arbitrary filesystem paths are rejected. The tool queues the image for final delivery and never returns raw bytes.".to_string(),
        input_schema: asset_uri_tool_schema(),
    }
}

/// Shared JSON schema for tools that accept one stored media URI.
///
/// Tool implementations still perform the security checks; the schema exists
/// to guide model arguments and reject unrelated JSON fields before execution.
pub(crate) fn asset_uri_tool_schema() -> ToolInputSchema {
    ToolInputSchema::object([ToolInputField::required(
        "uri",
        ToolInputValueSchema::string().description(
            "A stored Chudbot file:// media URI such as file://images/abc.jpg, file://videos/abc.mp4, file://audio/abc.ogg, or file://avatars/abc.png. Do not pass local filesystem paths or public URLs.",
        ),
    )])
}

#[tracing::instrument(
    name = "tool.media_access.read",
    skip_all,
    fields(tool_call = %call.id)
)]
pub(crate) async fn read_asset<M>(
    media_store: &M,
    call: ClientToolCall,
) -> Result<ClientToolOutput, BotToolError>
where
    M: MediaStore,
{
    let uri = media_uri_from_tool_input(&call.input)?;
    // The store lookup is the trust boundary between a syntactic file:// URI
    // and a real, supported media object.
    let media = media_store
        .media_from_uri(&uri)
        .await
        .map_err(|error| BotToolError::Media(error.to_string()))?;
    if !model_transcript_supports_media(media.as_ref()) {
        tracing::warn!(
            uri = %media.uri(),
            category = ?media.category(),
            mime_type = %media.mime_type(),
            "read rejected unsupported media asset"
        );
        return Err(BotToolError::InvalidInput(format!(
            "`read` only supports stored image assets with supported image MIME types; `{}` resolved as category `{:?}` with MIME type `{}`",
            media.uri(),
            media.category(),
            media.mime_type()
        )));
    }

    // The JSON result is for traceability and model instructions; the media
    // vector is what actually makes the image available to the provider.
    let value = media_access_metadata_json(
        media.as_ref(),
        serde_json::json!({
            "exists": true,
            "visible_to_model": true,
            "content": {
                "kind": "image",
                "delivery": "attached_to_next_model_step",
                "bytes_returned": false
            }
        }),
    );
    tracing::info!(
        uri = %media.uri(),
        mime_type = %media.mime_type(),
        size_bytes = media.size_bytes(),
        "read exposed stored image asset to model"
    );
    Ok(ClientToolOutput {
        result: ClientToolResultContent::Json {
            value: value.clone(),
        },
        media: vec![media],
        is_error: false,
        trace_response: value,
        usage: Vec::new(),
    })
}

#[tracing::instrument(
    name = "tool.media_access.stat",
    skip_all,
    fields(tool_call = %call.id)
)]
pub(crate) async fn stat_asset<M>(
    media_store: &M,
    call: ClientToolCall,
) -> Result<ClientToolOutput, BotToolError>
where
    M: MediaStore,
{
    let uri = media_uri_from_tool_input(&call.input)?;
    // Missing media is a successful stat result with `exists: false`, which
    // lets the model recover without turning an inspection miss into a tool
    // execution failure.
    let value = match media_store.media_from_uri(&uri).await {
        Ok(media) => media_access_metadata_json(
            media.as_ref(),
            serde_json::json!({
                "exists": true,
            }),
        ),
        Err(error) => serde_json::json!({
            "uri": uri.as_str(),
            "exists": false,
            "error": error.to_string(),
        }),
    };
    Ok(ClientToolOutput {
        result: ClientToolResultContent::Json {
            value: value.clone(),
        },
        media: Vec::new(),
        is_error: false,
        trace_response: value,
        usage: Vec::new(),
    })
}

#[tracing::instrument(
    name = "tool.media_access.public_url",
    skip_all,
    fields(tool_call = %call.id)
)]
pub(crate) async fn public_url_asset<M>(
    media_store: &M,
    call: ClientToolCall,
) -> Result<ClientToolOutput, BotToolError>
where
    M: MediaStore,
{
    let uri = media_uri_from_tool_input(&call.input)?;
    // Resolve availability as data. Unsupported MIME/category values and
    // unconfigured public URLs both return `available: false`.
    let value = match media_store.media_from_uri(&uri).await {
        Ok(media) if !public_url_supports_media(media.as_ref()) => media_access_metadata_json(
            media.as_ref(),
            serde_json::json!({
                "exists": true,
                "available": false,
                "public_url": null,
                "error": "unsupported media type for public_url",
            }),
        ),
        Ok(media) => match media.public_url().await {
            Ok(public_url) => media_access_metadata_json(
                media.as_ref(),
                serde_json::json!({
                    "exists": true,
                    "available": true,
                    "public_url": public_url.as_str(),
                }),
            ),
            Err(error) => media_access_metadata_json(
                media.as_ref(),
                serde_json::json!({
                    "exists": true,
                    "available": false,
                    "public_url": null,
                    "error": error.to_string(),
                }),
            ),
        },
        Err(error) => serde_json::json!({
            "uri": uri.as_str(),
            "exists": false,
            "available": false,
            "public_url": null,
            "error": error.to_string(),
        }),
    };
    Ok(ClientToolOutput {
        result: ClientToolResultContent::Json {
            value: value.clone(),
        },
        media: Vec::new(),
        is_error: false,
        trace_response: value,
        usage: Vec::new(),
    })
}

#[tracing::instrument(
    name = "tool.media_access.attach",
    skip_all,
    fields(tool_call = %call.id)
)]
pub(crate) async fn attach_asset<M>(
    media_store: &M,
    call: ClientToolCall,
) -> Result<ClientToolOutput, BotToolError>
where
    M: MediaStore,
{
    let uri = media_uri_from_tool_input(&call.input)?;
    // Validate now, but leave byte loading and final attachment sizing to the
    // reply-delivery path that consumes the successful tool trace.
    let media = media_store
        .media_from_uri(&uri)
        .await
        .map_err(|error| BotToolError::Media(error.to_string()))?;
    if !attach_supports_media(media.as_ref()) {
        tracing::warn!(
            uri = %media.uri(),
            category = ?media.category(),
            mime_type = %media.mime_type(),
            "attach rejected unsupported media asset"
        );
        return Err(BotToolError::InvalidInput(format!(
            "`attach` only supports stored image assets; `{}` resolved as category `{:?}` with MIME type `{}`",
            media.uri(),
            media.category(),
            media.mime_type()
        )));
    }

    // `attach` communicates through trace JSON only. Keeping `media` empty
    // prevents provider adapters from treating this as model-visible input.
    let value = media_access_metadata_json(
        media.as_ref(),
        serde_json::json!({
            "exists": true,
            "attached": true,
            "delivery": {
                "platform_reply": "The image will be attached to the final platform reply automatically. Do not paste media URIs, filenames, public URLs, or markdown image links in user-facing text.",
                "deduplication": "If this URI is already queued by generated media or another attach call, it will only be sent once."
            }
        }),
    );
    tracing::info!(
        uri = %media.uri(),
        mime_type = %media.mime_type(),
        size_bytes = media.size_bytes(),
        "queued stored image asset for final reply attachment"
    );
    Ok(ClientToolOutput {
        result: ClientToolResultContent::Json {
            value: value.clone(),
        },
        media: Vec::new(),
        is_error: false,
        trace_response: value,
        usage: Vec::new(),
    })
}

/// Parse the shared `uri` argument and reject non-stored URI forms early.
///
/// This is only a coarse scope check. It prevents public URLs and local path
/// strings from entering the tool flow, while the media store decides whether a
/// `file://...` value names a supported Chudbot media asset.
pub(crate) fn media_uri_from_tool_input(
    input: &serde_json::Value,
) -> Result<MediaUri, BotToolError> {
    let uri = tool_required_string(input, "uri")?;
    if !uri.starts_with("file://") {
        return Err(BotToolError::InvalidInput(
            "`uri` must be a stored file:// media URI".to_string(),
        ));
    }
    Ok(MediaUri::new(uri))
}

/// Build the common JSON result shape for media access tools.
///
/// The base fields identify the resolved stored object. Tool-specific fields
/// such as `exists`, `visible_to_model`, `available`, or `attached` are merged
/// into the same object so `result` and `trace_response` can stay identical.
pub(crate) fn media_access_metadata_json(
    media: &dyn chudbot_api::MediaRef,
    extra: serde_json::Value,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "uri": media.uri().as_str(),
        "category": media.category(),
        "name": media.name(),
        "mime_type": media.mime_type(),
        "size_bytes": media.size_bytes(),
    });
    if let (Some(value), serde_json::Value::Object(extra)) = (value.as_object_mut(), extra) {
        value.extend(extra);
    }
    value
}
