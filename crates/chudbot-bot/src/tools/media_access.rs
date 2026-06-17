//! Stored-media access tools: read, stat, public URL, and attach.

use super::*;

pub(crate) fn read_asset_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Read a stored Chudbot image asset by file:// URI. Only verified image assets already in media storage are accepted; videos, audio, PDFs, unknown MIME types, and arbitrary filesystem paths are rejected. The tool returns metadata and makes the image visible to the next model step, but never returns raw bytes.".to_string(),
        input_schema: asset_uri_tool_schema(),
    }
}

pub(crate) fn stat_asset_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Validate a stored Chudbot media URI and return whether it exists with MIME type and size metadata. This only checks media storage; it does not read or return file bytes.".to_string(),
        input_schema: asset_uri_tool_schema(),
    }
}

pub(crate) fn public_url_asset_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Resolve a supported stored Chudbot media URI to its configured public URL when one is available. Images, videos, audio, and avatars are supported; unknown/non-media MIME types are rejected. This only returns metadata and a URL; it does not read or return file bytes.".to_string(),
        input_schema: asset_uri_tool_schema(),
    }
}

pub(crate) fn attach_asset_spec() -> ClientToolSpec {
    ClientToolSpec {
        description: "Attach an existing stored Chudbot image asset to the final platform reply. Only verified image assets already in media storage are accepted; videos, audio, PDFs, unknown MIME types, public URLs, and arbitrary filesystem paths are rejected. The tool queues the image for final delivery and never returns raw bytes.".to_string(),
        input_schema: asset_uri_tool_schema(),
    }
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
    if let (Some(value), Some(extra)) = (value.as_object_mut(), extra.as_object()) {
        for (key, extra_value) in extra {
            value.insert(key.clone(), extra_value.clone());
        }
    }
    value
}

pub(crate) fn asset_uri_tool_schema() -> ToolInputSchema {
    ToolInputSchema::new(serde_json::json!({
        "type": "object",
        "required": ["uri"],
        "properties": {
            "uri": {
                "type": "string",
                "description": "A stored Chudbot file:// media URI such as file://images/abc.jpg, file://videos/abc.mp4, file://audio/abc.ogg, or file://avatars/abc.png. Do not pass local filesystem paths or public URLs."
            }
        },
        "additionalProperties": false
    }))
}
