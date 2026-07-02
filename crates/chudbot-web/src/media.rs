use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use chudbot_api::{MediaCategory, MediaStore};

use crate::api::ApiError;
use crate::server::{WebRuntimeTypes, WebState};
use crate::static_files::CACHE_IMMUTABLE;

#[tracing::instrument(name = "web.get_image", skip_all, fields(name = %name))]
pub(crate) async fn get_image<R>(
    State(state): State<WebState<R>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError>
where
    R: WebRuntimeTypes,
{
    load_media_response(&state.media_store, MediaCategory::Image, &name).await
}

#[tracing::instrument(name = "web.get_video", skip_all, fields(name = %name))]
pub(crate) async fn get_video<R>(
    State(state): State<WebState<R>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError>
where
    R: WebRuntimeTypes,
{
    load_media_response(&state.media_store, MediaCategory::Video, &name).await
}

#[tracing::instrument(name = "web.get_audio", skip_all, fields(name = %name))]
pub(crate) async fn get_audio<R>(
    State(state): State<WebState<R>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError>
where
    R: WebRuntimeTypes,
{
    load_media_response(&state.media_store, MediaCategory::Audio, &name).await
}

#[tracing::instrument(name = "web.get_avatar", skip_all, fields(name = %name))]
pub(crate) async fn get_avatar<R>(
    State(state): State<WebState<R>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError>
where
    R: WebRuntimeTypes,
{
    load_media_response(&state.media_store, MediaCategory::Avatar, &name).await
}

#[tracing::instrument(name = "web.get_guild_icon", skip_all, fields(name = %name))]
pub(crate) async fn get_guild_icon<R>(
    State(state): State<WebState<R>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError>
where
    R: WebRuntimeTypes,
{
    load_media_response(&state.media_store, MediaCategory::GuildIcon, &name).await
}

#[tracing::instrument(
    name = "web.load_media",
    skip_all,
    fields(category = ?category, name = %name)
)]
async fn load_media_response<M>(
    media_store: &M,
    category: MediaCategory,
    name: &str,
) -> Result<Response, ApiError>
where
    M: MediaStore,
{
    let media = media_store
        .media_from_name(category, name)
        .await
        .map_err(|error| ApiError::Media(error.to_string()))?;
    let loaded = media
        .load()
        .await
        .map_err(|error| ApiError::Media(error.to_string()))?;
    let content_type = HeaderValue::from_str(loaded.media.mime_type())
        .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream"));
    tracing::debug!(
        mime_type = loaded.media.mime_type(),
        bytes = loaded.bytes.len(),
        uri = %loaded.media.uri(),
        "loaded media response"
    );
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (
                header::CACHE_CONTROL,
                HeaderValue::from_static(CACHE_IMMUTABLE),
            ),
        ],
        loaded.bytes,
    )
        .into_response())
}
