use std::path::{Component, Path as FsPath, PathBuf};

use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use moka::future::Cache;
use tower_http::set_header::SetResponseHeaderLayer;

use crate::server::{WebRuntimeTypes, WebState};

pub(crate) const CACHE_IMMUTABLE: &str = "public, max-age=31536000, immutable";
pub(crate) const CACHE_NO_CACHE: &str = "no-cache, must-revalidate";
pub(crate) const CACHE_NO_STORE: &str = "no-store";
const FRONTEND_STATIC_CACHE_MAX_BYTES: u64 = 64 * 1024 * 1024;
const ROBOTS_TXT: &str = "\
# This host serves unauthenticated, UUID-gated conversation traces.
# Nothing here may be indexed, archived, or used for model training.
User-agent: *
Disallow: /

User-agent: GPTBot
Disallow: /

User-agent: OAI-SearchBot
Disallow: /

User-agent: ChatGPT-User
Disallow: /

User-agent: Google-Extended
Disallow: /

User-agent: anthropic-ai
Disallow: /

User-agent: ClaudeBot
Disallow: /

User-agent: Claude-Web
Disallow: /

User-agent: CCBot
Disallow: /

User-agent: PerplexityBot
Disallow: /

User-agent: Applebot-Extended
Disallow: /

User-agent: Bytespider
Disallow: /

User-agent: meta-externalagent
Disallow: /
";

#[derive(Clone)]
pub(crate) struct StaticFileCache {
    files: Cache<PathBuf, Bytes>,
}

impl StaticFileCache {
    pub(crate) fn new() -> Self {
        let files = Cache::builder()
            .name("frontend-static-files")
            .weigher(|_path, bytes: &Bytes| static_file_weight(bytes))
            .max_capacity(FRONTEND_STATIC_CACHE_MAX_BYTES)
            .build();
        Self { files }
    }

    pub(crate) async fn load(&self, path: PathBuf) -> Option<Bytes> {
        let load_path = path.clone();
        self.files
            .optionally_get_with(path, read_static_file(load_path))
            .await
    }
}

impl std::fmt::Debug for StaticFileCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticFileCache")
            .field("entry_count", &self.files.entry_count())
            .field("weighted_size", &self.files.weighted_size())
            .finish()
    }
}

pub(crate) async fn frontend_assets_root() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

#[tracing::instrument(name = "web.get_frontend_asset", skip_all, fields(path = %path))]
pub(crate) async fn get_frontend_asset<R>(
    State(state): State<WebState<R>>,
    Path(path): Path<String>,
) -> Response
where
    R: WebRuntimeTypes,
{
    let Some(relative_path) = static_relative_path(&path) else {
        tracing::debug!("invalid frontend asset path");
        return StatusCode::NOT_FOUND.into_response();
    };
    let path = state.config.frontend_dir.join("assets").join(relative_path);
    serve_cached_static_file(&state.static_files, path, CACHE_IMMUTABLE).await
}

#[tracing::instrument(name = "web.get_favicon", skip_all)]
pub(crate) async fn get_favicon<R>(State(state): State<WebState<R>>) -> Response
where
    R: WebRuntimeTypes,
{
    serve_configured_image(state.config.favicon_path.as_deref(), "favicon").await
}

#[tracing::instrument(name = "web.get_og_image", skip_all)]
pub(crate) async fn get_og_image<R>(State(state): State<WebState<R>>) -> Response
where
    R: WebRuntimeTypes,
{
    serve_configured_image(state.config.og_image_path.as_deref(), "og image").await
}

#[tracing::instrument(name = "web.get_robots")]
pub(crate) async fn get_robots() -> Response {
    (
        StatusCode::OK,
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            ),
            (
                header::CACHE_CONTROL,
                HeaderValue::from_static("public, max-age=86400"),
            ),
        ],
        ROBOTS_TXT,
    )
        .into_response()
}

pub(crate) fn cache_layer(value: &'static str) -> SetResponseHeaderLayer<HeaderValue> {
    SetResponseHeaderLayer::overriding(header::CACHE_CONTROL, HeaderValue::from_static(value))
}

async fn serve_configured_image(path: Option<&FsPath>, what: &'static str) -> Response {
    let Some(path) = path else {
        return (StatusCode::NOT_FOUND, format!("no {what} configured")).into_response();
    };
    match tokio::fs::read(path).await {
        Ok(bytes) => {
            let content_type = match path.extension().and_then(|e| e.to_str()) {
                Some("png") => "image/png",
                Some("svg") => "image/svg+xml",
                Some("gif") => "image/gif",
                Some("jpg") | Some("jpeg") => "image/jpeg",
                Some("webp") => "image/webp",
                _ => "image/x-icon",
            };
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, HeaderValue::from_static(content_type)),
                    (
                        header::CACHE_CONTROL,
                        HeaderValue::from_static(CACHE_NO_CACHE),
                    ),
                ],
                bytes,
            )
                .into_response()
        }
        Err(error) => {
            tracing::warn!(
                error = %error,
                path = %path.display(),
                what,
                "configured image could not be read"
            );
            (StatusCode::NOT_FOUND, format!("{what} not found")).into_response()
        }
    }
}

async fn serve_cached_static_file(
    cache: &StaticFileCache,
    path: PathBuf,
    cache_control: &'static str,
) -> Response {
    match cache.load(path.clone()).await {
        Some(bytes) => static_file_response(&path, bytes, cache_control),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

pub(crate) async fn read_static_file(path: PathBuf) -> Option<Bytes> {
    tokio::fs::read(path).await.ok().map(Bytes::from)
}

pub(crate) fn static_file_response(
    path: &FsPath,
    bytes: Bytes,
    cache_control: &'static str,
) -> Response {
    let mut response = (StatusCode::OK, bytes).into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, content_type_for_path(path));
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(cache_control),
    );
    response
}

fn static_relative_path(path: &str) -> Option<PathBuf> {
    // Accept URL-relative asset paths only; reject traversal and Windows
    // separators before joining.
    if path.is_empty() || path.contains('\\') {
        return None;
    }
    let mut relative_path = PathBuf::new();
    for component in FsPath::new(path).components() {
        match component {
            Component::Normal(segment) => relative_path.push(segment),
            Component::CurDir => {}
            _ => return None,
        }
    }
    (!relative_path.as_os_str().is_empty()).then_some(relative_path)
}

fn content_type_for_path(path: &FsPath) -> HeaderValue {
    mime_guess::from_path(path)
        .first_raw()
        .map(HeaderValue::from_static)
        .unwrap_or_else(|| HeaderValue::from_static("application/octet-stream"))
}

fn static_file_weight(bytes: &Bytes) -> u32 {
    bytes.len().try_into().unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    async fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!("chudbot-web-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&path)
            .await
            .expect("create temp dir");
        path
    }

    #[test]
    fn static_relative_path_rejects_unsafe_paths() {
        assert!(static_relative_path("../app.js").is_none());
        assert!(static_relative_path("nested/../../app.js").is_none());
        assert!(static_relative_path("nested\\app.js").is_none());
        assert!(static_relative_path("").is_none());
        assert_eq!(
            static_relative_path("./assets/app.js"),
            Some(PathBuf::from("assets").join("app.js"))
        );
    }

    #[tokio::test]
    async fn static_file_cache_reuses_loaded_asset_bytes() {
        let dir = temp_dir().await;
        let path = dir.join("app.123abc.js");
        tokio::fs::write(&path, "first")
            .await
            .expect("write first asset");

        let cache = StaticFileCache::new();
        let first = cache.load(path.clone()).await.expect("load first asset");
        tokio::fs::write(&path, "second")
            .await
            .expect("write second asset");
        let second = cache.load(path.clone()).await.expect("load cached asset");

        assert_eq!(&first[..], b"first");
        assert_eq!(&second[..], b"first");

        tokio::fs::remove_dir_all(dir)
            .await
            .expect("remove temp dir");
    }
}
