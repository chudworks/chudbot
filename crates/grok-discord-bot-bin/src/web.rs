//! Axum HTTP server: JSON API + SSE event stream + static React bundle.
//!
//! Route layout:
//!   - `/api/conversations/{uuid}` → full [`ConversationView`] as JSON.
//!   - `/api/conversations/{uuid}/events` → text/event-stream that
//!     emits one event per [`ConversationEvent`] whose `conversation_id`
//!     matches (and globals like avatar updates).
//!   - `/images/*`, `/videos/*`, `/avatars/*` → media `ServeDir`s.
//!     File names embed a UUID (images/videos) or `<user_id>_<hash>`
//!     (avatars) so contents at a given URL are stable — served with
//!     `Cache-Control: public, max-age=31536000, immutable`.
//!   - `/assets/*` → Vite-emitted JS/CSS bundles with hashed filenames.
//!     Same long-lived immutable cache headers.
//!   - everything else → the `spa_index` fallback returns `index.html`
//!     with a hard `200` (and `no-cache, must-revalidate`) so client
//!     routes like `/c/<uuid>` resolve and report success. Paths whose
//!     last segment looks like a file (e.g. `/favicon.ico`) get a real
//!     `404` instead of being masked as HTML. We deliberately avoid
//!     tower-http's `ServeDir::not_found_service(ServeFile(index))` SPA
//!     pattern: in 0.6 it serves index.html's body but leaks the
//!     original `404` status for multi-segment paths.
//!
//! Compression: deliberately not done at origin. Cloudflare in front
//! of the tunnel compresses dynamically (brotli when the client
//! supports it, gzip otherwise) — doing it ourselves would duplicate
//! that work and lock the CDN into whatever encoding we chose. Our
//! `Cache-Control` values also intentionally avoid `no-transform` so
//! Cloudflare is free to compress / minify on the way out.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures::Stream;
use grok_discord_bot_core::{ConversationView, DbError};
use thiserror::Error;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tower_http::services::ServeDir;
use tower_http::set_header::SetResponseHeaderLayer;
use uuid::Uuid;

use crate::app::{AppState, ConversationEvent};

/// Errors returned by the web layer's startup path. Per-request errors
/// use [`ApiError`] instead.
#[derive(Debug, Error)]
pub enum WebError {
    /// Failure binding or serving over TCP.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Interval at which SSE connections emit a comment-only keepalive
/// frame. Cloudflare's default idle timeout for non-WebSocket
/// connections is ~100s; 30s gives a comfortable margin.
const SSE_KEEPALIVE: Duration = Duration::from_secs(30);

/// `Cache-Control` for content whose URL is stable forever (Vite
/// hashed bundles, UUID-named media, hash-named avatars). One year is
/// what Cloudflare's docs recommend for `immutable`-tagged assets.
const CACHE_IMMUTABLE: &str = "public, max-age=31536000, immutable";

/// `Cache-Control` for index.html and any SPA fallback. We want
/// browsers to revalidate on every page load so a new deploy ships
/// to users without a hard refresh. The hashed asset URLs inside the
/// HTML are then served from the long-lived cache, so no actual
/// bandwidth is wasted.
const CACHE_NO_CACHE: &str = "no-cache, must-revalidate";

/// `Cache-Control` for the JSON API. Conversations mutate constantly;
/// caching is actively harmful here.
const CACHE_NO_STORE: &str = "no-store";

/// Entry point for the web half of `grok serve`. Builds the router,
/// binds `listen`, and serves until the shared cancellation token
/// fires.
pub async fn run(app: Arc<AppState>, listen: SocketAddr) -> Result<(), WebError> {
    // Make sure every static-served directory exists so ServeDir
    // doesn't 500 the first time someone hits a missing file. The
    // frontend dir is treated as fatal-if-missing — if it doesn't
    // exist, the operator has skipped the `bun run build` step.
    tokio::fs::create_dir_all(&app.storage.images_dir).await?;
    tokio::fs::create_dir_all(&app.storage.videos_dir).await?;
    tokio::fs::create_dir_all(&app.storage.avatars_dir).await?;
    if !app.web_frontend_dir.exists() {
        tracing::warn!(
            frontend_dir = %app.web_frontend_dir.display(),
            "frontend_dir does not exist — SPA routes will return empty / 404. \
             Run `serve.sh deploy` (or `bun run build` in frontend/ then copy \
             dist/ to this path) before hitting the viewer."
        );
    }

    // --- API: JSON + SSE, never cached ---
    let api_router = Router::new()
        .route("/api/config", get(get_site_config))
        .route("/api/conversations/{id}", get(get_conversation))
        .route("/api/conversations/{id}/events", get(conversation_events))
        .layer(cache_layer(CACHE_NO_STORE));

    // --- Media (uploaded + generated): immutable forever ---
    //
    // ServeDir alone doesn't set Cache-Control; we wrap each one with
    // a SetResponseHeaderLayer that injects the immutable header.
    let images_service = ServeDir::new(&app.storage.images_dir);
    let videos_service = ServeDir::new(&app.storage.videos_dir);
    let avatars_service = ServeDir::new(&app.storage.avatars_dir);

    // --- Vite-emitted hashed bundles: immutable forever ---
    //
    // Vite always emits hashed files under "assets/", so this whole
    // subtree can be cached aggressively. index.html and the SPA
    // client routes are handled by the `spa_index` fallback below.
    let assets_service = ServeDir::new(app.web_frontend_dir.join("assets"));

    let router = api_router
        .nest_service(
            "/images",
            tower::ServiceBuilder::new()
                .layer(cache_layer(CACHE_IMMUTABLE))
                .service(images_service),
        )
        .nest_service(
            "/videos",
            tower::ServiceBuilder::new()
                .layer(cache_layer(CACHE_IMMUTABLE))
                .service(videos_service),
        )
        .nest_service(
            "/avatars",
            tower::ServiceBuilder::new()
                .layer(cache_layer(CACHE_IMMUTABLE))
                .service(avatars_service),
        )
        .nest_service(
            "/assets",
            tower::ServiceBuilder::new()
                .layer(cache_layer(CACHE_IMMUTABLE))
                .service(assets_service),
        )
        // Everything else is a single-page-app route → serve the SPA
        // shell. `spa_index` sets its own status + cache headers.
        .fallback(spa_index)
        .with_state(Arc::clone(&app));

    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!(
        addr = %listen,
        images_dir = %app.storage.images_dir.display(),
        videos_dir = %app.storage.videos_dir.display(),
        avatars_dir = %app.storage.avatars_dir.display(),
        frontend_dir = %app.web_frontend_dir.display(),
        "web server listening"
    );

    let cancel = app.cancel.clone();
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            cancel.cancelled().await;
            tracing::info!("web server: cancellation requested, shutting down");
        })
        .await?;
    Ok(())
}

/// Build a `Cache-Control: <value>` layer to slap on a route group.
fn cache_layer(value: &'static str) -> SetResponseHeaderLayer<HeaderValue> {
    SetResponseHeaderLayer::overriding(header::CACHE_CONTROL, HeaderValue::from_static(value))
}

/// Per-handler error type. Convert to a JSON response that the React
/// frontend can branch on.
#[derive(Debug, Error)]
enum ApiError {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("conversation not found")]
    NotFound,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, "not found".to_string()),
            ApiError::Db(err) => {
                tracing::error!(error = %err, "db error in api handler");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal server error".to_string(),
                )
            }
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

/// Static front-end configuration the React bundle reads at startup.
/// Kept deliberately tiny — just the bits the operator can tune via
/// `config.toml` that the browser needs to know about (today: the
/// browser-tab title prefix).
async fn get_site_config(State(app): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "title_prefix": app.web_title_prefix }))
}

async fn get_conversation(
    State(app): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ConversationView>, ApiError> {
    let view = app
        .db
        .fetch_conversation_view(id)
        .await?
        .ok_or(ApiError::NotFound)?;
    Ok(Json(view))
}

/// SSE stream of [`ConversationEvent`]s for one conversation. Filters
/// the broadcast channel to events whose `conversation_id` matches the
/// URL parameter, plus globals (avatar updates apply to any open
/// view). Lagged subscribers (broadcast buffer overflow) emit a
/// special `event: lag` frame so the frontend can decide to do a
/// manual refetch; in practice it refetches on any event anyway.
async fn conversation_events(
    State(app): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    let rx = app.events.subscribe();
    let stream: BroadcastStream<ConversationEvent> = BroadcastStream::new(rx);
    let filtered = stream.filter_map(move |item| match item {
        Ok(ev) if ev.conversation_id == id || ev.is_global() => Some(Ok(event_payload(&ev))),
        Ok(_) => None,
        Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
            tracing::warn!(conversation_id = %id, skipped = n, "sse stream lagged");
            Some(Ok(Event::default().event("lag").data(format!("{n}"))))
        }
    });
    // The stream naturally ends when the client disconnects (drops the
    // SSE response) or when the broadcast channel is closed (process
    // shutdown). axum::serve's `with_graceful_shutdown` in `run()`
    // handles the final teardown; per-connection cancellation isn't
    // needed here.

    // Suppress proxy buffering. Cloudflare and nginx both honor
    // `X-Accel-Buffering: no`; without it some proxies hold SSE
    // frames until the connection closes.
    let mut response = Sse::new(typed_stream(filtered))
        .keep_alive(KeepAlive::new().interval(SSE_KEEPALIVE))
        .into_response();
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    response
}

// Helper to keep the SSE handler's return type clean (axum needs the
// stream item to be `Result<Event, _>`; collapsing the closure type
// requires a function-shaped boundary).
fn typed_stream<S>(s: S) -> impl Stream<Item = Result<Event, Infallible>>
where
    S: Stream<Item = Result<Event, Infallible>>,
{
    s
}

/// Format a [`ConversationEvent`] as an SSE frame. Event name is the
/// kind discriminator; data is a small JSON object the frontend can
/// inspect if it wants kind-specific behavior (it doesn't, today).
fn event_payload(ev: &ConversationEvent) -> Event {
    use crate::app::EventKind;
    let (name, extra) = match ev.kind {
        EventKind::Created => ("created", serde_json::json!({})),
        EventKind::TurnStarted => ("turn_started", serde_json::json!({})),
        EventKind::TurnUpdated => ("turn_updated", serde_json::json!({})),
        EventKind::ToolCallRecorded => ("tool_call_recorded", serde_json::json!({})),
        EventKind::ContextItemAdded => ("context_item_added", serde_json::json!({})),
        EventKind::TitleUpdated => ("title_updated", serde_json::json!({})),
        EventKind::UserAvatarUpdated { user_id } => (
            "user_avatar_updated",
            // Stringify the snowflake for the same reason the JSON API
            // does: a bare number would be rounded past 2^53 by the
            // browser's JSON parser. Consumers compare it against the
            // (string) user ids in the conversation payload.
            serde_json::json!({ "user_id": user_id.to_string() }),
        ),
    };
    Event::default().event(name).data(extra.to_string())
}

/// Axum fallback for any route not matched by the API or a static
/// asset subtree. Serves the SPA shell so client-side routes like
/// `/c/<uuid>` resolve.
async fn spa_index(State(app): State<Arc<AppState>>, uri: axum::http::Uri) -> Response {
    render_spa(&app.web_frontend_dir, uri.path()).await
}

/// Core of [`spa_index`], split out so it's testable without an
/// `AppState` / database.
///
/// - SPA client routes (path's last segment has no `.`) → `index.html`
///   with a hard `200` and `no-cache` headers.
/// - Asset-looking misses (last segment contains a `.`, e.g.
///   `/favicon.ico`, `/foo.js`) → real `404`, so we don't mask a
///   missing asset as HTML (which would only break the browser's
///   parsing and caching).
async fn render_spa(frontend_dir: &std::path::Path, request_path: &str) -> Response {
    let last_segment = request_path.rsplit('/').next().unwrap_or("");
    if last_segment.contains('.') {
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let index = frontend_dir.join("index.html");
    match tokio::fs::read(&index).await {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("text/html; charset=utf-8"),
                ),
                (
                    header::CACHE_CONTROL,
                    HeaderValue::from_static(CACHE_NO_CACHE),
                ),
            ],
            bytes,
        )
            .into_response(),
        Err(err) => {
            tracing::error!(
                error = %err,
                path = %index.display(),
                "failed to read index.html for SPA fallback"
            );
            (
                StatusCode::NOT_FOUND,
                "frontend not built (index.html missing)",
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    /// Make a unique temp dir containing an `index.html`. `marker`
    /// keeps concurrent tests from colliding on the same path.
    fn temp_frontend(marker: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("grok_spa_{}_{marker}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.html"), b"<!doctype html><title>spa</title>").unwrap();
        dir
    }

    #[tokio::test]
    async fn client_route_serves_index_with_200() {
        let dir = temp_frontend("client_route");
        // This is the exact shape that was 404ing in production.
        let resp = render_spa(&dir, "/c/26765603-92a8-4b4b-b0ad-748383d24708").await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/html; charset=utf-8"
        );
        assert_eq!(
            resp.headers().get(header::CACHE_CONTROL).unwrap(),
            CACHE_NO_CACHE
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert!(body.starts_with(b"<!doctype html>"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn root_serves_index_with_200() {
        let dir = temp_frontend("root");
        let resp = render_spa(&dir, "/").await;
        assert_eq!(resp.status(), StatusCode::OK);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn asset_looking_miss_is_404() {
        let dir = temp_frontend("asset_miss");
        let resp = render_spa(&dir, "/favicon.ico").await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn missing_index_is_404() {
        let dir = std::env::temp_dir().join(format!("grok_spa_{}_no_index", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let resp = render_spa(&dir, "/c/whatever").await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        std::fs::remove_dir_all(&dir).ok();
    }
}
