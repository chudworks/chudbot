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
//!   - everything else → tries to serve a static file under
//!     `app.web_frontend_dir`; if no such file exists, returns
//!     `index.html` (SPA fallback). Both index.html and the SPA shell
//!     are served `no-cache, must-revalidate` so a fresh deploy is
//!     picked up on the next page load without a hard refresh.
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
use tower_http::services::{ServeDir, ServeFile};
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
    // We split the frontend dir into "/assets" (long-cached, hashed
    // filenames) and the rest (no-cache, index.html + SPA fallback).
    // Vite always emits hashed files under "assets/" so this is a
    // clean cut.
    let assets_dir = app.web_frontend_dir.join("assets");
    let assets_service = ServeDir::new(&assets_dir);
    let index_html = app.web_frontend_dir.join("index.html");
    let spa_fallback =
        ServeDir::new(&app.web_frontend_dir).not_found_service(ServeFile::new(&index_html));

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
        // Everything else: SPA shell + fallback. Never cached so a
        // fresh deploy is visible on the next page load.
        .fallback_service(
            tower::ServiceBuilder::new()
                .layer(cache_layer(CACHE_NO_CACHE))
                .service(spa_fallback),
        )
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
            serde_json::json!({ "user_id": user_id }),
        ),
    };
    Event::default().event(name).data(extra.to_string())
}
