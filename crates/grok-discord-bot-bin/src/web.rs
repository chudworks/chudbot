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
//!   - `/robots.txt` → a site-wide `Disallow: /` (plus explicit blocks
//!     for AI crawlers that ignore the `*` group). See below.
//!   - everything else → the `spa_index` fallback returns `index.html`
//!     with a hard `200` (and `no-cache, must-revalidate`) so client
//!     routes like `/c/<uuid>` resolve and report success. Paths whose
//!     last segment looks like a file (e.g. `/favicon.ico`) get a real
//!     `404` instead of being masked as HTML. We deliberately avoid
//!     tower-http's `ServeDir::not_found_service(ServeFile(index))` SPA
//!     pattern: in 0.6 it serves index.html's body but leaks the
//!     original `404` status for multi-segment paths.
//!
//! Search-engine / crawler policy: this host is unauthenticated and
//! every conversation trace is guarded only by an unguessable UUID, so
//! NOTHING here should ever be indexed, cached, or scraped for training.
//! We enforce that in three layers, strongest first:
//!   1. `X-Robots-Tag: noindex, nofollow, noarchive, nosnippet` on EVERY
//!      response ([`x_robots_layer`]). Unlike a `<meta>` tag this also
//!      covers non-HTML bytes (images/videos) and crawlers that never run
//!      our JS — it is the load-bearing control.
//!   2. A hard `403` for any request whose User-Agent matches a known
//!      search-engine / AI / SEO crawler ([`block_crawlers`]). Social
//!      link-unfurl bots (Discord, Slack, Twitter, …) are intentionally
//!      NOT blocked so pasting a viewer URL into chat still previews.
//!   3. `/robots.txt` for the polite, compliant crawlers.
//!
//! The frontend's `index.html` also carries a `<meta name="robots">` tag
//! as a fourth, JS-rendered-crawler fallback.
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
use std::time::{Duration, Instant};

use axum::Json;
use axum::Router;
use axum::extract::{ConnectInfo, Path, Request, State};
use axum::http::{HeaderName, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use futures::Stream;
use grok_discord_bot_core::{ConversationView, DbError};
use http_body::Body as _;
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

/// `X-Robots-Tag` stamped on every response. `noindex` keeps the URL
/// out of result pages, `nofollow` stops crawlers following links out of
/// it, `noarchive` suppresses the "cached" copy, `nosnippet` blocks any
/// text/preview excerpt. Applies to non-HTML bytes too, which is the
/// whole reason we set it as a header rather than only a `<meta>` tag.
const X_ROBOTS_TAG: &str = "noindex, nofollow, noarchive, nosnippet";

/// Body served at `/robots.txt`. A wildcard `Disallow: /` covers
/// well-behaved search engines; the named groups below exist because
/// several AI / answer-engine crawlers (per their own docs) only honor a
/// directive addressed to their specific product token and ignore the
/// `*` group for training opt-out. Keep this list and
/// [`CRAWLER_UA_TOKENS`] roughly in sync.
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

/// Lowercased substrings that flag a search-engine, AI, or SEO crawler.
/// We hard-`403` any request whose User-Agent contains one of these (see
/// [`block_crawlers`]). Social-media link-unfurl bots (Discordbot,
/// Slackbot, Twitterbot, facebookexternalhit, TelegramBot, WhatsApp,
/// LinkedInBot) are DELIBERATELY absent: they build share previews, not
/// search-index entries, and blocking them would break the preview when
/// a viewer URL is pasted into a chat — which is exactly how these links
/// are meant to be shared. Match is case-insensitive substring, so e.g.
/// `"googlebot"` also catches `Googlebot-Image`.
const CRAWLER_UA_TOKENS: &[&str] = &[
    // Major search engines.
    "googlebot",
    "google-inspectiontool",
    "storebot-google",
    "bingbot",
    "bingpreview",
    "msnbot",
    "slurp", // Yahoo
    "duckduckbot",
    "duckassistbot",
    "baiduspider",
    "yandex",
    "sogou",
    "exabot",
    "seznambot",
    "petalbot", // Huawei
    "applebot", // also catches Applebot-Extended
    "ia_archiver",
    "archive.org_bot",
    // AI / answer-engine crawlers.
    "gptbot",
    "oai-searchbot",
    "chatgpt-user",
    "ccbot", // Common Crawl (feeds many training sets)
    "claudebot",
    "claude-web",
    "anthropic-ai",
    "perplexitybot",
    "perplexity-user",
    "amazonbot",
    "bytespider", // ByteDance
    "meta-externalagent",
    "cohere-ai",
    "diffbot",
    "google-extended",
    // Aggressive SEO / backlink scrapers.
    "semrushbot",
    "ahrefsbot",
    "mj12bot",
    "dotbot",
    "dataforseobot",
    "blexbot",
];

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
        // Favicon: served from an operator-configured path that lives
        // outside `frontend_dir` so it survives `serve.sh deploy`.
        // Browsers request `/favicon.ico` automatically, so no markup
        // change is needed. Sets its own cache header.
        .route("/favicon.ico", get(get_favicon))
        // Crawler opt-out for the polite, compliant bots. Sets its own
        // cache header; intentionally NOT behind `block_crawlers` (see
        // that fn) so a blocked crawler can still read the Disallow.
        .route("/robots.txt", get(get_robots))
        // Everything else is a single-page-app route → serve the SPA
        // shell. `spa_index` sets its own status + cache headers.
        .fallback(spa_index)
        .with_state(Arc::clone(&app))
        // Stamp `X-Robots-Tag: noindex…` on every response. Innermost of
        // the cross-cutting layers so it lands on real content; a 403
        // from `block_crawlers` short-circuits before this, but a 403 is
        // not indexable anyway. This is the load-bearing no-index control
        // (covers images/videos and JS-less crawlers, unlike a <meta>).
        .layer(x_robots_layer())
        // Hard-403 known search-engine / AI / SEO crawler User-Agents.
        .layer(middleware::from_fn(block_crawlers))
        // Access log wraps the whole router (added last → outermost),
        // so every request — API, media, SPA fallback, even a blocked
        // crawler's 403 — gets exactly one line.
        .layer(middleware::from_fn_with_state(
            app.web_trust_forwarded_for,
            access_log,
        ));

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
    // `into_make_service_with_connect_info` injects the TCP peer address
    // as a `ConnectInfo<SocketAddr>` request extension, which the access
    // logger reads when it isn't trusting `X-Forwarded-For`.
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        cancel.cancelled().await;
        tracing::info!("web server: cancellation requested, shutting down");
    })
    .await?;
    Ok(())
}

/// Longest User-Agent prefix we keep in the access log. The full string
/// (e.g. `Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:151.0)
/// Gecko/20100101 Firefox/151.0`) is noise; the leading product token
/// (`Mozilla/5.0`) is enough to tell humans from bots at a glance.
const UA_MAX_LEN: usize = 48;

/// Per-request access log middleware. Emits one `web::access` info line
/// per request with method, path, client IP, status, wall-clock
/// duration, request/response byte counts, and a trimmed User-Agent.
///
/// Body sizes come from `Body::size_hint().exact()`, so they're only
/// known for length-delimited bodies (most responses, and requests with
/// a `Content-Length`); streaming responses (the SSE endpoint) report
/// `0` since their bytes are produced after this middleware returns.
async fn access_log(State(trust_forwarded_for): State<bool>, req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let remote = client_ip(&req, trust_forwarded_for);
    let user_agent = req
        .headers()
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .map(short_user_agent)
        .unwrap_or_else(|| "-".to_owned());
    let input_bytes = req.body().size_hint().exact().unwrap_or(0);

    let start = Instant::now();
    let response = next.run(req).await;
    let duration = start.elapsed();

    let output_bytes = response.body().size_hint().exact().unwrap_or(0);

    tracing::info!(
        target: "web::access",
        %method,
        path,
        remote,
        status = response.status().as_u16(),
        duration_ms = duration.as_millis(),
        input_bytes,
        output_bytes,
        user_agent,
        "request"
    );

    response
}

/// Resolve the client IP for logging. When `trust_forwarded_for` is set
/// and an `X-Forwarded-For` header is present, use its first entry (the
/// original client; later entries are intermediary proxies). Otherwise
/// fall back to the TCP peer address from `ConnectInfo`.
fn client_ip(req: &Request, trust_forwarded_for: bool) -> String {
    if trust_forwarded_for
        && let Some(ip) = req
            .headers()
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(',').next())
            .map(str::trim)
            .filter(|s| !s.is_empty())
    {
        return ip.to_owned();
    }
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(addr)| addr.ip().to_string())
        .unwrap_or_else(|| "-".to_owned())
}

/// Trim a User-Agent down to its leading product token, capped at
/// [`UA_MAX_LEN`] bytes. Keeps the log readable without parsing the full
/// browser/OS soup.
fn short_user_agent(ua: &str) -> String {
    let token = ua.split_whitespace().next().unwrap_or(ua);
    token.chars().take(UA_MAX_LEN).collect()
}

/// Build a `Cache-Control: <value>` layer to slap on a route group.
fn cache_layer(value: &'static str) -> SetResponseHeaderLayer<HeaderValue> {
    SetResponseHeaderLayer::overriding(header::CACHE_CONTROL, HeaderValue::from_static(value))
}

/// Build the `X-Robots-Tag: <X_ROBOTS_TAG>` layer applied to the whole
/// router. `overriding` (not `if_not_present`) so no handler can
/// accidentally weaken it.
fn x_robots_layer() -> SetResponseHeaderLayer<HeaderValue> {
    SetResponseHeaderLayer::overriding(
        HeaderName::from_static("x-robots-tag"),
        HeaderValue::from_static(X_ROBOTS_TAG),
    )
}

/// True when `user_agent` looks like a crawler we want to hard-block.
/// Case-insensitive substring match against [`CRAWLER_UA_TOKENS`].
fn is_blocked_crawler(user_agent: &str) -> bool {
    let ua = user_agent.to_ascii_lowercase();
    CRAWLER_UA_TOKENS.iter().any(|token| ua.contains(token))
}

/// Middleware that returns `403` for known search-engine / AI / SEO
/// crawler User-Agents before they reach any handler. `/robots.txt` is
/// exempt so a compliant crawler can still fetch the site-wide
/// `Disallow` (and so we never answer a robots.txt fetch with a status
/// that some crawlers read as "retry later" rather than "stay out").
/// UA spoofing trivially defeats this, which is why [`x_robots_layer`]
/// and the unguessable UUID — not this — are the real protections; this
/// is defense-in-depth that also keeps crawl traffic off the box.
async fn block_crawlers(req: Request, next: Next) -> Response {
    if req.uri().path() != "/robots.txt"
        && let Some(ua) = req
            .headers()
            .get(header::USER_AGENT)
            .and_then(|v| v.to_str().ok())
        && is_blocked_crawler(ua)
    {
        return (
            StatusCode::FORBIDDEN,
            [(
                HeaderName::from_static("x-robots-tag"),
                HeaderValue::from_static(X_ROBOTS_TAG),
            )],
            "crawling and indexing of this host are not permitted\n",
        )
            .into_response();
    }
    next.run(req).await
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

/// Static front-end configuration the React bundle reads at startup:
/// the operator-tunable browser-tab title prefix, plus the running
/// server's build version (the ordered "vN" number resolved at startup
/// and the underlying `git describe` string) so every page can show it
/// in the footer.
async fn get_site_config(State(app): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "title_prefix": app.web_title_prefix,
        "version_number": app.app_version,
        "git_version": crate::VERSION,
    }))
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
    // End the stream when the client disconnects (drops the SSE
    // response) OR when the process is shutting down. The broadcast
    // sender lives inside the still-`Arc`'d `AppState`, so it is NOT
    // dropped on shutdown — without this `take_until`, the stream would
    // run forever and `axum::serve`'s `with_graceful_shutdown` in
    // `run()` would block on the open connection for the whole grace
    // period, making Ctrl+C appear to hang whenever a viewer is open.
    let filtered = futures::StreamExt::take_until(filtered, app.cancel.clone().cancelled_owned());

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
        EventKind::ConversationUpdated => ("conversation_updated", serde_json::json!({})),
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

/// Serve the operator-configured favicon at `/favicon.ico`. Returns
/// `404` when no `[web].favicon_path` is set or the file can't be read,
/// which is exactly the no-favicon default (browsers fall back to their
/// own icon). The content type is guessed from the file extension so an
/// operator can point this at a `.ico` or a `.png` interchangeably.
async fn get_favicon(State(app): State<Arc<AppState>>) -> Response {
    let Some(path) = app.web_favicon_path.as_deref() else {
        return (StatusCode::NOT_FOUND, "no favicon configured").into_response();
    };
    match tokio::fs::read(path).await {
        Ok(bytes) => {
            let content_type = match path.extension().and_then(|e| e.to_str()) {
                Some("png") => "image/png",
                Some("svg") => "image/svg+xml",
                Some("gif") => "image/gif",
                Some("jpg") | Some("jpeg") => "image/jpeg",
                // `.ico` and anything else fall through to the classic
                // favicon media type.
                _ => "image/x-icon",
            };
            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, HeaderValue::from_static(content_type)),
                    // `no-cache` (revalidate, don't blind-cache) so swapping
                    // the file shows up without users hard-refreshing —
                    // browsers cache favicons aggressively otherwise.
                    (
                        header::CACHE_CONTROL,
                        HeaderValue::from_static(CACHE_NO_CACHE),
                    ),
                ],
                bytes,
            )
                .into_response()
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                path = %path.display(),
                "configured favicon_path could not be read"
            );
            (StatusCode::NOT_FOUND, "favicon not found").into_response()
        }
    }
}

/// Serve `/robots.txt`. Static body ([`ROBOTS_TXT`]); cached for a day
/// so we're not re-serving it on every crawler pass, but short enough
/// that an operator edit propagates without a long wait.
async fn get_robots() -> Response {
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
    use test_case::test_case;

    // Search-engine + AI crawlers: blocked.
    #[test_case("Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)" ; "googlebot")]
    #[test_case("Mozilla/5.0 (compatible; Googlebot-Image/1.0; +http://www.google.com/bot.html)" ; "googlebot image variant")]
    #[test_case("Mozilla/5.0 (compatible; bingbot/2.0; +http://www.bing.com/bingbot.htm)" ; "bingbot")]
    #[test_case("Mozilla/5.0 (compatible; YandexBot/3.0)" ; "yandex")]
    #[test_case("Baiduspider+(+http://www.baidu.com/search/spider.htm)" ; "baidu")]
    #[test_case("GPTBot/1.0 (+https://openai.com/gptbot)" ; "gptbot")]
    #[test_case("Mozilla/5.0 (compatible; ClaudeBot/1.0; +claudebot@anthropic.com)" ; "claudebot")]
    #[test_case("CCBot/2.0 (https://commoncrawl.org/faq/)" ; "ccbot")]
    #[test_case("Mozilla/5.0 (compatible; PerplexityBot/1.0)" ; "perplexity")]
    #[test_case("Mozilla/5.0 (compatible; SemrushBot/7~bl)" ; "semrush")]
    #[test_case("GOOGLEBOT" ; "case insensitive")]
    fn blocked_crawler_user_agents(ua: &str) {
        assert!(is_blocked_crawler(ua), "expected {ua:?} to be blocked");
    }

    // Real browsers + social link-unfurl bots: allowed. Blocking the
    // unfurlers would kill the Discord/Slack/etc. preview that is the
    // whole point of sharing a viewer URL.
    #[test_case("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36" ; "chrome")]
    #[test_case("Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:151.0) Gecko/20100101 Firefox/151.0" ; "firefox")]
    #[test_case("Mozilla/5.0 (compatible; Discordbot/2.0; +https://discordapp.com)" ; "discord unfurl")]
    #[test_case("Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)" ; "slack unfurl")]
    #[test_case("Twitterbot/1.0" ; "twitter unfurl")]
    #[test_case("facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)" ; "facebook unfurl")]
    #[test_case("TelegramBot (like TwitterBot)" ; "telegram unfurl")]
    #[test_case("" ; "empty ua")]
    fn allowed_user_agents(ua: &str) {
        assert!(!is_blocked_crawler(ua), "expected {ua:?} to be allowed");
    }

    #[tokio::test]
    async fn robots_txt_disallows_everything() {
        let resp = get_robots().await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/plain; charset=utf-8"
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("User-agent: *"));
        assert!(text.contains("Disallow: /"));
        // AI crawlers that only honor a named group must be addressed.
        assert!(text.contains("GPTBot"));
        assert!(text.contains("ClaudeBot"));
    }

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
