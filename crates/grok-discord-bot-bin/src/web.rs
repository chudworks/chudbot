//! Axum web viewer.
//!
//! Renders one conversation per URL at `/c/{uuid}`. The UUID is the
//! only access control — links posted into Discord are unguessable, and
//! anyone with the link can read the trace. No auth, no login.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::Router;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use grok_discord_bot_core::{ContextItem, ConversationView, Db, DbError, TurnView, storage};
use maud::{DOCTYPE, Markup, PreEscaped, html};
use thiserror::Error;
use tower_http::services::ServeDir;
use uuid::Uuid;

/// Errors returned by the web layer. Map to HTTP responses via
/// [`IntoResponse`].
#[derive(Debug, Error)]
pub enum WebError {
    /// Failure binding or serving over TCP.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// State injected into every handler. Cheap to clone.
#[derive(Clone)]
struct WebState {
    db: Db,
}

/// Entry point for the `grok web` subcommand.
pub async fn run(
    db: Db,
    listen: SocketAddr,
    images_dir: PathBuf,
    videos_dir: PathBuf,
) -> Result<(), WebError> {
    // Ensure the dirs exist so ServeDir doesn't 500 on first hit
    // before any media has been written.
    tokio::fs::create_dir_all(&images_dir).await?;
    tokio::fs::create_dir_all(&videos_dir).await?;

    let state = WebState { db };
    let app = Router::new()
        .route("/", get(landing))
        .route("/c/{id}", get(view_conversation))
        .nest_service("/images", ServeDir::new(&images_dir))
        .nest_service("/videos", ServeDir::new(&videos_dir))
        .fallback(not_found)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(listen).await?;
    tracing::info!(
        addr = %listen,
        images_dir = %images_dir.display(),
        videos_dir = %videos_dir.display(),
        "web viewer listening"
    );
    axum::serve(listener, app).await?;
    Ok(())
}

/// Per-request error type. Distinct from [`WebError`] (which only covers
/// startup) so individual handlers can return either DB errors or
/// not-found cleanly.
#[derive(Debug, Error)]
enum HandlerError {
    #[error(transparent)]
    Db(#[from] DbError),
    #[error("conversation not found")]
    NotFound,
}

impl IntoResponse for HandlerError {
    fn into_response(self) -> Response {
        let (status, body) = match self {
            HandlerError::NotFound => {
                (StatusCode::NOT_FOUND, render_404().into_string())
            }
            HandlerError::Db(err) => {
                tracing::error!(error = %err, "db error in handler");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    render_error("Something went wrong loading this conversation.")
                        .into_string(),
                )
            }
        };
        (status, Html(body)).into_response()
    }
}

async fn landing() -> Html<String> {
    Html(
        html! {
            (DOCTYPE)
            html {
                head {
                    (head_common("grok"))
                }
                body {
                    main.center {
                        h1 { "grok viewer" }
                        p {
                            "Conversation traces are accessed by their unguessable \
                             UUID, surfaced as a link in Discord when the bot \
                             opens a new conversation."
                        }
                    }
                }
            }
        }
        .into_string(),
    )
}

async fn view_conversation(
    Path(id): Path<Uuid>,
    State(state): State<WebState>,
) -> Result<Html<String>, HandlerError> {
    let view = state
        .db
        .fetch_conversation_view(id)
        .await?
        .ok_or(HandlerError::NotFound)?;
    Ok(Html(render_conversation(&view).into_string()))
}

async fn not_found() -> Response {
    (StatusCode::NOT_FOUND, Html(render_404().into_string())).into_response()
}

fn render_conversation(view: &ConversationView) -> Markup {
    let title = view
        .conversation
        .title
        .clone()
        .unwrap_or_else(|| "Untitled conversation".to_string());
    let model = &view.conversation.model;
    let created = view.conversation.created_at;

    html! {
        (DOCTYPE)
        html {
            head {
                (head_common(&title))
            }
            body {
                header.conv-header {
                    h1 { (title) }
                    p.meta {
                        "Started " (created) " · model " code { (model) }
                    }
                }
                main.conv {
                    @for tv in &view.turns {
                        (render_turn(tv))
                    }
                    @if view.turns.is_empty() {
                        p.empty { "No turns yet." }
                    }
                }
            }
        }
    }
}

fn render_turn(tv: &TurnView) -> Markup {
    html! {
        section.turn {
            h2 {
                "Turn " (tv.turn.turn_index + 1)
                @match tv.turn.status.as_str() {
                    "completed" => span.badge.ok { "completed" },
                    "failed" => span.badge.err { "failed" },
                    other => span.badge { (other) },
                }
                @if let Some(p) = &tv.turn.persona_name {
                    " · persona " code { (p) }
                }
            }

            div.user {
                h3 { "User" }
                pre { (tv.turn.user_content) }
            }

            @if !tv.context.is_empty() {
                details.context {
                    summary { "Context fed to model (" (tv.context.len()) " items)" }
                    @for item in &tv.context {
                        article.context-item {
                            header {
                                span.role { (item.role) }
                                " · "
                                span.source { (item.source) }
                            }
                            (render_context_body(item))
                        }
                    }
                }
            }

            @if !tv.tool_calls.is_empty() {
                section.tools {
                    h3 { "Tool calls (" (tv.tool_calls.len()) ")" }
                    @for tc in &tv.tool_calls {
                        @let media = collect_media_uris(&tc.response);
                        article.tool-call {
                            header {
                                span.tool-name { (tc.tool_name) }
                            }
                            @if !media.is_empty() {
                                div.tool-images {
                                    @for m in &media {
                                        @match m {
                                            MediaUri::Image(uri) => @if let Some(p) = storage::to_web_path(uri) {
                                                img.context-image src=(p) alt=(tc.tool_name);
                                            },
                                            MediaUri::Video(uri) => @if let Some(p) = storage::to_web_path(uri) {
                                                video.context-video controls src=(p) {}
                                            },
                                        }
                                    }
                                }
                            }
                            details {
                                summary { "Request" }
                                pre { (PreEscaped(pretty_json(&tc.request))) }
                            }
                            details {
                                summary { "Response" }
                                pre { (PreEscaped(pretty_json(&tc.response))) }
                            }
                        }
                    }
                }
            }

            div.assistant {
                h3 { "Assistant" }
                @if let Some(content) = &tv.turn.assistant_content {
                    pre { (content) }
                } @else if tv.turn.status == "failed" {
                    pre.err { (tv.turn.error.as_deref().unwrap_or("(no error message)")) }
                } @else {
                    em { "(no response yet)" }
                }
            }
        }
    }
}

/// Render a context item's content. Media-typed items (per the
/// `file://images/…` or `file://videos/…` URI schemes) render as
/// inline `<img>` / `<video>` tags via the `/images/*` and `/videos/*`
/// static routes; everything else renders as preformatted text.
fn render_context_body(item: &ContextItem) -> Markup {
    if storage::is_image_uri(&item.content) {
        if let Some(web_path) = storage::to_web_path(&item.content) {
            return html! { img.context-image src=(web_path) alt="user attachment"; };
        }
    }
    if storage::is_video_uri(&item.content) {
        if let Some(web_path) = storage::to_web_path(&item.content) {
            return html! { video.context-video controls src=(web_path) {} };
        }
    }
    html! { pre { (item.content) } }
}

/// Media URI found in a JSON value, with kind tag so the renderer
/// knows whether to emit `<img>` or `<video>`.
#[derive(Debug, Clone)]
enum MediaUri {
    Image(String),
    Video(String),
}

/// Walk a JSON value and collect every string we recognise as a media
/// storage URI. Used to surface generated content embedded inside
/// tool-call responses (`generate_image` → `image_uri`,
/// `check_video_status` → `video_uri`, etc.).
fn collect_media_uris(value: &serde_json::Value) -> Vec<MediaUri> {
    let mut out = Vec::new();
    walk_for_media_uris(value, &mut out);
    out
}

fn walk_for_media_uris(value: &serde_json::Value, out: &mut Vec<MediaUri>) {
    match value {
        serde_json::Value::String(s) if storage::is_image_uri(s) => {
            out.push(MediaUri::Image(s.clone()));
        }
        serde_json::Value::String(s) if storage::is_video_uri(s) => {
            out.push(MediaUri::Video(s.clone()));
        }
        serde_json::Value::Array(arr) => arr.iter().for_each(|v| walk_for_media_uris(v, out)),
        serde_json::Value::Object(obj) => obj.values().for_each(|v| walk_for_media_uris(v, out)),
        _ => {}
    }
}

fn render_404() -> Markup {
    html! {
        (DOCTYPE)
        html {
            head {
                (head_common("not found"))
            }
            body {
                main.center {
                    h1 { "404" }
                    p { "No conversation here. The link may be wrong or the row was deleted." }
                }
            }
        }
    }
}

fn render_error(detail: &str) -> Markup {
    html! {
        (DOCTYPE)
        html {
            head {
                (head_common("error"))
            }
            body {
                main.center {
                    h1 { "500" }
                    p { (detail) }
                }
            }
        }
    }
}

fn head_common(title: &str) -> Markup {
    html! {
        meta charset="utf-8";
        meta name="viewport" content="width=device-width, initial-scale=1";
        title { (title) " · grok" }
        style { (PreEscaped(STYLE)) }
    }
}

fn pretty_json(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

const STYLE: &str = r#"
:root {
    color-scheme: light dark;
    --fg: #1a1a1a;
    --bg: #fafafa;
    --muted: #666;
    --border: #ddd;
    --card: #ffffff;
    --code-bg: #f1f1f1;
    --accent: #5b6cff;
    --err: #c0392b;
    --ok: #27ae60;
}
@media (prefers-color-scheme: dark) {
    :root {
        --fg: #eaeaea;
        --bg: #111;
        --muted: #999;
        --border: #2a2a2a;
        --card: #1a1a1a;
        --code-bg: #202020;
    }
}
* { box-sizing: border-box; }
body {
    margin: 0;
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--fg);
    background: var(--bg);
}
header.conv-header, main.conv, main.center {
    max-width: 860px;
    margin: 0 auto;
    padding: 2rem 1.25rem;
}
main.center { text-align: center; }
header.conv-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1rem;
}
header.conv-header h1 { margin: 0 0 .5rem; font-size: 1.6rem; }
.meta { color: var(--muted); font-size: .9rem; margin: 0; }
section.turn {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
}
section.turn h2 {
    font-size: 1.05rem;
    margin: 0 0 .75rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.badge {
    font-size: .75rem;
    font-weight: normal;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--code-bg);
    color: var(--muted);
}
.badge.ok { background: rgba(39,174,96,.12); color: var(--ok); }
.badge.err { background: rgba(192,57,43,.12); color: var(--err); }
.user h3, .assistant h3, .tools h3 {
    font-size: .85rem;
    text-transform: uppercase;
    letter-spacing: .04em;
    color: var(--muted);
    margin: 1rem 0 .35rem;
}
pre {
    background: var(--code-bg);
    padding: .75rem 1rem;
    border-radius: 6px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-word;
    margin: 0;
    font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
}
pre.err { color: var(--err); }
details {
    margin: .5rem 0;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .5rem .75rem;
}
details summary {
    cursor: pointer;
    color: var(--muted);
    font-size: .85rem;
}
details[open] summary { margin-bottom: .5rem; }
.context-item, .tool-call {
    margin: .5rem 0;
    padding: .5rem .75rem;
    border-left: 3px solid var(--border);
    background: var(--bg);
}
.context-item header, .tool-call header {
    font-size: .8rem;
    color: var(--muted);
    margin-bottom: .35rem;
}
.role { font-weight: 600; color: var(--accent); }
.tool-name { font-weight: 600; color: var(--accent); }
code {
    background: var(--code-bg);
    padding: 1px 6px;
    border-radius: 4px;
    font: 13px ui-monospace, SFMono-Regular, Menlo, monospace;
}
.empty { color: var(--muted); font-style: italic; }
.context-image, .context-video {
    max-width: 100%;
    max-height: 400px;
    border-radius: 6px;
    margin: .25rem 0;
    display: block;
}
.tool-images {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    margin: .5rem 0;
}
.tool-images .context-image,
.tool-images .context-video {
    max-height: 300px;
    flex: 0 0 auto;
}
"#;
