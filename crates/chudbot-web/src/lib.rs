//! Web trace viewer service.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::path::{Component, Path as FsPath, PathBuf};
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, Request};
use axum::extract::{Path, State};
use axum::http::{HeaderName, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use bytes::Bytes;
use chudbot_api::{
    BotStorage, ClientToolCall, ClientToolResult, ClientToolResultContent, ContextItem,
    Conversation, ConversationId, ConversationLookup, ConversationSnapshot, EventSink,
    GroundingMetadata, LiveEvent, LlmProviderRegistry, MediaCategory, MediaStore, MediaUri,
    ModelId, ModelInfo, ModelInfoRequest, ProviderName, ServerToolUse, ToolTrace, Turn, TurnAsset,
    TurnReasoning, TurnSnapshot, UsageRecord, UsageSubject, UserRef,
};
use futures::{Stream, StreamExt};
use http_body::Body as _;
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_stream::wrappers::BroadcastStream;
use tokio_util::sync::CancellationToken;
use tower_http::set_header::SetResponseHeaderLayer;
use uuid::Uuid;

const SSE_KEEPALIVE: Duration = Duration::from_secs(30);
const CACHE_IMMUTABLE: &str = "public, max-age=31536000, immutable";
const CACHE_NO_CACHE: &str = "no-cache, must-revalidate";
const CACHE_NO_STORE: &str = "no-store";
const FRONTEND_STATIC_CACHE_MAX_BYTES: u64 = 64 * 1024 * 1024;
const X_ROBOTS_TAG: &str = "noindex, nofollow, noarchive, nosnippet";
/// Accent color picked up by link-preview embeds; matches `--accent` in
/// `frontend/src/styles/main.scss`.
const EMBED_THEME_COLOR: &str = "#5b6cff";
const UA_MAX_LEN: usize = 48;
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
const CRAWLER_UA_TOKENS: &[&str] = &[
    // Major search engines.
    "googlebot",
    "google-inspectiontool",
    "storebot-google",
    "bingbot",
    "bingpreview",
    "msnbot",
    "slurp",
    "duckduckbot",
    "duckassistbot",
    "baiduspider",
    "yandex",
    "sogou",
    "exabot",
    "seznambot",
    "petalbot",
    "applebot",
    "ia_archiver",
    "archive.org_bot",
    // AI / answer-engine crawlers.
    "gptbot",
    "oai-searchbot",
    "chatgpt-user",
    "ccbot",
    "claudebot",
    "claude-web",
    "anthropic-ai",
    "perplexitybot",
    "perplexity-user",
    "amazonbot",
    "bytespider",
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

/// Web service configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConfig {
    /// Browser tab title prefix.
    pub title_prefix: String,
    /// Build/version label.
    pub version: String,
    /// Directory containing the built frontend bundle.
    pub frontend_dir: PathBuf,
    /// Optional favicon served at /favicon.ico.
    #[serde(default)]
    pub favicon_path: Option<PathBuf>,
    /// Public origin (e.g. `https://chudbot.example.com`) used to build the
    /// absolute URLs link-preview unfurlers require. When unset, previews
    /// omit `og:url`/`og:image`.
    #[serde(default)]
    pub public_base_url: Option<String>,
    /// Optional link-preview thumbnail served at /og-image.
    #[serde(default)]
    pub og_image_path: Option<PathBuf>,
    /// Whether access logs trust proxy-provided client IP headers.
    #[serde(default = "default_trust_forwarded_for")]
    pub trust_forwarded_for: bool,
}

/// State shared by web handlers.
#[derive(Debug, Clone)]
pub struct WebState<S, M, L> {
    storage: S,
    media_store: M,
    llms: L,
    events: EventBus,
    config: WebConfig,
    static_files: StaticFileCache,
    shutdown: CancellationToken,
}

impl<S, M, L> WebState<S, M, L> {
    /// Build web state from concrete services.
    pub fn new(storage: S, media_store: M, llms: L, events: EventBus, config: WebConfig) -> Self {
        tracing::debug!(
            frontend_dir = %config.frontend_dir.display(),
            title_prefix = %config.title_prefix,
            version = %config.version,
            "constructing web state"
        );
        Self {
            storage,
            media_store,
            llms,
            events,
            config,
            static_files: StaticFileCache::new(),
            shutdown: CancellationToken::new(),
        }
    }

    /// Borrow the event bus.
    pub fn events(&self) -> &EventBus {
        &self.events
    }

    fn with_shutdown_token(mut self, shutdown: CancellationToken) -> Self {
        self.shutdown = shutdown;
        self
    }
}

#[derive(Clone)]
struct StaticFileCache {
    files: Cache<PathBuf, Bytes>,
}

impl StaticFileCache {
    fn new() -> Self {
        let files = Cache::builder()
            .name("frontend-static-files")
            .weigher(|_path, bytes: &Bytes| static_file_weight(bytes))
            .max_capacity(FRONTEND_STATIC_CACHE_MAX_BYTES)
            .build();
        Self { files }
    }

    async fn load(&self, path: PathBuf) -> Option<Bytes> {
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

/// Broadcast event bus shared with the bot runtime.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: tokio::sync::broadcast::Sender<LiveEvent>,
}

impl EventBus {
    /// Construct a new event bus.
    pub fn new(capacity: usize) -> Self {
        let (sender, _receiver) = tokio::sync::broadcast::channel(capacity);
        tracing::debug!(capacity, "constructed web event bus");
        Self { sender }
    }

    /// Subscribe to live events.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<LiveEvent> {
        tracing::trace!(
            receivers = self.sender.receiver_count(),
            "subscribing to event bus"
        );
        self.sender.subscribe()
    }
}

impl EventSink for EventBus {
    fn publish(&self, event: LiveEvent) {
        let event_name = event.event_name();
        match &event {
            LiveEvent::Conversation {
                conversation_id,
                kind,
            } => tracing::trace!(
                event = event_name,
                conversation = %conversation_id,
                kind = ?kind,
                receivers = self.sender.receiver_count(),
                "publishing live event"
            ),
            LiveEvent::UserProfileUpdated { user } => tracing::trace!(
                event = event_name,
                platform = %user.platform,
                guild = ?user.guild_id,
                user = %user.user_id,
                receivers = self.sender.receiver_count(),
                "publishing live event"
            ),
        }
        if self.sender.send(event).is_err() {
            tracing::trace!("live event dropped because there are no subscribers");
        }
    }
}

/// Run the web server.
#[tracing::instrument(
    name = "web.run",
    skip_all,
    fields(
        listen = %listen,
        frontend_dir = %state.config.frontend_dir.display(),
    )
)]
pub async fn run<S, M, L>(
    state: WebState<S, M, L>,
    listen: SocketAddr,
) -> Result<(), WebServerError>
where
    S: BotStorage + Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
    L: LlmProviderRegistry + Clone + Send + Sync + 'static,
{
    run_until_shutdown(state, listen, std::future::pending::<()>()).await
}

/// Run the web server until the supplied shutdown future resolves.
#[tracing::instrument(
    name = "web.run_until_shutdown",
    skip_all,
    fields(
        listen = %listen,
        frontend_dir = %state.config.frontend_dir.display(),
    )
)]
pub async fn run_until_shutdown<S, M, L, F>(
    state: WebState<S, M, L>,
    listen: SocketAddr,
    shutdown: F,
) -> Result<(), WebServerError>
where
    S: BotStorage + Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
    L: LlmProviderRegistry + Clone + Send + Sync + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let listener = tokio::net::TcpListener::bind(listen).await?;
    let shutdown_token = CancellationToken::new();
    let state = state.with_shutdown_token(shutdown_token.clone());
    tracing::info!("web server listening");
    axum::serve(
        listener,
        router(state).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        shutdown.await;
        shutdown_token.cancel();
        tracing::info!("web server shutdown requested");
    })
    .await?;
    tracing::info!("web server stopped");
    Ok(())
}

/// Build an Axum router for the viewer.
pub fn router<S, M, L>(state: WebState<S, M, L>) -> Router
where
    S: BotStorage + Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
    L: LlmProviderRegistry + Clone + Send + Sync + 'static,
{
    tracing::debug!(
        frontend_dir = %state.config.frontend_dir.display(),
        "building web router"
    );
    let trust_forwarded_for = state.config.trust_forwarded_for;
    let api = Router::new()
        .route("/api/config", get(get_config::<S, M, L>))
        .route("/api/conversations/{id}", get(get_conversation::<S, M, L>))
        .route(
            "/api/conversations/{id}/events",
            get(conversation_events::<S, M, L>),
        )
        .layer(cache_layer(CACHE_NO_STORE));

    Router::new()
        .merge(api)
        .route("/videos/{name}", get(get_video::<S, M, L>))
        .route("/audio/{name}", get(get_audio::<S, M, L>))
        .route("/avatars/{name}", get(get_avatar::<S, M, L>))
        .route("/images/{name}", get(get_image::<S, M, L>))
        .route("/favicon.ico", get(get_favicon::<S, M, L>))
        .route("/og-image", get(get_og_image::<S, M, L>))
        .route("/robots.txt", get(get_robots))
        .route("/assets", get(frontend_assets_root))
        .route("/assets/{*path}", get(get_frontend_asset::<S, M, L>))
        .fallback(spa_index::<S, M, L>)
        .layer(x_robots_layer())
        .layer(middleware::from_fn(block_crawlers))
        .layer(middleware::from_fn_with_state(
            trust_forwarded_for,
            access_log,
        ))
        .with_state(state)
}

/// Web server startup error.
#[derive(Debug, Error)]
pub enum WebServerError {
    /// TCP/server I/O failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Error)]
enum ApiError {
    #[error("conversation not found")]
    NotFound,
    #[error("storage error: {0}")]
    Storage(String),
    #[error("media error: {0}")]
    Media(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::Storage(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Media(_) => StatusCode::NOT_FOUND,
        };
        match status {
            StatusCode::INTERNAL_SERVER_ERROR => {
                tracing::error!(error = %self, status = status.as_u16(), "api error")
            }
            _ => tracing::warn!(error = %self, status = status.as_u16(), "api error"),
        }
        let body = serde_json::json!({ "error": self.to_string() });
        (status, Json(body)).into_response()
    }
}

#[tracing::instrument(name = "web.get_config", skip_all)]
async fn get_config<S, M, L>(State(state): State<WebState<S, M, L>>) -> Json<serde_json::Value>
where
    S: Clone + Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    tracing::debug!("serving web config");
    Json(serde_json::json!({
        "title_prefix": state.config.title_prefix,
        "version": state.config.version,
    }))
}

#[tracing::instrument(
    name = "web.get_conversation",
    skip_all,
    fields(conversation = %id)
)]
async fn get_conversation<S, M, L>(
    State(state): State<WebState<S, M, L>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ConversationView>, ApiError>
where
    S: BotStorage + Clone + Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    L: LlmProviderRegistry + Clone + Send + Sync + 'static,
{
    let snapshot = state
        .storage
        .load_conversation(ConversationLookup::Id {
            id: ConversationId(id),
        })
        .await
        .map_err(|error| ApiError::Storage(error.to_string()))?
        .ok_or(ApiError::NotFound)?;
    tracing::debug!(
        turns = snapshot.turns.len(),
        stopped = snapshot.conversation.stopped_at.is_some(),
        "loaded conversation snapshot"
    );
    let users = user_metadata(&state.storage, &snapshot).await?;
    let model_info = model_info_for_snapshot(&state.llms, &snapshot).await;
    let turns = snapshot.turns.into_iter().map(TurnView::from).collect();
    Ok(Json(ConversationView {
        conversation: snapshot.conversation,
        turns,
        users,
        model_info,
    }))
}

/// Conversation read model served to the React viewer.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationView {
    /// Conversation metadata.
    pub conversation: Conversation,
    /// Ordered turn snapshots, shaped for the viewer.
    pub turns: Vec<TurnView>,
    /// User metadata keyed by `platform:guild:user` string.
    pub users: BTreeMap<String, UserMetadata>,
    /// Provider-reported model metadata keyed by provider/model pairs.
    pub model_info: Vec<ModelInfoView>,
}

/// Viewer-facing provider model metadata.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfoView {
    /// Provider registry key.
    pub provider: ProviderName,
    /// Model id used to request metadata.
    pub requested_model: ModelId,
    /// Provider-reported model id.
    pub model: ModelId,
    /// Maximum input/context tokens accepted by the model.
    pub context_window_tokens: Option<u64>,
    /// Maximum output tokens the model can produce, when reported separately.
    pub max_output_tokens: Option<u64>,
}

impl ModelInfoView {
    fn new(provider: ProviderName, requested_model: ModelId, info: ModelInfo) -> Self {
        Self {
            provider,
            requested_model,
            model: info.id,
            context_window_tokens: info.context_window_tokens,
            max_output_tokens: info.max_output_tokens,
        }
    }
}

/// One turn plus viewer-safe trace data.
#[derive(Debug, Clone, Serialize)]
pub struct TurnView {
    /// Turn metadata.
    pub turn: Turn,
    /// System/developer instructions used for this attempt/turn.
    pub system_instructions: Option<String>,
    /// Novel context items captured for this turn.
    pub context: Vec<ContextItem>,
    /// Tool/server/grounding trace events.
    pub tool_trace: Vec<ToolTraceView>,
    /// Assets that should be replayed with this turn.
    pub replay_assets: Vec<TurnAsset>,
    /// Usage/cost accumulated by this turn.
    pub usage: Vec<UsageRecord>,
    /// Viewer-safe reasoning summaries and token counts.
    pub reasoning: TurnReasoning,
}

impl From<TurnSnapshot> for TurnView {
    fn from(snapshot: TurnSnapshot) -> Self {
        let reasoning =
            TurnReasoning::from_model_steps_and_usage(&snapshot.model_steps, &snapshot.usage);
        Self {
            turn: snapshot.turn,
            system_instructions: snapshot.system_instructions,
            context: snapshot.context,
            tool_trace: snapshot
                .tool_trace
                .into_iter()
                .map(ToolTraceView::from)
                .collect(),
            replay_assets: snapshot.replay_assets,
            usage: snapshot.usage,
            reasoning,
        }
    }
}

async fn model_info_for_snapshot<L>(llms: &L, snapshot: &ConversationSnapshot) -> Vec<ModelInfoView>
where
    L: LlmProviderRegistry,
{
    let targets = model_info_targets(snapshot);
    let mut out = Vec::new();
    for (provider, model) in targets {
        let request = ModelInfoRequest {
            model: model.clone(),
            provider_options: None,
        };
        match llms.fetch_model_info(&provider, request).await {
            Ok(Some(info)) => out.push(ModelInfoView::new(provider, model, info)),
            Ok(None) => tracing::debug!(
                provider = %provider,
                model = %model,
                "model metadata unavailable"
            ),
            Err(error) => tracing::warn!(
                provider = %provider,
                model = %model,
                error = %error,
                "failed to fetch model metadata"
            ),
        }
    }
    out
}

fn model_info_targets(snapshot: &ConversationSnapshot) -> BTreeSet<(ProviderName, ModelId)> {
    let mut targets = BTreeSet::new();
    targets.insert((
        snapshot.conversation.provider.clone(),
        snapshot.conversation.initial_model.clone(),
    ));

    for turn in &snapshot.turns {
        if let (Some(provider), Some(model)) = (&turn.turn.provider, &turn.turn.model) {
            targets.insert((provider.clone(), model.clone()));
        }
        for step in &turn.model_steps {
            targets.insert((step.provider.clone(), step.model.clone()));
        }
        for usage in &turn.usage {
            if matches!(&usage.subject, UsageSubject::ModelStep)
                && let Some(model) = &usage.model
            {
                targets.insert((usage.provider.clone(), model.clone()));
            }
        }
    }

    targets
}

/// Viewer-facing tool trace event.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ToolTraceView {
    /// Client-side tool call/result.
    Client {
        /// Trace record.
        trace: ClientToolTraceView,
    },
    /// Provider-side tool use, with no client-furnished result.
    Server {
        /// Server tool use.
        tool: ServerToolUse,
    },
    /// Provider grounding/citation metadata.
    Grounding {
        /// Grounding metadata.
        metadata: GroundingMetadata,
    },
}

impl From<ToolTrace> for ToolTraceView {
    fn from(trace: ToolTrace) -> Self {
        match trace {
            ToolTrace::Client { trace } => Self::Client {
                trace: ClientToolTraceView::from(trace),
            },
            ToolTrace::Server { tool } => Self::Server { tool },
            ToolTrace::Grounding { metadata } => Self::Grounding { metadata },
        }
    }
}

/// Viewer-facing client-side tool trace.
#[derive(Debug, Clone, Serialize)]
pub struct ClientToolTraceView {
    /// Tool call requested by the model.
    pub call: ClientToolCall,
    /// Tool result furnished back to the model.
    pub result: ClientToolResult,
    /// Extra trace/debug payload, omitted when it duplicates the result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_payload: Option<serde_json::Value>,
    /// Usage/cost incurred by this client tool.
    pub usage: Vec<UsageRecord>,
}

impl From<chudbot_api::ClientToolTrace> for ClientToolTraceView {
    fn from(trace: chudbot_api::ClientToolTrace) -> Self {
        let trace_payload = if trace_response_matches_result(&trace.trace_response, &trace.result) {
            None
        } else {
            Some(trace.trace_response)
        };
        Self {
            call: trace.call,
            result: trace.result,
            trace_payload,
            usage: trace.usage,
        }
    }
}

fn trace_response_matches_result(
    trace_response: &serde_json::Value,
    result: &ClientToolResult,
) -> bool {
    if let Ok(content) = serde_json::to_value(&result.content)
        && trace_response == &content
    {
        return true;
    }

    match &result.content {
        ClientToolResultContent::Json { value } => trace_response == value,
        ClientToolResultContent::Text { text } => {
            trace_response.as_str() == Some(text.as_str())
                || trace_response
                    .as_object()
                    .filter(|object| object.len() == 1)
                    .and_then(|object| object.get("text"))
                    .and_then(serde_json::Value::as_str)
                    == Some(text.as_str())
        }
    }
}

/// User metadata for frontend rendering.
#[derive(Debug, Clone, Serialize)]
pub struct UserMetadata {
    /// Stable platform user reference.
    pub id: UserRef,
    /// Last platform username seen.
    pub username: String,
    /// Best display name seen by the bot.
    pub display_name: Option<String>,
    /// Resolved label the UI can render directly.
    pub label: String,
    /// Platform avatar URL, usually remote/CDN-backed.
    pub avatar_url: Option<String>,
    /// Cached local avatar media URI, when available.
    pub avatar_media_uri: Option<MediaUri>,
    /// Whether the platform marked this user as a bot.
    pub is_bot: bool,
}

async fn user_metadata<S>(
    storage: &S,
    snapshot: &chudbot_api::ConversationSnapshot,
) -> Result<std::collections::BTreeMap<String, UserMetadata>, ApiError>
where
    S: BotStorage,
{
    let mut users = std::collections::BTreeMap::<String, UserMetadata>::new();
    insert_user_fallback(&mut users, snapshot.conversation.created_by.clone(), None);
    if let Some(user) = snapshot.conversation.stopped_by.clone() {
        insert_user_fallback(&mut users, user, None);
    }
    for turn in &snapshot.turns {
        insert_user_fallback(
            &mut users,
            turn.turn.user.clone(),
            Some(turn.turn.user_display_name.clone()),
        );
    }

    let refs = users
        .values()
        .map(|user| user.id.clone())
        .collect::<Vec<_>>();
    let stored = storage
        .load_user_profiles(refs)
        .await
        .map_err(|error| ApiError::Storage(error.to_string()))?;
    for stored in stored {
        let key = user_key(&stored.profile.id);
        users.insert(
            key,
            UserMetadata {
                label: stored
                    .profile
                    .display_name
                    .clone()
                    .unwrap_or_else(|| stored.profile.username.clone()),
                username: stored.profile.username,
                display_name: stored.profile.display_name,
                avatar_url: stored.profile.avatar_url,
                avatar_media_uri: stored.avatar,
                is_bot: stored.profile.is_bot,
                id: stored.profile.id,
            },
        );
    }
    Ok(users)
}

fn insert_user_fallback(
    users: &mut std::collections::BTreeMap<String, UserMetadata>,
    user: UserRef,
    label: Option<String>,
) {
    let key = user_key(&user);
    users.entry(key).or_insert_with(|| {
        let label = label.unwrap_or_else(|| user.user_id.as_str().to_string());
        UserMetadata {
            id: user,
            username: label.clone(),
            display_name: Some(label.clone()),
            label,
            avatar_url: None,
            avatar_media_uri: None,
            is_bot: false,
        }
    });
}

fn user_key(user: &UserRef) -> String {
    format!(
        "{}:{}:{}",
        user.platform.as_str(),
        user.guild_id
            .as_ref()
            .map(|id| id.as_str())
            .unwrap_or("global"),
        user.user_id.as_str()
    )
}

#[tracing::instrument(
    name = "web.conversation_events",
    skip_all,
    fields(conversation = %id)
)]
async fn conversation_events<S, M, L>(
    State(state): State<WebState<S, M, L>>,
    Path(id): Path<Uuid>,
) -> Result<Response, ApiError>
where
    S: BotStorage + Clone + Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    let conversation_id = ConversationId(id);
    state
        .storage
        .load_conversation(ConversationLookup::Id {
            id: conversation_id,
        })
        .await
        .map_err(|error| ApiError::Storage(error.to_string()))?
        .ok_or(ApiError::NotFound)?;

    tracing::info!("opening conversation event stream");
    let stream = BroadcastStream::new(state.events.subscribe()).filter_map(move |item| {
        let event = match item {
            Ok(event) if event.applies_to_conversation(conversation_id) => event,
            Ok(_) => return futures::future::ready(None),
            Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
                tracing::warn!(
                    conversation = %conversation_id,
                    skipped = n,
                    "conversation event stream lagged"
                );
                return futures::future::ready(Some(Ok(Event::default()
                    .event("lag")
                    .data(n.to_string()))));
            }
        };
        tracing::trace!(
            conversation = %conversation_id,
            event = event.event_name(),
            "forwarding live event to SSE client"
        );
        let data = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
        futures::future::ready(Some(Ok(Event::default()
            .event(event.event_name())
            .data(data))))
    });
    let stream = stream.take_until(state.shutdown.clone().cancelled_owned());
    let mut response = Sse::new(typed_stream(stream))
        .keep_alive(KeepAlive::new().interval(SSE_KEEPALIVE))
        .into_response();
    response
        .headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    Ok(response)
}

fn typed_stream<S>(stream: S) -> impl Stream<Item = Result<Event, Infallible>>
where
    S: Stream<Item = Result<Event, Infallible>>,
{
    stream
}

async fn frontend_assets_root() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

#[tracing::instrument(name = "web.get_frontend_asset", skip_all, fields(path = %path))]
async fn get_frontend_asset<S, M, L>(
    State(state): State<WebState<S, M, L>>,
    Path(path): Path<String>,
) -> Response
where
    S: Clone + Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    let Some(relative_path) = static_relative_path(&path) else {
        tracing::debug!("invalid frontend asset path");
        return StatusCode::NOT_FOUND.into_response();
    };
    let path = state.config.frontend_dir.join("assets").join(relative_path);
    serve_cached_static_file(&state.static_files, path, CACHE_IMMUTABLE).await
}

#[tracing::instrument(name = "web.get_favicon", skip_all)]
async fn get_favicon<S, M, L>(State(state): State<WebState<S, M, L>>) -> Response
where
    S: Clone + Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    serve_configured_image(state.config.favicon_path.as_deref(), "favicon").await
}

#[tracing::instrument(name = "web.get_og_image", skip_all)]
async fn get_og_image<S, M, L>(State(state): State<WebState<S, M, L>>) -> Response
where
    S: Clone + Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    serve_configured_image(state.config.og_image_path.as_deref(), "og image").await
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

#[tracing::instrument(name = "web.get_robots")]
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

fn cache_layer(value: &'static str) -> SetResponseHeaderLayer<HeaderValue> {
    SetResponseHeaderLayer::overriding(header::CACHE_CONTROL, HeaderValue::from_static(value))
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

async fn read_static_file(path: PathBuf) -> Option<Bytes> {
    tokio::fs::read(path).await.ok().map(Bytes::from)
}

fn static_file_response(path: &FsPath, bytes: Bytes, cache_control: &'static str) -> Response {
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

fn x_robots_layer() -> SetResponseHeaderLayer<HeaderValue> {
    SetResponseHeaderLayer::overriding(
        HeaderName::from_static("x-robots-tag"),
        HeaderValue::from_static(X_ROBOTS_TAG),
    )
}

fn default_trust_forwarded_for() -> bool {
    true
}

async fn access_log(State(trust_forwarded_for): State<bool>, req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let remote = client_ip(&req, trust_forwarded_for);
    let user_agent = req
        .headers()
        .get(header::USER_AGENT)
        .and_then(|value| value.to_str().ok())
        .map(short_user_agent)
        .unwrap_or_else(|| "-".to_string());
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

fn client_ip(req: &Request, trust_forwarded_for: bool) -> String {
    if trust_forwarded_for && let Some(ip) = forwarded_client_ip(req) {
        return ip;
    }
    req.extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(addr)| addr.ip().to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn forwarded_client_ip(req: &Request) -> Option<String> {
    header_value(req, "cf-connecting-ip")
        .or_else(|| header_value(req, "true-client-ip"))
        .or_else(|| x_forwarded_for(req))
        .or_else(|| forwarded_for(req))
}

fn header_value(req: &Request, name: &'static str) -> Option<String> {
    req.headers()
        .get(HeaderName::from_static(name))
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn x_forwarded_for(req: &Request) -> Option<String> {
    req.headers()
        .get(HeaderName::from_static("x-forwarded-for"))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(',').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn forwarded_for(req: &Request) -> Option<String> {
    let forwarded = req
        .headers()
        .get(HeaderName::from_static("forwarded"))?
        .to_str()
        .ok()?;
    let first = forwarded.split(',').next()?;
    first.split(';').find_map(|field| {
        let (name, value) = field.split_once('=')?;
        if !name.trim().eq_ignore_ascii_case("for") {
            return None;
        }
        let value = value.trim().trim_matches('"').trim();
        (!value.is_empty()).then(|| value.to_string())
    })
}

fn short_user_agent(ua: &str) -> String {
    let token = ua.split_whitespace().next().unwrap_or(ua);
    token.chars().take(UA_MAX_LEN).collect()
}

fn is_blocked_crawler(user_agent: &str) -> bool {
    let ua = user_agent.to_ascii_lowercase();
    CRAWLER_UA_TOKENS.iter().any(|token| ua.contains(token))
}

async fn block_crawlers(req: Request, next: Next) -> Response {
    if req.uri().path() != "/robots.txt"
        && let Some(ua) = req
            .headers()
            .get(header::USER_AGENT)
            .and_then(|value| value.to_str().ok())
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

#[tracing::instrument(name = "web.get_image", skip_all, fields(name = %name))]
async fn get_image<S, M, L>(
    State(state): State<WebState<S, M, L>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError>
where
    S: Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    load_media_response(&state.media_store, MediaCategory::Image, &name).await
}

#[tracing::instrument(name = "web.get_video", skip_all, fields(name = %name))]
async fn get_video<S, M, L>(
    State(state): State<WebState<S, M, L>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError>
where
    S: Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    load_media_response(&state.media_store, MediaCategory::Video, &name).await
}

#[tracing::instrument(name = "web.get_audio", skip_all, fields(name = %name))]
async fn get_audio<S, M, L>(
    State(state): State<WebState<S, M, L>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError>
where
    S: Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    load_media_response(&state.media_store, MediaCategory::Audio, &name).await
}

#[tracing::instrument(name = "web.get_avatar", skip_all, fields(name = %name))]
async fn get_avatar<S, M, L>(
    State(state): State<WebState<S, M, L>>,
    Path(name): Path<String>,
) -> Result<Response, ApiError>
where
    S: Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    load_media_response(&state.media_store, MediaCategory::Avatar, &name).await
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

#[tracing::instrument(name = "web.spa_index", skip_all, fields(path = %uri.path()))]
async fn spa_index<S, M, L>(
    State(state): State<WebState<S, M, L>>,
    uri: axum::http::Uri,
) -> Response
where
    S: BotStorage + Clone + Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    let preview_meta = match conversation_path_id(uri.path()) {
        Some(id) => conversation_preview_meta(&state, id).await,
        None => None,
    };
    render_spa(
        &state.static_files,
        &state.config.frontend_dir,
        uri.path(),
        preview_meta.as_deref(),
    )
    .await
}

/// Extract the conversation id from `/c/<uuid>` viewer paths.
fn conversation_path_id(path: &str) -> Option<Uuid> {
    let id = path.strip_prefix("/c/")?.trim_end_matches('/');
    Uuid::try_parse(id).ok()
}

/// Load conversation metadata and render the `<head>` OpenGraph tags that
/// Discord-style link unfurlers read. Returns `None` (plain SPA index) when
/// the conversation is unknown or storage fails.
#[tracing::instrument(
    name = "web.conversation_preview_meta",
    skip_all,
    fields(conversation = %id)
)]
async fn conversation_preview_meta<S, M, L>(state: &WebState<S, M, L>, id: Uuid) -> Option<String>
where
    S: BotStorage + Clone + Send + Sync + 'static,
    M: Clone + Send + Sync + 'static,
    L: Clone + Send + Sync + 'static,
{
    let snapshot = state
        .storage
        .load_conversation(ConversationLookup::Id {
            id: ConversationId(id),
        })
        .await
        .unwrap_or_else(|error| {
            tracing::warn!(error = %error, "failed to load conversation for link preview");
            None
        })?;
    tracing::debug!(
        titled = snapshot.conversation.title.is_some(),
        turns = snapshot.turns.len(),
        "rendering link-preview metadata"
    );
    Some(preview_meta_html(
        &state.config,
        &snapshot.conversation,
        snapshot.turns.len(),
    ))
}

fn preview_meta_html(config: &WebConfig, conversation: &Conversation, turns: usize) -> String {
    let title = conversation
        .title
        .as_deref()
        .unwrap_or("Untitled conversation");
    let turn_word = if turns == 1 { "turn" } else { "turns" };
    let description = format!(
        "{turns} {turn_word} · started {}",
        conversation.created_at.date()
    );
    let mut meta = String::from("\n");
    push_meta(
        &mut meta,
        "property",
        "og:site_name",
        site_name(&config.title_prefix),
    );
    push_meta(&mut meta, "property", "og:type", "website");
    push_meta(&mut meta, "property", "og:title", title);
    push_meta(&mut meta, "property", "og:description", &description);
    if let Some(base) = trimmed_public_base_url(config) {
        push_meta(
            &mut meta,
            "property",
            "og:url",
            &format!("{base}/c/{}", conversation.id),
        );
        if config.og_image_path.is_some() {
            push_meta(
                &mut meta,
                "property",
                "og:image",
                &format!("{base}/og-image"),
            );
        }
    }
    push_meta(&mut meta, "name", "theme-color", EMBED_THEME_COLOR);
    meta
}

/// Derive the embed site name from the tab-title prefix, e.g.
/// "Chudbot QA - " -> "Chudbot QA".
fn site_name(title_prefix: &str) -> &str {
    let name = title_prefix
        .trim()
        .trim_end_matches(['-', '|', ':', '·', '—', '–'])
        .trim_end();
    if name.is_empty() { "Chudbot" } else { name }
}

fn trimmed_public_base_url(config: &WebConfig) -> Option<&str> {
    let base = config
        .public_base_url
        .as_deref()?
        .trim()
        .trim_end_matches('/');
    (!base.is_empty()).then_some(base)
}

fn push_meta(out: &mut String, key_attr: &str, key: &str, content: &str) {
    out.push_str("<meta ");
    out.push_str(key_attr);
    out.push_str("=\"");
    out.push_str(key);
    out.push_str("\" content=\"");
    out.push_str(&escape_attr(content));
    out.push_str("\">\n");
}

/// Minimal HTML attribute escaping for injected meta content.
fn escape_attr(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

#[tracing::instrument(
    name = "web.render_spa",
    skip_all,
    fields(frontend_dir = %frontend_dir.display(), request_path = %request_path)
)]
async fn render_spa(
    static_files: &StaticFileCache,
    frontend_dir: &FsPath,
    request_path: &str,
    head_meta: Option<&str>,
) -> Response {
    let last_segment = request_path.rsplit('/').next().unwrap_or("");
    if last_segment.contains('.') {
        tracing::debug!("asset-looking SPA fallback path not found");
        return (StatusCode::NOT_FOUND, "not found").into_response();
    }
    let index_path = frontend_dir.join("index.html");
    let index = if cache_spa_index() {
        static_files.load(index_path.clone()).await
    } else {
        read_static_file(index_path.clone()).await
    };
    match index {
        Some(bytes) => {
            let bytes = match head_meta {
                Some(meta) => inject_head_meta(&bytes, meta),
                None => bytes,
            };
            tracing::debug!(bytes = bytes.len(), "serving SPA index");
            let mut response = static_file_response(&index_path, bytes, CACHE_NO_CACHE);
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            );
            response
        }
        None => {
            tracing::warn!("SPA index is missing or unreadable");
            (
                StatusCode::NOT_FOUND,
                "frontend not built (index.html missing)",
            )
                .into_response()
        }
    }
}

/// Splice extra `<head>` markup in just before `</head>`. Serves the index
/// unmodified when the marker is missing.
fn inject_head_meta(index: &Bytes, meta: &str) -> Bytes {
    let needle = b"</head>";
    match index
        .windows(needle.len())
        .position(|window| window == needle)
    {
        Some(pos) => {
            let mut out = Vec::with_capacity(index.len() + meta.len());
            out.extend_from_slice(&index[..pos]);
            out.extend_from_slice(meta.as_bytes());
            out.extend_from_slice(&index[pos..]);
            Bytes::from(out)
        }
        None => {
            tracing::warn!("SPA index has no </head>; skipping link-preview metadata");
            index.clone()
        }
    }
}

fn cache_spa_index() -> bool {
    !cfg!(debug_assertions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use chudbot_api::{ChannelRef, ClientToolTrace, MessageRef, ToolName, ToolUseId};
    use serde_json::json;
    use test_case::test_case;

    fn request_with_peer(peer: SocketAddr) -> Request {
        let mut req = Request::builder().body(Body::empty()).unwrap();
        req.extensions_mut().insert(ConnectInfo(peer));
        req
    }

    async fn temp_dir() -> PathBuf {
        let path = std::env::temp_dir().join(format!("chudbot-web-{}", Uuid::new_v4()));
        tokio::fs::create_dir_all(&path)
            .await
            .expect("create temp dir");
        path
    }

    async fn response_bytes(response: Response) -> Bytes {
        to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read response body")
    }

    fn client_trace_view(
        content: ClientToolResultContent,
        trace_response: serde_json::Value,
    ) -> ClientToolTraceView {
        ClientToolTraceView::from(ClientToolTrace {
            call: ClientToolCall {
                id: ToolUseId::from("call-1"),
                name: ToolName::from("test_tool"),
                input: json!({ "prompt": "draw this" }),
            },
            result: ClientToolResult {
                tool_use_id: ToolUseId::from("call-1"),
                content,
                is_error: false,
            },
            trace_response,
            usage: Vec::new(),
        })
    }

    fn test_web_config() -> WebConfig {
        WebConfig {
            title_prefix: "Chudbot QA - ".to_string(),
            version: "test".to_string(),
            frontend_dir: PathBuf::from("frontend-build"),
            favicon_path: None,
            public_base_url: Some("https://chudbot.example.com/".to_string()),
            og_image_path: Some(PathBuf::from("chudbot.png")),
            trust_forwarded_for: true,
        }
    }

    fn test_conversation(title: Option<&str>) -> Conversation {
        let conversation_id = Uuid::try_parse("0626e4d2-d41f-44ea-aa7f-45489b48955c").unwrap();
        Conversation {
            id: ConversationId(conversation_id),
            created_at: time::OffsetDateTime::UNIX_EPOCH,
            channel: ChannelRef {
                platform: "discord".into(),
                guild_id: Some("1".into()),
                channel_id: "2".into(),
            },
            created_by: UserRef {
                platform: "discord".into(),
                guild_id: Some("1".into()),
                user_id: "3".into(),
            },
            root_message: MessageRef {
                platform: "discord".into(),
                guild_id: Some("1".into()),
                channel_id: "2".into(),
                message_id: "4".into(),
            },
            initial_model: "grok-4".into(),
            agent_name: "chud".to_string(),
            provider: "xai".into(),
            system_instructions: String::new(),
            title: title.map(str::to_string),
            stopped_at: None,
            stopped_by: None,
        }
    }

    #[test_case("/c/0626e4d2-d41f-44ea-aa7f-45489b48955c", true ; "conversation path")]
    #[test_case("/c/0626e4d2-d41f-44ea-aa7f-45489b48955c/", true ; "trailing slash")]
    #[test_case("/c/not-a-uuid", false ; "invalid uuid")]
    #[test_case("/c/", false ; "missing id")]
    #[test_case("/", false ; "root")]
    #[test_case("/c/0626e4d2-d41f-44ea-aa7f-45489b48955c/extra", false ; "nested path")]
    fn conversation_path_id_matches_viewer_paths(path: &str, expected: bool) {
        assert_eq!(conversation_path_id(path).is_some(), expected);
    }

    #[test_case("Chudbot QA - ", "Chudbot QA" ; "dash separator")]
    #[test_case("Chudbot | ", "Chudbot" ; "pipe separator")]
    #[test_case("Chudbot", "Chudbot" ; "no separator")]
    #[test_case("   ", "Chudbot" ; "blank prefix falls back")]
    fn site_name_strips_title_separators(prefix: &str, expected: &str) {
        assert_eq!(site_name(prefix), expected);
    }

    #[test]
    fn preview_meta_escapes_title_and_builds_absolute_urls() {
        let conversation = test_conversation(Some(r#"Tom & "Jerry" <3"#));

        let meta = preview_meta_html(&test_web_config(), &conversation, 3);

        assert!(meta.contains(r#"<meta property="og:site_name" content="Chudbot QA">"#));
        assert!(
            meta.contains(
                r#"<meta property="og:title" content="Tom &amp; &quot;Jerry&quot; &lt;3">"#
            )
        );
        assert!(meta.contains(
            r#"<meta property="og:description" content="3 turns · started 1970-01-01">"#
        ));
        assert!(meta.contains(
            r#"<meta property="og:url" content="https://chudbot.example.com/c/0626e4d2-d41f-44ea-aa7f-45489b48955c">"#
        ));
        assert!(
            meta.contains(
                r#"<meta property="og:image" content="https://chudbot.example.com/og-image">"#
            )
        );
        assert!(meta.contains(r##"<meta name="theme-color" content="#5b6cff">"##));
    }

    #[test]
    fn preview_meta_omits_urls_without_public_base() {
        let config = WebConfig {
            public_base_url: None,
            ..test_web_config()
        };
        let conversation = test_conversation(None);

        let meta = preview_meta_html(&config, &conversation, 1);

        assert!(meta.contains(r#"<meta property="og:title" content="Untitled conversation">"#));
        assert!(meta.contains(r#"content="1 turn · started 1970-01-01""#));
        assert!(!meta.contains("og:url"));
        assert!(!meta.contains("og:image"));
    }

    #[test]
    fn preview_meta_omits_image_without_configured_path() {
        let config = WebConfig {
            og_image_path: None,
            ..test_web_config()
        };

        let meta = preview_meta_html(&config, &test_conversation(None), 1);

        assert!(meta.contains("og:url"));
        assert!(!meta.contains("og:image"));
    }

    #[test]
    fn inject_head_meta_inserts_before_head_close() {
        let index = Bytes::from_static(b"<html><head><title>x</title></head><body></body></html>");

        let injected = inject_head_meta(&index, "<meta name=\"a\" content=\"b\">\n");

        let html = String::from_utf8(injected.to_vec()).expect("utf8 index");
        assert!(html.contains("<meta name=\"a\" content=\"b\">\n</head>"));
        assert!(html.ends_with("</html>"));
    }

    #[test]
    fn inject_head_meta_keeps_index_without_head() {
        let index = Bytes::from_static(b"no head here");

        assert_eq!(inject_head_meta(&index, "<meta>"), index);
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

    #[cfg(debug_assertions)]
    #[tokio::test]
    async fn render_spa_reads_index_from_disk_in_debug_builds() {
        let dir = temp_dir().await;
        let path = dir.join("index.html");
        tokio::fs::write(&path, "first")
            .await
            .expect("write first index");

        let cache = StaticFileCache::new();
        let first = render_spa(&cache, &dir, "/c/test", None).await;
        tokio::fs::write(&path, "second")
            .await
            .expect("write second index");
        let second = render_spa(&cache, &dir, "/c/test", None).await;

        assert_eq!(&response_bytes(first).await[..], b"first");
        assert_eq!(&response_bytes(second).await[..], b"second");

        tokio::fs::remove_dir_all(dir)
            .await
            .expect("remove temp dir");
    }

    #[test]
    fn viewer_trace_omits_duplicate_json_trace_payload() {
        let view = client_trace_view(
            ClientToolResultContent::Json {
                value: json!({ "ok": true }),
            },
            json!({ "ok": true }),
        );

        assert!(view.trace_payload.is_none());
        let value = serde_json::to_value(view).expect("serialize trace view");
        assert!(value.get("trace_payload").is_none());
    }

    #[test]
    fn viewer_trace_keeps_distinct_trace_payload() {
        let trace_payload = json!({
            "uri": "file://images/generated.png",
            "public_url": "https://media.example/generated.png"
        });
        let view = client_trace_view(
            ClientToolResultContent::Json {
                value: json!({ "uri": "file://images/generated.png" }),
            },
            trace_payload.clone(),
        );

        assert_eq!(view.trace_payload, Some(trace_payload));
    }

    #[test]
    fn client_ip_prefers_cloudflare_header_when_trusted() {
        let mut req = request_with_peer(SocketAddr::from(([10, 0, 0, 2], 443)));
        req.headers_mut().insert(
            HeaderName::from_static("cf-connecting-ip"),
            HeaderValue::from_static("203.0.113.42"),
        );
        req.headers_mut().insert(
            HeaderName::from_static("x-forwarded-for"),
            HeaderValue::from_static("198.51.100.7, 10.0.0.1"),
        );

        assert_eq!(client_ip(&req, true), "203.0.113.42");
    }

    #[test]
    fn client_ip_uses_first_x_forwarded_for_when_trusted() {
        let mut req = request_with_peer(SocketAddr::from(([10, 0, 0, 2], 443)));
        req.headers_mut().insert(
            HeaderName::from_static("x-forwarded-for"),
            HeaderValue::from_static("198.51.100.7, 10.0.0.1"),
        );

        assert_eq!(client_ip(&req, true), "198.51.100.7");
    }

    #[test]
    fn client_ip_uses_standard_forwarded_header_when_trusted() {
        let mut req = request_with_peer(SocketAddr::from(([10, 0, 0, 2], 443)));
        req.headers_mut().insert(
            HeaderName::from_static("forwarded"),
            HeaderValue::from_static("for=198.51.100.8;proto=https"),
        );

        assert_eq!(client_ip(&req, true), "198.51.100.8");
    }

    #[test]
    fn client_ip_ignores_forwarded_headers_when_untrusted() {
        let mut req = request_with_peer(SocketAddr::from(([10, 0, 0, 2], 443)));
        req.headers_mut().insert(
            HeaderName::from_static("cf-connecting-ip"),
            HeaderValue::from_static("203.0.113.42"),
        );

        assert_eq!(client_ip(&req, false), "10.0.0.2");
    }

    #[test]
    fn client_ip_returns_dash_without_peer_or_forwarded_header() {
        let req = Request::builder().body(Body::empty()).unwrap();

        assert_eq!(client_ip(&req, true), "-");
    }

    #[test]
    fn short_user_agent_keeps_first_token() {
        let ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:151.0)";

        assert_eq!(short_user_agent(ua), "Mozilla/5.0");
    }

    #[test]
    fn short_user_agent_caps_long_tokens() {
        let ua = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/1.0";

        assert_eq!(
            short_user_agent(ua),
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV"
        );
    }
}
