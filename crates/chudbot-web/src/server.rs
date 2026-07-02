use std::net::SocketAddr;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::middleware as axum_middleware;
use axum::routing::get;
use chudbot_api::{BotStorage, LlmProviderRegistry, MediaStore};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_util::sync::CancellationToken;

use crate::events::EventBus;
use crate::middleware::default_trust_forwarded_for;
use crate::static_files::StaticFileCache;
use crate::{api, events, media, middleware, spa, static_files};

/// Compile-time service types that keep web handlers statically dispatched over
/// storage, media, and LLM provider services.
pub trait WebRuntimeTypes: 'static {
    type Storage: BotStorage + Clone + Send + Sync + 'static;
    type Media: MediaStore + Clone + Send + Sync + 'static;
    type Llms: LlmProviderRegistry + Clone + Send + Sync + 'static;
}

/// Concrete dependencies used to run the web service.
pub struct WebRuntimeParts<R: WebRuntimeTypes> {
    pub storage: R::Storage,
    pub media_store: R::Media,
    pub llms: R::Llms,
    pub events: EventBus,
    pub config: WebConfig,
}

/// Runtime controls for the web service.
#[derive(Debug, Clone, Copy)]
pub struct WebRunOptions {
    /// How long graceful shutdown waits for active web connections to drain.
    pub drain_timeout: Duration,
}

/// Runtime dependencies shared by web handlers.
pub(crate) struct WebState<R: WebRuntimeTypes> {
    inner: Arc<WebStateInner<R>>,
    shutdown: CancellationToken,
}

/// Runtime dependencies behind [`WebState`].
#[derive(Debug)]
pub(crate) struct WebStateInner<R: WebRuntimeTypes> {
    pub(crate) storage: R::Storage,
    pub(crate) media_store: R::Media,
    pub(crate) llms: R::Llms,
    pub(crate) events: EventBus,
    pub(crate) config: WebConfig,
    pub(crate) static_files: StaticFileCache,
}

/// Run the web server until the supplied shutdown token is cancelled.
#[tracing::instrument(
    name = "web.run_until_shutdown",
    skip_all,
    fields(
        listen = ?listen,
        listener_count = listen.len(),
        frontend_dir = %parts.config.frontend_dir.display(),
    )
)]
pub async fn run_until_shutdown<R>(
    parts: WebRuntimeParts<R>,
    listen: Vec<SocketAddr>,
    shutdown: CancellationToken,
    options: WebRunOptions,
) -> Result<(), WebServerError>
where
    R: WebRuntimeTypes,
{
    if listen.is_empty() {
        return Err(WebServerError::NoListeners);
    }

    // Bind every configured address before building the router so startup
    // fails cleanly if any listener cannot be opened.
    let mut listeners = Vec::with_capacity(listen.len());
    for address in listen {
        let listener = tokio::net::TcpListener::bind(address)
            .await
            .map_err(|source| WebServerError::Bind {
                listen: address,
                source,
            })?;
        listeners.push((address, listener));
    }

    let WebRuntimeParts {
        storage,
        media_store,
        llms,
        events,
        config,
    } = parts;
    tracing::debug!(
        frontend_dir = %config.frontend_dir.display(),
        title_prefix = %config.title_prefix,
        version = %config.version,
        "constructing web server dependencies"
    );
    let state = WebState {
        inner: Arc::new(WebStateInner::<R> {
            storage,
            media_store,
            llms,
            events,
            config,
            static_files: StaticFileCache::new(),
        }),
        // Share the caller's service token with long-lived handlers such as SSE
        // streams so they stop when the web service is shutting down.
        shutdown: shutdown.clone(),
    };

    // Serve each listener over the same state; the select below turns an early
    // listener exit into a web-service shutdown for the whole listener group.
    let servers = listeners.into_iter().map(|(listen, listener)| {
        let state = state.clone();
        let shutdown = shutdown.clone();
        async move {
            tracing::info!(listen = %listen, "web server listening");
            axum::serve(
                listener,
                router(state).into_make_service_with_connect_info::<SocketAddr>(),
            )
            .with_graceful_shutdown(async move {
                shutdown.cancelled().await;
            })
            .await
            .map_err(|source| WebServerError::Serve { listen, source })?;
            tracing::info!(listen = %listen, "web listener stopped");
            Ok::<(), WebServerError>(())
        }
    });
    let all_servers = futures::future::try_join_all(servers);
    tokio::pin!(all_servers);

    tokio::select! {
        // Normal process shutdown: ask Axum to drain every listener, but do not
        // let stuck connections block process shutdown forever.
        _ = shutdown.cancelled() => {
            tracing::info!(
                timeout_ms = options.drain_timeout.as_millis(),
                "web server shutdown requested"
            );
            match tokio::time::timeout(options.drain_timeout, &mut all_servers).await {
                Ok(result) => {
                    result?;
                }
                Err(_elapsed) => {
                    tracing::warn!(
                        timeout_ms = options.drain_timeout.as_millis(),
                        "web server graceful shutdown timed out"
                    );
                }
            }
        }
        // Early listener exit or failure: cancel the shared token so sibling
        // listeners and SSE streams see the same web-service shutdown signal.
        result = &mut all_servers => {
            shutdown.cancel();
            result?;
        }
    }

    tracing::info!("web server stopped");
    Ok(())
}

/// Build the Axum router that wires API routes, media/static routes, the SPA
/// fallback, and shared middleware for the viewer.
fn router<R>(state: WebState<R>) -> Router
where
    R: WebRuntimeTypes,
{
    tracing::debug!(
        frontend_dir = %state.config.frontend_dir.display(),
        "building web router"
    );
    let trust_forwarded_for = state.config.trust_forwarded_for;
    let api = Router::new()
        .route("/api/config", get(api::get_config::<R>))
        .route("/api/conversations/{id}", get(api::get_conversation::<R>))
        .route(
            "/api/conversations/{id}/events",
            get(events::conversation_events::<R>),
        )
        .layer(static_files::cache_layer(static_files::CACHE_NO_STORE));

    Router::new()
        .merge(api)
        .route("/videos/{name}", get(media::get_video::<R>))
        .route("/audio/{name}", get(media::get_audio::<R>))
        .route("/avatars/{name}", get(media::get_avatar::<R>))
        .route("/guild-icons/{name}", get(media::get_guild_icon::<R>))
        .route("/images/{name}", get(media::get_image::<R>))
        .route("/favicon.ico", get(static_files::get_favicon::<R>))
        .route("/og-image", get(static_files::get_og_image::<R>))
        .route("/robots.txt", get(static_files::get_robots))
        .route("/assets", get(static_files::frontend_assets_root))
        .route(
            "/assets/{*path}",
            get(static_files::get_frontend_asset::<R>),
        )
        .fallback(spa::spa_index::<R>)
        .layer(middleware::x_robots_layer())
        .layer(axum_middleware::from_fn(middleware::block_crawlers))
        .layer(axum_middleware::from_fn_with_state(
            trust_forwarded_for,
            middleware::access_log,
        ))
        .with_state(state)
}

/// Web server startup error.
#[derive(Debug, Error)]
pub enum WebServerError {
    /// Returned when no TCP listen addresses are configured.
    #[error("no web listen addresses configured")]
    NoListeners,
    /// Returned before serving starts if a configured address cannot be bound.
    #[error("failed to bind web listener {listen}: {source}")]
    Bind {
        /// Address that failed to bind.
        listen: SocketAddr,
        /// I/O error returned by the listener bind.
        #[source]
        source: std::io::Error,
    },
    /// Returned after a listener is bound if Axum exits with an I/O error.
    #[error("web listener {listen} failed: {source}")]
    Serve {
        /// Address whose server task failed.
        listen: SocketAddr,
        /// I/O error returned by Axum.
        #[source]
        source: std::io::Error,
    },
}

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

impl<R> WebState<R>
where
    R: WebRuntimeTypes,
{
    pub(crate) fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }
}

impl<R> std::fmt::Debug for WebState<R>
where
    R: WebRuntimeTypes,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebState").finish_non_exhaustive()
    }
}

impl<R> Clone for WebState<R>
where
    R: WebRuntimeTypes,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            shutdown: self.shutdown.clone(),
        }
    }
}

impl<R> Deref for WebState<R>
where
    R: WebRuntimeTypes,
{
    type Target = WebStateInner<R>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
