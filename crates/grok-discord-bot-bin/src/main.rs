use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use grok_discord_bot_core::{
    AnyImageProvider, AnyProvider, AnyVideoProvider, Config, Db, ImageProviderKind,
    LlmProviderKind, VideoProviderKind,
};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

mod app;
mod avatars;
mod bot;
mod commands;
mod titles;
mod web;

use app::{AppState, new_event_channel};

const VERSION: &str = env!("GIT_VERSION");

/// How long the `serve` shutdown handler waits for in-flight tasks
/// (turn handlers, title generation, avatar fetches, slash command
/// dispatchers) to drain after Ctrl+C before exiting anyway.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(30);

#[derive(Debug, Parser)]
#[command(name = "grok")]
#[command(version = VERSION)]
#[command(about = "Discord bot + companion web viewer, integrating xAI Grok / Anthropic Claude")]
struct Args {
    /// Path to the TOML config file.
    #[arg(long, short, default_value = "config.toml", global = true)]
    config: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Run both the Discord gateway loop and the web viewer in one
    /// process. Background work (title generation, avatar caching) is
    /// drained on Ctrl+C with a 30s grace period.
    Serve,
    /// Apply pending database migrations.
    Migrate,
}

#[tokio::main]
async fn main() -> ExitCode {
    // Pin rustls' crypto provider before any TLS work. Several crates
    // in the tree (sqlx, reqwest, twilight, rustls-platform-verifier)
    // each enable rustls with potentially different provider features,
    // which leaves the process-wide default ambiguous and causes a
    // panic at first TLS handshake. Pick one explicitly.
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    init_tracing();
    let args = Args::parse();

    tracing::info!(
        version = VERSION,
        config = %args.config.display(),
        "grok-discord-bot starting"
    );

    match run(args).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            report_error(&*e);
            ExitCode::FAILURE
        }
    }
}

/// Pretty-print an error and its full source chain to stderr. The first
/// line carries the top-level Display; each `source()` link below is
/// indented by two spaces, which preserves the column alignment of
/// multi-line messages like `toml::de::Error`'s arrow diagrams.
fn report_error(e: &(dyn std::error::Error + 'static)) {
    eprintln!("Error: {e}");
    let mut src = e.source();
    while let Some(s) = src {
        eprintln!();
        eprintln!("Caused by:");
        for line in s.to_string().lines() {
            eprintln!("  {line}");
        }
        src = s.source();
    }
}

async fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::load(&args.config)?;

    match args.cmd {
        Cmd::Serve => serve(config).await,
        Cmd::Migrate => {
            let db = Db::connect(&config.postgres.url).await?;
            db.migrate().await?;
            tracing::info!("migrations applied");
            Ok(())
        }
    }
}

/// Build the shared [`AppState`], spawn the bot and web halves on the
/// shared [`TaskTracker`], and block until Ctrl+C. On signal, the
/// cancellation token is fired and the tracker is given
/// [`SHUTDOWN_GRACE`] to drain in-flight tasks.
async fn serve(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let db = Db::connect(&config.postgres.url).await?;

    let mut providers: HashMap<LlmProviderKind, AnyProvider> = HashMap::new();
    if let Some(cfg) = config.llm.xai.clone() {
        providers.insert(LlmProviderKind::Xai, AnyProvider::from(cfg));
    }
    if let Some(cfg) = config.llm.anthropic.clone() {
        providers.insert(LlmProviderKind::Anthropic, AnyProvider::from(cfg));
    }

    let mut image_providers: HashMap<ImageProviderKind, AnyImageProvider> = HashMap::new();
    if let Some(cfg) = config.image.xai.clone() {
        image_providers.insert(ImageProviderKind::Xai, AnyImageProvider::from(cfg));
    }

    let mut video_providers: HashMap<VideoProviderKind, AnyVideoProvider> = HashMap::new();
    if let Some(cfg) = config.video.xai.clone() {
        video_providers.insert(VideoProviderKind::Xai, AnyVideoProvider::from(cfg));
    }

    let listen: SocketAddr = SocketAddr::from_str(&config.web.listen)?;

    tracing::info!(
        providers = ?providers.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
        image_providers = ?image_providers.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
        video_providers = ?video_providers.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
        personas = ?config.personas.keys().collect::<Vec<_>>(),
        default_persona = %config.default_persona,
        listen = %listen,
        frontend_dir = %config.web.frontend_dir.display(),
        "starting serve"
    );

    let app = Arc::new(AppState {
        db,
        providers,
        image_providers,
        video_providers,
        personas: config.personas,
        default_persona: config.default_persona,
        default_privacy: config.default_privacy,
        web_base_url: config.web.base_url,
        web_frontend_dir: config.web.frontend_dir,
        web_title_prefix: config.web.title_prefix,
        web_favicon_path: config.web.favicon_path,
        storage: config.storage,
        download_http: reqwest::Client::new(),
        events: new_event_channel(),
        cancel: CancellationToken::new(),
        tracker: TaskTracker::new(),
    });

    // Both halves of the binary live on the same tracker so a Ctrl+C
    // drains them together. We clone the tracker/cancel out of `app`
    // before passing `app` into the spawned futures to avoid a
    // borrow-and-move conflict on the `app.tracker.spawn(...)` form.
    let tracker = app.tracker.clone();

    {
        let app_clone = Arc::clone(&app);
        tracker.spawn(async move {
            if let Err(err) = web::run(app_clone, listen).await {
                tracing::error!(error = %err, "web server exited with error");
            }
        });
    }

    {
        let app_clone = Arc::clone(&app);
        let token = config.discord.token.clone();
        let dev_guild_id = config.discord.dev_guild_id;
        tracker.spawn(async move {
            if let Err(err) = bot::run(app_clone, token, dev_guild_id).await {
                tracing::error!(error = %err, "bot loop exited with error");
            }
        });
    }

    // Close the tracker to new spawns from THIS scope — background
    // tasks spawned later (e.g. avatar fetcher, title gen) get tracked
    // via `app.tracker.spawn(...)` and the `tracker.wait()` below will
    // wait for those too. `close()` doesn't stop new spawns; it just
    // marks the tracker as "no longer accepting new work from outside
    // the running tasks" so the eventual `wait()` can return.

    shutdown_signal().await;

    app.cancel.cancel();
    app.tracker.close();

    match tokio::time::timeout(SHUTDOWN_GRACE, app.tracker.wait()).await {
        Ok(()) => tracing::info!("all background tasks drained cleanly"),
        Err(_) => tracing::warn!(
            grace_seconds = SHUTDOWN_GRACE.as_secs(),
            "shutdown grace period exceeded; exiting with in-flight work"
        ),
    }

    Ok(())
}

/// Wait for a shutdown signal. Resolves on either SIGINT (Ctrl+C, which
/// is what `serve.sh stop` sends via `tmux send-keys C-c`) or SIGTERM
/// (a plain `kill`, systemd, or a container stop). Handling both means
/// the graceful drain runs no matter how the process is asked to stop.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install SIGINT handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received SIGINT; shutting down"),
        () = terminate => tracing::info!("received SIGTERM; shutting down"),
    }
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    #[cfg(distribute)]
    fmt().json().with_env_filter(filter).init();

    #[cfg(not(distribute))]
    fmt().pretty().with_env_filter(filter).init();
}
