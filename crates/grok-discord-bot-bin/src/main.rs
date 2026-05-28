use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;

use clap::{Parser, Subcommand};
use grok_discord_bot_core::{
    AnyImageProvider, AnyProvider, AnyVideoProvider, Config, Db, ImageProviderKind,
    LlmProviderKind, VideoProviderKind,
};

mod bot;
mod commands;
mod web;

const VERSION: &str = env!("GIT_VERSION");

#[derive(Debug, Parser)]
#[command(name = "grok")]
#[command(version = VERSION)]
#[command(about = "Discord bot integrating xAI Grok / Anthropic Claude, with a companion web viewer")]
struct Args {
    /// Path to the TOML config file.
    #[arg(long, short, default_value = "config.toml", global = true)]
    config: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Run the Discord bot gateway loop.
    Bot,
    /// Run the web viewer HTTP server.
    Web,
    /// Apply pending database migrations.
    Migrate,
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
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
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            report_error(&*e);
            std::process::ExitCode::FAILURE
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
        Cmd::Bot => {
            let db = Db::connect(&config.postgres.url).await?;
            // Build one provider instance per kind we have credentials
            // for. Personas pick a (provider, model) pair at turn time
            // and route through the matching entry here.
            let mut providers: HashMap<LlmProviderKind, AnyProvider> = HashMap::new();
            if let Some(cfg) = config.llm.xai.clone() {
                providers.insert(LlmProviderKind::Xai, AnyProvider::from(cfg));
            }
            if let Some(cfg) = config.llm.anthropic.clone() {
                providers.insert(LlmProviderKind::Anthropic, AnyProvider::from(cfg));
            }
            // Image and video providers follow the same pattern: build
            // one instance per `[image.<kind>]` / `[video.<kind>]` block
            // that's actually configured. Personas pick a kind at turn
            // time; a persona that names a kind with no matching block
            // is rejected at config-validation time, so the map lookups
            // here are always satisfied at runtime.
            let mut image_providers: HashMap<ImageProviderKind, AnyImageProvider> = HashMap::new();
            if let Some(cfg) = config.image.xai.clone() {
                image_providers.insert(ImageProviderKind::Xai, AnyImageProvider::from(cfg));
            }
            let mut video_providers: HashMap<VideoProviderKind, AnyVideoProvider> = HashMap::new();
            if let Some(cfg) = config.video.xai.clone() {
                video_providers.insert(VideoProviderKind::Xai, AnyVideoProvider::from(cfg));
            }
            tracing::info!(
                providers = ?providers.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
                personas = ?config.personas.keys().collect::<Vec<_>>(),
                default_persona = %config.default_persona,
                image_providers = ?image_providers.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
                video_providers = ?video_providers.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
                "starting bot"
            );
            bot::run(
                config.discord.token.clone(),
                config.discord.dev_guild_id,
                db,
                providers,
                config.personas.clone(),
                config.default_persona.clone(),
                config.web.base_url.clone(),
                config.default_privacy.clone(),
                config.storage.clone(),
                image_providers,
                video_providers,
            )
            .await?;
        }
        Cmd::Web => {
            let db = Db::connect(&config.postgres.url).await?;
            let listen = SocketAddr::from_str(&config.web.listen)?;
            web::run(
                db,
                listen,
                config.storage.images_dir.clone(),
                config.storage.videos_dir.clone(),
            )
            .await?;
        }
        Cmd::Migrate => {
            let db = Db::connect(&config.postgres.url).await?;
            db.migrate().await?;
            tracing::info!("migrations applied");
        }
    }

    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    #[cfg(distribute)]
    fmt().json().with_env_filter(filter).init();

    #[cfg(not(distribute))]
    fmt().pretty().with_env_filter(filter).init();
}
