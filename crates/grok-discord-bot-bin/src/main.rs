use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use grok_discord_bot_core::{
    AnyProvider, Config, Db, LlmProviderKind, imagegen::ImageGenerator, videogen::VideoGenerator,
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
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
            // Image generation rides on the xAI API; expose the tool
            // whenever an xAI key is present, regardless of which
            // providers any persona happens to use for chat.
            let image_gen = config
                .llm
                .xai
                .as_ref()
                .map(|x| Arc::new(ImageGenerator::new(x.api_key.clone())));
            let video_gen = config
                .llm
                .xai
                .as_ref()
                .map(|x| Arc::new(VideoGenerator::new(x.api_key.clone())));
            tracing::info!(
                providers = ?providers.keys().map(|k| k.as_str()).collect::<Vec<_>>(),
                personas = ?config.personas.keys().collect::<Vec<_>>(),
                default_persona = %config.default_persona,
                image_gen = image_gen.is_some(),
                video_gen = video_gen.is_some(),
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
                image_gen,
                video_gen,
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
