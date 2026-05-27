use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;

use clap::{Parser, Subcommand};
use grok_discord_bot_core::{AnyProvider, Config, Db};

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
            let llm = AnyProvider::from_config(&config.llm)?;
            tracing::info!(model = %llm_name(&llm), "starting bot");
            bot::run(
                config.discord.token.clone(),
                db,
                llm,
                config.web.base_url.clone(),
                config.default_privacy.clone(),
            )
            .await?;
        }
        Cmd::Web => {
            let db = Db::connect(&config.postgres.url).await?;
            let listen = SocketAddr::from_str(&config.web.listen)?;
            web::run(db, listen).await?;
        }
        Cmd::Migrate => {
            let db = Db::connect(&config.postgres.url).await?;
            db.migrate().await?;
            tracing::info!("migrations applied");
        }
    }

    Ok(())
}

fn llm_name(p: &AnyProvider) -> &str {
    use grok_discord_bot_core::LlmProvider;
    p.name()
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    #[cfg(distribute)]
    fmt().json().with_env_filter(filter).init();

    #[cfg(not(distribute))]
    fmt().pretty().with_env_filter(filter).init();
}
