use clap::{Parser, Subcommand};

mod bot;
mod web;

const VERSION: &str = env!("GIT_VERSION");

#[derive(Debug, Parser)]
#[command(name = "grok")]
#[command(version = VERSION)]
#[command(about = "Discord bot integrating xAI Grok, with a companion web viewer")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Run the Discord bot gateway loop.
    Bot,
    /// Run the web viewer HTTP server.
    Web,
    /// Run pending database migrations.
    Migrate,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    tracing::info!(version = VERSION, "grok-discord-bot starting");

    let args = Args::parse();
    match args.cmd {
        Cmd::Bot => bot::run().await?,
        Cmd::Web => web::run().await?,
        Cmd::Migrate => {
            tracing::info!("no migrations defined yet");
        }
    }
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    #[cfg(distribute)]
    fmt().json().with_env_filter(filter).init();

    #[cfg(not(distribute))]
    fmt().pretty().with_env_filter(filter).init();
}
