mod config;
mod errors;
mod platforms;
mod runtime;
mod services;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;
use std::time::Duration;

use chudbot_bot::{BotRuntime, BotRuntimeParts};
use chudbot_storage_sqlx::SqlxStorage;
use chudbot_web::WebState;
use clap::{Parser, Subcommand};
use config::{LogFormat, LoggingConfig, LoggingFilterError, RuntimeConfig};
use errors::BinError;
use platforms::ConfiguredMessagePlatforms;
use runtime::run_runtime_services;
use services::ServicePlan;

const VERSION: &str = env!("GIT_VERSION");
const SHUTDOWN_GRACE_PERIOD: Duration = Duration::from_secs(30);
const PLATFORM_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Parser)]
#[command(name = "chudbot")]
#[command(version = VERSION)]
struct Args {
    /// Path to the TOML config file.
    #[arg(long, short, default_value = "config.toml", global = true)]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Load the configuration and verify static references.
    CheckConfig,
    /// Apply pending database migrations.
    Migrate,
    /// Build configured services and start the process.
    Serve,
}

#[tokio::main]
async fn main() -> ExitCode {
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    let args = Args::parse();
    let config = match RuntimeConfig::load(&args.config) {
        Ok(config) => config,
        Err(error) => {
            init_tracing(&LoggingConfig::default()).expect("default logging filter must be valid");
            let error = BinError::Config(error);
            tracing::error!(error = %error, "chudbot failed");
            report_error(&error);
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = init_tracing(&config.logging) {
        let error = BinError::from(error);
        report_error(&error);
        return ExitCode::FAILURE;
    }
    tracing::info!(
        version = VERSION,
        config = %args.config.display(),
        command = ?args.command,
        agents = config.bot.agents.len(),
        llm_providers = config.llm.len(),
        image_providers = config.image.len(),
        video_providers = config.video.len(),
        audio_providers = config.audio.len(),
        platforms = config.platforms.len(),
        memory_enabled = config.memory.enabled,
        "chudbot starting"
    );
    match run(args.command, args.config, config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(error = %error, "chudbot failed");
            report_error(&error);
            ExitCode::FAILURE
        }
    }
}

#[tracing::instrument(
    name = "bin.run",
    skip_all,
    fields(config = %config_path.display(), command = ?command)
)]
async fn run(
    command: Command,
    config_path: PathBuf,
    config: RuntimeConfig,
) -> Result<(), BinError> {
    match command {
        Command::CheckConfig => {
            config.validate()?;
            let plan = ServicePlan::build(&config)?;
            tracing::info!(
                llm_providers = plan.llms.configured_count(),
                image_providers = plan.images.configured_count(),
                video_providers = plan.videos.configured_count(),
                audio_providers = plan.audio.configured_count(),
                platforms = config.platforms.len(),
                agents = config.bot.agents.len(),
                "configuration is valid"
            );
            Ok(())
        }
        Command::Migrate => {
            config.validate_database()?;
            let storage = SqlxStorage::connect(&config.database.url).await?;
            storage.run_migrations().await?;
            tracing::info!("migrations applied");
            Ok(())
        }
        Command::Serve => {
            let mut config = config;
            config.validate()?;
            let storage = SqlxStorage::connect(&config.database.url)
                .await?
                .with_default_privacy(config.default_privacy.clone());
            let app_version = storage.register_app_version(VERSION).await?;
            let version_label = format!("v{}", app_version.id);
            tracing::info!(
                version_number = app_version.id,
                git_version = %app_version.git_version,
                first_seen = %app_version.first_seen_at,
                "resolved build version"
            );
            config.bot.version = version_label.clone();
            let mut plan = ServicePlan::build(&config)?;
            plan.web.version = format!("{version_label} ({VERSION})");
            let storage = storage.with_app_version_id(app_version.id);
            let platforms =
                ConfiguredMessagePlatforms::connect_from_config(&config.platforms).await?;
            let listen = SocketAddr::from_str(&config.web.listen)?;
            let bot = BotRuntime::new(
                BotRuntimeParts {
                    platforms,
                    storage: storage.clone(),
                    media_store: plan.media_store.clone(),
                    llms: plan.llms,
                    images: plan.images,
                    videos: plan.videos,
                    audio: plan.audio,
                    events: plan.events.clone(),
                    memory: config.memory,
                },
                config.bot,
            );
            let web = WebState::new(storage, plan.media_store, plan.events, plan.web);
            run_runtime_services(bot, web, listen).await
        }
    }
}

fn report_error(error: &BinError) {
    eprintln!("Error: {error}");
    let mut source = std::error::Error::source(error);
    while let Some(error) = source {
        eprintln!();
        eprintln!("Caused by:");
        for line in error.to_string().lines() {
            eprintln!("  {line}");
        }
        source = error.source();
    }
}

fn init_tracing(config: &LoggingConfig) -> Result<(), LoggingFilterError> {
    use tracing_subscriber::fmt;
    let filter = config.filter()?;
    match config.format {
        LogFormat::Pretty => fmt()
            .pretty()
            .with_ansi(config.ansi)
            .with_env_filter(filter)
            .init(),
        LogFormat::Compact => fmt()
            .compact()
            .with_ansi(config.ansi)
            .with_env_filter(filter)
            .init(),
        LogFormat::Json => fmt()
            .json()
            .with_ansi(config.ansi)
            .with_env_filter(filter)
            .init(),
    }
    Ok(())
}
