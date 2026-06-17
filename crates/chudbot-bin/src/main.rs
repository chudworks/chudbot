mod config;
mod diagnostics;
mod errors;
mod platforms;
mod runtime;
mod services;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;
use std::time::Duration;

use chudbot_bot::{BotRuntime, BotRuntimeParts};
use chudbot_storage_sqlx::SqlxStorage;
use chudbot_web::WebState;
use clap::{Parser, Subcommand};
use config::{LoadedRuntimeConfig, LogFormat, LoggingConfig, LoggingFilterError, RuntimeConfig};
use diagnostics::render_toml_error_for_stderr;
use errors::{BinError, ConfigError};
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
    let loaded = match RuntimeConfig::load_with_source(&args.config) {
        Ok(loaded) => loaded,
        Err(error) => {
            let error = BinError::Config(error);
            report_error(&error);
            return ExitCode::FAILURE;
        }
    };
    let check_config = matches!(args.command, Command::CheckConfig);
    if !check_config {
        if let Err(error) = init_tracing(&loaded.config.logging) {
            let error = BinError::from(error);
            report_error(&error);
            return ExitCode::FAILURE;
        }
        log_start(&args.config, &args.command, &loaded.config);
    }
    match run(args.command, args.config, loaded).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            if !matches!(error, BinError::ConfigValidation(_)) {
                tracing::error!(error = %error, "chudbot failed");
            }
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
    loaded: LoadedRuntimeConfig,
) -> Result<(), BinError> {
    let LoadedRuntimeConfig { config, source } = loaded;
    match command {
        Command::CheckConfig => {
            config.validate_all(&source)?;
            init_tracing(&config.logging)?;
            log_start(&config_path, &Command::CheckConfig, &config);
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
            config.validate_all(&source)?;
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

fn log_start(config_path: &Path, command: &Command, config: &RuntimeConfig) {
    tracing::info!(
        version = VERSION,
        config = %config_path.display(),
        command = ?command,
        agents = config.bot.agents.len(),
        llm_providers = config.llm.len(),
        image_providers = config.image.len(),
        video_providers = config.video.len(),
        audio_providers = config.audio.len(),
        platforms = config.platforms.len(),
        memory_enabled = config.memory.enabled,
        "chudbot starting"
    );
}

fn report_error(error: &BinError) {
    match error {
        BinError::ConfigValidation(report) => {
            eprint!("{}", report.render_for_stderr());
            return;
        }
        BinError::Config(ConfigError::Parse {
            path,
            content,
            source,
        }) => {
            eprint!(
                "{}",
                render_toml_error_for_stderr(path, content, source.as_ref())
            );
            return;
        }
        _ => {}
    }

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
