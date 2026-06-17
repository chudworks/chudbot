//! Thin process entrypoint for Chudbot.
//!
//! This crate owns CLI parsing, TOML loading, concrete provider/platform
//! registries, database migrations, and process startup. Bot behavior, Discord
//! I/O, storage, and the web viewer live in the workspace crates that this file
//! wires together.

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
use services::{BootstrapServices, ConfiguredBotRuntime};

/// Git version injected by `build.rs`.
const VERSION: &str = env!("GIT_VERSION");
/// Time allowed for bot work to drain after process shutdown is requested.
const SHUTDOWN_GRACE_PERIOD: Duration = Duration::from_secs(30);
/// Time allowed for platform event pumps to stop after platforms are asked to shut down.
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
    /// Apply pending database migrations without starting providers or platforms.
    Migrate,
    /// Build configured services and run the bot plus web viewer until shutdown.
    Serve,
}

/// Parse CLI input, load the config, and dispatch the selected command.
#[tokio::main]
async fn main() -> ExitCode {
    // Install the rustls backend before any provider or platform client can be
    // constructed.
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .expect("failed to install rustls crypto provider");

    let args = Args::parse();

    // Loading happens before tracing because parse errors need access to the
    // original TOML text for compiler-style diagnostics.
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
        // Runtime commands install logging immediately so startup failures are
        // visible through the configured tracing sink. `check-config` delays
        // this until validation succeeds so invalid configs print only the
        // diagnostic renderer.
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
            // Validation reports are already rendered below with TOML spans;
            // logging them again would duplicate the same errors without the
            // source context.
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
            // Static validation walks the full config graph and keeps all
            // errors on the span-aware diagnostics path before constructing any
            // runtime services.
            config.validate_all(&source)?;
            init_tracing(&config.logging)?;
            log_start(&config_path, &Command::CheckConfig, &config);

            // Provider/media registries and the media store are still built as
            // a bootstrap smoke test, but storage and platform network
            // connections are intentionally left untouched.
            let services = BootstrapServices::build(&config)?;
            tracing::info!(
                llm_providers = services.llms.configured_count(),
                image_providers = services.images.configured_count(),
                video_providers = services.videos.configured_count(),
                audio_providers = services.audio.configured_count(),
                platforms = config.platforms.len(),
                agents = config.bot.agents.len(),
                "configuration is valid"
            );
            Ok(())
        }
        Command::Migrate => {
            // Migrations need only a usable database URL. They do not require
            // provider credentials, platform tokens, or agent references to be
            // runnable.
            config.validate_database()?;
            let storage = SqlxStorage::connect(&config.database.url).await?;
            storage.run_migrations().await?;
            tracing::info!("migrations applied");
            Ok(())
        }
        Command::Serve => {
            let mut config = config;

            // Serve is the full runtime path: validate every static reference
            // before opening durable connections or spawning long-lived tasks.
            config.validate_all(&source)?;

            // Storage is shared by the bot and web viewer. The default privacy
            // fallback is attached before the runtime starts handling turns.
            let storage = SqlxStorage::connect(&config.database.url)
                .await?
                .with_default_privacy(config.default_privacy.clone());

            // Register the git build once per deployment and expose the
            // monotonic app-version id to bot replies, traces, and viewer UI.
            let app_version = storage.register_app_version(VERSION).await?;
            let version_label = format!("v{}", app_version.id);
            tracing::info!(
                version_number = app_version.id,
                git_version = %app_version.git_version,
                first_seen = %app_version.first_seen_at,
                "resolved build version"
            );
            config.bot.version = version_label.clone();

            // Build concrete provider/media registries from named config
            // entries. These are cheap Arc-backed registries until individual
            // requests call a provider.
            let mut services = BootstrapServices::build(&config)?;
            services.web.version = format!("{version_label} ({VERSION})");
            let storage = storage.with_app_version_id(app_version.id);

            // Platforms connect after validation and storage setup so incoming
            // events cannot race ahead of a usable runtime.
            let platforms =
                ConfiguredMessagePlatforms::connect_from_config(&config.platforms).await?;
            let listen = SocketAddr::from_str(&config.web.listen)?;

            // The web API also needs the LLM registry for model metadata, so
            // keep a clone before moving the concrete registries into the bot.
            let llms = services.llms.clone();
            let bot = BotRuntime::<ConfiguredBotRuntime>::new(
                BotRuntimeParts::<ConfiguredBotRuntime> {
                    platforms,
                    storage: storage.clone(),
                    media_store: services.media_store.clone(),
                    llms: services.llms,
                    images: services.images,
                    videos: services.videos,
                    audio: services.audio,
                    events: services.events.clone(),
                    memory: config.memory,
                },
                config.bot,
            );
            let web = WebState::<ConfiguredBotRuntime>::new(
                storage,
                services.media_store,
                llms,
                services.events,
                services.web,
            );

            // `runtime.rs` owns the select loop that runs bot and web tasks,
            // fans out cancellation, drains in-flight bot work, and joins both
            // services before returning.
            run_runtime_services(bot, web, listen).await
        }
    }
}

/// Emit the common startup record once tracing is installed.
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

/// Print rich config diagnostics when possible, then fall back to a plain error
/// plus source chain for operational failures.
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

/// Install the process tracing subscriber from `[logging]` config.
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
