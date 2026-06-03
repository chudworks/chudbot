use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use chudbot_api::{
    BotStorage, ChannelRef, EventSink, ExternalId, FetchMessages, GeneratedImage, ImageGenerator,
    ImageRequest, LlmBackend, MediaStore, MessagePlatform, MessageRef, ModelStep, ModelStepRequest,
    PlatformCommandDefinition, PlatformCommandResponse, PlatformEvent, PlatformMessage,
    PlatformMessageRelationship, PostedMessage, PrivacyMode, ProviderName, ReactionKind,
    SendMessage, UserProfile, VideoGenerator, VideoJobId, VideoJobStatus, VideoRequest,
};
use chudbot_asset_local::LocalMediaStore;
use chudbot_bot::{
    BotConfig, BotRunOptions, BotRuntime, BotRuntimeParts, ImageGeneratorRegistry,
    LlmProviderRegistry, MemoryConfig, MessagePlatformRegistry, VideoGeneratorRegistry,
};
use chudbot_storage_sqlx::SqlxStorage;
use chudbot_web::{EventBus, WebConfig, WebState};
use clap::{Parser, Subcommand};
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

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

/// Run fully constructed bot and web services until a process shutdown signal
/// or either service exits.
pub async fn run_runtime_services<P, S, M, L, I, V, E>(
    bot: BotRuntime<P, S, M, L, I, V, E>,
    web: WebState<S, M>,
    listen: SocketAddr,
) -> Result<(), BinError>
where
    P: MessagePlatformRegistry + Clone + 'static,
    S: BotStorage + Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
    L: LlmProviderRegistry + Clone + 'static,
    I: ImageGeneratorRegistry + Clone + 'static,
    V: VideoGeneratorRegistry + Clone + 'static,
    E: EventSink + Clone + 'static,
{
    let shutdown = CancellationToken::new();
    let bot_shutdown = shutdown.child_token();
    let web_shutdown = shutdown.child_token();
    let mut bot_task = tokio::spawn(async move {
        bot.run_with_options(
            bot_shutdown,
            BotRunOptions {
                drain_timeout: SHUTDOWN_GRACE_PERIOD,
            },
        )
        .await
        .map_err(BinError::Bot)
    });
    let mut web_task = tokio::spawn(async move {
        chudbot_web::run_until_shutdown(web, listen, async move {
            web_shutdown.cancelled().await;
        })
        .await
        .map_err(BinError::Web)
    });

    let mut bot_result = None;
    let mut web_result = None;
    tokio::select! {
        _ = shutdown_signal() => {
            tracing::info!("process shutdown requested");
            shutdown.cancel();
        }
        result = &mut bot_task => {
            tracing::info!("bot service exited; shutting down remaining services");
            shutdown.cancel();
            bot_result = Some(join_service_result("bot", result));
        }
        result = &mut web_task => {
            tracing::info!("web service exited; shutting down remaining services");
            shutdown.cancel();
            web_result = Some(join_service_result("web", result));
        }
    }

    let bot_result = match bot_result {
        Some(result) => result,
        None => join_service_result("bot", bot_task.await),
    };
    let web_result = match web_result {
        Some(result) => result,
        None => join_service_result("web", web_task.await),
    };
    bot_result?;
    web_result?;
    tracing::info!("all services stopped");
    Ok(())
}

fn join_service_result(
    task: &'static str,
    result: Result<Result<(), BinError>, JoinError>,
) -> Result<(), BinError> {
    match result {
        Ok(result) => result,
        Err(source) => {
            log_service_join_error(task, &source);
            Err(BinError::TaskJoin { task, source })
        }
    }
}

fn log_service_join_error(service: &'static str, error: &JoinError) {
    if error.is_cancelled() {
        tracing::warn!(service, error = %error, "service task was cancelled");
    } else if error.is_panic() {
        tracing::error!(service, error = %error, "service task panicked");
    } else {
        tracing::warn!(service, error = %error, "service task join failed");
    }
}

/// Wait for SIGINT or SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received SIGINT"),
        () = terminate => tracing::info!("received SIGTERM"),
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

/// Full process configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Postgres database connection config.
    pub database: DatabaseConfig,
    /// Process logging/tracing config.
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Bot agent/platform binding config.
    pub bot: BotConfig,
    /// User-memory runtime config.
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Deployment fallback privacy mode before a guild stores an override.
    #[serde(default = "default_privacy")]
    pub default_privacy: PrivacyMode,
    /// Named LLM provider configs.
    #[serde(default)]
    pub llm: BTreeMap<ProviderName, LlmProviderConfig>,
    /// Named image-generation provider configs.
    #[serde(default)]
    pub image: BTreeMap<ProviderName, ImageProviderConfig>,
    /// Named video-generation provider configs.
    #[serde(default)]
    pub video: BTreeMap<ProviderName, VideoProviderConfig>,
    /// Named message platform configs.
    #[serde(default)]
    pub platforms: BTreeMap<chudbot_api::PlatformName, MessagePlatformConfig>,
    /// Web viewer config.
    pub web: WebRuntimeConfig,
    /// Local media storage config.
    #[serde(default)]
    pub storage: LocalStorageConfig,
}

/// Process logging/tracing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Tracing filter expression, e.g. `info` or
    /// `info,chudbot=debug`.
    #[serde(default = "default_log_filter")]
    pub filter: String,
    /// Output format.
    #[serde(default)]
    pub format: LogFormat,
    /// Whether ANSI color/style codes are emitted.
    #[serde(default = "default_log_ansi")]
    pub ansi: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            filter: default_log_filter(),
            format: LogFormat::default(),
            ansi: default_log_ansi(),
        }
    }
}

impl LoggingConfig {
    fn filter(&self) -> Result<EnvFilter, LoggingFilterError> {
        EnvFilter::try_new(&self.filter).map_err(|source| LoggingFilterError {
            filter: self.filter.clone(),
            source,
        })
    }
}

#[derive(Debug, Error)]
#[error("invalid logging filter `{filter}`")]
pub struct LoggingFilterError {
    filter: String,
    #[source]
    source: tracing_subscriber::filter::ParseError,
}

/// Log output format.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// Pretty human-readable logs.
    #[default]
    Pretty,
    /// Compact line-oriented logs.
    Compact,
    /// JSON logs.
    Json,
}

fn default_log_filter() -> String {
    "info".to_string()
}

fn default_log_ansi() -> bool {
    true
}

fn default_privacy() -> PrivacyMode {
    PrivacyMode::OptIn
}

impl RuntimeConfig {
    /// Load config from TOML.
    #[tracing::instrument(name = "config.load", skip_all, fields(path = %path.display()))]
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        tracing::debug!("reading config file");
        let content = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        tracing::debug!(bytes = content.len(), "read config file");
        let mut config: Self = toml::from_str(&content).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        if config.bot.version.is_empty() {
            config.bot.version = VERSION.to_string();
            tracing::debug!(version = VERSION, "defaulted bot version from binary");
        }
        tracing::info!(
            agents = config.bot.agents.len(),
            llm_providers = config.llm.len(),
            image_providers = config.image.len(),
            video_providers = config.video.len(),
            platforms = config.platforms.len(),
            "loaded runtime config"
        );
        Ok(config)
    }

    /// Validate cross references.
    #[tracing::instrument(
        name = "config.validate",
        skip_all,
        fields(
            agents = self.bot.agents.len(),
            llm_providers = self.llm.len(),
            image_providers = self.image.len(),
            video_providers = self.video.len(),
            platforms = self.platforms.len(),
        )
    )]
    pub fn validate(&self) -> Result<(), BinError> {
        self.validate_database()?;
        self.logging.filter()?;
        self.bot.validate()?;
        self.memory.compaction_interval_seconds()?;
        self.memory.diary_backfill_window_seconds()?;
        self.memory.diary_interval_seconds()?;

        let provider_names = self.llm.keys().collect::<BTreeSet<_>>();
        let image_provider_names = self.image.keys().collect::<BTreeSet<_>>();
        let video_provider_names = self.video.keys().collect::<BTreeSet<_>>();
        if self.memory.enabled {
            if !provider_names.contains(&self.memory.provider) {
                tracing::warn!(
                    provider = %self.memory.provider,
                    "memory references missing provider config"
                );
                return Err(BinError::MissingMemoryProviderConfig {
                    provider: self.memory.provider.clone(),
                });
            }
            if matches!(
                self.llm.get(&self.memory.provider),
                Some(LlmProviderConfig::OpenAiCompat { .. })
            ) {
                tracing::warn!(
                    provider = %self.memory.provider,
                    kind = "openai_compat",
                    "memory references an LLM provider kind that is planned but not implemented"
                );
                return Err(BinError::UnimplementedMemoryProvider {
                    provider: self.memory.provider.clone(),
                    kind: "openai_compat",
                });
            }
        }
        for (agent_name, agent) in &self.bot.agents {
            if !provider_names.contains(&agent.provider) {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %agent.provider,
                    "agent references missing provider config"
                );
                return Err(BinError::MissingProviderConfig {
                    agent: agent_name.clone(),
                    provider: agent.provider.clone(),
                });
            }
            if matches!(
                self.llm.get(&agent.provider),
                Some(LlmProviderConfig::OpenAiCompat { .. })
            ) {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %agent.provider,
                    kind = "openai_compat",
                    "agent references an LLM provider kind that is planned but not implemented"
                );
                return Err(BinError::UnimplementedLlmProvider {
                    agent: agent_name.clone(),
                    provider: agent.provider.clone(),
                    kind: "openai_compat",
                });
            }
            if let Some(binding) = &agent.image_generation
                && !image_provider_names.contains(&binding.provider)
            {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %binding.provider,
                    model = %binding.model,
                    "agent references missing image provider config"
                );
                return Err(BinError::MissingImageProviderConfig {
                    agent: agent_name.clone(),
                    provider: binding.provider.clone(),
                });
            }
            if let Some(binding) = &agent.video_generation
                && !video_provider_names.contains(&binding.provider)
            {
                tracing::warn!(
                    agent = %agent_name,
                    provider = %binding.provider,
                    model = %binding.model,
                    "agent references missing video provider config"
                );
                return Err(BinError::MissingVideoProviderConfig {
                    agent: agent_name.clone(),
                    provider: binding.provider.clone(),
                });
            }
        }

        for platform in self.bot.platforms.keys() {
            if !self.platforms.contains_key(platform) {
                tracing::warn!(
                    platform = %platform,
                    "bot platform binding references missing platform config"
                );
                return Err(BinError::MissingPlatformConfig {
                    platform: platform.clone(),
                });
            }
        }

        SocketAddr::from_str(&self.web.listen)?;
        tracing::info!("runtime config validated");
        Ok(())
    }

    fn validate_database(&self) -> Result<(), BinError> {
        if self.database.url.trim().is_empty() {
            tracing::warn!("database URL is empty");
            return Err(BinError::MissingDatabaseUrl);
        }
        Ok(())
    }
}

/// Postgres database connection settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Standard `postgres://user:pass@host/db` URL.
    pub url: String,
}

/// Web listener plus viewer config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebRuntimeConfig {
    /// Socket address to listen on.
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Browser tab title prefix.
    pub title_prefix: String,
    /// Directory containing the built frontend bundle.
    pub frontend_dir: PathBuf,
    /// Optional favicon served at /favicon.ico.
    #[serde(default)]
    pub favicon_path: Option<PathBuf>,
}

impl WebRuntimeConfig {
    fn viewer_config(&self) -> WebConfig {
        WebConfig {
            title_prefix: self.title_prefix.clone(),
            version: VERSION.to_string(),
            frontend_dir: self.frontend_dir.clone(),
            favicon_path: self.favicon_path.clone(),
        }
    }
}

fn default_listen() -> String {
    "127.0.0.1:1860".to_string()
}

/// Local storage directories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalStorageConfig {
    /// Image directory.
    #[serde(default = "default_images_dir")]
    pub images_dir: PathBuf,
    /// Video directory.
    #[serde(default = "default_videos_dir")]
    pub videos_dir: PathBuf,
    /// Avatar directory.
    #[serde(default = "default_avatars_dir")]
    pub avatars_dir: PathBuf,
    /// Public base URL for media, usually the same host as the web viewer.
    #[serde(default)]
    pub public_base_url: Option<String>,
}

impl Default for LocalStorageConfig {
    fn default() -> Self {
        Self {
            images_dir: default_images_dir(),
            videos_dir: default_videos_dir(),
            avatars_dir: default_avatars_dir(),
            public_base_url: None,
        }
    }
}

fn default_images_dir() -> PathBuf {
    PathBuf::from("images")
}

fn default_videos_dir() -> PathBuf {
    PathBuf::from("videos")
}

fn default_avatars_dir() -> PathBuf {
    PathBuf::from("avatars")
}

/// Named LLM provider config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LlmProviderConfig {
    /// xAI provider.
    Xai {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
    /// OpenAI provider.
    #[serde(rename = "openai")]
    OpenAi {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
    /// Anthropic provider placeholder.
    Anthropic {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
    /// OpenAI-compatible provider placeholder.
    #[serde(rename = "openai_compat")]
    OpenAiCompat {
        /// Base URL.
        base_url: String,
        /// Optional API key.
        #[serde(default)]
        api_key: Option<String>,
    },
}

/// Named image-generation provider config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageProviderConfig {
    /// OpenAI image generation provider.
    #[serde(rename = "openai")]
    OpenAi {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
    /// xAI image generation provider.
    Xai {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
}

/// Named video-generation provider config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VideoProviderConfig {
    /// xAI video generation provider.
    Xai {
        /// API key.
        api_key: String,
        /// Optional base URL override.
        #[serde(default)]
        base_url: Option<String>,
    },
}

/// Named message platform config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MessagePlatformConfig {
    /// Discord platform placeholder.
    Discord {
        /// Bot token.
        token: String,
        /// Optional development guild.
        #[serde(default)]
        dev_guild_id: Option<String>,
    },
}

/// Services that can be built before storage/platform implementations exist.
#[derive(Debug)]
pub struct ServicePlan {
    /// LLM provider registry.
    pub llms: ConfiguredLlmProviders,
    /// Image generation registry.
    pub images: ConfiguredImageGenerators,
    /// Video generation registry.
    pub videos: ConfiguredVideoGenerators,
    /// Local media store.
    pub media_store: LocalMediaStore,
    /// Web event bus.
    pub events: EventBus,
    /// Web config.
    pub web: WebConfig,
}

impl ServicePlan {
    #[tracing::instrument(
        name = "services.build",
        skip_all,
        fields(
            images_dir = %config.storage.images_dir.display(),
            videos_dir = %config.storage.videos_dir.display(),
            avatars_dir = %config.storage.avatars_dir.display(),
            frontend_dir = %config.web.frontend_dir.display(),
        )
    )]
    fn build(config: &RuntimeConfig) -> Result<Self, BinError> {
        let media_store = LocalMediaStore::new(
            config.storage.images_dir.clone(),
            config.storage.videos_dir.clone(),
            config.storage.avatars_dir.clone(),
            config
                .storage
                .public_base_url
                .clone()
                .or_else(|| Some(config.bot.web_base_url.clone())),
        );
        let llms = ConfiguredLlmProviders::from_config(&config.llm);
        let images = ConfiguredImageGenerators::from_config(&config.image);
        let videos = ConfiguredVideoGenerators::from_config(&config.video);
        tracing::info!(
            llm_providers = llms.configured_count(),
            image_providers = images.configured_count(),
            video_providers = videos.configured_count(),
            event_capacity = 256,
            "built service plan"
        );
        Ok(Self {
            llms,
            images,
            videos,
            media_store,
            events: EventBus::new(256),
            web: config.web.viewer_config(),
        })
    }
}

/// Concrete named LLM provider registry for implemented 2.0 providers.
#[derive(Debug, Clone)]
pub struct ConfiguredLlmProviders {
    inner: Arc<ConfiguredLlmProvidersInner>,
}

#[derive(Debug, Default)]
struct ConfiguredLlmProvidersInner {
    anthropic: BTreeMap<ProviderName, chudbot_anthropic::AnthropicClient>,
    openai: BTreeMap<ProviderName, chudbot_openai::OpenAiClient>,
    xai: BTreeMap<ProviderName, chudbot_xai::XaiClient>,
}

impl Default for ConfiguredLlmProviders {
    fn default() -> Self {
        Self {
            inner: Arc::new(ConfiguredLlmProvidersInner::default()),
        }
    }
}

impl ConfiguredLlmProviders {
    #[tracing::instrument(
        name = "llm_registry.from_config",
        skip_all,
        fields(providers = config.len())
    )]
    fn from_config(config: &BTreeMap<ProviderName, LlmProviderConfig>) -> Self {
        let mut providers = ConfiguredLlmProvidersInner::default();
        for (name, provider) in config {
            match provider {
                LlmProviderConfig::Anthropic { api_key, base_url } => {
                    let mut client = chudbot_anthropic::AnthropicClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "anthropic",
                        base_url_override = base_url.is_some(),
                        "registered LLM provider"
                    );
                    providers.anthropic.insert(name.clone(), client);
                }
                LlmProviderConfig::OpenAi { api_key, base_url } => {
                    let mut client = chudbot_openai::OpenAiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "openai",
                        base_url_override = base_url.is_some(),
                        "registered LLM provider"
                    );
                    providers.openai.insert(name.clone(), client);
                }
                LlmProviderConfig::Xai { api_key, base_url } => {
                    let mut client = chudbot_xai::XaiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "xai",
                        base_url_override = base_url.is_some(),
                        "registered LLM provider"
                    );
                    providers.xai.insert(name.clone(), client);
                }
                LlmProviderConfig::OpenAiCompat { .. } => tracing::warn!(
                    provider = %name,
                    kind = "openai_compat",
                    "LLM provider kind is configured but not implemented in the 2.0 runtime yet"
                ),
            }
        }
        Self {
            inner: Arc::new(providers),
        }
    }

    fn configured_count(&self) -> usize {
        self.inner.anthropic.len() + self.inner.openai.len() + self.inner.xai.len()
    }
}

impl LlmProviderRegistry for ConfiguredLlmProviders {
    type Error = ConfiguredLlmError;

    fn contains_provider(&self, provider: &ProviderName) -> bool {
        let contains = self.inner.anthropic.contains_key(provider)
            || self.inner.openai.contains_key(provider)
            || self.inner.xai.contains_key(provider);
        tracing::trace!(provider = %provider, contains, "checking LLM provider registry");
        contains
    }

    #[tracing::instrument(
        name = "llm_registry.step",
        skip_all,
        fields(provider = %provider, model = %request.model)
    )]
    async fn step(
        &self,
        provider: &ProviderName,
        request: ModelStepRequest,
    ) -> Result<ModelStep, Self::Error> {
        if let Some(client) = self.inner.anthropic.get(provider) {
            tracing::debug!(kind = "anthropic", "dispatching model step");
            return LlmBackend::step(client, request)
                .await
                .map_err(ConfiguredLlmError::Anthropic);
        }
        if let Some(client) = self.inner.openai.get(provider) {
            tracing::debug!(kind = "openai", "dispatching model step");
            return LlmBackend::step(client, request)
                .await
                .map_err(ConfiguredLlmError::OpenAi);
        }
        if let Some(client) = self.inner.xai.get(provider) {
            tracing::debug!(kind = "xai", "dispatching model step");
            return LlmBackend::step(client, request)
                .await
                .map_err(ConfiguredLlmError::Xai);
        }
        tracing::warn!("requested provider is missing from registry");
        Err(ConfiguredLlmError::Missing(provider.clone()))
    }
}

/// Concrete named image-generation provider registry.
#[derive(Debug, Clone)]
pub struct ConfiguredImageGenerators {
    inner: Arc<ConfiguredImageGeneratorsInner>,
}

#[derive(Debug, Default)]
struct ConfiguredImageGeneratorsInner {
    openai: BTreeMap<ProviderName, chudbot_openai::OpenAiClient>,
    xai: BTreeMap<ProviderName, chudbot_xai::XaiClient>,
}

impl Default for ConfiguredImageGenerators {
    fn default() -> Self {
        Self {
            inner: Arc::new(ConfiguredImageGeneratorsInner::default()),
        }
    }
}

impl ConfiguredImageGenerators {
    #[tracing::instrument(
        name = "image_registry.from_config",
        skip_all,
        fields(providers = config.len())
    )]
    fn from_config(config: &BTreeMap<ProviderName, ImageProviderConfig>) -> Self {
        let mut providers = ConfiguredImageGeneratorsInner::default();
        for (name, provider) in config {
            match provider {
                ImageProviderConfig::OpenAi { api_key, base_url } => {
                    let mut client = chudbot_openai::OpenAiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "openai",
                        base_url_override = base_url.is_some(),
                        "registered image provider"
                    );
                    providers.openai.insert(name.clone(), client);
                }
                ImageProviderConfig::Xai { api_key, base_url } => {
                    let mut client = chudbot_xai::XaiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "xai",
                        base_url_override = base_url.is_some(),
                        "registered image provider"
                    );
                    providers.xai.insert(name.clone(), client);
                }
            }
        }
        Self {
            inner: Arc::new(providers),
        }
    }

    fn configured_count(&self) -> usize {
        self.inner.openai.len() + self.inner.xai.len()
    }
}

impl ImageGeneratorRegistry for ConfiguredImageGenerators {
    type Error = ConfiguredImageError;

    fn contains_generator(&self, provider: &ProviderName) -> bool {
        let contains =
            self.inner.openai.contains_key(provider) || self.inner.xai.contains_key(provider);
        tracing::trace!(provider = %provider, contains, "checking image provider registry");
        contains
    }

    #[tracing::instrument(
        name = "image_registry.generate",
        skip_all,
        fields(provider = %provider, model = ?request.model.as_ref())
    )]
    async fn generate_image(
        &self,
        provider: &ProviderName,
        request: ImageRequest,
    ) -> Result<GeneratedImage, Self::Error> {
        if let Some(client) = self.inner.openai.get(provider) {
            tracing::debug!(kind = "openai", "dispatching image generation");
            return ImageGenerator::generate_image(client, request)
                .await
                .map_err(ConfiguredImageError::OpenAi);
        }
        if let Some(client) = self.inner.xai.get(provider) {
            tracing::debug!(kind = "xai", "dispatching image generation");
            return ImageGenerator::generate_image(client, request)
                .await
                .map_err(ConfiguredImageError::Xai);
        }
        tracing::warn!("requested image provider is missing from registry");
        Err(ConfiguredImageError::Missing(provider.clone()))
    }
}

/// Concrete named video-generation provider registry.
#[derive(Debug, Clone)]
pub struct ConfiguredVideoGenerators {
    inner: Arc<ConfiguredVideoGeneratorsInner>,
}

#[derive(Debug, Default)]
struct ConfiguredVideoGeneratorsInner {
    xai: BTreeMap<ProviderName, chudbot_xai::XaiClient>,
}

impl Default for ConfiguredVideoGenerators {
    fn default() -> Self {
        Self {
            inner: Arc::new(ConfiguredVideoGeneratorsInner::default()),
        }
    }
}

impl ConfiguredVideoGenerators {
    #[tracing::instrument(
        name = "video_registry.from_config",
        skip_all,
        fields(providers = config.len())
    )]
    fn from_config(config: &BTreeMap<ProviderName, VideoProviderConfig>) -> Self {
        let mut providers = ConfiguredVideoGeneratorsInner::default();
        for (name, provider) in config {
            match provider {
                VideoProviderConfig::Xai { api_key, base_url } => {
                    let mut client = chudbot_xai::XaiClient::new(api_key.clone());
                    if let Some(base_url) = base_url {
                        client = client.with_base_url(base_url.clone());
                    }
                    tracing::info!(
                        provider = %name,
                        kind = "xai",
                        base_url_override = base_url.is_some(),
                        "registered video provider"
                    );
                    providers.xai.insert(name.clone(), client);
                }
            }
        }
        Self {
            inner: Arc::new(providers),
        }
    }

    fn configured_count(&self) -> usize {
        self.inner.xai.len()
    }
}

impl VideoGeneratorRegistry for ConfiguredVideoGenerators {
    type Error = ConfiguredVideoError;

    fn contains_generator(&self, provider: &ProviderName) -> bool {
        let contains = self.inner.xai.contains_key(provider);
        tracing::trace!(provider = %provider, contains, "checking video provider registry");
        contains
    }

    #[tracing::instrument(
        name = "video_registry.submit",
        skip_all,
        fields(provider = %provider, model = ?request.model.as_ref())
    )]
    async fn submit_video(
        &self,
        provider: &ProviderName,
        request: VideoRequest,
    ) -> Result<VideoJobId, Self::Error> {
        if let Some(client) = self.inner.xai.get(provider) {
            tracing::debug!(kind = "xai", "dispatching video submit");
            return VideoGenerator::submit_video(client, request)
                .await
                .map_err(ConfiguredVideoError::Xai);
        }
        tracing::warn!("requested video provider is missing from registry");
        Err(ConfiguredVideoError::Missing(provider.clone()))
    }

    #[tracing::instrument(name = "video_registry.check", skip_all, fields(provider = %provider, job = %job))]
    async fn check_video(
        &self,
        provider: &ProviderName,
        job: VideoJobId,
    ) -> Result<VideoJobStatus, Self::Error> {
        if let Some(client) = self.inner.xai.get(provider) {
            return VideoGenerator::check_video(client, job)
                .await
                .map_err(ConfiguredVideoError::Xai);
        }
        tracing::warn!("requested video provider is missing from registry");
        Err(ConfiguredVideoError::Missing(provider.clone()))
    }

    #[tracing::instrument(name = "video_registry.download", skip_all, fields(provider = %provider))]
    async fn download_video(
        &self,
        provider: &ProviderName,
        url: String,
    ) -> Result<Vec<u8>, Self::Error> {
        if let Some(client) = self.inner.xai.get(provider) {
            return VideoGenerator::download_video(client, url)
                .await
                .map_err(ConfiguredVideoError::Xai);
        }
        tracing::warn!("requested video provider is missing from registry");
        Err(ConfiguredVideoError::Missing(provider.clone()))
    }
}

/// Concrete named message platform registry.
#[derive(Clone)]
pub struct ConfiguredMessagePlatforms {
    inner: Arc<ConfiguredMessagePlatformsInner>,
}

struct ConfiguredMessagePlatformsInner {
    discord: BTreeMap<chudbot_api::PlatformName, ConfiguredDiscordPlatform>,
    events: tokio::sync::Mutex<
        tokio::sync::mpsc::Receiver<Result<PlatformEvent, ConfiguredPlatformError>>,
    >,
    event_pumps: tokio::sync::Mutex<Vec<PlatformEventPump>>,
}

struct ConfiguredDiscordPlatform {
    platform: chudbot_discord::DiscordPlatform,
    dev_guild_id: Option<ExternalId>,
}

struct PlatformEventPump {
    platform: chudbot_api::PlatformName,
    task: JoinHandle<()>,
}

impl Default for ConfiguredMessagePlatforms {
    fn default() -> Self {
        let (_events_tx, events) = tokio::sync::mpsc::channel(1);
        Self {
            inner: Arc::new(ConfiguredMessagePlatformsInner {
                discord: BTreeMap::new(),
                events: tokio::sync::Mutex::new(events),
                event_pumps: tokio::sync::Mutex::new(Vec::new()),
            }),
        }
    }
}

fn spawn_discord_event_pump(
    platform_name: chudbot_api::PlatformName,
    platform: chudbot_discord::DiscordPlatform,
    events: tokio::sync::mpsc::Sender<Result<PlatformEvent, ConfiguredPlatformError>>,
) -> PlatformEventPump {
    let handle_platform_name = platform_name.clone();
    let task = tokio::spawn(async move {
        let pump = run_discord_event_pump(platform_name.clone(), platform, events.clone());
        if let Err(payload) = AssertUnwindSafe(pump).catch_unwind().await {
            let message = panic_payload_message(payload.as_ref());
            tracing::error!(
                platform = %platform_name,
                panic = %message,
                "message platform event pump panicked"
            );
            let error = ConfiguredPlatformError::EventPumpPanic {
                platform: platform_name.clone(),
                message,
            };
            if events.send(Err(error)).await.is_err() {
                tracing::debug!(
                    platform = %platform_name,
                    "message platform event pump panic dropped because receiver closed"
                );
            }
        }
    });
    PlatformEventPump {
        platform: handle_platform_name,
        task,
    }
}

async fn run_discord_event_pump(
    platform_name: chudbot_api::PlatformName,
    platform: chudbot_discord::DiscordPlatform,
    events: tokio::sync::mpsc::Sender<Result<PlatformEvent, ConfiguredPlatformError>>,
) {
    loop {
        let event = MessagePlatform::next_event(&platform)
            .await
            .map_err(ConfiguredPlatformError::Discord);
        if let Err(error) = &event {
            tracing::warn!(
                platform = %platform_name,
                error = %error,
                "message platform event pump received an error"
            );
        }
        let should_stop = matches!(&event, Ok(PlatformEvent::Shutdown));
        if should_stop {
            match events.try_send(event) {
                Ok(()) => {}
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    tracing::debug!(
                        platform = %platform_name,
                        "message platform event pump stopped because receiver closed"
                    );
                }
                Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                    tracing::debug!(
                        platform = %platform_name,
                        "message platform shutdown event dropped because receiver was full"
                    );
                }
            }
            tracing::debug!(
                platform = %platform_name,
                "message platform event pump stopped after platform shutdown"
            );
            break;
        }
        if events.send(event).await.is_err() {
            tracing::debug!(
                platform = %platform_name,
                "message platform event pump stopped because receiver closed"
            );
            break;
        }
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn log_event_pump_join_result(platform: &chudbot_api::PlatformName, result: Result<(), JoinError>) {
    match result {
        Ok(()) => tracing::debug!(platform = %platform, "message platform event pump joined"),
        Err(error) if error.is_cancelled() => {
            tracing::debug!(
                platform = %platform,
                error = %error,
                "message platform event pump was cancelled"
            );
        }
        Err(error) if error.is_panic() => {
            tracing::error!(
                platform = %platform,
                error = %error,
                "message platform event pump panicked"
            );
        }
        Err(error) => {
            tracing::warn!(
                platform = %platform,
                error = %error,
                "message platform event pump join failed"
            );
        }
    }
}

impl ConfiguredMessagePlatforms {
    /// Connect every configured message platform.
    #[tracing::instrument(
        name = "platform_registry.connect",
        skip_all,
        fields(platforms = config.len())
    )]
    pub async fn connect_from_config(
        config: &BTreeMap<chudbot_api::PlatformName, MessagePlatformConfig>,
    ) -> Result<Self, ConfiguredPlatformError> {
        let mut discord = BTreeMap::new();
        let mut event_pumps = Vec::new();
        let (events_tx, events) = tokio::sync::mpsc::channel(256);
        for (name, platform) in config {
            match platform {
                MessagePlatformConfig::Discord {
                    token,
                    dev_guild_id,
                } => {
                    if dev_guild_id.is_some() {
                        tracing::debug!(
                            platform = %name,
                            "discord dev guild is configured for command registration"
                        );
                    }
                    let platform = chudbot_discord::DiscordPlatform::connect_named(
                        name.clone(),
                        token.clone(),
                    )
                    .await?;
                    tracing::info!(platform = %name, kind = "discord", "registered platform");
                    event_pumps.push(spawn_discord_event_pump(
                        name.clone(),
                        platform.clone(),
                        events_tx.clone(),
                    ));
                    discord.insert(
                        name.clone(),
                        ConfiguredDiscordPlatform {
                            platform,
                            dev_guild_id: dev_guild_id.clone().map(ExternalId::new),
                        },
                    );
                }
            }
        }
        drop(events_tx);
        Ok(Self {
            inner: Arc::new(ConfiguredMessagePlatformsInner {
                discord,
                events: tokio::sync::Mutex::new(events),
                event_pumps: tokio::sync::Mutex::new(event_pumps),
            }),
        })
    }

    /// Count connected platform services.
    pub fn configured_count(&self) -> usize {
        self.inner.discord.len()
    }

    fn discord(
        &self,
        platform: &chudbot_api::PlatformName,
    ) -> Result<&ConfiguredDiscordPlatform, ConfiguredPlatformError> {
        self.inner
            .discord
            .get(platform)
            .ok_or_else(|| ConfiguredPlatformError::Missing(platform.clone()))
    }

    async fn shutdown_platforms(&self) -> Result<(), ConfiguredPlatformError> {
        if self.inner.discord.is_empty() {
            return Ok(());
        }

        for (name, configured) in &self.inner.discord {
            tracing::debug!(platform = %name, "requesting message platform shutdown");
            configured.platform.request_shutdown();
        }

        let mut handles = {
            let mut event_pumps = self.inner.event_pumps.lock().await;
            std::mem::take(&mut *event_pumps)
        };
        if handles.is_empty() {
            return Ok(());
        }

        let deadline = tokio::time::sleep(PLATFORM_SHUTDOWN_TIMEOUT);
        tokio::pin!(deadline);
        let mut timed_out = false;
        for pump in &mut handles {
            let platform = pump.platform.clone();
            tokio::select! {
                result = &mut pump.task => {
                    log_event_pump_join_result(&platform, result);
                }
                () = &mut deadline => {
                    timed_out = true;
                    break;
                }
            }
        }

        if timed_out {
            let remaining = handles
                .iter()
                .filter(|pump| !pump.task.is_finished())
                .count();
            tracing::warn!(
                remaining,
                timeout_ms = PLATFORM_SHUTDOWN_TIMEOUT.as_millis(),
                "timed out waiting for message platform shutdown"
            );
            for pump in handles {
                if !pump.task.is_finished() {
                    tracing::debug!(
                        platform = %pump.platform,
                        "aborting message platform event pump after shutdown timeout"
                    );
                    pump.task.abort();
                }
            }
        }

        Ok(())
    }
}

impl MessagePlatformRegistry for ConfiguredMessagePlatforms {
    type Error = ConfiguredPlatformError;

    async fn bot_user(
        &self,
        platform: &chudbot_api::PlatformName,
    ) -> Result<UserProfile, Self::Error> {
        MessagePlatform::bot_user(&self.discord(platform)?.platform)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn register_commands(
        &self,
        commands: Vec<PlatformCommandDefinition>,
    ) -> Result<(), Self::Error> {
        for configured in self.inner.discord.values() {
            MessagePlatform::register_commands(
                &configured.platform,
                commands.clone(),
                configured.dev_guild_id.clone(),
            )
            .await
            .map_err(ConfiguredPlatformError::Discord)?;
        }
        Ok(())
    }

    async fn next_event(&self) -> Result<PlatformEvent, Self::Error> {
        if self.inner.discord.is_empty() {
            return Err(ConfiguredPlatformError::Empty);
        }
        self.inner
            .events
            .lock()
            .await
            .recv()
            .await
            .unwrap_or(Err(ConfiguredPlatformError::EventsClosed))
    }

    async fn shutdown(&self) -> Result<(), Self::Error> {
        self.shutdown_platforms().await
    }

    async fn respond_to_command(
        &self,
        response: PlatformCommandResponse,
    ) -> Result<(), Self::Error> {
        let platform = self.discord(&response.target.platform)?;
        MessagePlatform::respond_to_command(&platform.platform, response)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn send_message(&self, request: SendMessage) -> Result<PostedMessage, Self::Error> {
        let platform = self.discord(&request.channel.platform)?;
        MessagePlatform::send_message(&platform.platform, request)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn delete_message(&self, message: MessageRef) -> Result<(), Self::Error> {
        let platform = self.discord(&message.platform)?;
        MessagePlatform::delete_message(&platform.platform, message)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn add_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> Result<(), Self::Error> {
        let platform = self.discord(&message.platform)?;
        MessagePlatform::add_reaction(&platform.platform, message, reaction)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn remove_own_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> Result<(), Self::Error> {
        let platform = self.discord(&message.platform)?;
        MessagePlatform::remove_own_reaction(&platform.platform, message, reaction)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn typing(&self, channel: ChannelRef) -> Result<(), Self::Error> {
        let platform = self.discord(&channel.platform)?;
        MessagePlatform::typing(&platform.platform, channel)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn fetch_messages(
        &self,
        request: FetchMessages,
    ) -> Result<Vec<PlatformMessage>, Self::Error> {
        let platform = self.discord(&request.channel.platform)?;
        MessagePlatform::fetch_messages(&platform.platform, request)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn message_context(
        &self,
        message: &PlatformMessage,
        relationship: PlatformMessageRelationship,
    ) -> Result<serde_json::Value, Self::Error> {
        let platform = self.discord(&message.id.platform)?;
        MessagePlatform::message_context(&platform.platform, message, relationship)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn parent_channel(&self, channel: ChannelRef) -> Result<ChannelRef, Self::Error> {
        let platform = self.discord(&channel.platform)?;
        MessagePlatform::parent_channel(&platform.platform, channel)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }
}

/// Errors from the concrete provider registry.
#[derive(Debug, Error)]
pub enum ConfiguredLlmError {
    /// Provider was referenced but not implemented/configured.
    #[error("provider `{0}` is not available in the 2.0 runtime")]
    Missing(ProviderName),
    /// Anthropic request failed.
    #[error(transparent)]
    Anthropic(#[from] chudbot_anthropic::AnthropicError),
    /// OpenAI request failed.
    #[error(transparent)]
    OpenAi(#[from] chudbot_openai::OpenAiError),
    /// xAI request failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors from the concrete image-generation registry.
#[derive(Debug, Error)]
pub enum ConfiguredImageError {
    /// Provider was referenced but not implemented/configured.
    #[error("image provider `{0}` is not available in the 2.0 runtime")]
    Missing(ProviderName),
    /// OpenAI image generation failed.
    #[error(transparent)]
    OpenAi(#[from] chudbot_openai::OpenAiError),
    /// xAI image generation failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors from the concrete video-generation registry.
#[derive(Debug, Error)]
pub enum ConfiguredVideoError {
    /// Provider was referenced but not implemented/configured.
    #[error("video provider `{0}` is not available in the 2.0 runtime")]
    Missing(ProviderName),
    /// xAI video generation failed.
    #[error(transparent)]
    Xai(#[from] chudbot_xai::XaiError),
}

/// Errors from the concrete message-platform registry.
#[derive(Debug, Error)]
pub enum ConfiguredPlatformError {
    /// No platform exists for a requested platform name.
    #[error("message platform `{0}` is not available in the 2.0 runtime")]
    Missing(chudbot_api::PlatformName),
    /// The registry is empty.
    #[error("no message platforms are configured")]
    Empty,
    /// All event pump tasks stopped.
    #[error("all message platform event streams are closed")]
    EventsClosed,
    /// A platform event pump panicked.
    #[error("message platform `{platform}` event pump panicked: {message}")]
    EventPumpPanic {
        /// Platform name.
        platform: chudbot_api::PlatformName,
        /// Panic payload.
        message: String,
    },
    /// Discord platform failed.
    #[error(transparent)]
    Discord(#[from] chudbot_discord::DiscordError),
}

/// Top-level binary errors.
#[derive(Debug, Error)]
pub enum BinError {
    /// Config load failed.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// Bot config failed validation.
    #[error(transparent)]
    Bot(#[from] chudbot_bot::BotError),
    /// Web server failed.
    #[error(transparent)]
    Web(#[from] chudbot_web::WebServerError),
    /// SQLx storage failed.
    #[error(transparent)]
    Storage(#[from] chudbot_storage_sqlx::SqlxStorageError),
    /// Platform setup failed.
    #[error(transparent)]
    Platform(#[from] ConfiguredPlatformError),
    /// Logging filter failed validation.
    #[error(transparent)]
    LoggingFilter(#[from] LoggingFilterError),
    /// Memory config failed validation.
    #[error(transparent)]
    MemoryConfig(#[from] chudbot_bot::memory::MemoryConfigError),
    /// A service task failed to join.
    #[error("{task} service task join failed: {source}")]
    TaskJoin {
        /// Service name.
        task: &'static str,
        /// Join error.
        source: JoinError,
    },
    /// Agent provider has no matching `[llm.<name>]` config.
    #[error("agent `{agent}` uses provider `{provider}` but no matching [llm] entry exists")]
    MissingProviderConfig {
        /// Agent name.
        agent: String,
        /// Provider name.
        provider: ProviderName,
    },
    /// Memory provider has no matching `[llm.<name>]` config.
    #[error("memory uses provider `{provider}` but no matching [llm] entry exists")]
    MissingMemoryProviderConfig {
        /// Provider name.
        provider: ProviderName,
    },
    /// Agent image provider has no matching `[image.<name>]` config.
    #[error(
        "agent `{agent}` uses image provider `{provider}` but no matching [image] entry exists"
    )]
    MissingImageProviderConfig {
        /// Agent name.
        agent: String,
        /// Provider name.
        provider: ProviderName,
    },
    /// Agent video provider has no matching `[video.<name>]` config.
    #[error(
        "agent `{agent}` uses video provider `{provider}` but no matching [video] entry exists"
    )]
    MissingVideoProviderConfig {
        /// Agent name.
        agent: String,
        /// Provider name.
        provider: ProviderName,
    },
    /// An agent references an LLM provider kind with no runtime implementation.
    #[error(
        "agent `{agent}` uses llm provider `{provider}` with kind `{kind}`, which is planned but not implemented"
    )]
    UnimplementedLlmProvider {
        /// Agent name.
        agent: String,
        /// Provider name.
        provider: ProviderName,
        /// Provider kind.
        kind: &'static str,
    },
    /// Memory references an LLM provider kind with no runtime implementation.
    #[error(
        "memory uses llm provider `{provider}` with kind `{kind}`, which is planned but not implemented"
    )]
    UnimplementedMemoryProvider {
        /// Provider name.
        provider: ProviderName,
        /// Provider kind.
        kind: &'static str,
    },
    /// Platform binding has no matching platform config.
    #[error("platform `{platform}` is bound in [bot.platforms] but has no [platforms] entry")]
    MissingPlatformConfig {
        /// Platform name.
        platform: chudbot_api::PlatformName,
    },
    /// Database URL was omitted.
    #[error("database.url must not be empty")]
    MissingDatabaseUrl,
    /// Listen address failed to parse.
    #[error("invalid web listen address: {0}")]
    Listen(#[from] std::net::AddrParseError),
}

/// Config parse errors.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// Config file could not be read.
    #[error("could not read config file {}", path.display())]
    Read {
        /// Path.
        path: PathBuf,
        /// Source error.
        #[source]
        source: std::io::Error,
    },
    /// Config file could not be parsed.
    #[error("could not parse config file {}", path.display())]
    Parse {
        /// Path.
        path: PathBuf,
        /// Source error.
        #[source]
        source: toml::de::Error,
    },
}
