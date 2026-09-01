//! Concrete message-platform registry for this binary.
//!
//! `chudbot-bot` depends only on the platform-neutral
//! [`MessagePlatformRegistry`] contract. This module is the process-launcher
//! boundary that turns `[platforms.<name>]` config into concrete adapters,
//! currently Discord/Twilight through `chudbot-discord`, and keeps those
//! adapters addressable by the same platform names used in stored refs and
//! `[bot.platforms.<name>]` runtime bindings.

use std::collections::BTreeMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use chudbot_api::MessagePlatform;
use chudbot_api::{
    AttachmentCandidate, AttachmentPreflight, ChannelRef, FetchMessages, MessagePlatformEvents,
    MessagePlatformRegistry, MessageRef, PlatformCommandDefinition, PlatformCommandResponse,
    PlatformEvent, PlatformMessage, PlatformMessageRelationship, PostedMessage, ReactionKind,
    SendMessage, UserProfile, VibeIdentityProvider, VibeMembership, VibeOauthLogin,
};
use futures::FutureExt;
use tokio::task::{JoinError, JoinHandle};

use crate::PLATFORM_SHUTDOWN_TIMEOUT;
use crate::config::MessagePlatformConfig;
use crate::errors::ConfiguredPlatformError;

/// Concrete named message-platform registry used by `ConfiguredBotRuntime`.
///
/// Clones share the same platform clients for outbound operations. Runtime code
/// uses this value through [`MessagePlatformRegistry`], so no Twilight or
/// Discord types cross into `chudbot-bot`.
#[derive(Clone)]
pub struct ConfiguredMessagePlatforms {
    inner: Arc<ConfiguredMessagePlatformsInner>,
}

/// Unique platform event stream used by the bot runtime.
pub struct ConfiguredMessagePlatformEvents {
    platforms: ConfiguredMessagePlatforms,
    events: tokio::sync::mpsc::Receiver<Result<PlatformEvent, ConfiguredPlatformError>>,
    event_pumps: Vec<PlatformEventPump>,
}

/// Shared registry state behind cheap runtime clones.
struct ConfiguredMessagePlatformsInner {
    /// Discord adapters keyed by deployment-configured platform name.
    discord: BTreeMap<chudbot_api::PlatformName, ConfiguredDiscordPlatform>,
    vibe_identity:
        BTreeMap<chudbot_api::PlatformName, chudbot_discord::DiscordVibeIdentityProvider>,
}

/// Concrete Discord adapter stored under one configured platform name.
struct ConfiguredDiscordPlatform {
    platform: chudbot_discord::DiscordPlatform,
}

/// Background task forwarding one concrete platform stream into the registry.
struct PlatformEventPump {
    platform: chudbot_api::PlatformName,
    task: JoinHandle<()>,
}

/// Spawn a Discord event pump and convert pump panics into registry errors.
///
/// `DiscordPlatform` owns the Twilight gateway shard and already translates
/// gateway traffic into `PlatformEvent`. The pump only keeps that stream alive
/// and forwards it into the registry fan-in channel that `BotRuntime` polls.
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

/// Forward one Discord event stream into the shared registry channel.
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
            // Shutdown is terminal for this pump, so do not wait behind a full
            // queue after the runtime may have already stopped platform intake.
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

/// Normalize arbitrary panic payloads into a loggable error message.
fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Log the outcome of a platform pump task after shutdown joins it.
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

/// Connect every `[platforms.<name>]` service and build the runtime handles.
///
/// Each configured platform is connected once, stored by its configured name,
/// and given an event pump that forwards into one shared receiver.
/// `[bot.platforms.<name>]` agent bindings remain in `BotConfig`; the platform
/// name is the join key between those runtime bindings and these concrete
/// transport handles.
#[tracing::instrument(
    name = "platform_registry.connect",
    skip_all,
    fields(platforms = config.len())
)]
pub async fn connect_configured_message_platforms(
    config: &BTreeMap<chudbot_api::PlatformName, MessagePlatformConfig>,
    vibe_auth: Option<&chudbot_vibe::config::VibeAuthConfig>,
) -> Result<(ConfiguredMessagePlatforms, ConfiguredMessagePlatformEvents), ConfiguredPlatformError>
{
    let mut inner = ConfiguredMessagePlatformsInner {
        discord: BTreeMap::new(),
        vibe_identity: BTreeMap::new(),
    };
    let mut event_pumps = Vec::new();
    let (events_tx, events_rx) = tokio::sync::mpsc::channel(256);
    for (name, platform) in config {
        match platform {
            MessagePlatformConfig::Discord {
                token,
                dev_guild_id,
            } => {
                // Discord is the current concrete integration boundary:
                // this is where configured names become Twilight-backed
                // clients while the rest of the runtime keeps using
                // chudbot-api contracts.
                if dev_guild_id.is_some() {
                    tracing::warn!(
                        platform = %name,
                        "discord dev_guild_id is ignored; commands register globally"
                    );
                }
                let platform =
                    chudbot_discord::DiscordPlatform::connect_named(name.clone(), token.clone())
                        .await?;
                if let Some(auth) = vibe_auth {
                    inner.vibe_identity.insert(
                        name.clone(),
                        platform.vibe_identity_provider(
                            auth.client_id.clone(),
                            auth.client_secret.clone(),
                        ),
                    );
                }
                tracing::info!(platform = %name, kind = "discord", "registered platform");
                event_pumps.push(spawn_discord_event_pump(
                    name.clone(),
                    platform.clone(),
                    events_tx.clone(),
                ));
                inner
                    .discord
                    .insert(name.clone(), ConfiguredDiscordPlatform { platform });
            }
        }
    }
    drop(events_tx);
    let platforms = ConfiguredMessagePlatforms {
        inner: Arc::new(inner),
    };
    let platform_events = ConfiguredMessagePlatformEvents {
        platforms: platforms.clone(),
        events: events_rx,
        event_pumps,
    };
    Ok((platforms, platform_events))
}

impl VibeIdentityProvider for ConfiguredMessagePlatforms {
    type Error = ConfiguredPlatformError;

    fn authorization_url(&self, state: &str, callback_url: &str) -> Result<String, Self::Error> {
        self.primary_vibe_identity()
            .ok_or(ConfiguredPlatformError::MissingVibeIdentity)?
            .authorization_url(state, callback_url)
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn exchange_code(
        &self,
        code: &str,
        callback_url: &str,
    ) -> Result<VibeOauthLogin, Self::Error> {
        self.primary_vibe_identity()
            .ok_or(ConfiguredPlatformError::MissingVibeIdentity)?
            .exchange_code(code, callback_url)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }

    async fn guild_membership(
        &self,
        platform: &chudbot_api::PlatformName,
        guild_id: &chudbot_api::ExternalId,
        user_id: &chudbot_api::ExternalId,
    ) -> Result<Option<VibeMembership>, Self::Error> {
        self.inner
            .vibe_identity
            .get(platform)
            .ok_or(ConfiguredPlatformError::MissingVibeIdentity)?
            .guild_membership(platform, guild_id, user_id)
            .await
            .map_err(ConfiguredPlatformError::Discord)
    }
}

impl ConfiguredMessagePlatforms {
    fn primary_vibe_identity(&self) -> Option<&chudbot_discord::DiscordVibeIdentityProvider> {
        self.inner
            .vibe_identity
            .get(&chudbot_api::PlatformName::new("discord"))
            .or_else(|| self.inner.vibe_identity.values().next())
    }
}

impl ConfiguredMessagePlatforms {
    /// Look up the concrete Discord adapter named by a neutral platform ref.
    fn discord(
        &self,
        platform: &chudbot_api::PlatformName,
    ) -> Result<&ConfiguredDiscordPlatform, ConfiguredPlatformError> {
        self.inner
            .discord
            .get(platform)
            .ok_or_else(|| ConfiguredPlatformError::Missing(platform.clone()))
    }
}

impl ConfiguredMessagePlatformEvents {
    /// Request graceful shutdown for all platform clients and their pumps.
    async fn shutdown_platforms(&mut self) -> Result<(), ConfiguredPlatformError> {
        if self.platforms.inner.discord.is_empty() {
            return Ok(());
        }

        for (name, configured) in &self.platforms.inner.discord {
            tracing::debug!(platform = %name, "requesting message platform shutdown");
            configured.platform.request_shutdown();
        }

        // Take the task handles so repeated shutdown calls cannot attempt to
        // join the same pumps twice.
        let mut handles = std::mem::take(&mut self.event_pumps);
        if handles.is_empty() {
            return Ok(());
        }

        let deadline = tokio::time::sleep(PLATFORM_SHUTDOWN_TIMEOUT);
        tokio::pin!(deadline);
        let mut timed_out = false;
        for PlatformEventPump { platform, task } in &mut handles {
            // One deadline covers the whole platform shutdown phase, not each
            // pump independently, to keep process teardown bounded.
            tokio::select! {
                result = task => {
                    log_event_pump_join_result(platform, result);
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

impl MessagePlatformEvents for ConfiguredMessagePlatformEvents {
    type Error = ConfiguredPlatformError;

    async fn next_event(&mut self) -> Result<PlatformEvent, Self::Error> {
        if self.platforms.inner.discord.is_empty() {
            return Err(ConfiguredPlatformError::Empty);
        }
        // `BotRuntime` consumes one merged stream regardless of how many
        // concrete platform clients are configured.
        self.events
            .recv()
            .await
            .unwrap_or(Err(ConfiguredPlatformError::EventsClosed))
    }

    async fn shutdown(&mut self) -> Result<(), Self::Error> {
        self.shutdown_platforms().await
    }
}

/// Runtime-facing registry implementation.
///
/// The bot runtime calls this trait with neutral refs such as `PlatformName`,
/// `ChannelRef`, and `MessageRef`. Each method routes by that name, delegates
/// to the concrete adapter, and wraps adapter-specific errors at this binary
/// boundary.
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
        // Commands are process-global for each Discord application here; the
        // legacy dev-guild config knob is accepted but ignored during connect.
        for configured in self.inner.discord.values() {
            MessagePlatform::register_commands(&configured.platform, commands.clone(), None)
                .await
                .map_err(ConfiguredPlatformError::Discord)?;
        }
        Ok(())
    }

    async fn respond_to_command(
        &self,
        response: PlatformCommandResponse,
    ) -> Result<(), Self::Error> {
        // Outbound operations carry their target platform in the neutral API
        // type, so dispatch stays table-driven instead of leaking adapter
        // choices into the bot runtime.
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

    async fn preflight_attachment(
        &self,
        channel: ChannelRef,
        reply_to: Option<MessageRef>,
        candidate: AttachmentCandidate,
    ) -> Result<AttachmentPreflight, Self::Error> {
        let platform = self.discord(&channel.platform)?;
        MessagePlatform::preflight_attachment(&platform.platform, channel, reply_to, candidate)
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
