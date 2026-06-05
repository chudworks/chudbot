use std::collections::BTreeMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use chudbot_api::{
    ChannelRef, FetchMessages, MessagePlatform, MessageRef, PlatformCommandDefinition,
    PlatformCommandResponse, PlatformEvent, PlatformMessage, PlatformMessageRelationship,
    PostedMessage, ReactionKind, SendMessage, UserProfile,
};
use chudbot_bot::MessagePlatformRegistry;
use futures::FutureExt;
use tokio::task::{JoinError, JoinHandle};

use crate::PLATFORM_SHUTDOWN_TIMEOUT;
use crate::config::MessagePlatformConfig;
use crate::errors::ConfiguredPlatformError;

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
                        tracing::warn!(
                            platform = %name,
                            "discord dev_guild_id is ignored; commands register globally"
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
                    discord.insert(name.clone(), ConfiguredDiscordPlatform { platform });
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
            MessagePlatform::register_commands(&configured.platform, commands.clone(), None)
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
