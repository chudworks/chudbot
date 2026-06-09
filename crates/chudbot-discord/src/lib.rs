//! Discord platform implementation for chudbot.
//!
//! This crate is the only 2.0 crate that knows about Twilight and Discord
//! snowflakes. It converts gateway events and REST actions into the
//! platform-neutral contracts from `chudbot-api`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use chudbot_api::{
    AttachmentRef, ChannelRef, ExternalId, FetchMessages, MessagePlatform, MessageRef,
    OutgoingAttachment, PlatformCommand, PlatformCommandDefinition, PlatformCommandInput,
    PlatformCommandOption, PlatformCommandOptionKind, PlatformCommandResponse,
    PlatformCommandResponseTarget, PlatformCommandValue, PlatformEvent, PlatformMessage,
    PlatformMessageReference, PlatformMessageRelationship, PlatformName, PlatformReaction,
    PlatformReady, PostedMessage, ReactionKind, SendMessage, UserProfile, UserRef,
};
use thiserror::Error;
use time::OffsetDateTime;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use twilight_cache_inmemory::{DefaultInMemoryCache, ResourceType};
use twilight_gateway::{EventTypeFlags, Intents, Shard, ShardId, ShardState, StreamExt};
use twilight_http::Client as HttpClient;
use twilight_http::request::channel::reaction::RequestReactionType;
use twilight_model::application::command::{Command, CommandOption, CommandType};
use twilight_model::application::interaction::application_command::{
    CommandDataOption, CommandOptionValue as InteractionCommandOptionValue,
};
use twilight_model::application::interaction::{Interaction, InteractionData};
use twilight_model::channel::message::{
    EmojiReactionType, Mention, MessageFlags, MessageReference,
};
use twilight_model::channel::{ChannelType, Message};
use twilight_model::gateway::event::Event;
use twilight_model::gateway::payload::incoming::GuildCreate;
use twilight_model::gateway::{CloseFrame, GatewayReaction};
use twilight_model::guild::Permissions;
use twilight_model::http::attachment::Attachment as HttpAttachment;
use twilight_model::http::interaction::{
    InteractionResponse, InteractionResponseData, InteractionResponseType,
};
use twilight_model::id::Id;
use twilight_model::id::marker::{
    ApplicationMarker, ChannelMarker, EmojiMarker, GuildMarker, InteractionMarker, MessageMarker,
    UserMarker,
};
use twilight_model::user::{CurrentUser, User};
use twilight_util::builder::command::{
    ChannelBuilder, CommandBuilder, IntegerBuilder, StringBuilder, SubCommandBuilder,
};

const DEFAULT_PLATFORM_NAME: &str = "discord";
const DISCORD_MESSAGE_LIMIT: usize = 2000;
const DISCORD_ATTACHMENT_LIMIT: usize = 10;
const CODE_FENCE_MIN_WIDTH: usize = 3;
const GATEWAY_RECONNECT_BASE_DELAY: std::time::Duration = std::time::Duration::from_secs(5);
const GATEWAY_RECONNECT_MAX_DELAY: std::time::Duration = std::time::Duration::from_secs(60);
const GATEWAY_CLOSE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Discord platform runtime.
#[derive(Clone)]
pub struct DiscordPlatform {
    inner: Arc<DiscordPlatformInner>,
}

struct DiscordPlatformInner {
    platform: PlatformName,
    http: Arc<HttpClient>,
    token: String,
    intents: Intents,
    shard: Mutex<Shard>,
    bot_user: UserProfile,
    bot_user_id: Id<UserMarker>,
    application_id: Id<ApplicationMarker>,
    cache: DefaultInMemoryCache,
    ready_emitted: AtomicBool,
    event_flags: EventTypeFlags,
    shutdown: CancellationToken,
}

impl DiscordPlatform {
    /// Connect to Discord using the conventional platform name `discord`.
    pub async fn connect(token: impl Into<String>) -> Result<Self, DiscordError> {
        Self::connect_named(PlatformName::new(DEFAULT_PLATFORM_NAME), token).await
    }

    /// Connect to Discord with a deployment-configured platform name.
    #[tracing::instrument(name = "discord.connect", skip_all, fields(platform = %platform))]
    pub async fn connect_named(
        platform: PlatformName,
        token: impl Into<String>,
    ) -> Result<Self, DiscordError> {
        let token = token.into();
        let http = Arc::new(HttpClient::new(token.clone()));

        let current = http.current_user().await?.model().await?;
        let application = http.current_user_application().await?.model().await?;
        let bot_user = current_user_profile(&platform, &current);
        tracing::info!(
            platform = %platform,
            username = %current.name,
            user_id = %current.id,
            application_id = %application.id,
            "discord platform connected"
        );

        let intents = Intents::GUILDS
            | Intents::GUILD_MESSAGES
            | Intents::MESSAGE_CONTENT
            | Intents::DIRECT_MESSAGES
            | Intents::GUILD_MESSAGE_REACTIONS
            | Intents::DIRECT_MESSAGE_REACTIONS;
        let event_flags = EventTypeFlags::MESSAGE_CREATE
            | EventTypeFlags::INTERACTION_CREATE
            | EventTypeFlags::GUILD_CREATE
            | EventTypeFlags::REACTION_ADD
            | EventTypeFlags::REACTION_REMOVE;
        let cache = DefaultInMemoryCache::builder()
            .resource_types(
                ResourceType::GUILD
                    | ResourceType::CHANNEL
                    | ResourceType::USER
                    | ResourceType::MEMBER,
            )
            .build();

        Ok(Self {
            inner: Arc::new(DiscordPlatformInner {
                platform,
                http,
                token: token.clone(),
                intents,
                shard: Mutex::new(Shard::new(ShardId::ONE, token, intents)),
                bot_user,
                bot_user_id: current.id,
                application_id: application.id,
                cache,
                ready_emitted: AtomicBool::new(false),
                event_flags,
                shutdown: CancellationToken::new(),
            }),
        })
    }

    /// Borrow the platform name used in message refs.
    pub fn platform_name(&self) -> &PlatformName {
        &self.inner.platform
    }

    /// Borrow the underlying Twilight HTTP client.
    pub fn http(&self) -> &Arc<HttpClient> {
        &self.inner.http
    }

    /// Discord application id.
    pub fn application_id(&self) -> Id<ApplicationMarker> {
        self.inner.application_id
    }

    /// Discord bot user id.
    pub fn bot_user_id(&self) -> Id<UserMarker> {
        self.inner.bot_user_id
    }

    /// Request a clean Discord Gateway shutdown.
    ///
    /// The event loop owns the shard while awaiting gateway messages, so this
    /// method only signals that loop. The loop then sends Discord's normal
    /// websocket close frame, which invalidates the gateway session and marks
    /// the bot offline.
    pub fn request_shutdown(&self) {
        self.inner.shutdown.cancel();
    }

    async fn next_gateway_event(&self) -> Result<PlatformEvent, DiscordError> {
        let mut reconnect_delay = GATEWAY_RECONNECT_BASE_DELAY;
        loop {
            if self.inner.shutdown.is_cancelled() {
                return self.close_gateway().await;
            }
            let item = {
                let mut shard = self.inner.shard.lock().await;
                tokio::select! {
                    item = shard.next_event(self.inner.event_flags) => item,
                    _ = self.inner.shutdown.cancelled() => {
                        return close_gateway_shard(
                            &mut shard,
                            &self.inner.platform,
                            self.inner.event_flags,
                        )
                        .await;
                    }
                }
            };
            let Some(item) = item else {
                let delay = reconnect_delay;
                reconnect_delay = next_reconnect_delay(reconnect_delay);
                tracing::warn!(
                    platform = %self.inner.platform,
                    delay_ms = delay.as_millis(),
                    "discord gateway stream ended; reconnecting after backoff"
                );
                if !self.reconnect_gateway(delay).await {
                    return Ok(PlatformEvent::Shutdown);
                }
                continue;
            };
            let event = match item {
                Ok(event) => event,
                Err(error) => {
                    let delay = reconnect_delay;
                    reconnect_delay = next_reconnect_delay(reconnect_delay);
                    tracing::warn!(
                        platform = %self.inner.platform,
                        error = %error,
                        delay_ms = delay.as_millis(),
                        "discord gateway receive error; reconnecting after backoff"
                    );
                    if !self.reconnect_gateway(delay).await {
                        return Ok(PlatformEvent::Shutdown);
                    }
                    continue;
                }
            };
            reconnect_delay = GATEWAY_RECONNECT_BASE_DELAY;
            self.inner.cache.update(&event);
            match event {
                Event::MessageCreate(message) => {
                    let message = message.0;
                    let raw_reference = message.reference.as_ref();
                    tracing::debug!(
                        platform = %self.inner.platform,
                        guild = ?message.guild_id,
                        channel = %message.channel_id,
                        message = %message.id,
                        author = %message.author.id,
                        author_is_bot = message.author.bot,
                        mentions = message.mentions.len(),
                        attachments = message.attachments.len(),
                        content_chars = message.content.chars().count(),
                        raw_reference_kind = raw_reference
                            .map(|reference| reference.kind.name())
                            .unwrap_or("none"),
                        raw_reference_guild = ?raw_reference.and_then(|reference| reference.guild_id),
                        raw_reference_channel = ?raw_reference.and_then(|reference| reference.channel_id),
                        raw_reference_message = ?raw_reference.and_then(|reference| reference.message_id),
                        has_hydrated_reference = message.referenced_message.is_some(),
                        hydrated_reference_message = ?message
                            .referenced_message
                            .as_ref()
                            .map(|reference| reference.id),
                        "received discord message event"
                    );
                    let message = self.platform_message(message).await;
                    let reference = message.referenced_message_id();
                    tracing::debug!(
                        platform = %message.id.platform,
                        guild = ?message.id.guild_id.as_ref().map(ExternalId::as_str),
                        channel = %message.id.channel_id,
                        message = %message.id.message_id,
                        reference_kind = platform_message_reference_kind(&message.reference),
                        reference_guild = ?reference
                            .and_then(|reference| reference.guild_id.as_ref().map(ExternalId::as_str)),
                        reference_channel = ?reference.map(|reference| reference.channel_id.as_str()),
                        reference_message = ?reference.map(|reference| reference.message_id.as_str()),
                        has_hydrated_reference = message.referenced_message().is_some(),
                        "converted discord message event"
                    );
                    return Ok(PlatformEvent::MessageCreated {
                        message: Box::new(message),
                    });
                }
                Event::ReactionAdd(reaction) => {
                    let reaction = platform_reaction(&self.inner.platform, reaction.0);
                    return Ok(PlatformEvent::ReactionAdded { reaction });
                }
                Event::ReactionRemove(reaction) => {
                    let reaction = platform_reaction(&self.inner.platform, reaction.0);
                    return Ok(PlatformEvent::ReactionRemoved { reaction });
                }
                Event::InteractionCreate(interaction) => {
                    if let Some(command) = platform_command(&self.inner.platform, &interaction.0) {
                        return Ok(PlatformEvent::Command { command });
                    }
                    tracing::trace!("ignoring non-command discord interaction");
                }
                Event::GuildCreate(guild) => {
                    log_guild_create(&guild);
                }
                _ => {}
            }
        }
    }

    async fn close_gateway(&self) -> Result<PlatformEvent, DiscordError> {
        let mut shard = self.inner.shard.lock().await;
        close_gateway_shard(&mut shard, &self.inner.platform, self.inner.event_flags).await
    }

    async fn reconnect_gateway(&self, delay: std::time::Duration) -> bool {
        tokio::select! {
            () = tokio::time::sleep(delay) => {}
            () = self.inner.shutdown.cancelled() => return false,
        }
        let mut shard = self.inner.shard.lock().await;
        if self.inner.shutdown.is_cancelled() {
            return false;
        }
        *shard = Shard::new(ShardId::ONE, self.inner.token.clone(), self.inner.intents);
        tracing::info!(platform = %self.inner.platform, "discord gateway reconnecting");
        true
    }

    async fn platform_message(&self, message: Message) -> PlatformMessage {
        platform_message_with_guild(&self.inner.platform, message, None)
    }
}

async fn close_gateway_shard(
    shard: &mut Shard,
    platform: &PlatformName,
    event_flags: EventTypeFlags,
) -> Result<PlatformEvent, DiscordError> {
    if matches!(
        shard.state(),
        ShardState::Disconnected { .. } | ShardState::FatallyClosed
    ) {
        tracing::debug!(
            platform = %platform,
            state = ?shard.state(),
            "discord gateway already disconnected"
        );
        return Ok(PlatformEvent::Shutdown);
    }

    tracing::info!(
        platform = %platform,
        close_code = 1000,
        "closing discord gateway session"
    );
    shard.close(CloseFrame::NORMAL);

    let closed = tokio::time::timeout(GATEWAY_CLOSE_TIMEOUT, async {
        loop {
            match shard.next_event(event_flags).await {
                Some(Ok(Event::GatewayClose(frame))) => {
                    tracing::info!(
                        platform = %platform,
                        close_code = frame.as_ref().map(|frame| frame.code),
                        "discord gateway session closed"
                    );
                    break;
                }
                Some(Ok(event)) => {
                    tracing::trace!(
                        platform = %platform,
                        event = ?event.kind(),
                        "ignoring discord gateway event while closing"
                    );
                }
                Some(Err(error)) => {
                    tracing::warn!(
                        platform = %platform,
                        error = %error,
                        "discord gateway close encountered receive error"
                    );
                    break;
                }
                None => {
                    tracing::debug!(
                        platform = %platform,
                        "discord gateway stream ended during shutdown"
                    );
                    break;
                }
            }
        }
    })
    .await;

    if closed.is_err() {
        tracing::warn!(
            platform = %platform,
            timeout_ms = GATEWAY_CLOSE_TIMEOUT.as_millis(),
            "timed out waiting for discord gateway close"
        );
    }

    Ok(PlatformEvent::Shutdown)
}

fn next_reconnect_delay(delay: std::time::Duration) -> std::time::Duration {
    delay
        .checked_mul(2)
        .unwrap_or(GATEWAY_RECONNECT_MAX_DELAY)
        .min(GATEWAY_RECONNECT_MAX_DELAY)
}

impl MessagePlatform for DiscordPlatform {
    type Error = DiscordError;

    async fn bot_user(&self) -> Result<UserProfile, Self::Error> {
        Ok(self.inner.bot_user.clone())
    }

    async fn register_commands(
        &self,
        commands: Vec<PlatformCommandDefinition>,
        guild: Option<ExternalId>,
    ) -> Result<(), Self::Error> {
        let commands = commands
            .iter()
            .map(discord_command)
            .collect::<Result<Vec<_>, _>>()?;
        let interaction = self.inner.http.interaction(self.inner.application_id);
        match guild {
            Some(guild) => {
                let guild = parse_guild_id(&guild)?;
                interaction.set_guild_commands(guild, &commands).await?;
                tracing::info!(
                    guild = %guild,
                    commands = commands.len(),
                    "registered discord guild commands"
                );
            }
            None => {
                interaction.set_global_commands(&commands).await?;
                tracing::info!(
                    commands = commands.len(),
                    "registered discord global commands"
                );
            }
        }
        Ok(())
    }

    async fn next_event(&self) -> Result<PlatformEvent, Self::Error> {
        if !self.inner.ready_emitted.swap(true, Ordering::AcqRel) {
            return Ok(PlatformEvent::Ready {
                ready: PlatformReady {
                    bot: self.inner.bot_user.clone(),
                },
            });
        }
        self.next_gateway_event().await
    }

    async fn respond_to_command(
        &self,
        response: PlatformCommandResponse,
    ) -> Result<(), Self::Error> {
        let interaction_id = parse_interaction_id(&response.target.interaction_id)?;
        let token = response.target.token;
        let data = InteractionResponseData {
            content: Some(response.content),
            flags: response.ephemeral.then_some(MessageFlags::EPHEMERAL),
            ..InteractionResponseData::default()
        };
        let response = InteractionResponse {
            kind: InteractionResponseType::ChannelMessageWithSource,
            data: Some(data),
        };
        self.inner
            .http
            .interaction(self.inner.application_id)
            .create_response(interaction_id, &token, &response)
            .await?;
        Ok(())
    }

    #[tracing::instrument(
        name = "discord.send_message",
        skip_all,
        fields(
            platform = %request.channel.platform,
            guild = ?request.channel.guild_id,
            channel = %request.channel.channel_id,
            reply_to = ?request.reply_to.as_ref().map(|m| m.message_id.as_str()),
            content_chars = request.content.chars().count(),
            attachments = request.attachments.len(),
            open_thread = request.open_thread.is_some(),
        )
    )]
    async fn send_message(&self, request: SendMessage) -> Result<PostedMessage, Self::Error> {
        let source_channel = parse_channel_id(&request.channel.channel_id)?;
        let mut target_channel = source_channel;
        let mut reply_to = request
            .reply_to
            .as_ref()
            .map(|message| parse_message_id(&message.message_id))
            .transpose()?;

        if let Some(thread) = &request.open_thread {
            if let Some(message) = reply_to.take() {
                let thread = self
                    .inner
                    .http
                    .create_thread_from_message(source_channel, message, &thread.title)
                    .await?
                    .model()
                    .await?;
                if let Err(error) = self.inner.http.join_thread(thread.id).await {
                    tracing::warn!(
                        error = %error,
                        thread = %thread.id,
                        "failed to join newly created discord thread"
                    );
                }
                target_channel = thread.id;
                tracing::info!(
                    thread = %target_channel,
                    "opened discord thread for platform reply"
                );
            } else {
                tracing::warn!("thread request ignored because no reply target was provided");
            }
        }

        let chunks = split_discord_content(&request.content);
        let attachment_batches = discord_attachment_batches(&request.attachments);
        let attachment_message_count = attachment_batches.len().saturating_sub(1);
        let mut posted = Vec::with_capacity(chunks.len() + attachment_message_count);
        for (index, chunk) in chunks.iter().enumerate() {
            let is_last = index + 1 == chunks.len();
            let attachments = if is_last {
                attachment_batches.first().map(Vec::as_slice).unwrap_or(&[])
            } else {
                &[]
            };
            let mut builder = self
                .inner
                .http
                .create_message(target_channel)
                .content(chunk);
            if request.suppress_embeds {
                builder = builder.flags(MessageFlags::SUPPRESS_EMBEDS);
            }
            if index == 0
                && let Some(reply_to) = reply_to
            {
                builder = builder.reply(reply_to);
            }
            if !attachments.is_empty() {
                builder = builder.attachments(attachments);
            }
            let message = builder.await?.model().await?;
            tracing::trace!(
                message = %message.id,
                channel = %message.channel_id,
                chunk = index,
                chunks = chunks.len(),
                attachments = attachments.len(),
                "posted discord message chunk"
            );
            posted.push(message_ref_from_ids(
                &self.inner.platform,
                request.channel.guild_id.clone(),
                target_channel,
                message.id,
            ));
        }
        for (index, attachments) in attachment_batches.iter().enumerate().skip(1) {
            let mut builder = self.inner.http.create_message(target_channel).content("");
            if request.suppress_embeds {
                builder = builder.flags(MessageFlags::SUPPRESS_EMBEDS);
            }
            let message = builder.attachments(attachments).await?.model().await?;
            tracing::trace!(
                message = %message.id,
                channel = %message.channel_id,
                attachment_batch = index,
                attachment_batches = attachment_batches.len(),
                attachments = attachments.len(),
                "posted discord attachment batch"
            );
            posted.push(message_ref_from_ids(
                &self.inner.platform,
                request.channel.guild_id.clone(),
                target_channel,
                message.id,
            ));
        }

        let id = posted
            .first()
            .cloned()
            .ok_or(DiscordError::NoPostedMessage)?;
        let extra_messages = posted.into_iter().skip(1).collect();
        Ok(PostedMessage {
            id,
            channel: ChannelRef {
                platform: self.inner.platform.clone(),
                guild_id: request.channel.guild_id,
                channel_id: external_id(target_channel),
            },
            extra_messages,
        })
    }

    async fn delete_message(&self, message: MessageRef) -> Result<(), Self::Error> {
        let channel = parse_channel_id(&message.channel_id)?;
        let message = parse_message_id(&message.message_id)?;
        self.inner.http.delete_message(channel, message).await?;
        Ok(())
    }

    async fn add_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> Result<(), Self::Error> {
        let channel = parse_channel_id(&message.channel_id)?;
        let message = parse_message_id(&message.message_id)?;
        let reaction = request_reaction(&reaction)?;
        self.inner
            .http
            .create_reaction(channel, message, &reaction)
            .await?;
        Ok(())
    }

    async fn remove_own_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> Result<(), Self::Error> {
        let channel = parse_channel_id(&message.channel_id)?;
        let message = parse_message_id(&message.message_id)?;
        let reaction = request_reaction(&reaction)?;
        self.inner
            .http
            .delete_current_user_reaction(channel, message, &reaction)
            .await?;
        Ok(())
    }

    async fn typing(&self, channel: ChannelRef) -> Result<(), Self::Error> {
        let channel = parse_channel_id(&channel.channel_id)?;
        self.inner.http.create_typing_trigger(channel).await?;
        Ok(())
    }

    async fn fetch_messages(
        &self,
        request: FetchMessages,
    ) -> Result<Vec<PlatformMessage>, Self::Error> {
        let channel = parse_channel_id(&request.channel.channel_id)?;
        let before = request
            .before
            .as_ref()
            .map(|message| parse_message_id(&message.message_id))
            .transpose()?;

        let response = if let Some(before) = before {
            self.inner
                .http
                .channel_messages(channel)
                .before(before)
                .limit(request.limit)
                .await
        } else {
            self.inner
                .http
                .channel_messages(channel)
                .limit(request.limit)
                .await
        }?;
        let mut messages = response.models().await?;
        messages.reverse();

        Ok(messages
            .into_iter()
            .filter(|message| message.author.id != self.inner.bot_user_id)
            .map(|message| platform_message_with_guild(&self.inner.platform, message, None))
            .collect())
    }

    async fn message_context(
        &self,
        message: &PlatformMessage,
        relationship: PlatformMessageRelationship,
    ) -> Result<serde_json::Value, Self::Error> {
        Ok(discord_message_context_json(
            message,
            relationship,
            &self.inner.cache,
        ))
    }

    async fn parent_channel(&self, channel: ChannelRef) -> Result<ChannelRef, Self::Error> {
        let channel_id = parse_channel_id(&channel.channel_id)?;
        let discord_channel = self.inner.http.channel(channel_id).await?.model().await?;
        let parent = match discord_channel.kind {
            ChannelType::AnnouncementThread
            | ChannelType::PublicThread
            | ChannelType::PrivateThread => discord_channel.parent_id.unwrap_or(channel_id),
            _ => channel_id,
        };
        Ok(ChannelRef {
            platform: self.inner.platform.clone(),
            guild_id: channel.guild_id,
            channel_id: external_id(parent),
        })
    }
}

/// Discord platform errors.
#[derive(Debug, Error)]
pub enum DiscordError {
    /// Discord REST error.
    #[error("discord http: {0}")]
    Http(#[from] twilight_http::Error),
    /// Discord response body could not be decoded.
    #[error("discord deserialize: {0}")]
    Deserialize(#[from] twilight_http::response::DeserializeBodyError),
    /// Platform id was not a valid non-zero Discord snowflake.
    #[error("invalid discord {kind} id `{value}`")]
    InvalidId {
        /// ID kind.
        kind: &'static str,
        /// Bad value.
        value: String,
    },
    /// No Discord message was posted for a send request.
    #[error("discord send did not return any posted messages")]
    NoPostedMessage,
    /// Command definition could not be converted to Discord.
    #[error("invalid command option `{name}`: {message}")]
    InvalidCommandOption {
        /// Option name.
        name: String,
        /// Error detail.
        message: String,
    },
}

fn platform_message_with_guild(
    platform: &PlatformName,
    message: Message,
    fallback_guild_id: Option<ExternalId>,
) -> PlatformMessage {
    let guild_id = message.guild_id.map(external_id).or(fallback_guild_id);
    let reference_guild_id = message
        .reference
        .as_ref()
        .and_then(|reference| reference.guild_id.map(external_id))
        .or_else(|| guild_id.clone());
    let reference = match message.referenced_message {
        Some(message) => PlatformMessageReference::Hydrated(Box::new(platform_message_with_guild(
            platform,
            *message,
            reference_guild_id,
        ))),
        None => message
            .reference
            .as_ref()
            .and_then(|reference| message_ref_from_reference(platform, guild_id.clone(), reference))
            .map(PlatformMessageReference::Id)
            .unwrap_or_default(),
    };
    PlatformMessage {
        id: message_ref_from_ids(platform, guild_id.clone(), message.channel_id, message.id),
        author: user_profile(
            platform,
            guild_id.clone(),
            &message.author,
            message.member.as_ref(),
        ),
        content: message.content,
        mentions: message
            .mentions
            .iter()
            .map(|mention| UserRef {
                platform: platform.clone(),
                guild_id: guild_id.clone(),
                user_id: external_id(mention.id),
            })
            .collect(),
        mention_profiles: message
            .mentions
            .iter()
            .map(|mention| mention_user_profile(platform, guild_id.clone(), mention))
            .collect(),
        reference,
        attachments: message
            .attachments
            .iter()
            .map(|attachment| {
                attachment_ref(
                    attachment,
                    message
                        .flags
                        .is_some_and(|flags| flags.contains(MessageFlags::IS_VOICE_MESSAGE)),
                )
            })
            .collect(),
        created_at: timestamp_to_offset(message.timestamp),
    }
}

fn discord_message_context_json(
    message: &PlatformMessage,
    relationship: PlatformMessageRelationship,
    cache: &DefaultInMemoryCache,
) -> serde_json::Value {
    let author = cached_user_context(cache, &message.author.id);
    let guild_name = message
        .id
        .guild_id
        .as_ref()
        .and_then(|guild| cached_guild_name(cache, guild));
    serde_json::json!({
        "type": "discord_message",
        "relationship": discord_message_relationship(relationship),
        "platform": message.id.platform.as_str(),
        "guild": discord_entity_json(
            message.id.guild_id.as_ref(),
            guild_name.as_deref(),
        ),
        "channel": {
            "id": message.id.channel_id.as_str(),
            "name": cached_channel_name(cache, &message.id.channel_id),
        },
        "message": {
            "id": message.id.message_id.as_str(),
            "created_at": message.created_at.to_string(),
        },
        "author": {
            "id": message.author.id.user_id.as_str(),
            "username": message.author.username.as_str(),
            "global_name": message.author.name.as_deref().or(author.global_name.as_deref()),
            "guild_display_name": message
                .author
                .display_name
                .as_deref()
                .or(author.guild_display_name.as_deref()),
            "is_bot": message.author.is_bot,
        },
        "mentioned_users": message.mentions.iter().map(|mention| {
            let profile = message.mention_profiles.iter().find(|profile| {
                profile.id.platform == mention.platform
                    && profile.id.guild_id == mention.guild_id
                    && profile.id.user_id == mention.user_id
            });
            let cached = cached_user_context(cache, mention);
            serde_json::json!({
                "id": mention.user_id.as_str(),
                "mention": format!("<@{}>", mention.user_id.as_str()),
                "username": profile
                    .map(|profile| profile.username.as_str())
                    .or(cached.username.as_deref()),
                "global_name": profile
                    .and_then(|profile| profile.name.as_deref())
                    .or(cached.global_name.as_deref()),
                "guild_display_name": profile
                    .and_then(|profile| profile.display_name.as_deref())
                    .or(cached.guild_display_name.as_deref()),
                "is_bot": profile.map(|profile| profile.is_bot).or(cached.is_bot),
            })
        }).collect::<Vec<_>>(),
        "content": message.content.as_str(),
        "attachments": message.attachments.iter().map(|attachment| {
            serde_json::json!({
                "id": attachment.id.as_ref().map(ExternalId::as_str),
                "filename": attachment.filename.as_str(),
                "content_type": attachment.content_type.as_deref(),
                "size_bytes": attachment.size_bytes,
                "duration_seconds": attachment.duration_seconds,
                "is_voice_message": attachment.is_voice_message,
                "waveform": attachment.waveform.as_deref(),
            })
        }).collect::<Vec<_>>(),
    })
}

#[derive(Debug, Default)]
struct CachedUserContext {
    username: Option<String>,
    global_name: Option<String>,
    guild_display_name: Option<String>,
    is_bot: Option<bool>,
}

fn cached_guild_name(cache: &DefaultInMemoryCache, guild: &ExternalId) -> Option<String> {
    let guild = parse_guild_id(guild).ok()?;
    cache.guild(guild).map(|guild| guild.name().to_string())
}

fn cached_channel_name(cache: &DefaultInMemoryCache, channel: &ExternalId) -> Option<String> {
    let channel = parse_channel_id(channel).ok()?;
    cache
        .channel(channel)
        .and_then(|channel| channel.name.as_deref().map(str::to_string))
}

fn cached_user_context(cache: &DefaultInMemoryCache, user: &UserRef) -> CachedUserContext {
    let user_id = parse_user_id(&user.user_id).ok();
    let guild_id = user
        .guild_id
        .as_ref()
        .and_then(|guild| parse_guild_id(guild).ok());
    let mut context = CachedUserContext::default();

    if let (Some(guild_id), Some(user_id)) = (guild_id, user_id)
        && let Some(member) = cache.member(guild_id, user_id)
    {
        context.guild_display_name = member.nick().map(str::to_string);
    }

    if let Some(user_id) = user_id
        && let Some(user) = cache.user(user_id)
    {
        context.username = Some(user.name.clone());
        context.global_name.clone_from(&user.global_name);
        context.is_bot = Some(user.bot);
    }

    context
}

fn discord_message_relationship(relationship: PlatformMessageRelationship) -> &'static str {
    match relationship {
        PlatformMessageRelationship::Current => "current",
        PlatformMessageRelationship::Referenced => "referenced",
        PlatformMessageRelationship::Fetched => "fetched",
    }
}

fn discord_entity_json(id: Option<&ExternalId>, name: Option<&str>) -> serde_json::Value {
    id.map(|id| {
        serde_json::json!({
            "id": id.as_str(),
            "name": name,
        })
    })
    .unwrap_or(serde_json::Value::Null)
}

fn platform_reaction(platform: &PlatformName, reaction: GatewayReaction) -> PlatformReaction {
    let guild_id = reaction.guild_id.map(external_id);
    PlatformReaction {
        message: message_ref_from_ids(
            platform,
            guild_id.clone(),
            reaction.channel_id,
            reaction.message_id,
        ),
        user: UserRef {
            platform: platform.clone(),
            guild_id,
            user_id: external_id(reaction.user_id),
        },
        reaction: reaction_kind(&reaction.emoji),
    }
}

#[allow(deprecated)]
fn platform_command(platform: &PlatformName, interaction: &Interaction) -> Option<PlatformCommand> {
    let Some(InteractionData::ApplicationCommand(data)) = interaction.data.as_ref() else {
        return None;
    };
    let user = interaction.author()?;
    let channel_id = interaction
        .channel
        .as_ref()
        .map(|channel| channel.id)
        .or(interaction.channel_id)?;
    let guild_id = interaction.guild_id.map(external_id);
    Some(PlatformCommand {
        name: data.name.clone(),
        user: UserRef {
            platform: platform.clone(),
            guild_id: guild_id.clone(),
            user_id: external_id(user.id),
        },
        channel: ChannelRef {
            platform: platform.clone(),
            guild_id: guild_id.clone(),
            channel_id: external_id(channel_id),
        },
        options: platform_command_options(platform, guild_id.clone(), &data.options),
        is_admin: interaction
            .member
            .as_ref()
            .and_then(|member| member.permissions)
            .is_some_and(|permissions| permissions.contains(Permissions::ADMINISTRATOR)),
        response_target: PlatformCommandResponseTarget {
            platform: platform.clone(),
            interaction_id: external_id(interaction.id),
            token: interaction.token.clone(),
        },
    })
}

fn platform_command_options(
    platform: &PlatformName,
    guild_id: Option<ExternalId>,
    options: &[CommandDataOption],
) -> Vec<PlatformCommandInput> {
    options
        .iter()
        .map(|option| platform_command_option(platform, guild_id.clone(), option))
        .collect()
}

fn platform_command_option(
    platform: &PlatformName,
    guild_id: Option<ExternalId>,
    option: &CommandDataOption,
) -> PlatformCommandInput {
    match &option.value {
        InteractionCommandOptionValue::SubCommand(options)
        | InteractionCommandOptionValue::SubCommandGroup(options) => PlatformCommandInput {
            name: option.name.clone(),
            value: None,
            options: platform_command_options(platform, guild_id, options),
        },
        InteractionCommandOptionValue::String(value)
        | InteractionCommandOptionValue::Focused(value, _) => PlatformCommandInput {
            name: option.name.clone(),
            value: Some(PlatformCommandValue::String(value.clone())),
            options: Vec::new(),
        },
        InteractionCommandOptionValue::Integer(value) => PlatformCommandInput {
            name: option.name.clone(),
            value: Some(PlatformCommandValue::Integer(*value)),
            options: Vec::new(),
        },
        InteractionCommandOptionValue::Number(value) => PlatformCommandInput {
            name: option.name.clone(),
            value: Some(PlatformCommandValue::Number(*value)),
            options: Vec::new(),
        },
        InteractionCommandOptionValue::Boolean(value) => PlatformCommandInput {
            name: option.name.clone(),
            value: Some(PlatformCommandValue::Boolean(*value)),
            options: Vec::new(),
        },
        InteractionCommandOptionValue::Channel(channel_id) => PlatformCommandInput {
            name: option.name.clone(),
            value: Some(PlatformCommandValue::Channel(ChannelRef {
                platform: platform.clone(),
                guild_id,
                channel_id: external_id(*channel_id),
            })),
            options: Vec::new(),
        },
        InteractionCommandOptionValue::User(user_id) => PlatformCommandInput {
            name: option.name.clone(),
            value: Some(PlatformCommandValue::User(UserRef {
                platform: platform.clone(),
                guild_id,
                user_id: external_id(*user_id),
            })),
            options: Vec::new(),
        },
        InteractionCommandOptionValue::Role(role_id) => PlatformCommandInput {
            name: option.name.clone(),
            value: Some(PlatformCommandValue::Role(external_id(*role_id))),
            options: Vec::new(),
        },
        InteractionCommandOptionValue::Mentionable(id) => PlatformCommandInput {
            name: option.name.clone(),
            value: Some(PlatformCommandValue::Mentionable(external_id(*id))),
            options: Vec::new(),
        },
        InteractionCommandOptionValue::Attachment(id) => PlatformCommandInput {
            name: option.name.clone(),
            value: Some(PlatformCommandValue::Attachment(external_id(*id))),
            options: Vec::new(),
        },
    }
}

fn current_user_profile(platform: &PlatformName, user: &CurrentUser) -> UserProfile {
    UserProfile {
        id: UserRef {
            platform: platform.clone(),
            guild_id: None,
            user_id: external_id(user.id),
        },
        username: user.name.clone(),
        name: user.global_name.clone(),
        display_name: None,
        avatar_url: Some(match user.avatar {
            Some(hash) => avatar_url(user.id, hash.to_string()),
            None => default_avatar_url(user.id),
        }),
        is_bot: user.bot,
    }
}

fn user_profile(
    platform: &PlatformName,
    guild_id: Option<ExternalId>,
    user: &User,
    member: Option<&twilight_model::guild::PartialMember>,
) -> UserProfile {
    UserProfile {
        id: UserRef {
            platform: platform.clone(),
            guild_id,
            user_id: external_id(user.id),
        },
        username: user.name.clone(),
        name: user.global_name.clone(),
        display_name: member.and_then(|member| member.nick.clone()),
        avatar_url: Some(match user.avatar {
            Some(hash) => avatar_url(user.id, hash.to_string()),
            None => default_avatar_url(user.id),
        }),
        is_bot: user.bot,
    }
}

fn mention_user_profile(
    platform: &PlatformName,
    guild_id: Option<ExternalId>,
    mention: &Mention,
) -> UserProfile {
    UserProfile {
        id: UserRef {
            platform: platform.clone(),
            guild_id,
            user_id: external_id(mention.id),
        },
        username: mention.name.clone(),
        name: None,
        display_name: mention
            .member
            .as_ref()
            .and_then(|member| member.nick.clone()),
        avatar_url: Some(match mention.avatar {
            Some(hash) => avatar_url(mention.id, hash.to_string()),
            None => default_avatar_url(mention.id),
        }),
        is_bot: mention.bot,
    }
}

fn attachment_ref(
    attachment: &twilight_model::channel::Attachment,
    is_voice_message: bool,
) -> AttachmentRef {
    AttachmentRef {
        id: Some(external_id(attachment.id)),
        url: attachment.url.clone(),
        filename: attachment.filename.clone(),
        content_type: attachment.content_type.clone(),
        size_bytes: Some(attachment.size),
        duration_seconds: attachment.duration_secs,
        is_voice_message,
        waveform: attachment.waveform.clone(),
    }
}

fn reaction_kind(reaction: &EmojiReactionType) -> ReactionKind {
    match reaction {
        EmojiReactionType::Unicode { name } => ReactionKind::Unicode { name: name.clone() },
        EmojiReactionType::Custom { id, name, .. } => ReactionKind::Custom {
            id: external_id(*id),
            name: name.clone(),
        },
    }
}

fn request_reaction(reaction: &ReactionKind) -> Result<RequestReactionType<'_>, DiscordError> {
    match reaction {
        ReactionKind::Unicode { name } => Ok(RequestReactionType::Unicode { name }),
        ReactionKind::Custom { id, name } => Ok(RequestReactionType::Custom {
            id: parse_emoji_id(id)?,
            name: name.as_deref(),
        }),
    }
}

fn message_ref_from_ids(
    platform: &PlatformName,
    guild_id: Option<ExternalId>,
    channel_id: Id<ChannelMarker>,
    message_id: Id<MessageMarker>,
) -> MessageRef {
    MessageRef {
        platform: platform.clone(),
        guild_id,
        channel_id: external_id(channel_id),
        message_id: external_id(message_id),
    }
}

fn message_ref_from_reference(
    platform: &PlatformName,
    guild_id: Option<ExternalId>,
    reference: &MessageReference,
) -> Option<MessageRef> {
    Some(message_ref_from_ids(
        platform,
        reference.guild_id.map(external_id).or(guild_id),
        reference.channel_id?,
        reference.message_id?,
    ))
}

fn platform_message_reference_kind(reference: &PlatformMessageReference) -> &'static str {
    match reference {
        PlatformMessageReference::None => "none",
        PlatformMessageReference::Id(_) => "id",
        PlatformMessageReference::Hydrated(_) => "hydrated",
    }
}

fn avatar_url(user_id: Id<UserMarker>, hash: String) -> String {
    format!("https://cdn.discordapp.com/avatars/{user_id}/{hash}.png?size=128")
}

fn default_avatar_url(user_id: Id<UserMarker>) -> String {
    let bucket = (user_id.get() >> 22) % 6;
    format!("https://cdn.discordapp.com/embed/avatars/{bucket}.png")
}

fn discord_command(definition: &PlatformCommandDefinition) -> Result<Command, DiscordError> {
    let mut builder = CommandBuilder::new(
        &definition.name,
        &definition.description,
        CommandType::ChatInput,
    );
    if definition.admin_only {
        builder = builder.default_member_permissions(Permissions::ADMINISTRATOR);
    }
    for option in &definition.options {
        builder = builder.option(discord_command_option(option)?);
    }
    Ok(builder.build())
}

fn discord_command_option(option: &PlatformCommandOption) -> Result<CommandOption, DiscordError> {
    match option.kind {
        PlatformCommandOptionKind::SubCommand => {
            let mut builder = SubCommandBuilder::new(&option.name, &option.description);
            for child in &option.options {
                builder = builder.option(discord_command_option(child)?);
            }
            Ok(builder.build())
        }
        PlatformCommandOptionKind::String => {
            let mut builder =
                StringBuilder::new(&option.name, &option.description).required(option.required);
            if !option.choices.is_empty() {
                builder = builder.choices(
                    option
                        .choices
                        .iter()
                        .map(|choice| (choice.name.clone(), choice.value.clone())),
                );
            }
            Ok(builder.build())
        }
        PlatformCommandOptionKind::Integer => {
            let mut builder =
                IntegerBuilder::new(&option.name, &option.description).required(option.required);
            if let Some(min) = option.min_integer {
                builder = builder.min_value(min);
            }
            if let Some(max) = option.max_integer {
                builder = builder.max_value(max);
            }
            if !option.choices.is_empty() {
                let choices = option
                    .choices
                    .iter()
                    .map(|choice| {
                        choice
                            .value
                            .parse::<i64>()
                            .map(|value| (choice.name.clone(), value))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| DiscordError::InvalidCommandOption {
                        name: option.name.clone(),
                        message: error.to_string(),
                    })?;
                builder = builder.choices(choices);
            }
            Ok(builder.build())
        }
        PlatformCommandOptionKind::Channel => {
            Ok(ChannelBuilder::new(&option.name, &option.description)
                .required(option.required)
                .build())
        }
    }
}

fn timestamp_to_offset(timestamp: twilight_model::util::Timestamp) -> OffsetDateTime {
    OffsetDateTime::from_unix_timestamp(timestamp.as_secs()).unwrap_or(OffsetDateTime::UNIX_EPOCH)
}

fn http_attachments(attachments: &[OutgoingAttachment]) -> Vec<HttpAttachment> {
    attachments
        .iter()
        .enumerate()
        .map(|(index, attachment)| {
            HttpAttachment::from_bytes(
                attachment.filename.clone(),
                attachment.bytes.clone(),
                u64::try_from(index).unwrap_or(u64::MAX),
            )
        })
        .collect()
}

fn discord_attachment_batches(attachments: &[OutgoingAttachment]) -> Vec<Vec<HttpAttachment>> {
    attachments
        .chunks(DISCORD_ATTACHMENT_LIMIT)
        .map(http_attachments)
        .collect()
}

fn split_discord_content(content: &str) -> Vec<String> {
    if content.is_empty() {
        return vec![String::new()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    for segment in markdown_segments(content) {
        let pieces = match segment.kind {
            MarkdownSegmentKind::Text => split_text_segment(segment.text, DISCORD_MESSAGE_LIMIT),
            MarkdownSegmentKind::CodeFence => split_code_fence_segment(segment.text),
        };
        for piece in pieces {
            append_discord_chunk(&mut chunks, &mut current, &piece);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

fn append_discord_chunk(chunks: &mut Vec<String>, current: &mut String, piece: &str) {
    if piece.is_empty() {
        return;
    }
    if current.chars().count() + piece.chars().count() <= DISCORD_MESSAGE_LIMIT {
        current.push_str(piece);
        return;
    }
    if !current.is_empty() {
        chunks.push(std::mem::take(current));
    }
    if piece.chars().count() <= DISCORD_MESSAGE_LIMIT {
        current.push_str(piece);
        return;
    }
    for chunk in split_text_segment(piece, DISCORD_MESSAGE_LIMIT) {
        if chunk.chars().count() <= DISCORD_MESSAGE_LIMIT {
            chunks.push(chunk);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MarkdownSegment<'a> {
    kind: MarkdownSegmentKind,
    text: &'a str,
}

#[derive(Debug, Clone, Copy)]
enum MarkdownSegmentKind {
    Text,
    CodeFence,
}

#[derive(Debug, Clone, Copy)]
struct CodeFenceMarker {
    marker: char,
    width: usize,
}

impl CodeFenceMarker {
    fn closes(self, line: &str) -> bool {
        code_fence_marker(line).is_some_and(|candidate| {
            candidate.marker == self.marker && candidate.width >= self.width
        })
    }

    fn closing_line(self) -> String {
        self.marker.to_string().repeat(self.width)
    }
}

fn markdown_segments(content: &str) -> Vec<MarkdownSegment<'_>> {
    let mut segments = Vec::new();
    let mut text_start = 0usize;
    let mut line_start = 0usize;
    let mut active_fence: Option<(CodeFenceMarker, usize)> = None;

    while line_start < content.len() {
        let line_end = next_line_end(content, line_start);
        let line = &content[line_start..line_end];
        let line_without_ending = trim_line_ending(line);
        match active_fence {
            Some((marker, fence_start)) => {
                if marker.closes(line_without_ending) {
                    segments.push(MarkdownSegment {
                        kind: MarkdownSegmentKind::CodeFence,
                        text: &content[fence_start..line_end],
                    });
                    text_start = line_end;
                    active_fence = None;
                }
            }
            None => {
                if let Some(marker) = code_fence_marker(line_without_ending) {
                    if text_start < line_start {
                        segments.push(MarkdownSegment {
                            kind: MarkdownSegmentKind::Text,
                            text: &content[text_start..line_start],
                        });
                    }
                    active_fence = Some((marker, line_start));
                }
            }
        }
        line_start = line_end;
    }

    if let Some((_marker, fence_start)) = active_fence {
        segments.push(MarkdownSegment {
            kind: MarkdownSegmentKind::CodeFence,
            text: &content[fence_start..],
        });
    } else if text_start < content.len() {
        segments.push(MarkdownSegment {
            kind: MarkdownSegmentKind::Text,
            text: &content[text_start..],
        });
    }
    segments
}

fn split_text_segment(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = text;
    while remaining.chars().count() > max_chars {
        let split_at = find_text_split_point(remaining, max_chars);
        chunks.push(remaining[..split_at].to_string());
        remaining = &remaining[split_at..];
    }
    if !remaining.is_empty() {
        chunks.push(remaining.to_string());
    }
    chunks
}

fn split_code_fence_segment(segment: &str) -> Vec<String> {
    if segment.chars().count() <= DISCORD_MESSAGE_LIMIT {
        return vec![segment.to_string()];
    }

    let Some((opening, body, marker)) = code_fence_parts(segment) else {
        return split_text_segment(segment, DISCORD_MESSAGE_LIMIT);
    };
    let closing = marker.closing_line();
    let overhead = opening.chars().count() + closing.chars().count() + 1;
    if overhead >= DISCORD_MESSAGE_LIMIT {
        return split_text_segment(segment, DISCORD_MESSAGE_LIMIT);
    }

    split_code_body(body, DISCORD_MESSAGE_LIMIT - overhead)
        .into_iter()
        .map(|body| balanced_code_chunk(opening, &body, &closing))
        .collect()
}

fn code_fence_parts(segment: &str) -> Option<(&str, &str, CodeFenceMarker)> {
    let opening_end = next_line_end(segment, 0);
    let opening = &segment[..opening_end];
    let marker = code_fence_marker(trim_line_ending(opening))?;
    let mut line_start = opening_end;
    while line_start < segment.len() {
        let line_end = next_line_end(segment, line_start);
        let line = &segment[line_start..line_end];
        if marker.closes(trim_line_ending(line)) {
            return Some((opening, &segment[opening_end..line_start], marker));
        }
        line_start = line_end;
    }
    Some((opening, &segment[opening_end..], marker))
}

fn split_code_body(body: &str, max_chars: usize) -> Vec<String> {
    if body.chars().count() <= max_chars {
        return vec![body.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = body;
    while remaining.chars().count() > max_chars {
        let split_at = find_code_split_point(remaining, max_chars);
        chunks.push(remaining[..split_at].to_string());
        remaining = &remaining[split_at..];
    }
    if !remaining.is_empty() {
        chunks.push(remaining.to_string());
    }
    chunks
}

fn balanced_code_chunk(opening: &str, body: &str, closing: &str) -> String {
    let mut chunk = String::with_capacity(opening.len() + body.len() + closing.len() + 1);
    chunk.push_str(opening);
    chunk.push_str(body);
    if !chunk.ends_with('\n') {
        chunk.push('\n');
    }
    chunk.push_str(closing);
    chunk
}

fn find_text_split_point(text: &str, max_chars: usize) -> usize {
    let limit = byte_index_after_chars(text, max_chars);
    let candidate = &text[..limit];
    for separator in ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", ": ", " "] {
        if let Some(position) = candidate.rfind(separator) {
            let split_at = position + separator.len();
            if split_at > 0 {
                return split_at;
            }
        }
    }
    limit
}

fn find_code_split_point(text: &str, max_chars: usize) -> usize {
    let limit = byte_index_after_chars(text, max_chars);
    let candidate = &text[..limit];
    for separator in ["\n", " ", ",", ";", ":"] {
        if let Some(position) = candidate.rfind(separator) {
            let split_at = position + separator.len();
            if split_at > 0 {
                return split_at;
            }
        }
    }
    limit
}

fn byte_index_after_chars(text: &str, max_chars: usize) -> usize {
    text.char_indices()
        .nth(max_chars)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

fn next_line_end(text: &str, start: usize) -> usize {
    text[start..]
        .find('\n')
        .map(|offset| start + offset + 1)
        .unwrap_or(text.len())
}

fn trim_line_ending(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

fn code_fence_marker(line: &str) -> Option<CodeFenceMarker> {
    let trimmed = line.trim_start();
    let marker = trimmed.chars().next()?;
    if marker != '`' && marker != '~' {
        return None;
    }
    let width = trimmed.chars().take_while(|&ch| ch == marker).count();
    (width >= CODE_FENCE_MIN_WIDTH).then_some(CodeFenceMarker { marker, width })
}

fn parse_channel_id(id: &ExternalId) -> Result<Id<ChannelMarker>, DiscordError> {
    parse_id("channel", id)
}

fn parse_message_id(id: &ExternalId) -> Result<Id<MessageMarker>, DiscordError> {
    parse_id("message", id)
}

fn parse_emoji_id(id: &ExternalId) -> Result<Id<EmojiMarker>, DiscordError> {
    parse_id("emoji", id)
}

fn parse_guild_id(id: &ExternalId) -> Result<Id<GuildMarker>, DiscordError> {
    parse_id("guild", id)
}

fn parse_user_id(id: &ExternalId) -> Result<Id<UserMarker>, DiscordError> {
    parse_id("user", id)
}

fn parse_interaction_id(id: &ExternalId) -> Result<Id<InteractionMarker>, DiscordError> {
    parse_id("interaction", id)
}

fn parse_id<T>(kind: &'static str, id: &ExternalId) -> Result<Id<T>, DiscordError> {
    let value = id
        .as_str()
        .parse::<u64>()
        .map_err(|_| DiscordError::InvalidId {
            kind,
            value: id.as_str().to_string(),
        })?;
    Id::new_checked(value).ok_or_else(|| DiscordError::InvalidId {
        kind,
        value: id.as_str().to_string(),
    })
}

fn external_id<T>(id: Id<T>) -> ExternalId {
    ExternalId::new(id.get().to_string())
}

fn log_guild_create(event: &GuildCreate) {
    match event {
        GuildCreate::Available(guild) => tracing::info!(
            guild_id = %guild.id,
            guild_name = %guild.name,
            member_count = guild.member_count.unwrap_or(0),
            "discord bot is active in guild"
        ),
        GuildCreate::Unavailable(guild) => {
            tracing::warn!(guild_id = %guild.id, "discord guild is unavailable");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chudbot_api::{
        AttachmentRef, ExternalId, MessageRef, OutgoingAttachment, PlatformMessage,
        PlatformMessageReference, PlatformMessageRelationship, PlatformName, ReactionKind,
        UserProfile, UserRef,
    };
    use time::OffsetDateTime;
    use twilight_cache_inmemory::{DefaultInMemoryCache, ResourceType};
    use twilight_model::channel::message::{
        EmojiReactionType, MessageReference, MessageReferenceType,
    };
    use twilight_model::gateway::payload::incoming::{ChannelUpdate, GuildCreate};
    use twilight_model::id::Id;

    use super::{
        DISCORD_ATTACHMENT_LIMIT, DISCORD_MESSAGE_LIMIT, DiscordError, GATEWAY_RECONNECT_MAX_DELAY,
        discord_attachment_batches, discord_message_context_json, message_ref_from_reference,
        next_reconnect_delay, parse_channel_id, reaction_kind, split_discord_content,
    };

    #[test]
    fn split_discord_content_keeps_chunks_within_limit() {
        let input = "a".repeat(DISCORD_MESSAGE_LIMIT + 3);

        let chunks = split_discord_content(&input);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chars().count(), DISCORD_MESSAGE_LIMIT);
        assert_eq!(chunks[1], "aaa");
        assert_eq!(chunks.join(""), input);
    }

    #[test]
    fn split_discord_content_keeps_unicode_boundaries() {
        let input = "🙂".repeat(DISCORD_MESSAGE_LIMIT + 1);

        let chunks = split_discord_content(&input);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].chars().count(), DISCORD_MESSAGE_LIMIT);
        assert_eq!(chunks[1], "🙂");
        assert_eq!(chunks.join(""), input);
    }

    #[test]
    fn split_discord_content_prefers_line_breaks() {
        let input = format!(
            "{}\n{}",
            "a".repeat(DISCORD_MESSAGE_LIMIT - 10),
            "b".repeat(20)
        );

        let chunks = split_discord_content(&input);

        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].ends_with('\n'));
        assert_eq!(chunks.join(""), input);
    }

    #[test]
    fn split_discord_content_keeps_small_code_fence_together() {
        let input = format!(
            "{}\n```toml\n[dependencies]\nanchor-lang = \"0.29.0\"\nanchor-spl = \"0.29.0\"\n```\n",
            "a".repeat(DISCORD_MESSAGE_LIMIT - 20),
        );

        let chunks = split_discord_content(&input);

        assert_eq!(chunks.len(), 2);
        assert!(chunks[1].starts_with("```toml\n"));
        assert!(chunks[1].contains("anchor-spl = \"0.29.0\""));
        assert!(chunks[1].ends_with("```\n"));
        assert_eq!(chunks.join(""), input);
    }

    #[test]
    fn split_discord_content_rebalances_oversized_code_fence() {
        let input = format!(
            "```toml\n{}\n```\n",
            (0..150)
                .map(|index| {
                    format!(
                        "dependency-{index} = {{ version = \"1.{index}.0\", features = [\"one\", \"two\"] }}\n"
                    )
                })
                .collect::<String>(),
        );

        let chunks = split_discord_content(&input);

        assert!(chunks.len() > 1);
        for chunk in chunks {
            assert!(chunk.chars().count() <= DISCORD_MESSAGE_LIMIT);
            assert!(chunk.starts_with("```toml\n"));
            assert!(chunk.ends_with("```"));
        }
    }

    #[test]
    fn discord_attachment_batches_keep_each_message_within_limit() {
        let attachments = outgoing_attachments(23);

        let batches = discord_attachment_batches(&attachments);

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), DISCORD_ATTACHMENT_LIMIT);
        assert_eq!(batches[1].len(), DISCORD_ATTACHMENT_LIMIT);
        assert_eq!(batches[2].len(), 3);
        assert_eq!(batches[0][0].filename, "generated-0.png");
        assert_eq!(batches[1][0].filename, "generated-10.png");
        assert_eq!(batches[2][0].filename, "generated-20.png");
    }

    #[test]
    fn discord_attachment_batches_reset_attachment_ids_per_message() {
        let attachments = outgoing_attachments(DISCORD_ATTACHMENT_LIMIT + 1);

        let batches = discord_attachment_batches(&attachments);

        assert_eq!(
            batches[0]
                .iter()
                .map(|attachment| attachment.id)
                .collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );
        assert_eq!(batches[1][0].id, 0);
    }

    #[test]
    fn reaction_kind_converts_unicode_and_custom() {
        let unicode = EmojiReactionType::Unicode {
            name: "🔄".to_string(),
        };
        let custom = EmojiReactionType::Custom {
            animated: false,
            id: Id::new(42),
            name: Some("spin".to_string()),
        };

        assert_eq!(
            reaction_kind(&unicode),
            ReactionKind::Unicode {
                name: "🔄".to_string()
            }
        );
        assert_eq!(
            reaction_kind(&custom),
            ReactionKind::Custom {
                id: "42".into(),
                name: Some("spin".to_string())
            }
        );
    }

    fn outgoing_attachments(count: usize) -> Vec<OutgoingAttachment> {
        (0..count)
            .map(|index| OutgoingAttachment {
                filename: format!("generated-{index}.png"),
                content_type: "image/png".to_string(),
                bytes: vec![u8::try_from(index).unwrap_or(u8::MAX)],
            })
            .collect()
    }

    #[test]
    fn message_reference_maps_to_platform_message_ref() {
        let platform = PlatformName::new("discord");
        let reference = MessageReference {
            channel_id: Some(Id::new(111)),
            guild_id: Some(Id::new(222)),
            kind: MessageReferenceType::Default,
            message_id: Some(Id::new(333)),
            fail_if_not_exists: None,
        };

        let message = message_ref_from_reference(
            &platform,
            Some(ExternalId::new("fallback-guild")),
            &reference,
        )
        .expect("complete Discord reference should map to MessageRef");

        assert_eq!(message.platform, platform);
        assert_eq!(
            message.guild_id.as_ref().map(ExternalId::as_str),
            Some("222")
        );
        assert_eq!(message.channel_id.as_str(), "111");
        assert_eq!(message.message_id.as_str(), "333");
    }

    #[test]
    fn discord_message_context_uses_discord_vocabulary_and_cached_names() {
        let platform = PlatformName::new("discord");
        let guild = ExternalId::new("222");
        let channel = ExternalId::new("111");
        let cache = discord_context_test_cache();
        let message = PlatformMessage {
            id: MessageRef {
                platform: platform.clone(),
                guild_id: Some(guild.clone()),
                channel_id: channel.clone(),
                message_id: ExternalId::new("333"),
            },
            author: UserProfile {
                id: UserRef {
                    platform: platform.clone(),
                    guild_id: Some(guild.clone()),
                    user_id: ExternalId::new("444"),
                },
                username: "robert".to_string(),
                name: Some("Robert".to_string()),
                display_name: Some("Robert Guild".to_string()),
                avatar_url: None,
                is_bot: false,
            },
            content: "hello <@777>".to_string(),
            mentions: vec![UserRef {
                platform: platform.clone(),
                guild_id: Some(guild.clone()),
                user_id: ExternalId::new("777"),
            }],
            mention_profiles: vec![UserProfile {
                id: UserRef {
                    platform,
                    guild_id: Some(guild),
                    user_id: ExternalId::new("777"),
                },
                username: "trollzorftw808".to_string(),
                name: Some("Trollzor".to_string()),
                display_name: Some("Troll".to_string()),
                avatar_url: None,
                is_bot: false,
            }],
            reference: PlatformMessageReference::None,
            attachments: vec![AttachmentRef {
                id: Some(ExternalId::new("555")),
                url: "https://cdn.example/img.png".to_string(),
                filename: "img.png".to_string(),
                content_type: Some("image/png".to_string()),
                size_bytes: Some(123),
                duration_seconds: None,
                is_voice_message: false,
                waveform: None,
            }],
            created_at: OffsetDateTime::UNIX_EPOCH,
        };

        let value =
            discord_message_context_json(&message, PlatformMessageRelationship::Referenced, &cache);

        assert_eq!(value["type"].as_str(), Some("discord_message"));
        assert_eq!(value["relationship"].as_str(), Some("referenced"));
        assert_eq!(value["guild"]["name"].as_str(), Some("Test Guild"));
        assert_eq!(value["channel"]["name"].as_str(), Some("general"));
        assert_eq!(
            value["author"]["guild_display_name"].as_str(),
            Some("Robert Guild")
        );
        assert_eq!(
            value["attachments"][0]["filename"].as_str(),
            Some("img.png")
        );
        assert!(value["attachments"][0].get("url").is_none());
        assert_eq!(value["mentioned_users"][0]["id"].as_str(), Some("777"));
        assert_eq!(
            value["mentioned_users"][0]["username"].as_str(),
            Some("trollzorftw808")
        );
        assert_eq!(
            value["mentioned_users"][0]["guild_display_name"].as_str(),
            Some("Troll")
        );
    }

    fn discord_context_test_cache() -> DefaultInMemoryCache {
        let cache = DefaultInMemoryCache::builder()
            .resource_types(
                ResourceType::GUILD
                    | ResourceType::CHANNEL
                    | ResourceType::USER
                    | ResourceType::MEMBER,
            )
            .build();
        let guild: GuildCreate = serde_json::from_value(serde_json::json!({
            "id": "222",
            "afk_timeout": 900,
            "default_message_notifications": 1,
            "explicit_content_filter": 0,
            "features": [],
            "mfa_level": 0,
            "name": "Test Guild",
            "nsfw_level": 0,
            "owner_id": "444",
            "preferred_locale": "en-US",
            "premium_progress_bar_enabled": false,
            "roles": [],
            "system_channel_flags": 0,
            "verification_level": 0
        }))
        .expect("minimal guild create payload should deserialize");
        let channel: ChannelUpdate = serde_json::from_value(serde_json::json!({
            "id": "111",
            "guild_id": "222",
            "type": 0,
            "name": "general"
        }))
        .expect("minimal channel update payload should deserialize");

        cache.update(&guild);
        cache.update(&channel);
        cache
    }

    #[test]
    fn parse_snowflake_rejects_zero_and_non_numbers() {
        assert!(matches!(
            parse_channel_id(&"0".into()),
            Err(DiscordError::InvalidId {
                kind: "channel",
                ..
            })
        ));
        assert!(matches!(
            parse_channel_id(&"not-a-number".into()),
            Err(DiscordError::InvalidId {
                kind: "channel",
                ..
            })
        ));
    }

    #[test]
    fn gateway_reconnect_backoff_doubles_and_caps() {
        let mut delay = Duration::from_secs(5);
        delay = next_reconnect_delay(delay);
        assert_eq!(delay, Duration::from_secs(10));

        delay = next_reconnect_delay(delay);
        assert_eq!(delay, Duration::from_secs(20));

        delay = next_reconnect_delay(Duration::from_secs(60));
        assert_eq!(delay, GATEWAY_RECONNECT_MAX_DELAY);
    }
}
