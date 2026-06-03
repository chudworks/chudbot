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
    PlatformName, PlatformReaction, PlatformReady, PostedMessage, ReactionKind, SendMessage,
    UserProfile, UserRef,
};
use thiserror::Error;
use time::OffsetDateTime;
use tokio::sync::Mutex;
use twilight_gateway::{EventTypeFlags, Intents, Shard, ShardId, StreamExt};
use twilight_http::Client as HttpClient;
use twilight_http::request::channel::reaction::RequestReactionType;
use twilight_model::application::command::{Command, CommandOption, CommandType};
use twilight_model::application::interaction::application_command::{
    CommandDataOption, CommandOptionValue as InteractionCommandOptionValue,
};
use twilight_model::application::interaction::{Interaction, InteractionData};
use twilight_model::channel::message::{EmojiReactionType, Mention, MessageFlags};
use twilight_model::channel::{ChannelType, Message};
use twilight_model::gateway::GatewayReaction;
use twilight_model::gateway::event::Event;
use twilight_model::gateway::payload::incoming::GuildCreate;
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

/// Discord platform runtime.
#[derive(Clone)]
pub struct DiscordPlatform {
    inner: Arc<DiscordPlatformInner>,
}

struct DiscordPlatformInner {
    platform: PlatformName,
    http: Arc<HttpClient>,
    shard: Mutex<Shard>,
    bot_user: UserProfile,
    bot_user_id: Id<UserMarker>,
    application_id: Id<ApplicationMarker>,
    ready_emitted: AtomicBool,
    event_flags: EventTypeFlags,
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

        Ok(Self {
            inner: Arc::new(DiscordPlatformInner {
                platform,
                http,
                shard: Mutex::new(Shard::new(ShardId::ONE, token, intents)),
                bot_user,
                bot_user_id: current.id,
                application_id: application.id,
                ready_emitted: AtomicBool::new(false),
                event_flags,
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

    async fn next_gateway_event(&self) -> Result<PlatformEvent, DiscordError> {
        loop {
            let item = {
                let mut shard = self.inner.shard.lock().await;
                shard.next_event(self.inner.event_flags).await
            };
            let Some(item) = item else {
                tracing::info!("discord gateway stream ended");
                return Ok(PlatformEvent::Shutdown);
            };
            let event = match item {
                Ok(event) => event,
                Err(error) => {
                    tracing::warn!(error = %error, "discord gateway receive error");
                    continue;
                }
            };
            match event {
                Event::MessageCreate(message) => {
                    let message = platform_message(&self.inner.platform, message.0);
                    tracing::trace!(
                        message = %message.id.message_id,
                        channel = %message.id.channel_id,
                        "discord message event converted"
                    );
                    return Ok(PlatformEvent::MessageCreated { message });
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
                Event::GuildCreate(guild) => log_guild_create(&guild),
                _ => {}
            }
        }
    }
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
        let attachments = http_attachments(&request.attachments);
        let mut posted = Vec::with_capacity(chunks.len());
        for (index, chunk) in chunks.iter().enumerate() {
            let is_last = index + 1 == chunks.len();
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
            if is_last && !attachments.is_empty() {
                builder = builder.attachments(&attachments);
            }
            let message = builder.await?.model().await?;
            tracing::trace!(
                message = %message.id,
                channel = %message.channel_id,
                chunk = index,
                chunks = chunks.len(),
                "posted discord message chunk"
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
            .map(|message| platform_message(&self.inner.platform, message))
            .collect())
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

fn platform_message(platform: &PlatformName, message: Message) -> PlatformMessage {
    let guild_id = message.guild_id.map(external_id);
    let referenced_message = message
        .referenced_message
        .map(|message| Box::new(platform_message(platform, *message)));
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
        referenced_message,
        attachments: message.attachments.iter().map(attachment_ref).collect(),
        created_at: timestamp_to_offset(message.timestamp),
    }
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
        display_name: user.global_name.clone(),
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
        display_name: member
            .and_then(|member| member.nick.clone())
            .or_else(|| user.global_name.clone()),
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

fn attachment_ref(attachment: &twilight_model::channel::Attachment) -> AttachmentRef {
    AttachmentRef {
        id: Some(external_id(attachment.id)),
        url: attachment.url.clone(),
        filename: attachment.filename.clone(),
        content_type: attachment.content_type.clone(),
        size_bytes: Some(attachment.size),
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

fn split_discord_content(content: &str) -> Vec<String> {
    if content.is_empty() {
        return vec![String::new()];
    }

    let mut chunks = Vec::new();
    let mut chunk = String::new();
    let mut chars = 0usize;
    for ch in content.chars() {
        if chars == DISCORD_MESSAGE_LIMIT {
            chunks.push(std::mem::take(&mut chunk));
            chars = 0;
        }
        chunk.push(ch);
        chars += 1;
    }
    if !chunk.is_empty() {
        chunks.push(chunk);
    }
    chunks
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
    use chudbot_api::ReactionKind;
    use twilight_model::channel::message::EmojiReactionType;
    use twilight_model::id::Id;

    use super::{
        DISCORD_MESSAGE_LIMIT, DiscordError, parse_channel_id, reaction_kind, split_discord_content,
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
}
