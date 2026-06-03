//! Messaging platform contracts.

use std::future::Future;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::ids::{ChannelRef, ExternalId, MessageRef, PlatformName, UserRef};

/// User profile as seen by a messaging platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// Stable user reference.
    pub id: UserRef,
    /// Platform username.
    pub username: String,
    /// Platform-wide display/profile name.
    #[serde(default)]
    pub name: Option<String>,
    /// Display name at the event boundary.
    ///
    /// On Discord this is the guild display name/nickname when available.
    pub display_name: Option<String>,
    /// Optional avatar URL.
    pub avatar_url: Option<String>,
    /// Whether this user is a bot.
    pub is_bot: bool,
}

/// Attachment on an incoming platform message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachmentRef {
    /// Platform attachment id when available.
    pub id: Option<ExternalId>,
    /// Download URL.
    pub url: String,
    /// Original filename.
    pub filename: String,
    /// MIME type hint.
    pub content_type: Option<String>,
    /// Attachment size in bytes.
    pub size_bytes: Option<u64>,
}

/// Incoming platform message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformMessage {
    /// Message id.
    pub id: MessageRef,
    /// Platform guild/workspace/server name when known.
    #[serde(default)]
    pub guild_name: Option<String>,
    /// Platform channel name when known.
    #[serde(default)]
    pub channel_name: Option<String>,
    /// Author.
    pub author: UserProfile,
    /// Raw content.
    pub content: String,
    /// Mentioned users.
    pub mentions: Vec<UserRef>,
    /// Mentioned user profiles when the platform supplies them.
    #[serde(default)]
    pub mention_profiles: Vec<UserProfile>,
    /// Message quoted/replied to by this message, if provided by the platform.
    #[serde(default)]
    pub reference: PlatformMessageReference,
    /// Attachments.
    pub attachments: Vec<AttachmentRef>,
    /// Creation timestamp.
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
}

impl PlatformMessage {
    /// Message id quoted/replied to by this message, if known.
    pub fn referenced_message_id(&self) -> Option<&MessageRef> {
        self.reference.message_id()
    }

    /// Hydrated message quoted/replied to by this message, if the platform
    /// supplied it.
    pub fn referenced_message(&self) -> Option<&PlatformMessage> {
        self.reference.hydrated_message()
    }
}

/// Reference data for a platform message reply/quote.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", content = "message", rename_all = "snake_case")]
pub enum PlatformMessageReference {
    /// No referenced message.
    #[default]
    None,
    /// The platform supplied only a referenced message id.
    Id(MessageRef),
    /// The platform supplied the full referenced message payload.
    Hydrated(Box<PlatformMessage>),
}

impl PlatformMessageReference {
    /// Referenced message id, whether the platform supplied only an id or a
    /// fully hydrated message.
    pub fn message_id(&self) -> Option<&MessageRef> {
        match self {
            Self::None => None,
            Self::Id(message) => Some(message),
            Self::Hydrated(message) => Some(&message.id),
        }
    }

    /// Hydrated referenced message payload, if available.
    pub fn hydrated_message(&self) -> Option<&PlatformMessage> {
        match self {
            Self::None | Self::Id(_) => None,
            Self::Hydrated(message) => Some(message),
        }
    }
}

/// Reaction kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReactionKind {
    /// Unicode emoji.
    Unicode {
        /// Emoji string.
        name: String,
    },
    /// Platform custom emoji/reaction.
    Custom {
        /// Custom reaction id.
        id: ExternalId,
        /// Optional display name.
        name: Option<String>,
    },
}

/// Platform reaction event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformReaction {
    /// Reacted message.
    pub message: MessageRef,
    /// User who reacted.
    pub user: UserRef,
    /// Reaction.
    pub reaction: ReactionKind,
}

/// Platform command invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCommand {
    /// Command name.
    pub name: String,
    /// Invoking user.
    pub user: UserRef,
    /// Channel where command was invoked.
    pub channel: ChannelRef,
    /// Platform-normalized command options.
    #[serde(default)]
    pub options: Vec<PlatformCommandInput>,
    /// Whether the invoking member has platform administrator privileges.
    #[serde(default)]
    pub is_admin: bool,
    /// Target needed to send a platform command response.
    pub response_target: PlatformCommandResponseTarget,
}

/// One supplied command option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCommandInput {
    /// Option or subcommand name.
    pub name: String,
    /// Scalar option value. Subcommands carry nested options instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<PlatformCommandValue>,
    /// Nested options for subcommands and groups.
    #[serde(default)]
    pub options: Vec<PlatformCommandInput>,
}

/// Platform-normalized command option value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum PlatformCommandValue {
    /// String option.
    String(String),
    /// Integer option.
    Integer(i64),
    /// Floating-point number option.
    Number(f64),
    /// Boolean option.
    Boolean(bool),
    /// Channel option.
    Channel(ChannelRef),
    /// User option.
    User(UserRef),
    /// Role option.
    Role(ExternalId),
    /// Mentionable option.
    Mentionable(ExternalId),
    /// Attachment option.
    Attachment(ExternalId),
}

/// Target needed to respond to a platform command interaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCommandResponseTarget {
    /// Messaging platform.
    pub platform: PlatformName,
    /// Platform interaction id.
    pub interaction_id: ExternalId,
    /// Platform interaction response token.
    pub token: String,
}

/// Bot response to a platform command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCommandResponse {
    /// Interaction target.
    pub target: PlatformCommandResponseTarget,
    /// Message content.
    pub content: String,
    /// Whether only the invoking user should see the response.
    pub ephemeral: bool,
}

/// Platform command definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCommandDefinition {
    /// Command name.
    pub name: String,
    /// Command description.
    pub description: String,
    /// Whether the platform should restrict the command to administrators by
    /// default.
    pub admin_only: bool,
    /// Top-level options.
    pub options: Vec<PlatformCommandOption>,
}

/// Platform command option kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlatformCommandOptionKind {
    /// Subcommand.
    SubCommand,
    /// String value.
    String,
    /// Integer value.
    Integer,
    /// Channel value.
    Channel,
}

/// Platform command option choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCommandOptionChoice {
    /// User-visible choice label.
    pub name: String,
    /// Stored choice value.
    pub value: String,
}

/// Platform command option definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCommandOption {
    /// Option name.
    pub name: String,
    /// Option description.
    pub description: String,
    /// Option kind.
    pub kind: PlatformCommandOptionKind,
    /// Whether the option is required.
    pub required: bool,
    /// String/integer choices.
    pub choices: Vec<PlatformCommandOptionChoice>,
    /// Nested options for subcommands.
    pub options: Vec<PlatformCommandOption>,
    /// Integer minimum.
    pub min_integer: Option<i64>,
    /// Integer maximum.
    pub max_integer: Option<i64>,
}

/// Platform ready event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformReady {
    /// Bot profile.
    pub bot: UserProfile,
}

/// Incoming platform event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PlatformEvent {
    /// Platform is connected and ready.
    Ready {
        /// Ready payload.
        ready: PlatformReady,
    },
    /// New message.
    MessageCreated {
        /// Message payload.
        message: PlatformMessage,
    },
    /// Reaction added.
    ReactionAdded {
        /// Reaction payload.
        reaction: PlatformReaction,
    },
    /// Reaction removed.
    ReactionRemoved {
        /// Reaction payload.
        reaction: PlatformReaction,
    },
    /// Slash/chat command.
    Command {
        /// Command payload.
        command: PlatformCommand,
    },
    /// Platform stream shut down.
    Shutdown,
}

/// Outgoing attachment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutgoingAttachment {
    /// Filename to upload.
    pub filename: String,
    /// MIME type.
    pub content_type: String,
    /// Bytes to upload.
    pub bytes: Vec<u8>,
}

/// Request to open a thread while posting a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadRequest {
    /// Thread title.
    pub title: String,
}

/// Send message request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SendMessage {
    /// Target channel.
    pub channel: ChannelRef,
    /// Optional message to reply to.
    pub reply_to: Option<MessageRef>,
    /// Message content.
    pub content: String,
    /// Attachments to upload.
    pub attachments: Vec<OutgoingAttachment>,
    /// Suppress rich embeds when supported.
    pub suppress_embeds: bool,
    /// Optional platform thread request.
    pub open_thread: Option<ThreadRequest>,
}

/// Message returned after posting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostedMessage {
    /// Posted message id.
    pub id: MessageRef,
    /// Channel the message actually landed in.
    pub channel: ChannelRef,
    /// Any extra platform messages posted to deliver the same logical reply.
    ///
    /// Some platforms impose a hard message-length cap. Implementations may
    /// split one [`SendMessage`] request into several physical messages and
    /// report the additional ids here. `id` is always the first message.
    #[serde(default)]
    pub extra_messages: Vec<MessageRef>,
}

/// Fetch recent messages from a platform channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchMessages {
    /// Channel to fetch.
    pub channel: ChannelRef,
    /// Max messages to return.
    pub limit: u16,
    /// Fetch messages older than this id.
    pub before: Option<MessageRef>,
}

/// Messaging platform implementation.
pub trait MessagePlatform: Send + Sync {
    /// Platform error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Fetch the bot user profile.
    fn bot_user(&self) -> impl Future<Output = Result<UserProfile, Self::Error>> + Send;

    /// Register platform commands.
    fn register_commands(
        &self,
        commands: Vec<PlatformCommandDefinition>,
        guild: Option<ExternalId>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Read the next event.
    fn next_event(&self) -> impl Future<Output = Result<PlatformEvent, Self::Error>> + Send;

    /// Respond to a platform command.
    fn respond_to_command(
        &self,
        response: PlatformCommandResponse,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Send a message.
    fn send_message(
        &self,
        request: SendMessage,
    ) -> impl Future<Output = Result<PostedMessage, Self::Error>> + Send;

    /// Delete a message.
    fn delete_message(
        &self,
        message: MessageRef,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Add a reaction.
    fn add_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Remove the bot's own reaction.
    fn remove_own_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Trigger typing indicator.
    fn typing(&self, channel: ChannelRef) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Fetch recent channel messages.
    fn fetch_messages(
        &self,
        request: FetchMessages,
    ) -> impl Future<Output = Result<Vec<PlatformMessage>, Self::Error>> + Send;

    /// Resolve a platform channel's parent, if any. Non-thread channels can
    /// return themselves.
    fn parent_channel(
        &self,
        channel: ChannelRef,
    ) -> impl Future<Output = Result<ChannelRef, Self::Error>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_reference_exposes_message_id_without_hydrated_message() {
        let id = MessageRef {
            platform: PlatformName::new("discord"),
            guild_id: Some(ExternalId::new("guild-1")),
            channel_id: ExternalId::new("channel-1"),
            message_id: ExternalId::new("message-1"),
        };
        let reference = PlatformMessageReference::Id(id.clone());

        assert_eq!(reference.message_id(), Some(&id));
        assert!(reference.hydrated_message().is_none());
    }
}
