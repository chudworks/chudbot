//! Messaging platform contracts.
//!
//! This module defines the neutral boundary between native chat services and
//! the bot runtime. Platform adapters normalize their gateway/webhook payloads
//! into these types, while the runtime drives replies, reactions, commands,
//! history fetches, and model-visible context through [`MessagePlatform`].
//!
//! High-level flow:
//!
//! 1. A platform implementation emits [`PlatformEvent`] values from its native
//!    event stream.
//! 2. The bot runtime handles those events, persists conversation state, and
//!    asks the same platform abstraction for side effects such as
//!    [`SendMessage`], reactions, command responses, and typing indicators.
//! 3. When a message becomes model context, the runtime passes the normalized
//!    [`PlatformMessage`] back to the adapter through
//!    [`MessagePlatform::message_context`] so platform-specific vocabulary and
//!    mention syntax stay outside the provider-neutral crates.
//!
//! Identifiers such as [`UserRef`], [`ChannelRef`], and [`MessageRef`] carry
//! opaque platform ids plus optional workspace scope. Concrete transport types
//! from Discord, Telegram, Slack, or any other platform must not leak into this
//! crate.

use std::future::Future;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::ids::{ChannelRef, ExternalId, MessageRef, PlatformName, UserRef};

/// User profile as seen by a messaging platform.
///
/// Profiles are captured at platform boundaries and stored in traces, memory
/// metadata, and model context. `username`, `name`, and `display_name` are kept
/// separate because platforms often distinguish stable handles, global profile
/// names, and per-workspace nicknames.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// Stable user reference.
    pub id: UserRef,
    /// Platform username.
    pub username: String,
    /// Platform-wide display/profile name.
    #[serde(default)]
    pub name: Option<String>,
    /// Display name at the event boundary, often scoped to a guild/workspace.
    pub display_name: Option<String>,
    /// Optional avatar URL.
    pub avatar_url: Option<String>,
    /// Whether this user is a bot.
    pub is_bot: bool,
}

/// Attachment on an incoming platform message.
///
/// This is the adapter-supplied description of remote media before the bot has
/// downloaded or stored anything in its own media layer.
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
    /// Audio duration in seconds when the platform supplies it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f64>,
    /// Whether the platform marks this attachment as a voice message.
    #[serde(default)]
    pub is_voice_message: bool,
    /// Platform-supplied waveform preview when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub waveform: Option<String>,
}

/// Incoming platform message.
///
/// Adapters produce this shape from native message payloads. The bot treats ids
/// as opaque, uses `content` as the original platform text, and relies on
/// [`MessagePlatform::message_context`] for any platform-specific rendering
/// needed by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformMessage {
    /// Message id.
    pub id: MessageRef,
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

/// Reference data for a platform message reply/quote.
///
/// Some platforms include the quoted message inline, while others only include
/// its id or require a separate fetch that the adapter may choose not to do.
/// Keeping both cases explicit lets privacy and transcript code distinguish
/// "known target" from "available context".
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
            // 1. No reply/quote relationship was present.
            Self::None => None,
            // 2. Id-only references preserve linkage without forcing a fetch.
            Self::Id(message) => Some(message),
            // 3. Hydrated references expose the nested payload's own id.
            Self::Hydrated(message) => Some(&message.id),
        }
    }

    /// Hydrated referenced message payload, if available.
    pub fn hydrated_message(&self) -> Option<&PlatformMessage> {
        match self {
            // Id-only references are intentionally not resolved here; adapters
            // decide whether to hydrate before constructing PlatformMessage.
            Self::None | Self::Id(_) => None,
            Self::Hydrated(message) => Some(message),
        }
    }
}

impl PlatformMessage {
    /// Message id quoted/replied to by this message, if known.
    ///
    /// This works for both id-only references and hydrated references.
    pub fn referenced_message_id(&self) -> Option<&MessageRef> {
        // Keep the convenience API on PlatformMessage while centralizing the
        // reference-kind handling in PlatformMessageReference.
        self.reference.message_id()
    }

    /// Hydrated message quoted/replied to by this message, if the platform
    /// supplied it.
    ///
    /// This accessor never performs platform I/O; id-only references remain
    /// id-only.
    pub fn referenced_message(&self) -> Option<&PlatformMessage> {
        // Hydration is a payload property, not a lazy fetch operation.
        self.reference.hydrated_message()
    }
}

/// Relationship between a platform message and the current model turn.
///
/// The runtime passes this into [`MessagePlatform::message_context`] so the
/// adapter can label the resulting JSON as the initiating message, an explicit
/// quote/reply target, or fetched background context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlatformMessageRelationship {
    /// The user message that directly started the current turn.
    Current,
    /// A platform message explicitly referenced/quoted by the current turn.
    Referenced,
    /// A platform message fetched as recent channel context.
    Fetched,
}

/// Reaction kind.
///
/// Reactions are platform-normalized enough for the bot to request add/remove
/// operations, but custom reaction ids remain platform-specific opaque values.
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
///
/// Carries the minimum normalized data needed to map platform reactions back to
/// stored messages and users.
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
///
/// This is the inbound half of the command contract. Adapters translate native
/// slash/chat command payloads into a tree of [`PlatformCommandInput`] values
/// and include the response target needed for
/// [`MessagePlatform::respond_to_command`].
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
///
/// Subcommands and option groups are represented by leaving `value` empty and
/// filling `options`.
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
///
/// Values are intentionally scalar and opaque. Platform adapters own conversion
/// from native command option types into these variants.
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
///
/// The runtime stores and echoes this target without understanding the native
/// interaction token semantics.
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
///
/// Command responses are separate from normal [`SendMessage`] replies because
/// interaction-based platforms often require a platform token response rather
/// than a channel post.
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
///
/// This is the outbound half of the command contract. The bot defines commands
/// once in platform-neutral terms; adapters translate them into the native
/// registration API.
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
///
/// This covers the option kinds currently used by Chudbot command definitions,
/// not every possible kind every platform may expose.
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
///
/// Choices are represented as strings because the current command surface uses
/// string-backed enumerations even when a platform has richer native choice
/// metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCommandOptionChoice {
    /// User-visible choice label.
    pub name: String,
    /// Stored choice value.
    pub value: String,
}

/// Platform command option definition.
///
/// Nested `options` are used for subcommands and subcommand groups. Numeric
/// bounds are represented only for integer options because that is the current
/// bot command need.
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
///
/// Signals that an adapter can accept side-effect calls and identifies the bot
/// user profile visible on that platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformReady {
    /// Bot profile.
    pub bot: UserProfile,
}

/// Incoming platform event.
///
/// This is the stream consumed by the bot runtime. Events are already
/// normalized; adapter-specific reconnect, acknowledgement, and webhook details
/// remain inside the concrete platform crate.
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
        message: Box<PlatformMessage>,
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
///
/// Bytes are already loaded by the bot before they cross into the platform
/// adapter. The adapter owns any upload chunking or platform-specific file
/// metadata conversion.
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
///
/// Thread support is platform-dependent. Adapters that support it decide how
/// to map this neutral title onto native thread/forum primitives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadRequest {
    /// Thread title.
    pub title: String,
}

/// Send message request.
///
/// This is the runtime's single request for a visible platform reply. Adapters
/// may split it into multiple physical platform messages and report that through
/// [`PostedMessage::extra_messages`].
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
///
/// The first message id is treated as the canonical platform reply for storage
/// links and future reply targets.
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
///
/// Used by history tools and privacy-aware context gathering. The adapter
/// returns messages in the order expected by the caller for that fetch
/// operation; it does not persist them.
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
///
/// Implement this trait for one concrete platform service. A higher-level
/// registry fans events in from all configured platforms and routes side
/// effects by the platform name embedded in [`ChannelRef`] and [`MessageRef`].
///
/// The trait uses native async return-position `impl Future` so implementors
/// can stay concrete and avoid boxing platform clients behind broad service
/// objects.
pub trait MessagePlatform: Send + Sync {
    /// Platform error type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Fetch the bot user profile as seen by this platform.
    fn bot_user(&self) -> impl Future<Output = Result<UserProfile, Self::Error>> + Send;

    /// Register platform commands.
    ///
    /// `guild` is a platform workspace/server scope when a platform supports
    /// scoped registration. `None` means the adapter should use its global or
    /// deployment-wide registration path.
    fn register_commands(
        &self,
        commands: Vec<PlatformCommandDefinition>,
        guild: Option<ExternalId>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Read the next normalized event from this platform.
    ///
    /// Implementations own reconnects, acknowledgements, and any native
    /// long-polling or gateway mechanics behind this call.
    fn next_event(&self) -> impl Future<Output = Result<PlatformEvent, Self::Error>> + Send;

    /// Respond to a platform command interaction.
    fn respond_to_command(
        &self,
        response: PlatformCommandResponse,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Send a visible platform message.
    fn send_message(
        &self,
        request: SendMessage,
    ) -> impl Future<Output = Result<PostedMessage, Self::Error>> + Send;

    /// Delete a platform message previously identified by [`MessageRef`].
    fn delete_message(
        &self,
        message: MessageRef,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Add a reaction to a platform message.
    fn add_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Remove the bot's own reaction from a platform message.
    fn remove_own_reaction(
        &self,
        message: MessageRef,
        reaction: ReactionKind,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Trigger a typing indicator or equivalent presence hint.
    fn typing(&self, channel: ChannelRef) -> impl Future<Output = Result<(), Self::Error>> + Send;

    /// Fetch recent channel messages without mutating bot storage.
    fn fetch_messages(
        &self,
        request: FetchMessages,
    ) -> impl Future<Output = Result<Vec<PlatformMessage>, Self::Error>> + Send;

    /// Render a platform message into the JSON value shown to the model.
    ///
    /// Platform implementations own the vocabulary here, such as whether a
    /// workspace is called a server, guild, team, room, or channel. This method
    /// is also where native mention syntax, cached display names, and
    /// attachment hints can be projected into model-facing context.
    fn message_context(
        &self,
        message: &PlatformMessage,
        relationship: PlatformMessageRelationship,
    ) -> impl Future<Output = Result<serde_json::Value, Self::Error>> + Send;

    /// Resolve a platform channel's parent, if any.
    ///
    /// Thread-like channels should return their parent conversation scope so
    /// runtime settings can inherit from the right channel. Non-thread channels
    /// can return themselves.
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
