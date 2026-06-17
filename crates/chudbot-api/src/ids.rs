//! Shared identifier types for provider-neutral API contracts.
//!
//! The API crate has to name platforms, providers, models, tools, stored
//! conversations, turns, and platform objects without depending on Discord,
//! SQLx, Reqwest, Axum, or any concrete provider crate. This module keeps that
//! boundary explicit by using:
//!
//! - string-backed wrappers for names and opaque ids owned by another system,
//! - UUID-backed ids for Chudbot-owned persistent records, and
//! - scoped refs that combine a platform name with the external ids needed to
//!   address a user, channel, or message.
//!
//! The string wrappers intentionally do not validate syntax. Validation belongs
//! to the config parser, platform adapter, provider adapter, or tool registry
//! that owns the namespace.

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Keep the newtype pattern uniform so callers get strong typing without
// changing the serialized wire/storage shape from the underlying string.
macro_rules! string_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        ///
        /// This is a transparent wrapper around the raw string. Constructing it
        /// with [`Self::new`] preserves the input exactly; any trimming,
        /// normalization, or semantic validation must happen at the boundary
        /// that understands this identifier's namespace.
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(
            /// Raw identifier text as it appears in config, platform payloads,
            /// provider payloads, or tool traces.
            pub String,
        );

        impl $name {
            /// Wrap a string-like value without normalizing or validating it.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            /// Borrow the raw identifier text.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_string())
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
    };
}

// Named runtime/configuration namespaces.
string_id! {
    /// Name of a configured messaging platform.
    ///
    /// The bot and storage layers route neutral platform operations by this
    /// name before a concrete adapter, such as Discord, parses any external
    /// ids. A single deployment may mount multiple platform entries, so this
    /// is a configured service key rather than a hard-coded vendor enum.
    PlatformName
}

string_id! {
    /// Opaque identifier assigned by a platform or provider.
    ///
    /// `ExternalId` values are not globally meaningful by themselves. Combine
    /// them with a [`PlatformName`] and the relevant scope fields before using
    /// them as durable keys.
    ExternalId
}

string_id! {
    /// Name of a configured model/media provider.
    ///
    /// This is the registry key for a runtime backend, not necessarily the
    /// public vendor name. For example, two OpenAI-compatible hosts can be
    /// registered under different provider names while sharing one provider
    /// crate.
    ProviderName
}

string_id! {
    /// Provider-specific model identifier.
    ///
    /// The API layer stores and forwards model ids exactly as selected by an
    /// agent or returned by a backend. Provider-specific aliases, availability
    /// checks, and pricing interpretation live in provider/config code.
    ModelId
}

// Tool and async media identifiers that cross model/provider boundaries.
string_id! {
    /// Registered client-tool name exposed to model backends.
    ///
    /// Tool names identify the tool definition selected from Chudbot's tool
    /// registry. They are separate from [`ToolUseId`], which identifies one
    /// concrete call made by a model response.
    ToolName
}

string_id! {
    /// Identifier for one tool call in a provider response.
    ///
    /// Providers generate these ids so later tool outputs can be matched back
    /// to the requested calls. Treat them as response/turn-local unless a
    /// provider contract explicitly gives them a wider scope.
    ToolUseId
}

string_id! {
    /// Identifier for a submitted video-generation job.
    ///
    /// This is whatever job token the configured video backend returns.
    /// Callers pair it with the generator/provider that accepted the request
    /// when checking status or fetching results.
    VideoJobId
}

// Chudbot-owned persistent ids.
/// Stable internal conversation id.
///
/// This id is generated by Chudbot and is independent of any platform message,
/// channel, or provider identifier. Storage uses it as the primary conversation
/// handle, and the web viewer exposes it in trace URLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConversationId(pub Uuid);

impl ConversationId {
    /// Generate a fresh conversation id.
    pub fn new() -> Self {
        // Conversation ids must be unguessable enough for unauthenticated trace
        // URLs and collision-resistant across every platform adapter.
        Self(Uuid::new_v4())
    }
}

impl Default for ConversationId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ConversationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Stable internal turn id.
///
/// A turn id names one persisted user-message attempt within a conversation.
/// It is not derived from the platform message id, because retries, storage
/// links, trace records, and cancellation bookkeeping all need a Chudbot-owned
/// handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnId(pub Uuid);

impl TurnId {
    /// Generate a fresh turn id.
    pub fn new() -> Self {
        // Keep turn identity stable even when multiple platform messages,
        // status updates, tool traces, or retries point at the same turn.
        Self(Uuid::new_v4())
    }
}

impl Default for TurnId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TurnId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// Platform-scoped refs used at adapter, storage, and viewer boundaries.
/// User id scoped to a platform and optionally a guild/workspace/server.
///
/// The compound `(platform, guild_id, user_id)` is the neutral identity the bot
/// can store and compare without importing platform SDK types. `guild_id` is
/// optional so direct-message or workspace-less platforms can still use the
/// same contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserRef {
    /// Messaging platform, e.g. `discord`.
    pub platform: PlatformName,
    /// Platform guild/workspace/server id, if the platform has one.
    pub guild_id: Option<ExternalId>,
    /// Platform user id.
    pub user_id: ExternalId,
}

/// Channel id scoped to a platform and optionally a guild/workspace/server.
///
/// Platform adapters may interpret `channel_id` as a text channel, thread,
/// direct-message channel, or equivalent surface. Parent-channel lookup and
/// other vendor-specific structure stay in the platform adapter.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelRef {
    /// Messaging platform, e.g. `discord`.
    pub platform: PlatformName,
    /// Platform guild/workspace/server id, if the platform has one.
    pub guild_id: Option<ExternalId>,
    /// Platform channel id.
    pub channel_id: ExternalId,
}

/// Message id scoped to a concrete platform channel.
///
/// Even when a platform's message ids are globally unique, keeping the channel
/// in the ref gives neutral code enough information to fetch, delete, reply to,
/// or link the message through the platform registry without guessing adapter
/// requirements.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageRef {
    /// Messaging platform, e.g. `discord`.
    pub platform: PlatformName,
    /// Platform guild/workspace/server id, if the platform has one.
    pub guild_id: Option<ExternalId>,
    /// Platform channel id.
    pub channel_id: ExternalId,
    /// Platform message id.
    pub message_id: ExternalId,
}
