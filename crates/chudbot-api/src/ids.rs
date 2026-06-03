//! Shared identifiers.

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

macro_rules! string_id {
    ($name:ident) => {
        #[doc = concat!("String-backed identifier: `", stringify!($name), "`.")]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            /// Construct from any string-like value.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            /// Borrow the underlying string.
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

string_id!(PlatformName);
string_id!(ExternalId);
string_id!(ProviderName);
string_id!(ModelId);
string_id!(ToolName);
string_id!(ToolUseId);
string_id!(VideoJobId);

/// Stable internal conversation id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConversationId(pub Uuid);

impl ConversationId {
    /// Generate a fresh conversation id.
    pub fn new() -> Self {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnId(pub Uuid);

impl TurnId {
    /// Generate a fresh turn id.
    pub fn new() -> Self {
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

/// User id scoped to a platform and optionally a guild/workspace/server.
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
