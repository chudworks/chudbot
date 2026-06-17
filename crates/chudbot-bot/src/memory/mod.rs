//! User-memory runtime facade.
//!
//! This module is the public entry point for the bot's user-memory subsystem. It
//! exposes the memory tool names that conversation agents can call, the
//! `MemoryConfig` and `MemoryRuntime` types used by the process launcher, and
//! the platform-neutral key helpers shared by the tool and background paths.
//!
//! Memory data moves through three related surfaces:
//!
//! - [`LOOKUP_USER_MEMORY_TOOL`] reads the durable compact profile, raw memory
//!   events that have not yet been compacted, and a small recent diary slice for
//!   the target user.
//! - `MEMORY_DIARY_AGENT` jobs summarize bounded transcript windows into diary
//!   entries, preserving context that is useful but too noisy to store directly
//!   as profile facts.
//! - `MEMORY_COMPACT_AGENT` jobs fold explicit remember/forget events and
//!   diary entries into the compact Markdown profile returned by lookup.
//!
//! The scheduler and prompt construction live in submodules; this file keeps the
//! names and key conversions that other bot modules depend on in one place.

use chudbot_api::{ExternalId, UserMemoryKey, UserProfile, UserRef};
use tokio_util::sync::CancellationToken;

use crate::{BotRuntime, BotRuntimeTypes, spawn_background_task};

mod compact;
mod config;
mod diary;
mod runtime;

/// Reserved agent name used by background compaction jobs.
pub use compact::MEMORY_COMPACT_AGENT;
/// Runtime configuration for the user-memory scheduler and worker limits.
pub use config::MemoryConfig;
/// Error returned while parsing or validating memory configuration.
pub use config::MemoryConfigError;
/// Parse the human-readable duration strings accepted by memory config fields.
pub use config::parse_duration_seconds;
/// Reserved agent name used by background diary jobs.
pub use diary::MEMORY_DIARY_AGENT;
/// Error returned by the background memory runtime.
pub use runtime::MemoryError;
/// Background scheduler that creates and runs diary and compaction jobs.
pub use runtime::MemoryRuntime;

pub(crate) use config::resolve_memory_agent;

/// Client tool name for current or target user memory lookup.
pub const LOOKUP_USER_MEMORY_TOOL: &str = "lookup_user_memory";
/// Client tool name for appending a raw remember event.
pub const REMEMBER_USER_MEMORY_TOOL: &str = "remember_user_memory";
/// Client tool name for appending a raw forget/tombstone event.
pub const FORGET_USER_MEMORY_TOOL: &str = "forget_user_memory";

const EMPTY_MEMORY: &str = "(no stored memory)";

/// Prompt guidance inserted into top-level memory-enabled agents.
///
/// The guidance tells agents to lookup each visible user before answering, write
/// explicit remember/forget events through the tools, and treat lookup output as
/// a combination of compact profile, pending raw events, and recent diaries.
pub const PROMPT_GUIDANCE: &str = "CRITICAL: Memory System\n\
- CRITICAL: If a user is the `author` of a message, you MUST load memory about that user. Do not respond to a user if you do not load their memory document first. Use the `lookup_user_memory` any time you see a user for the first time.\n\
- CRITICAL: If a user's memory has not been loaded, then any **mention** of a user should trigger a `lookup_user_memory` call, even if they are not the author.\n\
- The `lookup_user_memory` tool gives you a memory document about a user, recent events, and recent diary entries. These recent events can be `remember` or `forget`.\n\
- Use the `remember_user_memory` tool to store facts about a user. If there's something you think would be useful in the future, you should use this tool to remember it.\n\
- There is a `forget_user_memory` which works like `remember_user_memory`, but instead stores a fact to forget about a user.\n\
- If a user asks you explicitly to remember or forget something about themselves, then you should absolutely use the tools to store the user's preference and respect their humanity!\n\
- If a user tells you a fact about another user, you are allowed to remember / forget it. Take memories from 3rd parties with a \"grain of salt\".\n\
- If the current message conflicts with stored memory, trust the current message and remember the correction when appropriate.\n\
- Avoid repeating or storing any memory which reveals sensitive personal information (credit card, physical address, legal name, SSN, etc)\n\n\n\
IT IS CRITICAL TO USE THE MEMORY SYSTEM PROACTIVELY! The tool calls are cheap, use the tools!\n\
VERY IMPORTANT: If a user is the `author` of a message, you MUST load memory about that user. Do not respond to a user if you do not load their memory document first. Use the `lookup_user_memory` any time you see a user for the first time.\n";

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Start the background memory scheduler when memory is enabled.
    ///
    /// The scheduler owns diary-window summarization and profile compaction; the
    /// foreground tools only read memory or append raw events.
    pub(crate) fn spawn_memory_runtime(&self, shutdown: CancellationToken) {
        if !self.memory_config.enabled {
            return;
        }
        // Resolve reserved agents once so each job uses the same configured
        // model, prompt, and limits for the lifetime of this runtime.
        let memory_agents = self
            .memory_config
            .resolve_agent_set(&self.config.agents, self.config.limits);
        let runtime = MemoryRuntime::new(
            self.storage.clone(),
            self.llms.clone(),
            self.media_store.clone(),
            self.memory_config.clone(),
            memory_agents,
        );
        spawn_background_task(&self.background, "memory runtime", async move {
            if let Err(error) = runtime.run_until_shutdown(shutdown).await {
                tracing::warn!(error = %error, "memory runtime stopped with error");
            }
        });
    }
}

/// Build the neutral memory key for a platform user.
///
/// Guild-scoped users are stored under `guild:<id>`; direct messages or other
/// non-guild contexts use `global`. The resulting key is consumed by storage,
/// lookup tools, diary jobs, and compaction jobs.
pub fn key_from_user_ref(user: &UserRef) -> UserMemoryKey {
    UserMemoryKey {
        platform: user.platform.clone(),
        scope_key: scope_key(user.guild_id.as_ref().map(chudbot_api::ExternalId::as_str)),
        user_key: user.user_id.as_str().to_string(),
    }
}

/// Convert an optional guild id into the storage scope segment.
fn scope_key(guild_id: Option<&str>) -> String {
    guild_id
        .map(|guild| format!("guild:{guild}"))
        .unwrap_or_else(|| "global".to_string())
}

/// Return a human-readable scope id for logs, stripping the internal prefix.
fn memory_scope_id(scope_key: &str) -> &str {
    scope_key.strip_prefix("guild:").unwrap_or(scope_key)
}

/// Recover a guild id from the storage scope when the key is guild-scoped.
fn memory_guild_id(scope_key: &str) -> Option<&str> {
    scope_key.strip_prefix("guild:")
}

/// Rebuild a platform user reference from a stored memory key.
fn memory_user_ref(key: &UserMemoryKey) -> UserRef {
    UserRef {
        platform: key.platform.clone(),
        guild_id: memory_guild_id(&key.scope_key).map(ExternalId::new),
        user_id: ExternalId::new(key.user_key.clone()),
    }
}

/// Pick the best non-id label for memory-job tracing.
fn memory_profile_display_name(profile: &UserProfile, user_key: &str) -> Option<String> {
    let name = profile
        .display_name
        .as_deref()
        .or(profile.name.as_deref())
        .unwrap_or(profile.username.as_str())
        .trim();
    (!name.is_empty() && name != user_key).then(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use chudbot_api::{ExternalId, PlatformName, UserMemoryKey, UserProfile, UserRef};

    use super::*;

    #[test]
    fn builds_guild_scoped_memory_key() {
        let key = key_from_user_ref(&UserRef {
            platform: PlatformName::new("discord"),
            guild_id: Some(ExternalId::new("guild-1")),
            user_id: ExternalId::new("user-1"),
        });

        assert_eq!(key.platform.as_str(), "discord");
        assert_eq!(key.scope_key, "guild:guild-1");
        assert_eq!(key.user_key, "user-1");
        assert_eq!(key.memory_key(), "discord:guild:guild-1:user-1");
    }

    #[test]
    fn memory_user_ref_extracts_guild_scope() {
        let user = memory_user_ref(&UserMemoryKey {
            platform: PlatformName::new("discord"),
            scope_key: "guild:guild-1".to_string(),
            user_key: "user-1".to_string(),
        });

        assert_eq!(user.platform.as_str(), "discord");
        assert_eq!(
            user.guild_id.as_ref().map(ExternalId::as_str),
            Some("guild-1")
        );
        assert_eq!(user.user_id.as_str(), "user-1");
    }

    #[test]
    fn memory_profile_display_name_prefers_readable_names() {
        let profile = UserProfile {
            id: UserRef {
                platform: PlatformName::new("discord"),
                guild_id: Some(ExternalId::new("guild-1")),
                user_id: ExternalId::new("user-1"),
            },
            username: "alice_global".to_string(),
            name: Some("Alice Global".to_string()),
            display_name: Some("Alice Guild".to_string()),
            avatar_url: None,
            is_bot: false,
        };

        assert_eq!(
            memory_profile_display_name(&profile, "user-1").as_deref(),
            Some("Alice Guild")
        );
    }

    #[test]
    fn memory_profile_display_name_omits_id_fallback() {
        let profile = UserProfile {
            id: UserRef {
                platform: PlatformName::new("discord"),
                guild_id: Some(ExternalId::new("guild-1")),
                user_id: ExternalId::new("user-1"),
            },
            username: "user-1".to_string(),
            name: None,
            display_name: None,
            avatar_url: None,
            is_bot: false,
        };

        assert_eq!(memory_profile_display_name(&profile, "user-1"), None);
    }

    #[test]
    fn prompt_guidance_names_tools_and_proactive_policy() {
        let guidance = PROMPT_GUIDANCE;

        assert!(guidance.contains(LOOKUP_USER_MEMORY_TOOL));
        assert!(guidance.contains(REMEMBER_USER_MEMORY_TOOL));
        assert!(guidance.contains(FORGET_USER_MEMORY_TOOL));
        assert!(guidance.contains("CRITICAL: Memory System"));
        assert!(guidance.contains("`author` of a message"));
        assert!(guidance.contains("MUST load memory about that user"));
        assert!(guidance.contains("Do not respond to a user"));
        assert!(guidance.contains("any time you see a user for the first time"));
        assert!(guidance.contains("any **mention** of a user"));
        assert!(guidance.contains("IT IS CRITICAL TO USE THE MEMORY SYSTEM PROACTIVELY"));
        assert!(guidance.contains("The tool calls are cheap"));
        assert!(guidance.contains("respect their humanity"));
        assert!(guidance.contains("grain of salt"));
        assert!(guidance.contains("trust the current message"));
        assert!(guidance.contains("sensitive personal information"));
    }
}
