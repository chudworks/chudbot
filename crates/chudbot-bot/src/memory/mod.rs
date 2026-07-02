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
/// The runtime injects persistent memory notes for conversation participants
/// as system messages, so this guidance covers the parts the model still owns:
/// refreshing memory on request, writing remember/forget events with correct
/// targets, and keeping sensitive data out of memory. Concrete example calls
/// are included because they anchor weaker models better than abstract rules.
pub const PROMPT_GUIDANCE: &str = "Memory System:\n\
Memory notes for conversation participants are injected automatically as system messages: the first time a user sends a message, is mentioned, or has a message quoted in this conversation, a note labeled \"User memory note\" with their memory document appears. Read the notes before answering. You normally do not need `lookup_user_memory`.\n\
- Call `lookup_user_memory` only to fetch fresh or missing memory: when a user asks you to re-load their memory, when a note seems out of date, or for a user discussed by name who has no note (you need their numeric id from the message JSON). The result contains the remembered profile, pending remember/forget events, and recent diary entries.\n\
- Use `remember_user_memory` proactively whenever you learn a stable fact worth keeping: preferences, relationships, projects, corrections, recurring facts, running jokes. Store one concise third-person fact per call, e.g. {\"memory\": \"Prefers Rust over Python\"}.\n\
- When the fact is about someone other than the author, pass that user's numeric id as `target_user_id`, e.g. {\"target_user_id\": \"123456789012345678\", \"memory\": \"Runs the weekly movie night\"}. Copy ids from `author.id` or `mentioned_users[].id` in the message JSON; never pass usernames.\n\
- Use `forget_user_memory` to retract a stored fact. Its `memory` field describes what to stop using, e.g. {\"memory\": \"The claim that they live in Ohio\", \"reason\": \"User corrected this\"}.\n\
- If a user explicitly asks you to remember or forget something, always honor it with the matching tool call.\n\
- Facts told by one user about another may be remembered for the subject user; attribute second-hand facts in the text, e.g. {\"memory\": \"According to Alice, is afraid of geese\"}.\n\
- If the current message conflicts with stored memory, trust the current message and remember the correction.\n\
- Never store or repeat sensitive personal information (credit cards, physical addresses, legal names, government IDs).\n\n";

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
        assert!(guidance.contains("Memory System"));
        assert!(guidance.contains("injected automatically as system messages"));
        assert!(guidance.contains("You normally do not need `lookup_user_memory`"));
        assert!(guidance.contains("re-load their memory"));
        assert!(guidance.contains("target_user_id"));
        assert!(guidance.contains("mentioned_users[].id"));
        assert!(guidance.contains("never pass usernames"));
        assert!(guidance.contains("proactively"));
        assert!(guidance.contains("trust the current message"));
        assert!(guidance.contains("sensitive personal information"));
    }
}
