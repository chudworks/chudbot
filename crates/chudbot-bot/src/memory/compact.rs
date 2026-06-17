//! Compaction job prompt and input construction for user memory.
//!
//! The memory runtime schedules `MemoryJobKind::Compact` after enough raw memory
//! material accumulates for a subject user. This module owns the static fallback
//! prompt and the Markdown input shape sent to the reserved compaction agent.
//! Runtime code handles leasing, model execution, retry/failure accounting, and
//! storing the returned Markdown as the next complete memory document revision.

use std::collections::BTreeMap;

use chudbot_api::{
    AgentLimits, UserMemoryDiaryEntry, UserMemoryDocument, UserMemoryEvent, UserMemoryEventKind,
    UserMemoryKey,
};

use crate::config::{AgentConfig, SystemAgentConfig};

use super::{EMPTY_MEMORY, resolve_memory_agent};

/// Reserved agent name for memory compaction jobs.
pub const MEMORY_COMPACT_AGENT: &str = "memory_compact";

/// Fallback instruction set for the reserved compaction agent.
///
/// Deployments can override this by defining `[bot.agents.memory_compact]`.
/// When they do not, memory compaction still has a stable default: rewrite the
/// whole user profile from the previous profile plus pending ledger events and
/// generated diary entries.
const COMPACTOR_PROMPT: &str = "You maintain a compact Markdown memory profile for one \
Chudbot user in one server/workspace. Produce a complete replacement profile, not a diff. \
Use explicit memory events, diary entries, corrections, and forget requests. Keep the \
profile short, normally 1-3 KB. Remove or rewrite forgotten and stale facts. Preserve \
useful uncertainty. Output Markdown only.";

/// Resolve the system agent used by compaction jobs.
///
/// A configured `memory_compact` agent wins. Otherwise this returns the default
/// prompt/model settings used by the background memory runtime.
pub(in crate::memory) fn resolve_agent(
    agents: &BTreeMap<String, AgentConfig>,
    default_limits: AgentLimits,
) -> SystemAgentConfig {
    resolve_memory_agent(
        MEMORY_COMPACT_AGENT,
        COMPACTOR_PROMPT,
        default_max_output_tokens(),
        agents,
        default_limits,
    )
}

/// Fallback output budget for complete replacement memory profiles.
fn default_max_output_tokens() -> u32 {
    2048
}

/// Build the single user-message payload for a compaction model run.
///
/// The result is intentionally plain Markdown text rather than a tool schema:
/// the compactor must see the subject key, the current full profile, every
/// pending raw memory event, and every pending diary entry before it returns the
/// complete replacement profile. `run_compact_job` persists the model's text
/// verbatim as a new `UserMemoryDocument` revision and advances the source
/// cutoffs to the included events and diaries.
pub(in crate::memory) fn compact_input(
    key: &UserMemoryKey,
    document: Option<&UserMemoryDocument>,
    events: &[UserMemoryEvent],
    diaries: &[UserMemoryDiaryEntry],
) -> String {
    let mut out = String::new();
    out.push_str("# Subject\n");
    out.push_str(&format!(
        "platform: {}\nscope: {}\nuser: {}\n\n",
        key.platform, key.scope_key, key.user_key
    ));
    out.push_str("# Current Memory Profile\n");
    out.push_str(
        document
            .map(|document| document.markdown.trim())
            .filter(|markdown| !markdown.is_empty())
            .unwrap_or(EMPTY_MEMORY),
    );
    out.push_str("\n\n# New Raw Memory Events\n");
    // Ledger events are the authoritative correction/forget stream, so they
    // stay separate from softer diary summaries in the prompt.
    if events.is_empty() {
        out.push_str(EMPTY_MEMORY);
        out.push('\n');
    } else {
        for event in events {
            out.push_str(&format!(
                "\n- id: {}\n  kind: {}\n  created_at: {}\n  body: {}\n",
                event.id,
                event_kind_label(event.kind),
                event.created_at,
                event.body.replace('\n', "\n    ")
            ));
        }
    }
    out.push_str("\n# New Diary Entries\n");
    // Diary entries compress recent turns; compaction folds them into the
    // stable profile only after preserving uncertainty and removing stale notes.
    if diaries.is_empty() {
        out.push_str(EMPTY_MEMORY);
        out.push('\n');
    } else {
        for diary in diaries {
            out.push_str(&format!(
                "\n## Diary {} ({} - {})\n{}\n",
                diary.id, diary.window_start, diary.window_end, diary.markdown
            ));
        }
    }
    out.push_str("\n# Required Profile Headings\n");
    // The headings keep future lookup output predictable even though the model
    // can omit empty sections under each heading.
    out.push_str(
        "# User Memory\n\n## Identity And Names\n## Relationships\n## Preferences\n## Projects And Interests\n## Server Lore\n## Roast Material\n## Boundaries And Avoidances\n## Uncertain Or Low-Confidence Notes\n",
    );
    out
}

/// Convert the typed ledger event kind into the prompt-facing label.
///
/// These strings mirror the persisted snake-case names so the compactor can
/// distinguish memory additions, corrections, forget requests, diary-sourced
/// observations, and operator notes.
fn event_kind_label(kind: UserMemoryEventKind) -> &'static str {
    match kind {
        UserMemoryEventKind::Remember => "remember",
        UserMemoryEventKind::Correction => "correction",
        UserMemoryEventKind::Forget => "forget",
        UserMemoryEventKind::DiaryObservation => "diary_observation",
        UserMemoryEventKind::OperatorNote => "operator_note",
    }
}
