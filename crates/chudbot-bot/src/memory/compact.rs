//! Compaction job input construction for user memory.

use std::collections::BTreeMap;

use chudbot_api::{
    AgentLimits, UserMemoryDiaryEntry, UserMemoryDocument, UserMemoryEvent, UserMemoryEventKind,
    UserMemoryKey,
};

use crate::config::{AgentConfig, SystemAgentConfig};

use super::{EMPTY_MEMORY, resolve_memory_agent};

/// Reserved agent name for memory compaction jobs.
pub const MEMORY_COMPACT_AGENT: &str = "memory_compact";

const COMPACTOR_PROMPT: &str = "You maintain a compact Markdown memory profile for one \
Chudbot user in one server/workspace. Produce a complete replacement profile, not a diff. \
Use explicit memory events, diary entries, corrections, and forget requests. Keep the \
profile short, normally 1-3 KB. Remove or rewrite forgotten and stale facts. Preserve \
useful uncertainty. Output Markdown only.";

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

fn default_max_output_tokens() -> u32 {
    2048
}

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
    out.push_str(
        "# User Memory\n\n## Identity And Names\n## Relationships\n## Preferences\n## Projects And Interests\n## Server Lore\n## Roast Material\n## Boundaries And Avoidances\n## Uncertain Or Low-Confidence Notes\n",
    );
    out
}

fn event_kind_label(kind: UserMemoryEventKind) -> &'static str {
    match kind {
        UserMemoryEventKind::Remember => "remember",
        UserMemoryEventKind::Correction => "correction",
        UserMemoryEventKind::Forget => "forget",
        UserMemoryEventKind::DiaryObservation => "diary_observation",
        UserMemoryEventKind::OperatorNote => "operator_note",
    }
}
