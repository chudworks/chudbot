//! Platform-neutral glue for message identity, mentions, replies, and tracing labels.
//!
//! The bot runtime handles conversations in terms of `chudbot_api` platform
//! contracts. This module keeps the small, reusable translation rules in one
//! place so turn orchestration, tools, and prompt assembly do not each encode
//! their own idea of channel identity, user display names, reply formatting, or
//! stable event labels.

use crate::prelude::*;
use crate::*;

/// Return the storage/runtime scope key for a guild-backed message.
///
/// Direct-message events have no guild scope and therefore return `None`.
pub(crate) fn guild_key(message: &MessageRef) -> Option<String> {
    message.guild_id.as_ref().map(|id| id.as_str().to_string())
}

/// Build the channel identity that contains a platform message.
///
/// This preserves the message platform and guild so callers can address the
/// same channel without depending on a concrete platform adapter type.
pub(crate) fn channel_from_message(message: &MessageRef) -> ChannelRef {
    ChannelRef {
        platform: message.platform.clone(),
        guild_id: message.guild_id.clone(),
        channel_id: message.channel_id.clone(),
    }
}

/// Pick the best user-facing author name available on a platform message.
///
/// The order matches how conversation transcripts should read: server-local
/// display name, platform profile name, then the stable username fallback.
pub(crate) fn display_name(message: &PlatformMessage) -> String {
    message
        .author
        .display_name
        .clone()
        .or_else(|| message.author.name.clone())
        .unwrap_or_else(|| message.author.username.clone())
}

/// Compare two users by platform identity only.
///
/// Profile details can be stale or absent; the platform and user id are the
/// invariant pair used for ownership and mention matching.
pub(crate) fn same_platform_user(
    left: &chudbot_api::UserRef,
    right: &chudbot_api::UserRef,
) -> bool {
    left.platform == right.platform && left.user_id == right.user_id
}

/// Prepare inbound message text for the transcript and model prompt.
///
/// The wake-up mention for the bot is removed, while mentions of other users
/// are expanded to include a readable label next to the original platform
/// token. Keeping the token matters because downstream reply logic can still
/// preserve platform-addressable identity.
pub(crate) fn normalize_mention_content(
    content: &str,
    bot_user: &chudbot_api::UserRef,
    mentions: &[chudbot_api::UserRef],
    profiles: &[UserProfile],
) -> String {
    // First remove the bot mention that caused the turn so the model sees the
    // user's actual request instead of the routing syntax.
    let mut out = strip_user_mention(content, bot_user).trim().to_string();

    // Then annotate every non-bot mention with the best profile label available
    // in this event payload, falling back to the raw platform id if needed.
    for mention in mentions {
        if same_platform_user(mention, bot_user) {
            continue;
        }
        let label = profiles
            .iter()
            .find(|profile| same_platform_user(&profile.id, mention))
            .map(display_name_for_profile)
            .unwrap_or_else(|| mention.user_id.as_str().to_string());
        let replacement = format!("{label} (<@{}>)", mention.user_id.as_str());
        out = out
            .replace(&format!("<@{}>", mention.user_id.as_str()), &replacement)
            .replace(&format!("<@!{}>", mention.user_id.as_str()), &replacement);
    }
    out
}

/// Remove both supported mention spellings for a single user.
///
/// Some platforms or clients emit nickname-style `<@!id>` mentions while others
/// emit plain `<@id>`. Both refer to the same user identity here.
pub(crate) fn strip_user_mention(content: &str, user: &chudbot_api::UserRef) -> String {
    content
        .replace(&format!("<@{}>", user.user_id.as_str()), "")
        .replace(&format!("<@!{}>", user.user_id.as_str()), "")
}

/// Pick the best user-facing name from a stored or hydrated user profile.
pub(crate) fn display_name_for_profile(profile: &UserProfile) -> String {
    profile
        .display_name
        .clone()
        .or_else(|| profile.name.clone())
        .unwrap_or_else(|| profile.username.clone())
}

/// Decide whether a stored platform message already replays as the assistant.
///
/// Quoted/referenced assistant replies are skipped when they already belong to
/// the same conversation transcript, avoiding duplicate assistant context.
pub(crate) fn message_link_replays_as_assistant(
    link: &MessageLink,
    conversation_id: ConversationId,
) -> bool {
    link.conversation_id == conversation_id && link.role == "assistant"
}

/// Convert model-emitted bare user ids into platform mention syntax.
///
/// Models sometimes answer with `@123...` instead of `<@123...>`. Only
/// snowflake-length digit runs are wrapped, and already wrapped mentions are
/// left intact.
pub(crate) fn fix_bare_mentions(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '@' {
            out.push(ch);
            continue;
        }

        // Capture the digit run immediately after `@`; any non-digit suffix is
        // left for the outer loop to copy unchanged.
        let mut digits = String::new();
        while let Some(&next) = chars.peek() {
            if next.is_ascii_digit() {
                digits.push(next);
                chars.next();
            } else {
                break;
            }
        }
        let is_snowflake = (17..=20).contains(&digits.len());
        let already_wrapped = out.ends_with('<');

        // Avoid producing `<<@...>>` when the answer already used the canonical
        // platform mention spelling.
        if is_snowflake && !already_wrapped {
            out.push_str("<@");
            out.push_str(&digits);
            out.push('>');
        } else {
            out.push('@');
            out.push_str(&digits);
        }
    }
    out
}

/// Parse an optional fetch-messages channel override from tool input.
///
/// The tool may narrow or redirect within the same platform and guild scope,
/// but it cannot use input JSON to cross to another platform or guild.
pub(crate) fn requested_channel(
    default_channel: &ChannelRef,
    input: &serde_json::Value,
) -> Result<ChannelRef, BotToolError> {
    // Missing channel_id means the tool call targets the current conversation
    // channel.
    let Some(channel_id) = input.get("channel_id").and_then(serde_json::Value::as_str) else {
        return Ok(default_channel.clone());
    };
    if channel_id.trim().is_empty() {
        return Err(BotToolError::InvalidInput(
            "`channel_id` cannot be empty".to_string(),
        ));
    }

    // Keep platform and guild anchored to the default channel; only the channel
    // id is caller-controlled.
    Ok(ChannelRef {
        platform: default_channel.platform.clone(),
        guild_id: default_channel.guild_id.clone(),
        channel_id: channel_id.into(),
    })
}

/// Decide whether a reply should open a platform thread.
///
/// Only first replies for new conversations can open threads. Continuation
/// replies stay in the existing conversation surface regardless of length.
pub(crate) fn should_thread(
    is_new: bool,
    content: &str,
    char_threshold: usize,
    line_threshold: usize,
) -> bool {
    if !is_new {
        return false;
    }

    // Prefer the cheap character threshold first; line counting walks every
    // rendered line and only matters near the size boundary.
    if content.chars().count() > char_threshold {
        return true;
    }
    rendered_line_count(content) > line_threshold
}

/// Estimate platform-rendered line count for thread threshold decisions.
///
/// Empty logical lines still occupy one rendered line. Non-empty lines are
/// measured against the configured wrap width because very long single-line
/// answers need threads too.
pub(crate) fn rendered_line_count(content: &str) -> usize {
    content
        .split('\n')
        .map(|line| {
            let chars = line.chars().count();
            if chars == 0 {
                1
            } else {
                chars.div_ceil(THREAD_REPLY_WRAP_WIDTH)
            }
        })
        .sum()
}

/// Apply final text-only reply formatting before sending to a platform.
///
/// Mention repair runs for every reply. The trace-viewer link is appended only
/// to the first reply of a new conversation, where the user has no prior link
/// to the stored trace.
pub(crate) fn format_reply_content(
    text: &str,
    is_new: bool,
    conversation_id: ConversationId,
    web_base_url: &str,
) -> String {
    let text = fix_bare_mentions(text);
    if !is_new {
        return text;
    }
    format!(
        "{text}\n\n{}",
        full_trace_link_markdown(web_base_url, conversation_id)
    )
}

/// Build the unauthenticated trace-viewer URL for a conversation.
pub(crate) fn full_trace_url(web_base_url: &str, conversation_id: ConversationId) -> String {
    let base = web_base_url.trim_end_matches('/');
    format!("{base}/c/{conversation_id}")
}

/// Build the Discord-friendly trace link line appended to new replies.
///
/// The exact prefix keeps the line visually subdued in Discord, so callers that
/// need the user-facing trace link should use this helper instead of formatting
/// their own Markdown.
pub(crate) fn full_trace_link_markdown(
    web_base_url: &str,
    conversation_id: ConversationId,
) -> String {
    format!(
        "-# 🔎 [full trace]({})",
        full_trace_url(web_base_url, conversation_id)
    )
}

/// Build prompt guidance for trace-link requests in this conversation.
///
/// The trace URL is safe to reveal only when the user asks for it. The guidance
/// includes the exact user-facing line so the model does not invent a different
/// Discord rendering or expose the UUID unprompted.
pub(crate) fn trace_link_prompt_guidance(
    web_base_url: &str,
    conversation_id: ConversationId,
) -> String {
    let url = full_trace_url(web_base_url, conversation_id);
    let link = full_trace_link_markdown(web_base_url, conversation_id);
    format!(
        concat!(
            "Full trace URL for this conversation: {url}. This is an allowed ",
            "user-facing trace link even though it contains the conversation UUID. ",
            "Do not volunteer it, ",
            "but if the user asks for the trace link, full trace, or full trace link, ",
            "provide this Discord-friendly line exactly: {link}\n"
        ),
        url = url,
        link = link
    )
}

/// Derive a compact platform-thread title for a new turn.
///
/// User text supplies the first eight words when available; agent name is the
/// fallback for media-only or otherwise empty turns. Titles are capped before
/// crossing into platform-specific length limits.
pub(crate) fn thread_title(execution: &TurnExecution) -> String {
    let mut title = execution
        .turn
        .user_content
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ");
    if title.is_empty() {
        title = execution.agent_name.clone();
    }
    title.chars().take(80).collect()
}

/// Return the stable tracing label for a platform event.
pub(crate) fn platform_event_kind(event: &PlatformEvent) -> &'static str {
    match event {
        PlatformEvent::Ready { .. } => "ready",
        PlatformEvent::MessageCreated { .. } => "message_created",
        PlatformEvent::ReactionAdded { .. } => "reaction_added",
        PlatformEvent::ReactionRemoved { .. } => "reaction_removed",
        PlatformEvent::Command { .. } => "command",
        PlatformEvent::Shutdown => "shutdown",
    }
}

/// Return the stable trace label for a streamed model-step terminal event.
pub(crate) fn model_step_kind_from_event(event: &ModelStepEvent) -> Option<&'static str> {
    match event {
        ModelStepEvent::Finished { kind, .. } => Some(match kind {
            ModelStepKind::Final => "final",
            ModelStepKind::ClientTools => "use_client_tools",
            ModelStepKind::Continue => "continue",
        }),
        ModelStepEvent::Delta(_)
        | ModelStepEvent::Continuation(_)
        | ModelStepEvent::ServerToolUse(_)
        | ModelStepEvent::Grounding(_)
        | ModelStepEvent::Usage(_) => None,
    }
}

/// Return the stable trace label for a tool-call trace entry.
pub(crate) fn tool_trace_kind(trace: &chudbot_api::ToolTrace) -> &'static str {
    match trace {
        chudbot_api::ToolTrace::Client { .. } => "client",
        chudbot_api::ToolTrace::Server { .. } => "server",
        chudbot_api::ToolTrace::Grounding { .. } => "grounding",
    }
}

/// Return the stable live-event label for a conversation event.
pub(crate) fn conversation_event_kind(kind: ConversationEventKind) -> &'static str {
    match kind {
        ConversationEventKind::Created => "created",
        ConversationEventKind::TurnStarted => "turn_started",
        ConversationEventKind::TurnUpdated => "turn_updated",
        ConversationEventKind::ToolTraceRecorded => "tool_trace_recorded",
        ConversationEventKind::ContextRecorded => "context_recorded",
        ConversationEventKind::TitleUpdated => "title_updated",
        ConversationEventKind::ConversationUpdated => "conversation_updated",
    }
}

/// Return the stable tracing label for a privacy mode.
pub(crate) fn privacy_mode_kind(mode: &PrivacyMode) -> &'static str {
    match mode {
        PrivacyMode::Open { .. } => "open",
        PrivacyMode::ChannelOnly { .. } => "channel_only",
        PrivacyMode::OptIn => "opt_in",
        PrivacyMode::ConversationOnly => "conversation_only",
    }
}

/// Return the stable tracing label for a platform message reference payload.
pub(crate) fn platform_message_reference_kind(
    reference: &PlatformMessageReference,
) -> &'static str {
    match reference {
        PlatformMessageReference::None => "none",
        PlatformMessageReference::Id(_) => "id",
        PlatformMessageReference::Hydrated(_) => "hydrated",
    }
}
