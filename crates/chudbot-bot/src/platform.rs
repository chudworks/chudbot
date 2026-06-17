//! Platform-neutral helpers for message identity, mentions, replies, and tracing labels.

use crate::prelude::*;
use crate::*;

pub(crate) fn guild_key(message: &MessageRef) -> Option<String> {
    message.guild_id.as_ref().map(|id| id.as_str().to_string())
}

pub(crate) fn channel_from_message(message: &MessageRef) -> ChannelRef {
    ChannelRef {
        platform: message.platform.clone(),
        guild_id: message.guild_id.clone(),
        channel_id: message.channel_id.clone(),
    }
}

pub(crate) fn display_name(message: &PlatformMessage) -> String {
    message
        .author
        .display_name
        .clone()
        .or_else(|| message.author.name.clone())
        .unwrap_or_else(|| message.author.username.clone())
}

pub(crate) fn same_platform_user(
    left: &chudbot_api::UserRef,
    right: &chudbot_api::UserRef,
) -> bool {
    left.platform == right.platform && left.user_id == right.user_id
}

pub(crate) fn normalize_mention_content(
    content: &str,
    bot_user: &chudbot_api::UserRef,
    mentions: &[chudbot_api::UserRef],
    profiles: &[UserProfile],
) -> String {
    let mut out = strip_user_mention(content, bot_user).trim().to_string();
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

pub(crate) fn strip_user_mention(content: &str, user: &chudbot_api::UserRef) -> String {
    content
        .replace(&format!("<@{}>", user.user_id.as_str()), "")
        .replace(&format!("<@!{}>", user.user_id.as_str()), "")
}

pub(crate) fn display_name_for_profile(profile: &UserProfile) -> String {
    profile
        .display_name
        .clone()
        .or_else(|| profile.name.clone())
        .unwrap_or_else(|| profile.username.clone())
}

pub(crate) fn message_link_replays_as_assistant(
    link: &MessageLink,
    conversation_id: ConversationId,
) -> bool {
    link.conversation_id == conversation_id && link.role == "assistant"
}

pub(crate) fn fix_bare_mentions(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '@' {
            out.push(ch);
            continue;
        }
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

pub(crate) fn requested_channel(
    default_channel: &ChannelRef,
    input: &serde_json::Value,
) -> Result<ChannelRef, BotToolError> {
    let Some(channel_id) = input.get("channel_id").and_then(serde_json::Value::as_str) else {
        return Ok(default_channel.clone());
    };
    if channel_id.trim().is_empty() {
        return Err(BotToolError::InvalidInput(
            "`channel_id` cannot be empty".to_string(),
        ));
    }
    Ok(ChannelRef {
        platform: default_channel.platform.clone(),
        guild_id: default_channel.guild_id.clone(),
        channel_id: channel_id.into(),
    })
}

pub(crate) fn should_thread(
    is_new: bool,
    content: &str,
    char_threshold: usize,
    line_threshold: usize,
) -> bool {
    if !is_new {
        return false;
    }
    if content.chars().count() > char_threshold {
        return true;
    }
    rendered_line_count(content) > line_threshold
}

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

pub(crate) fn full_trace_url(web_base_url: &str, conversation_id: ConversationId) -> String {
    let base = web_base_url.trim_end_matches('/');
    format!("{base}/c/{conversation_id}")
}

pub(crate) fn full_trace_link_markdown(
    web_base_url: &str,
    conversation_id: ConversationId,
) -> String {
    format!(
        "-# 🔎 [full trace]({})",
        full_trace_url(web_base_url, conversation_id)
    )
}

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

pub(crate) fn model_step_kind(step: &ModelStep) -> &'static str {
    match step {
        ModelStep::Final { .. } => "final",
        ModelStep::UseClientTools { .. } => "use_client_tools",
        ModelStep::Continue { .. } => "continue",
    }
}

pub(crate) fn agent_outcome_kind(outcome: &AgentOutcome) -> &'static str {
    match outcome {
        AgentOutcome::Completed { .. } => "completed",
        AgentOutcome::Failed { .. } => "failed",
        AgentOutcome::IterationLimit { .. } => "iteration_limit",
        AgentOutcome::Cancelled { .. } => "cancelled",
    }
}

pub(crate) fn tool_trace_kind(trace: &chudbot_api::ToolTrace) -> &'static str {
    match trace {
        chudbot_api::ToolTrace::Client { .. } => "client",
        chudbot_api::ToolTrace::Server { .. } => "server",
        chudbot_api::ToolTrace::Grounding { .. } => "grounding",
    }
}

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

pub(crate) fn privacy_mode_kind(mode: &PrivacyMode) -> &'static str {
    match mode {
        PrivacyMode::Open { .. } => "open",
        PrivacyMode::ChannelOnly { .. } => "channel_only",
        PrivacyMode::OptIn => "opt_in",
        PrivacyMode::ConversationOnly => "conversation_only",
    }
}

pub(crate) fn platform_message_reference_kind(
    reference: &PlatformMessageReference,
) -> &'static str {
    match reference {
        PlatformMessageReference::None => "none",
        PlatformMessageReference::Id(_) => "id",
        PlatformMessageReference::Hydrated(_) => "hydrated",
    }
}
