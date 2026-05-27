//! Discord bot event loop.
//!
//! Connects to the gateway with twilight, listens for `MessageCreate`
//! and `InteractionCreate` events. For any `@<bot>` mention:
//!   1. reacts 👀
//!   2. resolves which conversation this belongs to
//!   3. builds the initial context (system prompt, prior turns,
//!      and — where the privacy mode allows — the Discord-reply-quoted
//!      message)
//!   4. drives the model through the agentic loop in [`core::agent`],
//!      with `fetch_messages` exposed as a client-side tool the model
//!      can call to pull more channel history on demand
//!   5. replies inline, or in a new thread when the answer is long
//!   6. reacts ✅ / ❌
//!
//! Interactions (slash commands) are dispatched to [`crate::commands`].

use std::sync::Arc;

use grok_discord_bot_core::{
    AgentRun, AnyProvider, ChatTurn, ContextItem, Conversation, Db, LlmProvider, MessageRole,
    PrivacyMode, ToolDefinition, ToolError, ToolExecutor, TurnBlock, run_agent,
};
use serde::Serialize;
use serde_json::{Value, json};
use thiserror::Error;
use twilight_gateway::{EventTypeFlags, Intents, Shard, ShardId, StreamExt};
use twilight_http::Client as HttpClient;
use twilight_http::request::channel::reaction::RequestReactionType;
use twilight_model::channel::Message;
use twilight_model::gateway::event::Event;
use twilight_model::id::Id;
use twilight_model::id::marker::{
    ApplicationMarker, ChannelMarker, GuildMarker, MessageMarker, UserMarker,
};

use crate::commands;

const SYSTEM_PROMPT: &str = "You are a helpful AI assistant in a private Discord \
server. Be direct and concise. When asked to verify a claim, use the web search \
tool to ground your answer in current sources and cite URLs. When you need more \
context about an ongoing conversation in this channel (for example: \"what did \
they decide?\", \"what's the discussion been about?\"), call the `fetch_messages` \
tool to pull recent messages from the channel. Don't fetch speculatively — only \
when you actually need extra context to answer.";

/// Discord messages have a hard 2000-char limit; we auto-thread when the
/// answer exceeds this. Threading is also skipped for follow-ups inside
/// an existing conversation — we just reply inline.
const REPLY_LENGTH_THRESHOLD: usize = 1500;

/// Soft cap on the model's reply tokens per step. Anthropic requires
/// `max_tokens`; xAI tolerates an unused field.
const MAX_OUTPUT_TOKENS: u32 = 4096;

/// Safety cap on the agent's tool-use loop. Most turns finish in 1-3
/// iterations; this is a runaway guard.
const MAX_AGENT_ITERATIONS: u32 = 6;

/// Errors returned by the bot loop. Logged + surfaced as a ❌ reaction.
#[derive(Debug, Error)]
pub enum BotError {
    /// Discord HTTP / gateway transport.
    #[error("discord http: {0}")]
    Http(#[from] twilight_http::Error),
    /// Failure deserializing a Discord response body.
    #[error("discord deserialize: {0}")]
    Deserialize(#[from] twilight_http::response::DeserializeBodyError),
    /// Database error.
    #[error(transparent)]
    Db(#[from] grok_discord_bot_core::DbError),
    /// LLM provider error.
    #[error(transparent)]
    Llm(#[from] grok_discord_bot_core::LlmError),
}

/// State shared across all message-handler tasks.
struct State {
    http: Arc<HttpClient>,
    db: Db,
    llm: AnyProvider,
    web_base_url: String,
    bot_user_id: Id<UserMarker>,
    app_id: Id<ApplicationMarker>,
    default_privacy: PrivacyMode,
}

/// Entry point for the `grok bot` subcommand.
pub async fn run(
    discord_token: String,
    db: Db,
    llm: AnyProvider,
    web_base_url: String,
    default_privacy: PrivacyMode,
) -> Result<(), BotError> {
    let intents = Intents::GUILDS
        | Intents::GUILD_MESSAGES
        | Intents::MESSAGE_CONTENT
        | Intents::DIRECT_MESSAGES;

    let http = Arc::new(HttpClient::new(discord_token.clone()));

    let current = http.current_user().await?.model().await?;
    let application = http.current_user_application().await?.model().await?;
    tracing::info!(
        user = %current.name,
        id = %current.id,
        app_id = %application.id,
        "discord bot ready"
    );

    if let Err(err) = commands::register(&http, application.id).await {
        tracing::error!(error = %err, "failed to register slash commands; continuing without them");
    }

    let state = Arc::new(State {
        http,
        db,
        llm,
        web_base_url,
        bot_user_id: current.id,
        app_id: application.id,
        default_privacy,
    });

    let mut shard = Shard::new(ShardId::ONE, discord_token, intents);
    let watched = EventTypeFlags::MESSAGE_CREATE | EventTypeFlags::INTERACTION_CREATE;

    while let Some(item) = shard.next_event(watched).await {
        let event = match item {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!(error = %err, "gateway receive error");
                continue;
            }
        };

        match event {
            Event::MessageCreate(msg) => {
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    handle_message(state, msg.0).await;
                });
            }
            Event::InteractionCreate(boxed) => {
                let state = Arc::clone(&state);
                tokio::spawn(async move {
                    commands::handle(
                        Arc::clone(&state.http),
                        state.db.clone(),
                        state.default_privacy.clone(),
                        state.app_id,
                        boxed.0,
                    )
                    .await;
                });
            }
            _ => {}
        }
    }

    Ok(())
}

/// Top-level handler for one mention. Resolves the privacy mode, gates
/// on ChannelOnly, sets the 👀 reaction, calls [`process`], then
/// transitions the reaction to ✅ or ❌.
async fn handle_message(state: Arc<State>, msg: Message) {
    if msg.author.bot {
        return;
    }
    if !msg.mentions.iter().any(|u| u.id == state.bot_user_id) {
        return;
    }

    let guild_id_opt = msg.guild_id.map(|g| i64::try_from(g.get()).unwrap_or(i64::MAX));
    let privacy_mode = match guild_id_opt {
        Some(gid) => match state
            .db
            .guild_privacy_mode_or(gid, &state.default_privacy)
            .await
        {
            Ok(m) => m,
            Err(err) => {
                tracing::error!(error = %err, "failed to load guild privacy mode; falling back to default");
                state.default_privacy.clone()
            }
        },
        None => state.default_privacy.clone(),
    };

    if let PrivacyMode::ChannelOnly { channel_id, .. } = &privacy_mode {
        if msg.channel_id.get() != *channel_id {
            tracing::debug!(
                channel = %msg.channel_id,
                allowed = *channel_id,
                "ignoring mention outside allowed channel (channel_only mode)"
            );
            return;
        }
    }

    let working = RequestReactionType::Unicode { name: "👀" };
    let done = RequestReactionType::Unicode { name: "✅" };
    let failed = RequestReactionType::Unicode { name: "❌" };

    let _ = state
        .http
        .create_reaction(msg.channel_id, msg.id, &working)
        .await;

    let result = process(&state, &msg, &privacy_mode).await;

    let _ = state
        .http
        .delete_current_user_reaction(msg.channel_id, msg.id, &working)
        .await;

    match result {
        Ok(()) => {
            let _ = state
                .http
                .create_reaction(msg.channel_id, msg.id, &done)
                .await;
        }
        Err(err) => {
            tracing::error!(error = %err, "message handler failed");
            let _ = state
                .http
                .create_reaction(msg.channel_id, msg.id, &failed)
                .await;
            let snippet = err.to_string();
            let snippet = if snippet.len() > 500 {
                format!("{}…", &snippet[..500])
            } else {
                snippet
            };
            let _ = state
                .http
                .create_message(msg.channel_id)
                .content(&format!("⚠️ {snippet}"))
                .reply(msg.id)
                .await;
        }
    }
}

/// Full happy-path: resolve conversation, build initial context, run
/// the agent loop with `fetch_messages` available, persist everything,
/// post the reply.
async fn process(
    state: &State,
    msg: &Message,
    privacy_mode: &PrivacyMode,
) -> Result<(), BotError> {
    let (conversation, is_new) = resolve_conversation(state, msg).await?;
    let user_content = strip_mentions(&msg.content, state.bot_user_id);

    let initial_context =
        build_context(state, msg, &conversation, is_new, &user_content, privacy_mode).await?;

    let turn = state
        .db
        .start_turn(
            conversation.id,
            i64::try_from(msg.id.get()).unwrap_or(i64::MAX),
            &user_content,
        )
        .await?;

    for item in &initial_context {
        state.db.record_context_item(turn.id, item).await?;
    }

    // Build the LLM-facing chat history from the initial context items.
    let messages: Vec<ChatTurn> = initial_context
        .iter()
        .map(|c| ChatTurn::text(MessageRole::from_str_lossy(&c.role), c.content.clone()))
        .collect();

    // Tools available to the model for this turn. fetch_messages is
    // exposed in every mode except ConversationOnly; that mode's whole
    // point is "don't reach beyond the conversation."
    let tools = build_tool_definitions(privacy_mode);

    let executor = BotToolExecutor {
        http: Arc::clone(&state.http),
        db: state.db.clone(),
        bot_user_id: state.bot_user_id,
        default_channel_id: msg.channel_id,
        guild_id: conversation.discord_guild_id,
        privacy_mode: privacy_mode.clone(),
    };

    let agent_result = run_agent(
        &state.llm,
        messages,
        tools,
        &executor,
        true, // server-side web search always enabled
        MAX_OUTPUT_TOKENS,
        MAX_AGENT_ITERATIONS,
    )
    .await;

    let agent_run: AgentRun = match agent_result {
        Ok(r) => r,
        Err(e) => {
            state.db.fail_turn(turn.id, &e.to_string()).await.ok();
            return Err(e.into());
        }
    };

    // Persist all tool calls (server + client) in execution order.
    for (i, tc) in agent_run.tool_calls.iter().enumerate() {
        state
            .db
            .record_tool_call(turn.id, i32::try_from(i).unwrap_or(0), tc)
            .await?;
    }

    let reply_text =
        format_reply(&agent_run.content, is_new, &conversation, &state.web_base_url);
    let reply_msg = post_reply(state, msg, &reply_text, is_new).await?;

    state
        .db
        .complete_turn(
            turn.id,
            &agent_run.content,
            i64::try_from(reply_msg.id.get()).unwrap_or(i64::MAX),
        )
        .await?;

    state
        .db
        .record_message_link(
            i64::try_from(msg.id.get()).unwrap_or(i64::MAX),
            conversation.discord_guild_id,
            conversation.id,
            turn.id,
            "user",
        )
        .await?;
    state
        .db
        .record_message_link(
            i64::try_from(reply_msg.id.get()).unwrap_or(i64::MAX),
            conversation.discord_guild_id,
            conversation.id,
            turn.id,
            "assistant",
        )
        .await?;

    Ok(())
}

/// Decide whether this @mention starts a new conversation or continues
/// an existing one. See [`Db::lookup_conversation_by_message`].
async fn resolve_conversation(
    state: &State,
    msg: &Message,
) -> Result<(Conversation, bool), BotError> {
    if let Some(referenced) = &msg.referenced_message {
        let parent_id = i64::try_from(referenced.id.get()).unwrap_or(i64::MAX);
        if let Some(conv_id) = state.db.lookup_conversation_by_message(parent_id).await? {
            if let Some(conv) = state.db.get_conversation(conv_id).await? {
                return Ok((conv, false));
            }
        }
    }

    let channel_id = i64::try_from(msg.channel_id.get()).unwrap_or(i64::MAX);
    if let Some(conv_id) = state.db.lookup_conversation_by_message(channel_id).await? {
        if let Some(conv) = state.db.get_conversation(conv_id).await? {
            return Ok((conv, false));
        }
    }

    let conv = state
        .db
        .create_conversation(
            msg.guild_id
                .map(|g| i64::try_from(g.get()).unwrap_or(0))
                .unwrap_or(0),
            i64::try_from(msg.channel_id.get()).unwrap_or(i64::MAX),
            i64::try_from(msg.author.id.get()).unwrap_or(i64::MAX),
            i64::try_from(msg.id.get()).unwrap_or(i64::MAX),
            state.llm.name(),
            None,
        )
        .await?;
    Ok((conv, true))
}

/// Assemble the initial prompt for the agent loop. The model can
/// always pull more channel history on demand via `fetch_messages`, so
/// this only needs to include:
///   - the system prompt;
///   - prior turns of the conversation (when continuing);
///   - the Discord-reply-quoted message, gated by the privacy mode;
///   - the user's current `@`-mention.
async fn build_context(
    state: &State,
    msg: &Message,
    conversation: &Conversation,
    is_new: bool,
    user_content: &str,
    privacy_mode: &PrivacyMode,
) -> Result<Vec<ContextItem>, BotError> {
    let mut items = Vec::new();
    let mut pos: i32 = 0;

    push_item(
        &mut items,
        &mut pos,
        "system".to_string(),
        "system",
        SYSTEM_PROMPT.to_string(),
        None,
    );

    if !is_new {
        let history = state.db.load_conversation_history(conversation.id).await?;
        for turn in history {
            push_item(
                &mut items,
                &mut pos,
                format!("turn:{}:user", turn.id),
                "user",
                turn.user_content,
                Some(turn.user_discord_message_id),
            );
            if let Some(answer) = turn.assistant_content {
                push_item(
                    &mut items,
                    &mut pos,
                    format!("turn:{}:assistant", turn.id),
                    "assistant",
                    answer,
                    turn.assistant_discord_message_id,
                );
            }
        }
    } else if let Some(referenced) = &msg.referenced_message {
        if !referenced.author.bot
            && quoted_message_allowed(state, conversation, referenced, privacy_mode).await?
        {
            push_item(
                &mut items,
                &mut pos,
                format!("discord:msg:{}", referenced.id),
                "user",
                format!(
                    "[Quoted message from {}]: {}",
                    referenced.author.name, referenced.content
                ),
                Some(i64::try_from(referenced.id.get()).unwrap_or(i64::MAX)),
            );
        }
    }

    push_item(
        &mut items,
        &mut pos,
        format!("discord:msg:{}", msg.id),
        "user",
        user_content.to_string(),
        Some(i64::try_from(msg.id.get()).unwrap_or(i64::MAX)),
    );

    Ok(items)
}

fn push_item(
    items: &mut Vec<ContextItem>,
    pos: &mut i32,
    source: String,
    role: &str,
    content: String,
    discord_message_id: Option<i64>,
) {
    items.push(ContextItem {
        position: *pos,
        source,
        role: role.to_string(),
        content,
        discord_message_id,
    });
    *pos += 1;
}

/// Privacy gate for the quoted message that arrives as part of a
/// Discord reply. The active mode decides:
async fn quoted_message_allowed(
    state: &State,
    conversation: &Conversation,
    referenced: &Message,
    mode: &PrivacyMode,
) -> Result<bool, BotError> {
    match mode {
        PrivacyMode::Open { .. } | PrivacyMode::ChannelOnly { .. } => Ok(true),
        PrivacyMode::ConversationOnly => Ok(false),
        PrivacyMode::OptIn => {
            // Messages inside a Grok-owned thread are always visible
            // (participation implies consent).
            let channel_as_msg =
                i64::try_from(referenced.channel_id.get()).unwrap_or(i64::MAX);
            if state
                .db
                .lookup_conversation_by_message(channel_as_msg)
                .await?
                .is_some()
            {
                return Ok(true);
            }
            let author_id = i64::try_from(referenced.author.id.get()).unwrap_or(i64::MAX);
            Ok(state
                .db
                .user_opted_in(conversation.discord_guild_id, author_id)
                .await?)
        }
    }
}

/// Tool definitions exposed to the model for this turn. `fetch_messages`
/// is omitted in `ConversationOnly` mode — that mode's whole purpose is
/// to NOT reach beyond the current conversation.
fn build_tool_definitions(mode: &PrivacyMode) -> Vec<ToolDefinition> {
    if matches!(mode, PrivacyMode::ConversationOnly) {
        return Vec::new();
    }
    vec![ToolDefinition {
        name: "fetch_messages".to_string(),
        description: "Fetch recent messages from a Discord channel for additional \
context. Use this when you need to see surrounding conversation that wasn't \
quoted directly — for example when the user asks \"what was the discussion?\" \
or \"what did they decide?\". By default this fetches the most recent messages \
from the current channel."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel ID (snowflake as a string). Omit to use the current channel."
                },
                "limit": {
                    "type": "integer",
                    "description": "How many recent messages to fetch (1-100). Defaults to 20.",
                    "minimum": 1,
                    "maximum": 100
                },
                "before_message_id": {
                    "type": "string",
                    "description": "Fetch messages older than this message ID (snowflake as a string). Use for paginating further back."
                }
            },
            "additionalProperties": false
        }),
    }]
}

/// [`ToolExecutor`] backing `fetch_messages` plus any future tools.
/// Owned per-turn so it can capture the channel + guild context.
struct BotToolExecutor {
    http: Arc<HttpClient>,
    db: Db,
    bot_user_id: Id<UserMarker>,
    default_channel_id: Id<ChannelMarker>,
    guild_id: i64,
    privacy_mode: PrivacyMode,
}

impl ToolExecutor for BotToolExecutor {
    async fn execute(&self, name: &str, input: Value) -> Result<Value, ToolError> {
        match name {
            "fetch_messages" => self.fetch_messages(input).await,
            other => Err(ToolError::Unknown(other.to_string())),
        }
    }
}

#[derive(Serialize)]
struct FetchedMessage {
    id: String,
    channel_id: String,
    author: String,
    author_id: String,
    content: String,
    created_at: String,
    /// `false` = visible content; `true` = author has not opted in (in
    /// OptIn mode) and the content has been redacted from the result.
    redacted: bool,
}

impl BotToolExecutor {
    async fn fetch_messages(&self, input: Value) -> Result<Value, ToolError> {
        let channel_id_input = input
            .get("channel_id")
            .and_then(Value::as_str)
            .map(parse_snowflake)
            .transpose()
            .map_err(|e| ToolError::InvalidInput(format!("channel_id: {e}")))?;
        let channel_id: Id<ChannelMarker> = match channel_id_input {
            Some(id) => Id::new(id),
            None => self.default_channel_id,
        };

        // ChannelOnly mode: don't let the model fetch from arbitrary
        // channels; it can only see the configured one.
        if let PrivacyMode::ChannelOnly {
            channel_id: allowed,
            ..
        } = &self.privacy_mode
        {
            if channel_id.get() != *allowed {
                return Err(ToolError::InvalidInput(format!(
                    "this server is in channel_only mode; fetch_messages can only target channel {allowed}"
                )));
            }
        }

        let limit_i64 = input
            .get("limit")
            .and_then(Value::as_i64)
            .unwrap_or(20)
            .clamp(1, 100);
        let limit = limit_i64 as u16;

        let before_input = input
            .get("before_message_id")
            .and_then(Value::as_str)
            .map(parse_snowflake)
            .transpose()
            .map_err(|e| ToolError::InvalidInput(format!("before_message_id: {e}")))?;

        // Twilight's `.before()` switches the builder type so we can't
        // mutate the same variable through both branches.
        let req = self.http.channel_messages(channel_id);
        let resp = if let Some(b) = before_input {
            req.before(Id::<MessageMarker>::new(b)).limit(limit).await
        } else {
            req.limit(limit).await
        }
        .map_err(|e| ToolError::Execution(format!("discord http: {e}")))?;
        let raw = resp
            .models()
            .await
            .map_err(|e| ToolError::Execution(format!("discord deserialize: {e}")))?;

        // Reverse to chronological order (Discord returns newest-first).
        let mut messages: Vec<Message> = raw;
        messages.reverse();

        let mut out: Vec<FetchedMessage> = Vec::with_capacity(messages.len());
        for m in messages {
            if m.author.id == self.bot_user_id {
                continue;
            }
            let visible = self.is_visible(&m).await;
            out.push(FetchedMessage {
                id: m.id.get().to_string(),
                channel_id: m.channel_id.get().to_string(),
                author: m.author.name.clone(),
                author_id: m.author.id.get().to_string(),
                content: if visible {
                    m.content.clone()
                } else {
                    "[redacted: author has not opted in]".to_string()
                },
                created_at: m.timestamp.iso_8601().to_string(),
                redacted: !visible,
            });
        }

        Ok(serde_json::to_value(&out).unwrap_or(Value::Array(vec![])))
    }

    /// Returns true if the message's content should be visible to the
    /// model under the active privacy mode.
    async fn is_visible(&self, m: &Message) -> bool {
        match &self.privacy_mode {
            PrivacyMode::Open { .. } | PrivacyMode::ChannelOnly { .. } => true,
            PrivacyMode::ConversationOnly => false,
            PrivacyMode::OptIn => {
                let channel_as_msg = i64::try_from(m.channel_id.get()).unwrap_or(i64::MAX);
                if self
                    .db
                    .lookup_conversation_by_message(channel_as_msg)
                    .await
                    .unwrap_or(None)
                    .is_some()
                {
                    return true;
                }
                let author_id = i64::try_from(m.author.id.get()).unwrap_or(i64::MAX);
                self.db
                    .user_opted_in(self.guild_id, author_id)
                    .await
                    .unwrap_or(false)
            }
        }
    }
}

fn parse_snowflake(s: &str) -> Result<u64, String> {
    s.parse::<u64>()
        .map_err(|e| format!("not a valid snowflake: {e}"))
}

fn format_reply(
    answer: &str,
    is_new: bool,
    conversation: &Conversation,
    web_base_url: &str,
) -> String {
    if is_new {
        format!(
            "{answer}\n\n-# 🔎 [full trace]({base}/c/{id})",
            base = web_base_url.trim_end_matches('/'),
            id = conversation.id,
        )
    } else {
        answer.to_string()
    }
}

/// Send the reply. For a new conversation with a long answer we open a
/// public thread off the user's message and post inside it; otherwise
/// we reply inline.
async fn post_reply(
    state: &State,
    user_msg: &Message,
    body: &str,
    is_new: bool,
) -> Result<Message, BotError> {
    if is_new && body.len() > REPLY_LENGTH_THRESHOLD {
        let title = make_thread_title(&user_msg.content);
        let thread = state
            .http
            .create_thread_from_message(user_msg.channel_id, user_msg.id, &title)
            .await?
            .model()
            .await?;
        let trimmed = truncate(body, 1990);
        let reply = state
            .http
            .create_message(thread.id)
            .content(&trimmed)
            .await?
            .model()
            .await?;
        Ok(reply)
    } else {
        let trimmed = truncate(body, 1990);
        let reply = state
            .http
            .create_message(user_msg.channel_id)
            .content(&trimmed)
            .reply(user_msg.id)
            .await?
            .model()
            .await?;
        Ok(reply)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut cutoff = max.saturating_sub(1);
        while !s.is_char_boundary(cutoff) && cutoff > 0 {
            cutoff -= 1;
        }
        format!("{}…", &s[..cutoff])
    }
}

fn make_thread_title(content: &str) -> String {
    let stripped = strip_bracketed_tokens(content);
    let joined: String = stripped
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ");
    let title = truncate(&joined, 95);
    if title.trim().is_empty() {
        "Grok".to_string()
    } else {
        title
    }
}

/// Drop Discord-style `<@id>`, `<@!id>`, `<#channel_id>`, `<:emoji:id>`
/// tokens entirely.
fn strip_bracketed_tokens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '<' && matches!(chars.peek(), Some('@' | '#' | ':')) {
            for inner in chars.by_ref() {
                if inner == '>' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn strip_mentions(content: &str, bot_user_id: Id<UserMarker>) -> String {
    let plain = format!("<@{}>", bot_user_id.get());
    let nick = format!("<@!{}>", bot_user_id.get());
    content
        .replace(&plain, "")
        .replace(&nick, "")
        .trim()
        .to_string()
}

// Reference the unused marker types so rustc's dead-code linter doesn't
// complain when only one of them is needed in the future.
#[allow(dead_code)]
fn _force_marker_imports(
    _g: Id<GuildMarker>,
    _c: Id<ChannelMarker>,
    _m: Id<MessageMarker>,
    _u: Id<UserMarker>,
    _a: Id<ApplicationMarker>,
    _t: TurnBlock,
) {
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_both_mention_forms() {
        let bot: Id<UserMarker> = Id::new(123456789);
        assert_eq!(strip_mentions("<@123456789> hello", bot), "hello");
        assert_eq!(strip_mentions("hi <@!123456789> there", bot), "hi  there");
    }

    #[test]
    fn truncate_respects_utf8_boundary() {
        let s = "héllo wörld";
        let t = truncate(s, 6);
        assert!(t.ends_with('…'));
        assert!(t.len() <= 8);
    }

    #[test]
    fn thread_title_falls_back_when_only_mentions() {
        assert_eq!(make_thread_title("<@123>"), "Grok");
        assert_eq!(make_thread_title("<@123> what is rust"), "what is rust");
    }

    #[test]
    fn fetch_tool_definition_only_when_allowed() {
        assert!(!build_tool_definitions(&PrivacyMode::OptIn).is_empty());
        assert!(!build_tool_definitions(&PrivacyMode::Open { history_size: 20 }).is_empty());
        assert!(build_tool_definitions(&PrivacyMode::ConversationOnly).is_empty());
    }
}
