//! Discord bot event loop.
//!
//! Connects to the gateway with twilight, listens for `MessageCreate`
//! events, and for any `@<bot>` mention:
//!   1. reacts 👀
//!   2. resolves which conversation this belongs to (Discord reply to a
//!      bot message → continue; in a bot-owned thread → continue;
//!      otherwise → create new)
//!   3. builds the context fed to the LLM, calls it with server-side
//!      web search enabled, persists the turn + every tool call
//!   4. replies inline, or in a new thread when the answer is long
//!   5. reacts ✅ / ❌
//!
//! Each handled message runs in its own task so a slow LLM call doesn't
//! block the gateway.

use std::sync::Arc;

use grok_discord_bot_core::{
    AnyProvider, ChatMessage, CompletionRequest, ContextItem, Conversation, Db, LlmProvider,
    MessageRole, PrivacyMode,
};
use thiserror::Error;
use twilight_gateway::{EventTypeFlags, Intents, Shard, ShardId, StreamExt};
use twilight_http::Client as HttpClient;
use twilight_http::request::channel::reaction::RequestReactionType;
use twilight_model::channel::Message;
use twilight_model::gateway::event::Event;
use twilight_model::id::Id;
use twilight_model::id::marker::{ApplicationMarker, ChannelMarker, MessageMarker, UserMarker};

use crate::commands;

const SYSTEM_PROMPT: &str = "You are a helpful AI assistant in a private Discord \
server. Be direct and concise. When asked to verify a claim or look something up, \
use the web search tool to ground your answer in current sources. Cite URLs where \
relevant.";

/// Discord messages have a hard 2000-char limit; we auto-thread when the
/// answer exceeds this. Threading is also skipped for follow-ups inside
/// an existing conversation — we just reply inline.
const REPLY_LENGTH_THRESHOLD: usize = 1500;

/// Soft cap on the model's reply tokens. Anthropic requires `max_tokens`;
/// xAI tolerates an unused field.
const MAX_OUTPUT_TOKENS: u32 = 4096;

/// Errors returned by the bot loop. We don't propagate these — each
/// handler logs and reacts ❌ on failure — but having a typed error
/// makes the code paths easier to follow.
#[derive(Debug, Error)]
pub enum BotError {
    /// Underlying HTTP / gateway transport failure.
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

/// Top-level handler for one mention. Sets the 👀 reaction, calls
/// [`process`], then transitions the reaction to ✅ or ❌.
async fn handle_message(state: Arc<State>, msg: Message) {
    if msg.author.bot {
        return;
    }
    if !msg.mentions.iter().any(|u| u.id == state.bot_user_id) {
        return;
    }

    // Resolve the active privacy mode for this guild. DMs have no
    // guild_id and use the config-supplied default.
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

    // Design 2: if the bot is confined to a single channel, ignore
    // mentions anywhere else.
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

    // Best-effort: even if the reaction request fails, keep going.
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
            // Try to surface the error in-channel so the user knows
            // something went wrong rather than the bot silently dropping.
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

/// Full happy-path: resolve conversation, build context, call LLM,
/// persist everything, post the reply.
async fn process(
    state: &State,
    msg: &Message,
    privacy_mode: &PrivacyMode,
) -> Result<(), BotError> {
    let (conversation, is_new) = resolve_conversation(state, msg).await?;
    let user_content = strip_mentions(&msg.content, state.bot_user_id);

    let context =
        build_context(state, msg, &conversation, is_new, &user_content, privacy_mode).await?;

    let turn = state
        .db
        .start_turn(
            conversation.id,
            i64::try_from(msg.id.get()).unwrap_or(i64::MAX),
            &user_content,
        )
        .await?;

    for item in &context {
        state.db.record_context_item(turn.id, item).await?;
    }

    let chat = context
        .iter()
        .map(|c| ChatMessage {
            role: MessageRole::from_str_lossy(&c.role),
            content: c.content.clone(),
        })
        .collect();

    let response = match state
        .llm
        .complete(CompletionRequest {
            messages: chat,
            enable_web_search: true,
            max_tokens: MAX_OUTPUT_TOKENS,
        })
        .await
    {
        Ok(r) => r,
        Err(e) => {
            state.db.fail_turn(turn.id, &e.to_string()).await.ok();
            return Err(e.into());
        }
    };

    for (i, tc) in response.tool_calls.iter().enumerate() {
        state
            .db
            .record_tool_call(turn.id, i32::try_from(i).unwrap_or(0), tc)
            .await?;
    }

    let reply_text = format_reply(&response.content, is_new, &conversation, &state.web_base_url);
    let reply_msg = post_reply(state, msg, &reply_text, is_new).await?;

    state
        .db
        .complete_turn(
            turn.id,
            &response.content,
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
/// an existing one. The continuation paths are:
///   - Discord reply to a message we have a link for (typically the
///     bot's prior reply).
///   - Message posted inside a thread that itself was started off a
///     message we have a link for (channel_id of the message equals the
///     parent message id for threads created from messages).
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

    // Public threads created off a message share the parent message's id
    // as their channel id. If this channel_id is in message_links, we're
    // in a bot-owned thread.
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

/// Assemble the prompt fed to the LLM and recorded into `context_items`.
///
/// Structure: system prompt → (continuation? prior turns : extra context
/// per privacy mode) → user's current message.
///
/// Privacy modes affect the "extra context" middle section for new
/// conversations:
///   - Open / ChannelOnly → bulk-fetch recent channel messages
///   - OptIn → include the quoted message only if its author has opted
///     in or the message lives inside a Grok-owned thread
///   - ConversationOnly → no extra context at all
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
    } else {
        match privacy_mode {
            PrivacyMode::Open { history_size }
            | PrivacyMode::ChannelOnly { history_size, .. } => {
                let history =
                    fetch_channel_history(state, msg.channel_id, msg.id, *history_size).await?;
                if !history.is_empty() {
                    push_item(
                        &mut items,
                        &mut pos,
                        "system:channel_history_header".to_string(),
                        "system",
                        "Recent messages from this Discord channel are included below \
                         as context. Each line is of the form \"[author]: content\"."
                            .to_string(),
                        None,
                    );
                    for m in &history {
                        let body =
                            format!("[{author}]: {content}", author = m.author.name, content = m.content);
                        push_item(
                            &mut items,
                            &mut pos,
                            format!("discord:msg:{}", m.id),
                            "user",
                            body,
                            Some(i64::try_from(m.id.get()).unwrap_or(i64::MAX)),
                        );
                    }
                }
            }
            PrivacyMode::OptIn => {
                if let Some(referenced) = &msg.referenced_message {
                    if !referenced.author.bot
                        && is_referenced_visible_opt_in(state, conversation, referenced).await?
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
            }
            PrivacyMode::ConversationOnly => {
                // Privacy-maxxing: never include external context, even
                // a Discord-reply quote.
            }
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

/// For OptIn mode: a quoted (Discord-reply target) message is visible
/// if its author has opted in for this guild, or if the quoted message
/// itself lives in a Grok-owned thread (participation implies consent).
async fn is_referenced_visible_opt_in(
    state: &State,
    conversation: &Conversation,
    referenced: &Message,
) -> Result<bool, BotError> {
    // Property 2: messages inside a Grok-owned thread are always
    // visible. Threads created from a message have channel_id == that
    // parent message's id, so the thread's existence shows up in
    // message_links.
    let channel_as_msg = i64::try_from(referenced.channel_id.get()).unwrap_or(i64::MAX);
    if state
        .db
        .lookup_conversation_by_message(channel_as_msg)
        .await?
        .is_some()
    {
        return Ok(true);
    }

    // Otherwise, check the author's per-guild opt-in.
    let author_id = i64::try_from(referenced.author.id.get()).unwrap_or(i64::MAX);
    state
        .db
        .user_opted_in(conversation.discord_guild_id, author_id)
        .await
        .map_err(BotError::from)
}

/// Pull `limit` recent messages from `channel_id`, ending just before
/// `before`. Bot's own messages are filtered out — they're already in
/// the conversation history (when applicable) and reposting them as
/// "context" confuses the model.
async fn fetch_channel_history(
    state: &State,
    channel_id: Id<ChannelMarker>,
    before: Id<MessageMarker>,
    limit: u32,
) -> Result<Vec<Message>, BotError> {
    let capped = limit.min(100) as u16;
    let mut msgs = state
        .http
        .channel_messages(channel_id)
        .before(before)
        .limit(capped)
        .await?
        .models()
        .await?;
    msgs.retain(|m| m.author.id != state.bot_user_id && !m.content.is_empty());
    // Discord returns newest-first; reverse so the LLM sees them in
    // chronological order, which is how humans read them.
    msgs.reverse();
    Ok(msgs)
}

/// Append the viewer URL to the answer when starting a new conversation
/// so the user can click through to the full trace.
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
        // Post the answer inside the new thread. We truncate just under
        // the 2000-char hard limit; the full content is always in the DB
        // and viewer.
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
/// tokens entirely (vs. just deleting the `<` / `>` / `@` characters,
/// which leaves the numeric ID behind and pollutes thread titles).
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
}
