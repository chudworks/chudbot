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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use grok_discord_bot_core::{
    AgentRun, AnyImageProvider, AnyProvider, AnyVideoProvider, ChatTurn, ContextItem, Conversation,
    Db, ImageProvider, LlmProvider, MessageRole, NoopObserver, Persona, PrivacyMode,
    ProviderOptions, StepRequest, StepResponse, ToolDefinition, ToolError, ToolExecutor, Turn,
    TurnBlock, VideoProvider, imagegen::ImageGenRequest, run_agent, storage,
    videogen::VideoGenRequest,
};

use crate::app::{AppState, EventKind};
use serde::Serialize;
use serde_json::{Value, json};
use thiserror::Error;
use twilight_gateway::{EventTypeFlags, Intents, Shard, ShardId, StreamExt};
use twilight_http::Client as HttpClient;
use twilight_http::request::channel::reaction::RequestReactionType;
use twilight_model::channel::message::{EmojiReactionType, Mention, MessageFlags};
use twilight_model::channel::{ChannelType, Message};
use twilight_model::gateway::GatewayReaction;
use twilight_model::gateway::event::Event;
use twilight_model::gateway::payload::incoming::GuildCreate;
use twilight_model::http::attachment::Attachment as HttpAttachment;
use twilight_model::id::Id;
use twilight_model::id::marker::{
    ApplicationMarker, ChannelMarker, GuildMarker, MessageMarker, UserMarker,
};
use uuid::Uuid;

use crate::commands;

/// Auto-thread when the new-conversation reply is heavy enough that
/// inlining it would dominate the channel. We trigger threading on
/// EITHER signal:
///   - total characters across all chunks > [`REPLY_LENGTH_THRESHOLD`]
///   - "rendered" line count > [`REPLY_RENDERED_LINES_THRESHOLD`]
///
/// "Rendered" counts each `\n`-delimited line plus extra rows for
/// lines that auto-wrap on a typical Discord client (~80 chars wide).
/// The line-based check catches replies like a 10-row numbered list
/// where the char count is low but the vertical footprint is huge.
/// Threading is also skipped for follow-ups inside an existing
/// conversation — we just reply inline.
const REPLY_LENGTH_THRESHOLD: usize = 1500;
const REPLY_RENDERED_LINES_THRESHOLD: usize = 20;
const REPLY_WRAP_WIDTH: usize = 80;

/// Soft cap on the model's reply tokens per step. Anthropic requires
/// `max_tokens`; xAI tolerates an unused field.
const MAX_OUTPUT_TOKENS: u32 = 4096;

/// Safety cap on the agent's tool-use loop. Most turns finish in 1-3
/// iterations; this is a runaway guard.
const MAX_AGENT_ITERATIONS: u32 = 8;

/// Cap on how many prior-turn images (user-uploaded + model-generated)
/// are replayed into the model context, keeping the most recent. Images
/// are the priciest part of the context; this bounds a long thread's
/// per-turn image bill. Prompt caching makes re-sending the survivors
/// cheap, but the count still needs a ceiling. Dropped images are
/// logged so a silent truncation never looks like full coverage.
const MAX_REPLAYED_IMAGES: usize = 32;

/// Discord free-tier upload size cap. Files larger than this are
/// linked rather than attached (avoids a Discord-side reject).
const DISCORD_FREE_UPLOAD_LIMIT_BYTES: u64 = 25 * 1024 * 1024;

/// Unicode emoji a user reacts with to retry a failed turn. Reacting it on
/// either the user's @mention (which carries the ❌) or the bot's error
/// reply re-runs the turn — see [`handle_reaction`].
const RETRY_EMOJI: &str = "🔄";

/// System prompt for the pre-flight moderation classifier. The bot
/// runs in **private friends-only servers**, so the default is ALLOW
/// and the refusal list is narrowly scoped to genuinely illegal or
/// near-illegal content. Banter, slurs, dark humor, political shit-
/// talking, and edgy art requests are explicitly *not* refusals.
const MODERATION_PROMPT: &str = "You are a TOS compliance classifier for a \
private friends-only Discord server. Each message you classify is prefixed \
with the sender's Discord display name as `[name]: ` — use it to apply any \
sender-specific rules below, but classify the message text itself for the \
categories. The DEFAULT IS ALLOW. Only REFUSE the narrowly listed categories \
below.

REFUSE these:
- CSAM or any sexualization of minors
- Doxxing — sharing someone's non-public personal info (home address, \
  workplace, phone, financial info) with apparent intent to harm
- Credible, specific threats of violence against a real, identifiable person
- Coordinated incitement to suicide or self-harm directed at a specific person
- Illegal arrangements: drug or weapon sales, human trafficking, \
  exploitation rings
- Malware, phishing payloads, or coordinated large-scale spam campaigns

ALLOW EVERYTHING ELSE. The friends in this server routinely use slurs, \
dark humor, harsh insults, and call each other names like \"retarded\" or \
worse as banter. That is NOT a refusal — it is the normal tone of the \
server. Likewise ALLOW: profanity, edgy jokes, sarcasm, political \
opinions of any flavor, criticism of public figures, conspiracy theory \
discussion, dark or NSFW jokes that don't involve minors, requests to \
generate edgy art, asking about news/current events, and asking me to \
explain why someone is wrong about something.

When in any doubt, ALLOW. Unless it's panther. That guy is always trying to get us banned. \
Be extra careful with panther.

Respond with EXACTLY one token: ALLOW or REFUSE. No punctuation. No \
explanation.";

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

/// Disposition of a finished turn. Returned by [`run_turn_and_reply`] (and
/// thus `process` / `handle_reaction`) so the caller sets the right
/// reaction WITHOUT posting another message — the turn machinery already
/// posted exactly one user-facing reply and finalized the turn row. `Err`
/// is reserved for infrastructure failures that happened *before* we could
/// post anything, which the caller surfaces with a generic fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnOutcome {
    /// Clean success (or media delivered despite a late transient error).
    Completed,
    /// The turn posted a user-facing failure message and was marked
    /// `failed` in the DB.
    Failed,
    /// Refused by upstream safety; no user-facing message was posted.
    Refused,
}

/// State shared across all message-handler tasks.
///
/// Thin wrapper over the cross-process [`AppState`] plus Discord-only
/// fields that don't make sense on the web side. Implements
/// [`std::ops::Deref`] to `AppState` so existing call sites that read
/// `state.db`, `state.providers`, etc. keep working.
struct State {
    /// Shared application state — db, providers, storage, event bus,
    /// shutdown handles.
    app: Arc<AppState>,
    /// twilight HTTP client (Discord REST API).
    http: Arc<HttpClient>,
    /// This bot's own Discord user id (used to detect self-mentions
    /// and filter out the bot's own messages from history fetches).
    bot_user_id: Id<UserMarker>,
    /// Discord application id (used for registering slash commands and
    /// responding to interactions).
    app_id: Id<ApplicationMarker>,
}

impl std::ops::Deref for State {
    type Target = AppState;

    fn deref(&self) -> &AppState {
        &self.app
    }
}

/// Entry point for the Discord half of `grok serve`. Connects to the
/// gateway, registers slash commands, then loops until the shared
/// `AppState::cancel` token fires or the gateway returns end-of-stream.
pub async fn run(
    app: Arc<AppState>,
    discord_token: String,
    dev_guild_id: Option<u64>,
) -> Result<(), BotError> {
    let intents = Intents::GUILDS
        | Intents::GUILD_MESSAGES
        | Intents::MESSAGE_CONTENT
        | Intents::DIRECT_MESSAGES
        // Reaction intents power the 🔄-to-retry affordance. Neither is
        // privileged, so no Developer-Portal toggle is required.
        | Intents::GUILD_MESSAGE_REACTIONS
        | Intents::DIRECT_MESSAGE_REACTIONS;

    let http = Arc::new(HttpClient::new(discord_token.clone()));

    let current = http.current_user().await?.model().await?;
    let application = http.current_user_application().await?.model().await?;
    tracing::info!(
        user = %current.name,
        id = %current.id,
        app_id = %application.id,
        "discord bot ready"
    );

    if let Err(err) = commands::register(&http, application.id, dev_guild_id).await {
        tracing::error!(error = %err, "failed to register slash commands; continuing without them");
    }

    let state = Arc::new(State {
        app: Arc::clone(&app),
        http,
        bot_user_id: current.id,
        app_id: application.id,
    });

    let mut shard = Shard::new(ShardId::ONE, discord_token, intents);
    let watched = EventTypeFlags::MESSAGE_CREATE
        | EventTypeFlags::INTERACTION_CREATE
        | EventTypeFlags::GUILD_CREATE
        | EventTypeFlags::REACTION_ADD;

    let cancel = app.cancel.clone();
    let tracker = app.tracker.clone();

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                tracing::info!("bot loop: cancellation requested, exiting gateway read");
                break;
            }
            item = shard.next_event(watched) => {
                let Some(item) = item else {
                    tracing::info!("bot loop: gateway stream ended");
                    break;
                };
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
                        tracker.spawn(async move {
                            handle_message(state, msg.0).await;
                        });
                    }
                    Event::InteractionCreate(boxed) => {
                        let state = Arc::clone(&state);
                        tracker.spawn(async move {
                            commands::handle(
                                Arc::clone(&state.http),
                                state.db.clone(),
                                state.default_privacy.clone(),
                                state.personas.clone(),
                                state.default_persona.clone(),
                                state.app_id,
                                boxed.0,
                            )
                            .await;
                        });
                    }
                    Event::ReactionAdd(boxed) => {
                        let state = Arc::clone(&state);
                        tracker.spawn(async move {
                            handle_reaction(state, boxed.0).await;
                        });
                    }
                    Event::GuildCreate(boxed) => log_guild_create(&boxed),
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

/// Log every guild the bot becomes active in. Fires once per guild
/// when the gateway connects (and again whenever the bot joins a new
/// server). Useful for grabbing the `dev_guild_id` you need in
/// `config.toml` without enabling Developer Mode in the Discord client.
fn log_guild_create(event: &GuildCreate) {
    match event {
        GuildCreate::Available(g) => {
            tracing::info!(
                guild_id = %g.id,
                guild_name = %g.name,
                member_count = g.member_count.unwrap_or(0),
                "bot is active in guild"
            );
        }
        GuildCreate::Unavailable(u) => {
            tracing::warn!(guild_id = %u.id, "guild is unavailable (outage)");
        }
    }
}

/// Ask the configured LLM whether the user's message violates Discord
/// TOS. One short call with `temperature=0` and a tight prompt that
/// asks for a single ALLOW/REFUSE token. **Fails open** on transient
/// errors so a broken classifier doesn't silently DOS the bot — except
/// when the upstream itself refuses for safety reasons (e.g. xAI's
/// server-side SAFETY_CHECK_TYPE_* 403), which IS a refusal signal
/// and we honor it directly.
#[tracing::instrument(name = "moderation", skip_all)]
async fn moderation_allows(state: &State, content: &str) -> bool {
    // Route the classifier through the default persona's provider +
    // model. That's the bot's baseline voice and the cheapest stable
    // route — we don't want a persona override to silently change the
    // moderation surface.
    let persona = state
        .personas
        .get(&state.default_persona)
        .expect("default_persona is validated at startup");
    let Some(provider) = state.providers.get(&persona.provider) else {
        tracing::warn!(
            provider = persona.provider.as_str(),
            "moderation: default persona's provider is not initialized; failing open"
        );
        return true;
    };
    let request = StepRequest {
        model: persona.model.clone(),
        messages: vec![
            ChatTurn::text(MessageRole::System, MODERATION_PROMPT),
            ChatTurn::text(
                MessageRole::User,
                format!("Message to classify:\n<<<\n{content}\n>>>"),
            ),
        ],
        tools: Vec::new(),
        enable_web_search: false,
        max_tokens: 8,
        temperature: Some(0.0),
        top_p: None,
        // Moderation is one tight ALLOW/REFUSE token — we never want
        // the model to burn tokens reasoning here, even if the
        // persona normally requests high effort.
        provider_options: ProviderOptions::default(),
        // No cache key: this is a one-shot classification with a short,
        // static prefix (xAI caches that automatically). A constant key
        // would only funnel every guild's moderation traffic onto one
        // server for negligible gain — affinity is worth it only for the
        // long, growing per-conversation prefix in the main agent loop.
        cache_key: None,
    };

    match provider.step(request).await {
        Ok(StepResponse::Final {
            content: verdict, ..
        }) => {
            let normalized = verdict.trim().to_ascii_uppercase();
            // Treat anything containing REFUSE as a refusal; anything
            // else (including empty / unexpected) as ALLOW. We don't
            // want a borked classifier to silently DOS the bot.
            let allowed = !normalized.starts_with("REFUSE")
                && !normalized.contains(" REFUSE")
                && normalized != "REFUSE";
            tracing::info!(verdict = %normalized, allowed, "moderation: classified");
            allowed
        }
        Ok(_) => {
            tracing::warn!("moderation: classifier returned tool-use; failing open");
            true
        }
        Err(err) if is_upstream_safety_refusal(&err) => {
            tracing::info!(
                error = %err,
                "moderation: upstream refused the classifier prompt itself; treating as REFUSE"
            );
            false
        }
        Err(err) => {
            tracing::warn!(error = %err, "moderation: classifier errored; failing open");
            true
        }
    }
}

/// Detect xAI-style safety refusals from a provider error.
fn is_upstream_safety_refusal(err: &grok_discord_bot_core::LlmError) -> bool {
    matches!(
        err,
        grok_discord_bot_core::LlmError::Api { status: 403, body }
            if body_indicates_safety_refusal(body)
    )
}

/// Substring match for xAI's safety-refusal response bodies — the same
/// language appears whether the refusal came from the chat API
/// (response body of a 403) or the image API (error string surfaced
/// through a tool call's response JSON).
///
/// Example body:
/// `{"error":"Content violates usage guidelines. … Failed check: SAFETY_CHECK_TYPE_CSAM"}`
fn body_indicates_safety_refusal(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("safety_check") || lower.contains("violates usage guidelines")
}

/// Top-level handler for one mention. Resolves the privacy mode, gates
/// on ChannelOnly, sets the 👀 reaction, calls [`process`], then
/// transitions the reaction to ✅ or ❌.
#[tracing::instrument(
    skip_all,
    fields(
        message_id = %msg.id,
        channel = %msg.channel_id,
        guild = ?msg.guild_id.map(|g| g.get()),
        author = %msg.author.name,
    )
)]
async fn handle_message(state: Arc<State>, msg: Message) {
    if msg.author.bot {
        return;
    }

    let is_mention = msg.mentions.iter().any(|u| u.id == state.bot_user_id);
    if !is_mention {
        // Log at DEBUG so `RUST_LOG=grok=debug` surfaces "we did
        // receive this message but ignored it" diagnostics, which is
        // crucial when triaging "the bot didn't respond" reports.
        tracing::debug!(
            mentioned_user_ids = ?msg.mentions.iter().map(|u| u.id.get()).collect::<Vec<_>>(),
            bot_id = %state.bot_user_id,
            content_preview = %msg.content.chars().take(80).collect::<String>(),
            "ignoring message (bot not @-mentioned)"
        );
        return;
    }

    let guild_id_opt = msg
        .guild_id
        .map(|g| i64::try_from(g.get()).unwrap_or(i64::MAX));
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
        let in_allowed_channel = msg.channel_id.get() == *channel_id;
        // A message inside a Grok-owned thread is by definition rooted
        // in a channel where the bot already accepted a turn, so it
        // shouldn't be filtered out even if `msg.channel_id` (which
        // for a thread message is the thread's own id, not the
        // parent's) doesn't match the configured allowed channel.
        let in_grok_thread = if in_allowed_channel {
            false
        } else {
            let channel_as_msg = i64::try_from(msg.channel_id.get()).unwrap_or(i64::MAX);
            state
                .db
                .lookup_conversation_by_message(channel_as_msg)
                .await
                .ok()
                .flatten()
                .is_some()
        };
        if !in_allowed_channel && !in_grok_thread {
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
    let refused = RequestReactionType::Unicode { name: "❓" };

    let _ = state
        .http
        .create_reaction(msg.channel_id, msg.id, &working)
        .await;

    // Pre-flight moderation check — refuse without replying if the
    // message clearly violates Discord TOS. Hand the classifier the
    // same `[display name]: text` form the main model sees (see
    // build_context), so author-targeted rules in MODERATION_PROMPT can
    // act on who is speaking, not just what was said.
    let stripped = strip_mentions(&msg.content, state.bot_user_id);
    if !stripped.is_empty()
        && !moderation_allows(
            &state,
            &format!("[{}]: {stripped}", best_display_name(&msg)),
        )
        .await
    {
        tracing::info!(
            author = %msg.author.name,
            channel = %msg.channel_id,
            preview = %stripped.chars().take(80).collect::<String>(),
            "turn: refused by moderation"
        );
        let _ = state
            .http
            .delete_current_user_reaction(msg.channel_id, msg.id, &working)
            .await;
        let _ = state
            .http
            .create_reaction(msg.channel_id, msg.id, &refused)
            .await;
        return;
    }

    let result = process(&state, &msg, &privacy_mode).await;

    let _ = state
        .http
        .delete_current_user_reaction(msg.channel_id, msg.id, &working)
        .await;

    match result {
        Ok(TurnOutcome::Completed) => {
            let _ = state
                .http
                .create_reaction(msg.channel_id, msg.id, &done)
                .await;
        }
        Ok(TurnOutcome::Refused) => {
            tracing::info!("message refused by upstream safety check; reacting ❓");
            let _ = state
                .http
                .create_reaction(msg.channel_id, msg.id, &refused)
                .await;
        }
        Ok(TurnOutcome::Failed) => {
            // `process` already posted the single user-facing error
            // message and marked the turn failed in the DB; we only set
            // the reaction here (no second message).
            let _ = state
                .http
                .create_reaction(msg.channel_id, msg.id, &failed)
                .await;
        }
        Err(err) => {
            // An infrastructure failure *before* the turn machinery posted
            // a reply (e.g. a DB write mid-setup). Surface a generic
            // fallback so the user isn't left with a bare 👀→nothing.
            tracing::error!(error = %err, "message handler failed before a reply was posted");
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
) -> Result<TurnOutcome, BotError> {
    let preview_chars = msg.content.chars().take(80).collect::<String>();
    tracing::info!(
        author = %msg.author.name,
        author_id = %msg.author.id,
        channel = %msg.channel_id,
        guild = ?msg.guild_id,
        attachments = msg.attachments.len(),
        preview = %preview_chars,
        "turn: mention received"
    );

    // Two-phase conversation resolution: first look up whether this
    // message extends an existing conversation, then resolve the persona
    // (which can be conversation-scoped), then either fetch or create
    // the conversation row stamped with the right persona's model.
    let existing = lookup_existing_conversation(state, msg).await?;
    let conversation_id_for_persona = existing.as_ref().map(|c| c.id);
    let guild_id_for_persona = msg
        .guild_id
        .map(|g| i64::try_from(g.get()).unwrap_or(i64::MAX));
    let channel_id_i64 = i64::try_from(msg.channel_id.get()).unwrap_or(i64::MAX);
    // For channel-scoped persona lookups, threads roll up to their
    // parent channel — operators set the override on the channel they
    // can see, not on individual auto-opened threads.
    let persona_channel_id = parent_channel_id(&state.http, msg.channel_id).await;
    let persona_channel_id_i64 = i64::try_from(persona_channel_id.get()).unwrap_or(i64::MAX);
    let user_id_i64 = i64::try_from(msg.author.id.get()).unwrap_or(i64::MAX);

    // Capture the author's identity NOW (before any DB writes that need
    // to attribute to them). Picks guild nickname → global display name
    // → username, in priority order. `discord_users` is upserted so
    // later turns referring to this user resolve to current name +
    // avatar in the viewer; the chosen display name is also stamped on
    // the turn row below so historical attribution is durable.
    let display_name = best_display_name(msg).to_string();
    let avatar_hash = msg.author.avatar.map(|h| h.to_string());
    let prior_user = state.db.get_discord_user(user_id_i64).await?;
    let user_row = state
        .db
        .upsert_discord_user(
            user_id_i64,
            &msg.author.name,
            msg.author.global_name.as_deref(),
            avatar_hash.as_deref(),
        )
        .await?;
    let needs_avatar_fetch = match (&prior_user, &user_row.avatar_local_path) {
        // First time we see this user — fetch whatever they have (or
        // resolve to a default avatar).
        (None, _) => true,
        // Hash changed since last fetch → re-download.
        (Some(prev), _) if prev.avatar_hash != user_row.avatar_hash => true,
        // We never successfully wrote a file.
        (Some(_), None) => true,
        _ => false,
    };

    let resolved_persona_name = state
        .db
        .resolve_persona(
            conversation_id_for_persona,
            guild_id_for_persona,
            persona_channel_id_i64,
            user_id_i64,
        )
        .await?
        .unwrap_or_else(|| state.default_persona.clone());
    // If the stored persona name no longer exists in config (e.g. the
    // operator renamed/removed it), fall back to default rather than
    // panic. The user can fix the override later with /grok-persona.
    let (persona_name, persona) = match state.personas.get(&resolved_persona_name) {
        Some(p) => (resolved_persona_name, p),
        None => {
            tracing::warn!(
                stored = %resolved_persona_name,
                "persona resolved to a name not in current config; using default"
            );
            (
                state.default_persona.clone(),
                state
                    .personas
                    .get(&state.default_persona)
                    .expect("default_persona validated at startup"),
            )
        }
    };
    let Some(provider) = state.providers.get(&persona.provider) else {
        tracing::error!(
            persona = %persona_name,
            provider = persona.provider.as_str(),
            "no provider initialized for resolved persona; this should have failed validation"
        );
        return Err(BotError::Llm(grok_discord_bot_core::LlmError::Transport(
            format!(
                "persona `{persona_name}` references provider `{}` but no credentials are loaded",
                persona.provider.as_str()
            ),
        )));
    };

    let (conversation, is_new) = match existing {
        Some(c) => (c, false),
        None => {
            let conv = state
                .db
                .create_conversation(
                    msg.guild_id
                        .map(|g| i64::try_from(g.get()).unwrap_or(0))
                        .unwrap_or(0),
                    channel_id_i64,
                    user_id_i64,
                    i64::try_from(msg.id.get()).unwrap_or(i64::MAX),
                    &persona.model,
                    None,
                )
                .await?;
            (conv, true)
        }
    };
    tracing::info!(
        conversation = %conversation.id,
        is_new,
        persona = %persona_name,
        provider = persona.provider.as_str(),
        model = %persona.model,
        "turn: conversation resolved"
    );

    let user_content = resolve_user_mentions(
        &strip_mentions(&msg.content, state.bot_user_id),
        msg,
        state.bot_user_id,
    );

    // Persist any image attachments before recording context items so
    // every image gets its own `discord:msg:<id>:image:<i>` context row
    // for the viewer trace. Keep the original Discord CDN URL in memory
    // to pass to the LLM (it's still fresh; cheaper than base64).
    let saved_images = save_image_attachments(state, msg).await;

    // Resolve the per-turn media providers from the persona. A persona
    // that doesn't name an image/video provider (or names one with no
    // matching `[image.<kind>]` / `[video.<kind>]` credentials block)
    // simply doesn't get the corresponding tool — same as before, just
    // now decided per-persona instead of per-deployment. Resolved here
    // (before context building) so the composed system prompt can list
    // exactly the capabilities whose tools will be declared this turn.
    let image_provider: Option<AnyImageProvider> = persona
        .image_provider
        .and_then(|kind| state.image_providers.get(&kind).cloned());
    let video_provider: Option<AnyVideoProvider> = persona
        .video_provider
        .and_then(|kind| state.video_providers.get(&kind).cloned());

    // The system prompt = the persona's voice + a dynamically-built
    // operational block (build version, model tuple, the capabilities
    // actually enabled this turn) + any operator-global addendum. Built
    // per turn because persona/privacy/capabilities are only known after
    // resolution; see `compose_system_prompt`.
    let system_prompt = compose_system_prompt(
        persona,
        privacy_mode,
        image_provider.is_some(),
        video_provider.is_some(),
        state.app_version,
        state.extra_system_prompt.as_deref(),
    );

    let mut initial_context = build_context(
        state,
        msg,
        &conversation,
        is_new,
        &user_content,
        &display_name,
        privacy_mode,
        &system_prompt,
    )
    .await?;

    let next_pos = initial_context.last().map(|c| c.position + 1).unwrap_or(0);
    for (i, image) in saved_images.iter().enumerate() {
        initial_context.push(ContextItem {
            position: next_pos + i32::try_from(i).unwrap_or(0),
            source: format!("discord:msg:{}:image:{i}", msg.id),
            role: "user".to_string(),
            content: image.stored_uri.clone(),
            discord_message_id: Some(i64::try_from(msg.id.get()).unwrap_or(i64::MAX)),
        });
    }

    let turn = state
        .db
        .start_turn(
            conversation.id,
            i64::try_from(msg.id.get()).unwrap_or(i64::MAX),
            &user_content,
            user_id_i64,
            &display_name,
            state.app_version,
        )
        .await?;
    state.publish(conversation.id, EventKind::TurnStarted);
    if is_new {
        state.publish(conversation.id, EventKind::Created);
    }
    if needs_avatar_fetch {
        crate::avatars::spawn_fetch(Arc::clone(&state.app), user_id_i64);
    }
    // Stamp the persona on the turn *before* the agent runs so the
    // model used is recoverable even if a later step fails. The web
    // viewer picks this up via the `persona_name` column on `turns`.
    if let Err(err) = state.db.set_turn_persona(turn.id, &persona_name).await {
        tracing::warn!(
            turn = %turn.id,
            error = %err,
            "failed to stamp persona on turn row; continuing"
        );
    }
    // Snapshot the fully-composed system prompt for the viewer. Best-effort
    // (viewer-only data) — a failure here must not sink the turn.
    if let Err(err) = state
        .db
        .record_turn_system_prompt(turn.id, &system_prompt)
        .await
    {
        tracing::warn!(
            turn = %turn.id,
            error = %err,
            "failed to snapshot system prompt; continuing"
        );
    }
    tracing::info!(
        conversation = %conversation.id,
        turn = %turn.id,
        turn_index = turn.turn_index,
        context_items = initial_context.len(),
        images = saved_images.len(),
        "turn: started"
    );

    // Record the user's @mention as a message_link IMMEDIATELY, before
    // anything that could fail later. Thread continuation depends on
    // this link existing: when the bot auto-threads its reply, the
    // thread's channel_id equals this user message id, so an
    // in-thread @mention later must be able to look up the
    // conversation from this row even if the rest of the turn dies.
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

    // Persist only items that are NOVEL to this turn — the user's
    // @-mention, any Discord-quoted message, and saved image
    // attachments. The composed system prompt is snapshotted separately
    // (in `turn_system_prompts`, above) and prior-turn user/assistant
    // content is already stored verbatim in the `turns` table, so
    // re-stamping them into `context_items` every turn would just
    // duplicate data and grow the table quadratically with conversation
    // length.
    for item in initial_context
        .iter()
        .filter(|i| i.source.starts_with("discord:msg:"))
    {
        state.db.record_context_item(turn.id, item).await?;
    }
    state.publish(conversation.id, EventKind::ContextItemAdded);

    // Assemble the LLM-facing chat history from the context items, with
    // this turn's freshly-uploaded images attached via their live Discord
    // URLs. Extracted so the 🔄-retry path reuses identical assembly.
    let messages = assemble_messages(&initial_context, &saved_images, &state.web_base_url);

    // Resolve tools + executor, drive the agent loop, post exactly one
    // reply, and finalize the turn row. Shared with the retry path.
    run_turn_and_reply(
        state,
        &conversation,
        &turn,
        is_new,
        msg.channel_id,
        msg.id,
        &msg.content,
        privacy_mode,
        &persona_name,
        persona,
        provider,
        image_provider,
        video_provider,
        messages,
    )
    .await
}

/// Build the LLM-facing chat history from a turn's ordered context items.
///
/// - Text items map 1:1 to chat turns.
/// - Image items attach as [`TurnBlock::Image`] to the user turn they
///   belong to (`build_context` / the retry builder order each turn's
///   image rows immediately after its user-text row):
///   - `turn:<id>:image:*` are served from our own storage via
///     [`storage::to_public_url`] (the URL outlives Discord's expiring CDN
///     links). The retry path relabels *this* turn's uploads to this form
///     too, since the original Discord URL is long gone.
///   - `discord:msg:*:image:*` are skipped here — on the live gateway path
///     they're attached below from `saved_images`, also minted through
///     [`storage::to_public_url`] so the URL is byte-identical to the
///     `turn:*` form this same image takes on later turns (keeping xAI's
///     prompt cache matching). Falls back to the live Discord URL only when
///     no served URL can be minted.
///
/// A reference annotation lists every in-context image by its stable
/// `file://` id so the model can pass one to `generate_image`'s
/// `reference_images` to edit or restyle it.
fn assemble_messages(
    initial_context: &[ContextItem],
    saved_images: &[SavedImage],
    web_base_url: &str,
) -> Vec<ChatTurn> {
    let mut messages: Vec<ChatTurn> = Vec::new();
    // A `turn:<id>:reasoning` item carries the opaque {provider, data}
    // continuation blob for the assistant message that immediately
    // follows it; hold it until we build that assistant turn.
    let mut pending_reasoning: Option<TurnBlock> = None;
    for c in initial_context {
        if c.source.starts_with("turn:") && c.source.ends_with(":reasoning") {
            match serde_json::from_str::<serde_json::Value>(&c.content) {
                Ok(mut blob) => {
                    let provider_name = blob
                        .get("provider")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let data = blob
                        .get_mut("data")
                        .map(serde_json::Value::take)
                        .unwrap_or(serde_json::Value::Null);
                    pending_reasoning = Some(TurnBlock::Reasoning {
                        provider_name,
                        data,
                    });
                }
                Err(e) => tracing::warn!(error = %e, "skipping unparseable reasoning replay blob"),
            }
            continue;
        }
        if c.source.starts_with("turn:") && c.source.contains(":image:") {
            match (
                messages.last_mut(),
                storage::to_public_url(&c.content, web_base_url),
            ) {
                (Some(last), Some(url)) => last.blocks.push(TurnBlock::Image {
                    url,
                    mime_type: None,
                }),
                (_, None) => tracing::warn!(
                    uri = %c.content,
                    "skipping replay image: URI has no servable public URL"
                ),
                (None, _) => {}
            }
            continue;
        }
        if c.source.contains(":image:") {
            // This turn's upload — handled by the `saved_images` block below.
            continue;
        }
        let role = MessageRole::from_str_lossy(&c.role);
        let mut blocks: Vec<TurnBlock> = Vec::new();
        // Reasoning leads the assistant turn it was captured before.
        if role == MessageRole::Assistant
            && let Some(reasoning) = pending_reasoning.take()
        {
            blocks.push(reasoning);
        }
        blocks.push(TurnBlock::Text(c.content.clone()));
        messages.push(ChatTurn { role, blocks });
    }
    // Reference annotation: list every image currently in context by its
    // stable `file://` id. The pixels are already *visible* (as image
    // blocks), but vision blocks aren't quotable strings — the model needs
    // the ids in text to pass one to `generate_image`. `file://` is the
    // most robust reference: `generate_image` base64-encodes it from disk,
    // so editing works without our server being publicly reachable and
    // never trips over Discord's CDN expiry.
    let reference_lines: Vec<String> = initial_context
        .iter()
        .filter(|c| c.source.contains(":image:"))
        .map(|c| {
            let origin = if c.source.starts_with("turn:") {
                "from earlier in this conversation"
            } else {
                "attached to this message"
            };
            format!("- {} ({origin})", c.content)
        })
        .collect();
    if let Some(last) = messages.last_mut() {
        if !reference_lines.is_empty() {
            let annotation = format!(
                "\n\n[Images in this conversation you can edit or restyle with \
                 generate_image — pass the exact file:// id below as a \
                 reference_images entry (never invent paths). When you pass two \
                 or three references, refer to them in your prompt as <IMAGE_0>, \
                 <IMAGE_1>, … in the order you list them.]\n{}",
                reference_lines.join("\n")
            );
            match last.blocks.first_mut() {
                Some(TurnBlock::Text(text)) => text.push_str(&annotation),
                _ => last.blocks.insert(0, TurnBlock::Text(annotation)),
            }
        }
        // Vision for this turn's uploads: mint the same served URL the
        // cross-turn replay path produces from `stored_uri`, not the live
        // Discord link. The Discord URL carries expiring query params and
        // differs from the served form this image takes on later turns, so
        // reusing it here would bust xAI's cache prefix from the image
        // onward; a stable URL keeps the prefix matching. Costs us needing
        // `web.base_url` publicly reachable (as prior-turn images already
        // do). Fall back to the live URL only when none can be minted.
        // (On retry this slice is empty — those uploads were relabeled
        // `turn:*:image:*` and replayed above.)
        for image in saved_images {
            let url = storage::to_public_url(&image.stored_uri, web_base_url)
                .unwrap_or_else(|| image.live_url.clone());
            last.blocks.push(TurnBlock::Image {
                url,
                mime_type: image.mime_type.clone(),
            });
        }
    }
    messages
}

/// Keeps the Discord "chudbot is typing…" indicator alive in a channel for
/// the duration of a turn. Discord's indicator lasts ~10s per trigger, so a
/// background task re-pings every 8s until the guard is dropped (turn done or
/// early `?` return). Pings are best-effort: a failed trigger is a cosmetic
/// blip and must never disturb the turn's single user-facing failure path.
struct TypingGuard(tokio::task::JoinHandle<()>);

impl TypingGuard {
    fn start(http: Arc<HttpClient>, channel_id: Id<ChannelMarker>) -> Self {
        Self(tokio::spawn(async move {
            // `interval` fires its first tick immediately, so the indicator
            // shows right away with no startup delay.
            let mut tick =
                tokio::time::interval(std::time::Duration::from_secs(8));
            loop {
                tick.tick().await;
                let _ = http.create_typing_trigger(channel_id).await;
            }
        }))
    }
}

impl Drop for TypingGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Shared tail for running a turn: build tools + executor, drive the agent
/// loop, persist tool calls, post **exactly one** user-facing reply, and
/// finalize the turn row. Used by both the gateway create path
/// (`process`) and the 🔄-reaction retry path (`handle_reaction`).
///
/// Returns the [`TurnOutcome`] so the caller sets the reaction emoji
/// WITHOUT posting another message — this is the single seam that owns all
/// user-facing turn output, which is why a failed turn now posts one
/// message (not the old two) and is marked `failed` (not `completed`).
#[allow(clippy::too_many_arguments)]
#[tracing::instrument(
    name = "turn",
    skip_all,
    fields(
        conversation = %conversation.id,
        turn = %turn.id,
        persona = %persona_name,
        model = %persona.model,
    )
)]
async fn run_turn_and_reply(
    state: &State,
    conversation: &Conversation,
    turn: &Turn,
    is_new: bool,
    channel_id: Id<ChannelMarker>,
    reply_to: Id<MessageMarker>,
    user_content: &str,
    privacy_mode: &PrivacyMode,
    persona_name: &str,
    persona: &Persona,
    provider: &AnyProvider,
    image_provider: Option<AnyImageProvider>,
    video_provider: Option<AnyVideoProvider>,
    messages: Vec<ChatTurn>,
) -> Result<TurnOutcome, BotError> {
    // Tools available this turn:
    //   - fetch_messages: every mode except ConversationOnly
    //   - generate_image / generate_video: only when the resolved persona
    //     names a backend that's actually configured
    //   - post_status_message: always available
    let tools = build_tool_definitions(
        privacy_mode,
        image_provider.is_some(),
        video_provider.is_some(),
    );

    let executor = BotToolExecutor {
        http: Arc::clone(&state.http),
        db: state.db.clone(),
        bot_user_id: state.bot_user_id,
        default_channel_id: channel_id,
        user_msg_id: reply_to,
        guild_id: conversation.discord_guild_id,
        conversation_id: conversation.id,
        turn_id: turn.id,
        privacy_mode: privacy_mode.clone(),
        image_provider,
        video_provider,
        images_dir: state.storage.images_dir.clone(),
        videos_dir: state.storage.videos_dir.clone(),
        last_status_text: Mutex::new(None),
    };

    // Show "chudbot is typing…" while the agent loop runs. Dropped at the end
    // of this fn (or on any early return) which aborts the re-trigger task.
    let _typing = TypingGuard::start(Arc::clone(&state.http), channel_id);

    let agent_run: AgentRun = run_agent(
        provider,
        persona.model.clone(),
        messages,
        tools,
        &executor,
        &NoopObserver,
        true, // server-side web search always enabled
        MAX_OUTPUT_TOKENS,
        persona.temperature,
        persona.top_p,
        ProviderOptions {
            xai: persona.xai.clone(),
            anthropic: persona.anthropic.clone(),
        },
        // Stable cache-routing key: the conversation UUID. Every
        // agent-loop iteration and every later turn re-sends the
        // growing prefix, so pinning all of them to one
        // `prompt_cache_key` keeps xAI's prefix cache hitting.
        Some(conversation.id.to_string()),
        MAX_AGENT_ITERATIONS,
    )
    .await;

    // Persist all tool calls (server + client) in execution order — even
    // on a failed run, so the trace shows the failed generate_image etc.
    for (i, tc) in agent_run.tool_calls.iter().enumerate() {
        state
            .db
            .record_tool_call(turn.id, i32::try_from(i).unwrap_or(0), tc)
            .await?;
        state.publish(conversation.id, EventKind::ToolCallRecorded);
    }

    // Collect any media the agent generated this turn for upload as
    // Discord attachments on the outgoing reply.
    let generated_attachments = collect_generated_attachments(
        &state.storage.images_dir,
        &state.storage.videos_dir,
        &agent_run.tool_calls,
    )
    .await;

    // --- Safety refusal: a media tool refused, OR the chat call itself
    // refused (a 403 whose body trips the safety matcher). No user-facing
    // message — mark the turn failed and react ❓. Checked first so a
    // safety 403 never gets a ❌ + raw-error reply.
    let media_safety_refused = agent_run.tool_calls.iter().any(|tc| {
        matches!(tc.tool_name.as_str(), "generate_image" | "generate_video")
            && tc
                .response
                .get("error")
                .and_then(|v| v.as_str())
                .map(body_indicates_safety_refusal)
                .unwrap_or(false)
    });
    let chat_safety_refused = agent_run
        .error
        .as_deref()
        .map(body_indicates_safety_refusal)
        .unwrap_or(false);
    if media_safety_refused || chat_safety_refused {
        tracing::info!("turn refused by upstream safety; reacting ❓");
        state
            .db
            .fail_turn(turn.id, "refused by upstream safety")
            .await
            .ok();
        state.publish(conversation.id, EventKind::TurnUpdated);
        return Ok(TurnOutcome::Refused);
    }

    // --- Detect "media generation attempted but produced no output" — a
    // generate_image call with no image_uri, or a generate_video with no
    // video_uri, in any response.
    let attempted_image_gen = agent_run
        .tool_calls
        .iter()
        .any(|tc| tc.tool_name == "generate_image");
    let attempted_video_gen = agent_run
        .tool_calls
        .iter()
        .any(|tc| tc.tool_name == "generate_video");
    let image_gen_failed = attempted_image_gen
        && !agent_run
            .tool_calls
            .iter()
            .any(|tc| tc.tool_name == "generate_image" && tc.response.get("image_uri").is_some());
    let video_gen_failed = attempted_video_gen
        && !agent_run
            .tool_calls
            .iter()
            .any(|tc| tc.tool_name == "generate_video" && tc.response.get("video_uri").is_some());
    let media_gen_failed = image_gen_failed || video_gen_failed;
    let failure_label = match (image_gen_failed, video_gen_failed) {
        (true, true) => "Image and video generation",
        (true, false) => "Image generation",
        (false, true) => "Video generation",
        (false, false) => "Media generation",
    };

    let agent_loop_error = agent_run.error.as_deref();
    let have_media = !generated_attachments.is_empty();
    // What the model actually said (may be empty). If media generated but
    // the loop died before the closing line, fall back to a minimal ack.
    let final_content: String = if !agent_run.content.is_empty() {
        agent_run.content.clone()
    } else if have_media && agent_loop_error.is_some() {
        "Here you go.".to_string()
    } else {
        agent_run.content.clone()
    };

    // A failure is: media attempted-but-empty, OR a bare loop error with
    // no media to salvage. (Media delivered despite a late transient error
    // is still a success — the user got their picture/video.)
    let is_failure = media_gen_failed || (agent_loop_error.is_some() && !have_media);

    let answer_text = if media_gen_failed {
        format!("⚠️ {failure_label} failed.\n\n{final_content}")
    } else if agent_loop_error.is_some() && !have_media {
        let snippet = agent_loop_error
            .unwrap_or_default()
            .chars()
            .take(200)
            .collect::<String>();
        format!("⚠️ {snippet}")
    } else {
        // Success, or media delivered despite a late error.
        if let Some(err) = agent_loop_error {
            tracing::warn!(
                error = %err,
                "agent loop errored after media generation succeeded; \
                 short reply + attachment"
            );
        }
        final_content.clone()
    };

    // Post exactly one reply.
    let formatted = format_reply(&answer_text, is_new, conversation, &state.web_base_url);
    let chunks = assemble_chunks(&formatted);
    let total_rendered_lines: usize = chunks.iter().map(|c| rendered_line_count(c)).sum();
    let threaded = should_open_thread(is_new, &chunks);
    let reply_msgs = post_reply_chunks(
        state,
        channel_id,
        reply_to,
        user_content,
        &chunks,
        is_new,
        &generated_attachments,
        conversation,
        turn.id,
    )
    .await?;
    let reply_msg = reply_msgs
        .last()
        .cloned()
        .expect("post_reply_chunks always returns at least one message");
    let reply_msg_id = i64::try_from(reply_msg.id.get()).unwrap_or(i64::MAX);
    tracing::info!(
        reply_msg = %reply_msg.id,
        threaded,
        chunks = reply_msgs.len(),
        reply_chars = agent_run.content.len(),
        rendered_lines = total_rendered_lines,
        tool_calls = agent_run.tool_calls.len(),
        is_failure,
        "turn: reply posted"
    );

    // Finalize the turn row. Failure path stores the REAL underlying error
    // (not the cosmetic ⚠️ text) plus any salvaged content, so the viewer
    // shows the error in red AND whatever the model managed to say.
    if is_failure {
        let real_error = if media_gen_failed {
            match agent_loop_error {
                Some(err) => format!("{failure_label} produced no output; {err}"),
                None => format!("{failure_label} produced no output"),
            }
        } else {
            agent_loop_error.unwrap_or("unknown error").to_string()
        };
        let salvaged = (!final_content.is_empty()).then_some(final_content.as_str());
        state
            .db
            .fail_turn_with_reply(turn.id, &real_error, salvaged, Some(reply_msg_id))
            .await?;
        state.publish(conversation.id, EventKind::TurnUpdated);
        // Add a 🔄 affordance to our own failure message so the user can
        // one-click retry. `handle_reaction` resolves the reacted message
        // back to this turn (the reply is message-linked), and the bot's
        // own reaction here is self-ignored — it just sits there clickable
        // until a human adds theirs. Reacts on the reply's actual channel
        // so it's correct even in the rare threaded case.
        let _ = state
            .http
            .create_reaction(
                reply_msg.channel_id,
                reply_msg.id,
                &RequestReactionType::Unicode { name: RETRY_EMOJI },
            )
            .await;
        tracing::warn!(error = %real_error, "turn: marked failed");
        return Ok(TurnOutcome::Failed);
    }

    state
        .db
        .complete_turn(
            turn.id,
            &agent_run.content,
            reply_msg_id,
            // Opaque, provider-tagged reasoning continuation — replayed
            // before this turn's answer on later turns to keep the prompt
            // cache warm. NULL for non-reasoning providers/models.
            agent_run.provider_state.as_ref(),
        )
        .await?;
    state.publish(conversation.id, EventKind::TurnUpdated);

    // First completed turn on a conversation → schedule background title
    // generation. The task drops itself if a title already exists, so this
    // is safe to fire from a retry of turn 0 too.
    if turn.turn_index == 0 {
        crate::titles::spawn_generate(
            Arc::clone(&state.app),
            conversation.id,
            persona_name.to_string(),
        );
    }

    Ok(TurnOutcome::Completed)
}

/// Gate an incoming reaction: act only on the 🔄 retry emoji (and never on
/// our own reactions), resolve the reacted message to its turn, and hand
/// off to [`retry_turn`]. Everything else is a silent no-op.
#[tracing::instrument(
    skip_all,
    fields(message_id = %reaction.message_id, user = %reaction.user_id)
)]
async fn handle_reaction(state: Arc<State>, reaction: GatewayReaction) {
    let is_retry = matches!(
        &reaction.emoji,
        EmojiReactionType::Unicode { name } if name == RETRY_EMOJI
    );
    if !is_retry || reaction.user_id == state.bot_user_id {
        return;
    }

    let message_id = i64::try_from(reaction.message_id.get()).unwrap_or(i64::MAX);
    let turn_id = match state.db.lookup_turn_by_message(message_id).await {
        Ok(Some(id)) => id,
        // Reaction on a message that isn't part of any turn (or a DB
        // hiccup) — nothing to retry.
        Ok(None) => return,
        Err(err) => {
            tracing::warn!(error = %err, "retry: turn lookup failed");
            return;
        }
    };

    if let Err(err) = retry_turn(&state, turn_id, reaction.channel_id).await {
        tracing::error!(error = %err, %turn_id, "retry: failed to re-run turn");
    }
}

/// Re-run a previously-failed turn (triggered by a 🔄 reaction). Confirms
/// it's the latest failed turn (atomic), cleans up the prior failed reply,
/// reconstructs the LLM history from the DB (no live gateway `Message`),
/// and drives the shared [`run_turn_and_reply`] tail. Manages the reaction
/// on the user's original message (❌ → 👀 → ✅/❌/❓).
#[tracing::instrument(name = "retry", skip_all, fields(turn = %turn_id, conversation = tracing::field::Empty))]
async fn retry_turn(
    state: &State,
    turn_id: Uuid,
    channel_id: Id<ChannelMarker>,
) -> Result<(), BotError> {
    let Some(turn) = state.db.get_turn(turn_id).await? else {
        return Ok(());
    };
    let Some(conversation) = state.db.get_conversation(turn.conversation_id).await? else {
        return Ok(());
    };
    tracing::Span::current().record("conversation", tracing::field::display(conversation.id));

    // Atomic gate: only the LATEST turn, and only while it's `failed`.
    // The same statement flips it to `pending`, so a double 🔄 or a stale
    // reaction on an older turn is a silent no-op.
    if !state
        .db
        .reset_turn_for_retry(turn.id, conversation.id)
        .await?
    {
        tracing::info!(
            status = %turn.status,
            "retry: turn not eligible (not failed, or not the latest turn); ignoring"
        );
        return Ok(());
    }
    tracing::info!("retry: re-running failed turn");
    state.publish(conversation.id, EventKind::TurnUpdated);

    let working = RequestReactionType::Unicode { name: "👀" };
    let done = RequestReactionType::Unicode { name: "✅" };
    let failed = RequestReactionType::Unicode { name: "❌" };
    let refused = RequestReactionType::Unicode { name: "❓" };

    // The user's original message — what we reply to and react on.
    let Some(user_msg_id) = u64::try_from(turn.user_discord_message_id)
        .ok()
        .and_then(Id::<MessageMarker>::new_checked)
    else {
        tracing::warn!("retry: turn has no valid user message id; aborting");
        state
            .db
            .fail_turn(turn.id, "retry: invalid user message id")
            .await
            .ok();
        state.publish(conversation.id, EventKind::TurnUpdated);
        return Ok(());
    };

    // Re-resolve everything the turn needs — same as `process` does for a
    // live mention, but sourced from the DB. All fallible DB reads happen
    // BEFORE we touch Discord, so a propagated error never leaves a
    // dangling 👀.
    let guild_id = conversation.discord_guild_id;
    let guild_opt = (guild_id != 0).then_some(guild_id);
    let privacy_mode = state
        .db
        .guild_privacy_mode_or(guild_id, &state.default_privacy)
        .await
        .unwrap_or_else(|err| {
            tracing::warn!(error = %err, "retry: privacy mode load failed; using default");
            state.default_privacy.clone()
        });

    let user_id = turn.discord_user_id.unwrap_or(0);
    let persona_channel = parent_channel_id(&state.http, channel_id).await;
    let persona_channel_i64 = i64::try_from(persona_channel.get()).unwrap_or(i64::MAX);
    let resolved_persona_name = state
        .db
        .resolve_persona(
            Some(conversation.id),
            guild_opt,
            persona_channel_i64,
            user_id,
        )
        .await?
        .unwrap_or_else(|| state.default_persona.clone());
    let (persona_name, persona) = match state.personas.get(&resolved_persona_name) {
        Some(p) => (resolved_persona_name, p),
        None => (
            state.default_persona.clone(),
            state
                .personas
                .get(&state.default_persona)
                .expect("default_persona validated at startup"),
        ),
    };
    let Some(provider) = state.providers.get(&persona.provider) else {
        tracing::error!(
            persona = %persona_name,
            provider = persona.provider.as_str(),
            "retry: no provider initialized for resolved persona"
        );
        state
            .db
            .fail_turn(turn.id, "retry: persona's provider not configured")
            .await
            .ok();
        state.publish(conversation.id, EventKind::TurnUpdated);
        let _ = state
            .http
            .create_reaction(channel_id, user_msg_id, &failed)
            .await;
        return Ok(());
    };

    let image_provider = persona
        .image_provider
        .and_then(|kind| state.image_providers.get(&kind).cloned());
    let video_provider = persona
        .video_provider
        .and_then(|kind| state.video_providers.get(&kind).cloned());

    let system_prompt = compose_system_prompt(
        persona,
        &privacy_mode,
        image_provider.is_some(),
        video_provider.is_some(),
        state.app_version,
        state.extra_system_prompt.as_deref(),
    );
    // Re-stamp persona + system-prompt snapshot for the viewer (overwrites
    // the failed attempt's). Best-effort.
    let _ = state.db.set_turn_persona(turn.id, &persona_name).await;
    let _ = state
        .db
        .record_turn_system_prompt(turn.id, &system_prompt)
        .await;

    // Rebuild the LLM history from the DB: system prompt + prior completed
    // turns + this turn's own persisted novel items.
    let mut context: Vec<ContextItem> = Vec::new();
    let mut pos: i32 = 0;
    push_item(
        &mut context,
        &mut pos,
        "system".to_string(),
        "system",
        system_prompt,
        None,
    );
    history_context_items(state, &conversation, &mut context, &mut pos).await?;
    for item in state.db.load_turn_context(turn.id).await? {
        // Relabel this turn's image uploads `discord:msg:<id>:image:<i>` →
        // `turn:<turnid>:image:<i>` so `assemble_messages` serves them from
        // our own storage — the original Discord CDN URL has long expired.
        let source = match item.source.split_once(":image:") {
            Some((_, idx)) => format!("turn:{}:image:{idx}", turn.id),
            None => item.source,
        };
        push_item(
            &mut context,
            &mut pos,
            source,
            &item.role,
            item.content,
            item.discord_message_id,
        );
    }
    let messages = assemble_messages(&context, &[], &state.web_base_url);

    // Wipe the failed attempt's tool-call rows so the re-run's fresh rows
    // don't collide on (turn_id, ordinal). Last fallible op before we touch
    // Discord.
    state.db.delete_turn_tool_calls(turn.id).await?;

    // Discord side effects: drop the prior (failed) reply message(s) so the
    // retry doesn't stack a second reply, clear the ❌, and show 👀.
    match state.db.assistant_message_ids_for_turn(turn.id).await {
        Ok(ids) => {
            for mid in ids {
                if let Some(id) = u64::try_from(mid)
                    .ok()
                    .and_then(Id::<MessageMarker>::new_checked)
                {
                    let _ = state.http.delete_message(channel_id, id).await;
                }
            }
        }
        Err(err) => {
            tracing::warn!(error = %err, "retry: couldn't list prior reply messages");
        }
    }
    let _ = state
        .http
        .delete_current_user_reaction(channel_id, user_msg_id, &failed)
        .await;
    let _ = state
        .http
        .create_reaction(channel_id, user_msg_id, &working)
        .await;

    let user_content = turn.user_content.clone();
    let result = run_turn_and_reply(
        state,
        &conversation,
        &turn,
        false, // is_new: a retry replies inline (no viewer-URL footer / auto-thread)
        channel_id,
        user_msg_id,
        &user_content,
        &privacy_mode,
        &persona_name,
        persona,
        provider,
        image_provider,
        video_provider,
        messages,
    )
    .await;

    let _ = state
        .http
        .delete_current_user_reaction(channel_id, user_msg_id, &working)
        .await;
    match &result {
        Ok(TurnOutcome::Completed) => {
            let _ = state
                .http
                .create_reaction(channel_id, user_msg_id, &done)
                .await;
        }
        Ok(TurnOutcome::Refused) => {
            let _ = state
                .http
                .create_reaction(channel_id, user_msg_id, &refused)
                .await;
        }
        Ok(TurnOutcome::Failed) => {
            let _ = state
                .http
                .create_reaction(channel_id, user_msg_id, &failed)
                .await;
        }
        Err(err) => {
            // The run errored before it could finalize the turn (and post
            // its own message). Restore the `failed` status so the turn
            // stays retryable, and react ❌.
            tracing::error!(error = %err, "retry: run failed before finalizing");
            state.db.fail_turn(turn.id, &err.to_string()).await.ok();
            state.publish(conversation.id, EventKind::TurnUpdated);
            let _ = state
                .http
                .create_reaction(channel_id, user_msg_id, &failed)
                .await;
        }
    }
    Ok(())
}

/// Map a Discord channel id to the "user-facing" channel id that
/// persona selections key off of. For threads, that's the parent
/// channel — operators set `/grok-persona scope:channel` expecting it
/// to apply to the visible channel and all threads under it. For
/// non-thread channels, it's the channel itself.
///
/// Falls back to the raw channel id on any lookup error so we never
/// block a turn on a transient Discord API hiccup.
async fn parent_channel_id(http: &HttpClient, channel_id: Id<ChannelMarker>) -> Id<ChannelMarker> {
    let Ok(resp) = http.channel(channel_id).await else {
        return channel_id;
    };
    let Ok(channel) = resp.model().await else {
        return channel_id;
    };
    match channel.kind {
        ChannelType::AnnouncementThread
        | ChannelType::PublicThread
        | ChannelType::PrivateThread => channel.parent_id.unwrap_or(channel_id),
        _ => channel_id,
    }
}

/// Look up whether this @mention extends an existing conversation,
/// without creating one. Returns `None` when the message is a fresh
/// root and the caller needs to create a new conversation row.
///
/// Lookup order:
///   1. Discord reply parent — if the user replied to a bot/user
///      message we already tracked, that conversation continues.
///   2. Channel id — when the bot opened a thread for an answer, the
///      thread's channel id matches the user's original message id and
///      lives in `message_links`; @mentions inside the thread should
///      continue the same conversation.
async fn lookup_existing_conversation(
    state: &State,
    msg: &Message,
) -> Result<Option<Conversation>, BotError> {
    if let Some(referenced) = &msg.referenced_message {
        let parent_id = i64::try_from(referenced.id.get()).unwrap_or(i64::MAX);
        if let Some(conv_id) = state.db.lookup_conversation_by_message(parent_id).await?
            && let Some(conv) = state.db.get_conversation(conv_id).await?
        {
            return Ok(Some(conv));
        }
    }

    let channel_id = i64::try_from(msg.channel_id.get()).unwrap_or(i64::MAX);
    if let Some(conv_id) = state.db.lookup_conversation_by_message(channel_id).await?
        && let Some(conv) = state.db.get_conversation(conv_id).await?
    {
        return Ok(Some(conv));
    }

    Ok(None)
}

/// Append a conversation's prior **completed** turns (user text, replayed
/// images, assistant text) to `items`, advancing `pos`. Factored out of
/// [`build_context`] so the 🔄-retry path can reconstruct an identical
/// history without a live Discord [`Message`]. Replay images are capped to
/// the most recent [`MAX_REPLAYED_IMAGES`] (oldest dropped, logged).
async fn history_context_items(
    state: &State,
    conversation: &Conversation,
    items: &mut Vec<ContextItem>,
    pos: &mut i32,
) -> Result<(), BotError> {
    let history = state.db.load_conversation_history(conversation.id).await?;

    // Replayable images from earlier turns (user-uploaded + model-
    // generated), capped to the most recent N. Capping from the
    // chronological front drops the OLDEST first. Grouped by turn so
    // each image re-attaches to the user message it belongs to; the
    // message-assembly step turns these rows into `Image` blocks.
    let mut replay = state
        .db
        .load_conversation_image_uris(conversation.id)
        .await?;
    if replay.len() > MAX_REPLAYED_IMAGES {
        let dropped = replay.len() - MAX_REPLAYED_IMAGES;
        tracing::info!(
            conversation = %conversation.id,
            dropped,
            cap = MAX_REPLAYED_IMAGES,
            "replaying only the most recent images; older ones dropped from context"
        );
        replay.drain(0..dropped);
    }
    let mut images_by_turn: HashMap<Uuid, Vec<String>> = HashMap::new();
    for img in replay {
        images_by_turn.entry(img.turn_id).or_default().push(img.uri);
    }

    for turn in history {
        // Prefix prior turns' user text with the historical display
        // name pinned on the turn row. Falls back to "user" for
        // legacy turns predating the identity-tracking feature.
        let prior_name = turn.discord_user_name.as_deref().unwrap_or("user");
        push_item(
            items,
            pos,
            format!("turn:{}:user", turn.id),
            "user",
            format!("[{prior_name}]: {}", turn.user_content),
            Some(turn.user_discord_message_id),
        );
        // Re-attach this turn's surviving images immediately after
        // its user text. `content` carries the stored `file://` URI;
        // message assembly resolves it to a served URL. These rows
        // are NOT persisted (only `discord:msg:` items are), so they
        // never feed back into `load_conversation_image_uris`.
        if let Some(uris) = images_by_turn.get(&turn.id) {
            for (i, uri) in uris.iter().enumerate() {
                push_item(
                    items,
                    pos,
                    format!("turn:{}:image:{i}", turn.id),
                    "user",
                    uri.clone(),
                    None,
                );
            }
        }
        if let Some(answer) = turn.assistant_content {
            // Replay this turn's opaque reasoning continuation (if any)
            // immediately before its answer — the position the provider
            // emitted it, and where the cache prefix expects it. Carried
            // as a transient item (source not `discord:msg:`, so never
            // persisted); `assemble_messages` decodes the {provider,data}
            // blob and attaches it to the reconstructed assistant turn.
            if let Some(state) = &turn.provider_state {
                push_item(
                    items,
                    pos,
                    format!("turn:{}:reasoning", turn.id),
                    "assistant",
                    state.to_string(),
                    None,
                );
            }
            push_item(
                items,
                pos,
                format!("turn:{}:assistant", turn.id),
                "assistant",
                answer,
                turn.assistant_discord_message_id,
            );
        }
    }
    Ok(())
}

/// Assemble the initial prompt for the agent loop. The model can
/// always pull more channel history on demand via `fetch_messages`, so
/// this only needs to include:
/// - the system prompt;
/// - prior turns of the conversation (when continuing);
/// - the Discord-reply-quoted message, gated by the privacy mode;
/// - the user's current `@`-mention.
#[allow(clippy::too_many_arguments)]
async fn build_context(
    state: &State,
    msg: &Message,
    conversation: &Conversation,
    is_new: bool,
    user_content: &str,
    user_display_name: &str,
    privacy_mode: &PrivacyMode,
    system_prompt: &str,
) -> Result<Vec<ContextItem>, BotError> {
    let mut items = Vec::new();
    let mut pos: i32 = 0;

    push_item(
        &mut items,
        &mut pos,
        "system".to_string(),
        "system",
        system_prompt.to_string(),
        None,
    );

    if !is_new {
        // Prior completed turns + their replayable images. Shared with
        // the retry path so the two reconstruct identical history.
        history_context_items(state, conversation, &mut items, &mut pos).await?;
    } else if let Some(referenced) = &msg.referenced_message
        && !referenced.author.bot
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

    push_item(
        &mut items,
        &mut pos,
        format!("discord:msg:{}", msg.id),
        "user",
        format!("[{user_display_name}]: {user_content}"),
        Some(i64::try_from(msg.id.get()).unwrap_or(i64::MAX)),
    );

    Ok(items)
}

/// Build the full system prompt for a resolved turn:
///   1. the persona's own `system_prompt` (its voice — operator-authored);
///   2. a dynamically-generated operational block — build version, the
///      model actually answering, and a one-line pointer to each
///      capability whose tool is declared *this* turn (the HOW stays in
///      the tool descriptions; these lines just make the model aware the
///      capability exists), plus cross-cutting conventions;
///   3. any global operator addendum from config (`extra_system_prompt`,
///      e.g. the Discord ToS).
///
/// The operational block and operator policy are placed AFTER the persona
/// so their rules win on conflict and survive an adversarial or careless
/// persona prompt. Output is stable within a deployment (build version),
/// persona, and privacy mode, so it caches cleanly.
fn compose_system_prompt(
    persona: &Persona,
    privacy_mode: &PrivacyMode,
    image_enabled: bool,
    video_enabled: bool,
    version_number: i32,
    extra: Option<&str>,
) -> String {
    // One pointer line per capability whose tool is declared this turn.
    // Order/conditions mirror `build_tool_definitions` + the always-on
    // server-side web search, so the prompt never advertises a tool the
    // model wasn't given.
    let mut capabilities = vec!["- Web search: look up current information when it helps."];
    if image_enabled {
        capabilities.push("- Image generation & editing: via the generate_image tool.");
    }
    if video_enabled {
        capabilities.push("- Video generation: via the generate_video tool.");
    }
    if !matches!(privacy_mode, PrivacyMode::ConversationOnly) {
        capabilities.push("- Recent channel messages: via the fetch_messages tool.");
    }

    let mut out = String::new();
    if let Some(extra) = extra.map(str::trim).filter(|s| !s.is_empty()) {
        out.push_str("— Operator policy —\n");
        out.push_str(extra);
        out.push_str("\n\n");
    }
    out.push_str("— Operational context (always applies; not part of your persona) —\n");
    out.push_str(&format!(
        "Bot build: v{} ({}). You are answering as model `{}` via the {} API.\n\n",
        version_number,
        crate::VERSION,
        persona.model,
        persona.provider.as_str(),
    ));
    out.push_str("Capabilities available this turn:\n");
    out.push_str(&capabilities.join("\n"));
    out.push_str(
        "\n\nConventions:\n\
         - Some messages carry bracketed notes we insert (e.g. \"[Quoted message from …]\" or \
         \"[Images in this conversation you can edit …]\"). They are context for you, not the \
         user's own words — act on them, but never echo the bracketed text and never surface \
         internal ids, URLs, or file:// paths to the user.\n\
         - Mentioning people: to ping or notify someone, emit the literal token `<@USER_ID>` — \
         Discord renders it as a clickable mention. You learn a user's ID from context (a person \
         shows up as `Name (<@123…>)`) and from fetch_messages results (each `author_id`). Prefer \
         pinging a person with their `<@ID>` over typing their bare name whenever you know the ID \
         and are addressing them or calling them out. This wrapped mention token is the ONE \
         identifier you should output verbatim — it is the deliberate exception to the no-raw-ids \
         rule above; never expose the digits any other way (no plain `@123…`, no `(<@123…>)`).\n\
         - For anything that takes more than a moment (image or video generation), call \
         post_status_message in the same response so the user sees progress.\n\
         - Write for Discord: concise, minimal markdown; don't re-link or re-describe media you \
         have already attached.",
    );

    out.push_str("\n\n");
    out.push_str(persona.system_prompt.trim_end());

    out
}

/// A single image attachment that's been persisted to local disk, ready
/// to hand to the LLM this turn as a served (cache-stable) URL.
struct SavedImage {
    /// `file://images/<uuid>.<ext>` — recorded in `context_items`. The
    /// LLM sees this minted into a served URL (stable across turns, so the
    /// prompt cache keeps matching).
    stored_uri: String,
    /// Original Discord CDN URL. Only a fallback now, for the rare upload
    /// whose `stored_uri` can't be minted into a served URL. Don't store;
    /// these signed URLs expire after ~24h.
    live_url: String,
    /// `content_type` from the Discord attachment metadata, if any.
    mime_type: Option<String>,
}

/// Download every image-typed attachment on `msg` to the configured
/// images dir. Failures are logged and skipped — a broken attachment
/// shouldn't fail the whole reply.
async fn save_image_attachments(state: &State, msg: &Message) -> Vec<SavedImage> {
    let mut out = Vec::new();
    for att in &msg.attachments {
        if !looks_like_image(att.content_type.as_deref(), &att.filename) {
            continue;
        }
        match storage::save_image_from_url(
            &state.download_http,
            &att.url,
            att.content_type.as_deref(),
            &state.storage.images_dir,
        )
        .await
        {
            Ok(stored_uri) => {
                tracing::info!(
                    uri = %stored_uri,
                    filename = %att.filename,
                    size = att.size,
                    "saved image attachment"
                );
                out.push(SavedImage {
                    stored_uri,
                    live_url: att.url.clone(),
                    mime_type: att.content_type.clone(),
                });
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    url = %att.url,
                    filename = %att.filename,
                    "failed to persist image attachment; skipping"
                );
            }
        }
    }
    out
}

fn looks_like_image(content_type: Option<&str>, filename: &str) -> bool {
    if let Some(ct) = content_type
        && ct.starts_with("image/")
    {
        return true;
    }
    let ext = filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "heic" | "heif"
    )
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
            let channel_as_msg = i64::try_from(referenced.channel_id.get()).unwrap_or(i64::MAX);
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

/// Tool definitions exposed to the model for this turn.
///   - `fetch_messages` — declared except in `ConversationOnly` privacy
///     mode, which deliberately doesn't reach beyond the conversation.
///   - `generate_image` — declared only when an xAI key is configured
///     (the only image-gen backend we support today).
fn build_tool_definitions(
    mode: &PrivacyMode,
    image_gen_available: bool,
    video_gen_available: bool,
) -> Vec<ToolDefinition> {
    let mut tools = Vec::new();
    if !matches!(mode, PrivacyMode::ConversationOnly) {
        tools.push(fetch_messages_tool());
    }
    if image_gen_available {
        tools.push(generate_image_tool());
    }
    if video_gen_available {
        tools.push(generate_video_tool());
    }
    // post_status_message is always available. The model is meant to
    // call it in the same response as a slow tool (generate_video
    // especially) so the user gets a status update before the wait.
    tools.push(post_status_message_tool());
    tools
}

fn post_status_message_tool() -> ToolDefinition {
    ToolDefinition {
        name: "post_status_message".to_string(),
        description: "Post a short interim status message into the Discord \
channel as a reply to the user. Call this whenever you're about to do \
something that takes more than a few seconds — especially generate_video \
(60-120s) and generate_image (3-10s). The status reaches the user \
immediately; the slow tool then runs.

Best practice: include `post_status_message` in the SAME RESPONSE as the \
slow tool. Both tool calls fire in one round-trip, so the user sees the \
status before the wait starts.

Examples of good status messages:
- \"Working on your video, takes about a minute…\"
- \"Cooking up that image for you…\"
- \"Searching the web for current info…\"

Don't use it for the final answer (the final answer is plain assistant \
text). Don't spam it — one message per long step is plenty."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["text"],
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Plain-text message body. Discord markdown is fine. Max ~1900 chars."
                }
            },
            "additionalProperties": false
        }),
    }
}

fn fetch_messages_tool() -> ToolDefinition {
    ToolDefinition {
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
    }
}

fn generate_video_tool() -> ToolDefinition {
    ToolDefinition {
        name: "generate_video".to_string(),
        description: "Generate a short video with xAI's Grok Imagine Video \
model. Synchronous: submits the job, polls until done, returns the saved \
video URI. Typical wait is ~60-120 seconds.

**ALWAYS call `post_status_message` in the SAME RESPONSE as this tool** \
so the user sees \"Working on your video, takes about a minute…\" right \
away. Both tool calls fire in one round-trip — the status posts \
immediately, then this tool blocks until the video is ready.

When the call returns successfully, the bot auto-attaches the video to \
your final reply. DO NOT include placeholders like \"[video attached]\", \
\"(see attached)\", or link to the URI — just write a short natural \
message; the user sees the actual video file under it.

For image-to-video, pass an EXACT image URL in `image_url` (same rules as \
generate_image — never invent paths). Max duration 15s. 480p is cheap; \
720p costs more."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string"},
                "image_url": {
                    "type": "string",
                    "description": "Optional image URL/URI to animate from."
                },
                "duration_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 15
                },
                "aspect_ratio": {
                    "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
                },
                "resolution": {
                    "enum": ["480p", "720p"],
                    "description": "Defaults to 480p (cheaper)."
                }
            },
            "additionalProperties": false
        }),
    }
}

fn generate_image_tool() -> ToolDefinition {
    ToolDefinition {
        name: "generate_image".to_string(),
        description: "Generate an image with xAI's Grok Imagine model. Use \
this whenever the user asks for an image, picture, drawing, illustration, or \
visual. Takes ~3-10 seconds.

Best practice: call `post_status_message` in the SAME RESPONSE as this \
tool (e.g. \"Cooking up that image…\") so the user knows you're working.

To edit, restyle, or combine images already in the conversation (e.g. \
\"make my teeth whiter\", \"turn this photo into a pencil sketch\"), pass \
their EXACT `file://images/...` id(s) in `reference_images`. Those ids are \
listed in the bracketed \"Images in this conversation you can edit\" note on \
the user's turn — pick the one(s) the user is referring to (you can see the \
images themselves above). NEVER invent or guess a path; only use ids that \
appear verbatim in that note. When you pass two or three references, refer \
to them in your prompt as <IMAGE_0>, <IMAGE_1>, … matching the order you \
list them. For a faithful edit, describe the whole desired result and what \
to preserve (e.g. \"the same person and photo, only the teeth whitened\") — \
the model regenerates rather than masking, so spell out what stays the same. \
If no real id applies, omit `reference_images` and generate from text alone.

The generated image is attached to your reply automatically — don't link to \
it in your text. Returns the saved image's URI so you can reference it in \
chained generations on later turns."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed description of the image to generate."
                },
                "reference_images": {
                    "type": "array",
                    "description": "Optional list of 1-3 image ids to edit/restyle/combine. Use the exact file://images/... ids from the 'Images in this conversation you can edit' note on the user's turn (https:// URLs also work). For 2-3 refs, the prompt references them as <IMAGE_0>, <IMAGE_1>, … in this array's order.",
                    "maxItems": 3,
                    "items": { "type": "string" }
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Optional aspect ratio. Default 1:1.",
                    "enum": ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2"]
                },
                "quality": {
                    "type": "string",
                    "description": "Quality tier. 'standard' is fast/cheap; 'quality' is slower/higher fidelity. Default 'standard'.",
                    "enum": ["standard", "quality"]
                }
            },
            "additionalProperties": false
        }),
    }
}

/// [`ToolExecutor`] backing the client-side tools. Owned per-turn so
/// it can capture the channel + guild + media-gen context plus the
/// specific turn identifiers we need for `video_jobs` persistence and
/// `post_status_message` reply targeting.
struct BotToolExecutor {
    http: Arc<HttpClient>,
    db: Db,
    bot_user_id: Id<UserMarker>,
    default_channel_id: Id<ChannelMarker>,
    user_msg_id: Id<MessageMarker>,
    guild_id: i64,
    conversation_id: Uuid,
    turn_id: Uuid,
    privacy_mode: PrivacyMode,
    /// Resolved image backend for the *current* persona, or `None` if
    /// the persona doesn't have one — in which case the
    /// `generate_image` tool isn't declared this turn either.
    image_provider: Option<AnyImageProvider>,
    /// Same for video.
    video_provider: Option<AnyVideoProvider>,
    images_dir: PathBuf,
    videos_dir: PathBuf,
    /// Last `post_status_message` text actually posted this turn.
    /// Used to silently drop duplicate/near-duplicate consecutive
    /// status messages — the model sometimes re-narrates the same
    /// line across multiple agent loop iterations, which spams the
    /// channel.
    last_status_text: Mutex<Option<String>>,
}

impl ToolExecutor for BotToolExecutor {
    async fn execute(&self, name: &str, input: Value) -> Result<Value, ToolError> {
        match name {
            "fetch_messages" => self.fetch_messages(input).await,
            "generate_image" => self.generate_image(input).await,
            "generate_video" => self.generate_video(input).await,
            "post_status_message" => self.post_status_message(input).await,
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
    #[tracing::instrument(name = "tool.fetch_messages", skip_all)]
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
            && channel_id.get() != *allowed
        {
            return Err(ToolError::InvalidInput(format!(
                "this server is in channel_only mode; fetch_messages can only target channel {allowed}"
            )));
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

    #[tracing::instrument(name = "tool.generate_image", skip_all)]
    async fn generate_image(&self, input: Value) -> Result<Value, ToolError> {
        let Some(provider) = self.image_provider.as_ref() else {
            return Err(ToolError::Execution(
                "image generation isn't configured for this persona".to_string(),
            ));
        };

        let prompt = input
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("prompt is required".to_string()))?
            .to_string();
        let references: Vec<String> = input
            .get("reference_images")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let aspect_ratio = input
            .get("aspect_ratio")
            .and_then(Value::as_str)
            .map(str::to_string);
        // Free-form model knob — each provider interprets the string
        // against its own catalog. xAI accepts `"standard"` / `"quality"`
        // and the underlying ids; future backends define their own.
        let model = input
            .get("quality")
            .or_else(|| input.get("model"))
            .and_then(Value::as_str)
            .map(str::to_string);

        let req = ImageGenRequest {
            prompt: prompt.clone(),
            references,
            aspect_ratio,
            model,
            images_dir: self.images_dir.clone(),
        };

        let generated = provider
            .generate(req)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;

        let extension = storage::extension_for_mime(&generated.mime_type);
        let uri = storage::save_image_bytes(&generated.bytes, extension, &self.images_dir)
            .await
            .map_err(|e| ToolError::Execution(format!("save: {e}")))?;

        tracing::info!(
            uri = %uri,
            model = %generated.model,
            mime = %generated.mime_type,
            bytes = generated.bytes.len(),
            "imagegen: generated image"
        );

        Ok(json!({
            "image_uri": uri,
            "model": generated.model,
            "mime_type": generated.mime_type,
            "revised_prompt": generated.revised_prompt,
        }))
    }

    /// Generate a video synchronously: submit + poll + download
    /// happen inside the tool call. The tool blocks for ~60-120s;
    /// the agent loop sees one tool call go in and one result come
    /// out. Status messages reach the user via `post_status_message`
    /// fired in the same response as this call.
    #[tracing::instrument(name = "tool.generate_video", skip_all)]
    async fn generate_video(&self, input: Value) -> Result<Value, ToolError> {
        let Some(provider) = self.video_provider.as_ref() else {
            return Err(ToolError::Execution(
                "video generation isn't configured for this persona".to_string(),
            ));
        };
        let prompt = input
            .get("prompt")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("prompt is required".to_string()))?
            .to_string();
        let image_url = input
            .get("image_url")
            .and_then(Value::as_str)
            .map(str::to_string);
        let duration = input
            .get("duration_seconds")
            .and_then(Value::as_i64)
            .map(|n| n.clamp(1, 15) as u8);
        let aspect_ratio = input
            .get("aspect_ratio")
            .and_then(Value::as_str)
            .map(str::to_string);
        let resolution = input
            .get("resolution")
            .and_then(Value::as_str)
            .map(str::to_string);
        let model = input
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string);

        let req = VideoGenRequest {
            prompt: prompt.clone(),
            image_url,
            duration_seconds: duration,
            aspect_ratio,
            resolution,
            model,
        };

        // Submit, record the row, then poll-and-download. State is
        // persisted: the video_jobs row exists from the moment the
        // backend returns request_id, so a bot crash mid-poll leaves
        // the request discoverable for a future restart-resume.
        let request_id = provider
            .submit(&req)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        let job = self
            .db
            .create_video_job(self.turn_id, &request_id, &prompt)
            .await
            .map_err(|e| ToolError::Execution(format!("db: {e}")))?;
        tracing::info!(
            request_id = %request_id,
            job_id = %job.id,
            "videogen: job submitted and persisted; polling inline"
        );

        // Poll-until-done using the lower-level primitives so we can
        // update the DB row on terminal transitions without
        // rebuilding the whole loop.
        let video_meta = loop {
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            let status = provider
                .check_once(&request_id)
                .await
                .map_err(|e| ToolError::Execution(e.to_string()))?;
            match status {
                grok_discord_bot_core::videogen::JobStatus::Pending => continue,
                grok_discord_bot_core::videogen::JobStatus::Done(meta) => break meta,
                grok_discord_bot_core::videogen::JobStatus::Failed(msg) => {
                    self.db
                        .update_video_job_status(&request_id, "failed", None, Some(&msg))
                        .await
                        .ok();
                    return Err(ToolError::Execution(format!(
                        "video generation failed: {msg}"
                    )));
                }
                grok_discord_bot_core::videogen::JobStatus::Expired => {
                    self.db
                        .update_video_job_status(&request_id, "expired", None, Some("expired"))
                        .await
                        .ok();
                    return Err(ToolError::Execution(
                        "video generation job expired".to_string(),
                    ));
                }
            }
        };

        let bytes = provider
            .download_bytes(&video_meta.url)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        let extension = extension_from_video_url(&video_meta.url);
        let uri = storage::save_video_bytes(&bytes, extension, &self.videos_dir)
            .await
            .map_err(|e| ToolError::Execution(format!("save: {e}")))?;
        self.db
            .update_video_job_status(&request_id, "done", Some(&uri), None)
            .await
            .map_err(|e| ToolError::Execution(format!("db: {e}")))?;

        tracing::info!(
            request_id = %request_id,
            uri = %uri,
            bytes = bytes.len(),
            "videogen: completed and persisted"
        );

        Ok(json!({
            "request_id": request_id,
            "video_uri": uri,
            "duration_seconds": video_meta.duration.unwrap_or(0.0),
        }))
    }

    #[tracing::instrument(name = "tool.post_status_message", skip_all)]
    async fn post_status_message(&self, input: Value) -> Result<Value, ToolError> {
        let text = input
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidInput("text is required".to_string()))?;
        let trimmed = truncate(text, 1990);
        if trimmed.trim().is_empty() {
            return Err(ToolError::InvalidInput("text is empty".to_string()));
        }

        // Drop near-duplicates of the previous status this turn. The
        // model sometimes re-emits the same "Generating…" line on
        // every loop iteration, which spams the channel. Compare
        // case-insensitively after trimming trailing punctuation/
        // ellipses so "Generating the image…" and "Generating the
        // image..." collapse together.
        let normalized = normalize_status_for_dedup(&trimmed);
        {
            let mut last = self.last_status_text.lock().unwrap();
            if last.as_deref() == Some(normalized.as_str()) {
                tracing::info!(
                    chars = trimmed.len(),
                    "status message suppressed (duplicate of previous)"
                );
                return Ok(json!({
                    "skipped": true,
                    "reason": "duplicate of previous status this turn",
                }));
            }
            *last = Some(normalized);
        }

        let posted = self
            .http
            .create_message(self.default_channel_id)
            .content(&trimmed)
            .reply(self.user_msg_id)
            .flags(twilight_model::channel::message::MessageFlags::SUPPRESS_EMBEDS)
            .await
            .map_err(|e| ToolError::Execution(format!("discord http: {e}")))?
            .model()
            .await
            .map_err(|e| ToolError::Execution(format!("discord deserialize: {e}")))?;
        let _ = self
            .db
            .record_message_link(
                i64::try_from(posted.id.get()).unwrap_or(i64::MAX),
                self.guild_id,
                self.conversation_id,
                self.turn_id,
                "assistant_status",
            )
            .await;
        tracing::info!(
            message_id = %posted.id,
            chars = trimmed.len(),
            "status message posted (via tool)"
        );
        Ok(json!({
            "posted_message_id": posted.id.get().to_string(),
            "chars": trimmed.len(),
        }))
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

/// Discord hard caps messages at 2000 chars. We aim for a hair under
/// so multi-byte characters don't push us over the line.
const DISCORD_MESSAGE_BUDGET: usize = 1990;

/// `body` is the model's answer text. `footer` is an optional final
/// suffix (today: the trace-link line for new conversations); kept
/// separate so the splitter can preferentially attach it to the LAST
/// chunk, falling back to its own chunk only if it won't fit.
struct FormattedReply {
    body: String,
    footer: Option<String>,
}

fn format_reply(
    answer: &str,
    is_new: bool,
    conversation: &Conversation,
    web_base_url: &str,
) -> FormattedReply {
    let body = fix_bare_mentions(answer);
    let footer = is_new.then(|| {
        format!(
            "\n\n-# 🔎 [full trace]({base}/c/{id})",
            base = web_base_url.trim_end_matches('/'),
            id = conversation.id,
        )
    });
    FormattedReply { body, footer }
}

/// Split `text` into Discord-sized chunks, preferring nice breakpoints
/// (paragraph → line → sentence → word → hard char boundary). Each
/// returned chunk's `len()` is <= `max_per_chunk`. The algorithm walks
/// remaining text repeatedly: from the longest acceptable slice, find
/// the latest preferred break inside it and emit everything up to that
/// break as one chunk. Falls through to coarser breaks when no fine
/// break is available within budget.
fn split_into_messages(text: &str, max_per_chunk: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    if text.len() <= max_per_chunk {
        return vec![text.to_string()];
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut remaining = text;
    loop {
        if remaining.len() <= max_per_chunk {
            if !remaining.trim().is_empty() {
                chunks.push(remaining.to_string());
            }
            break;
        }
        let split_at = find_split_point(remaining, max_per_chunk);
        // `find_split_point` always returns a valid char boundary > 0.
        let chunk = remaining[..split_at].trim_end().to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        remaining = remaining[split_at..].trim_start();
    }
    chunks
}

/// Locate the best split offset within the first `max` bytes of `s`.
/// Tries paragraph break → newline → sentence terminator → word →
/// hard char-boundary cut, in that preference order. Returns a byte
/// offset on a UTF-8 char boundary, guaranteed > 0 when input is
/// longer than `max`.
fn find_split_point(s: &str, max: usize) -> usize {
    let limit = max.min(s.len());
    // Walk `limit` back to a char boundary so candidate slicing is safe.
    let mut limit = limit;
    while limit > 0 && !s.is_char_boundary(limit) {
        limit -= 1;
    }
    let candidate = &s[..limit];

    if let Some(pos) = candidate.rfind("\n\n") {
        return pos + 2;
    }
    if let Some(pos) = candidate.rfind('\n') {
        return pos + 1;
    }
    for sep in [". ", "! ", "? "] {
        if let Some(pos) = candidate.rfind(sep) {
            return pos + sep.len();
        }
    }
    if let Some(pos) = candidate.rfind(' ') {
        return pos + 1;
    }
    // Last resort: hard cut at the (char-aligned) limit. If even that
    // is 0 (input starts with a multi-byte char larger than `max`),
    // bump up one char to avoid infinite-looping.
    if limit > 0 {
        return limit;
    }
    s.chars().next().map(|c| c.len_utf8()).unwrap_or(1)
}

/// Build the final list of message chunks ready to post, including
/// the optional footer. The footer joins the last chunk when it fits;
/// otherwise it becomes its own trailing chunk.
fn assemble_chunks(reply: &FormattedReply) -> Vec<String> {
    let mut chunks = split_into_messages(&reply.body, DISCORD_MESSAGE_BUDGET);
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    if let Some(footer) = reply.footer.as_ref() {
        let last = chunks.last_mut().expect("at least one chunk");
        if last.len() + footer.len() <= DISCORD_MESSAGE_BUDGET {
            last.push_str(footer);
        } else {
            chunks.push(footer.trim_start().to_string());
        }
    }
    // Discard any chunk that ended up empty after trimming.
    chunks.retain(|c| !c.is_empty());
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks
}

/// Rewrite bare `@<snowflake>` runs into proper Discord mention syntax
/// `<@<snowflake>>` so the recipient actually gets pinged. Models that
/// learn user IDs from `fetch_messages` results tend to emit the raw
/// `@<digits>` form (verbatim from any chat conversation they were
/// trained on) which Discord renders as inert text.
///
/// We only act on runs of 17-20 ASCII digits (Discord snowflake range
/// today; the upper bound has headroom for future ID growth) and skip
/// ones that are already preceded by a literal `<` so we don't
/// double-wrap an existing `<@id>` mention.
fn fix_bare_mentions(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '@' {
            out.push(c);
            continue;
        }
        // Collect the following digit run without consuming non-digits.
        let mut digits = String::new();
        while let Some(&d) = chars.peek() {
            if d.is_ascii_digit() {
                digits.push(d);
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

/// Walk the agent's tool-call trace and load any generated media
/// (images and videos) from disk, in order. Each becomes a Discord
/// file attachment on the outgoing reply, skipping anything that
/// exceeds Discord's free-tier upload size limit.
async fn collect_generated_attachments(
    images_dir: &std::path::Path,
    videos_dir: &std::path::Path,
    tool_calls: &[grok_discord_bot_core::ToolCallRecord],
) -> Vec<HttpAttachment> {
    let mut attachments: Vec<HttpAttachment> = Vec::new();
    for (i, call) in tool_calls.iter().enumerate() {
        let (uri_field, dir, fallback_filename) = match call.tool_name.as_str() {
            "generate_image" => ("image_uri", images_dir, format!("image-{i}.png")),
            "generate_video" => ("video_uri", videos_dir, format!("video-{i}.mp4")),
            _ => continue,
        };
        let Some(uri) = call.response.get(uri_field).and_then(Value::as_str) else {
            continue;
        };
        let Some(local_path) = storage::file_uri_to_local_path(uri, dir) else {
            continue;
        };
        match tokio::fs::read(&local_path).await {
            Ok(bytes) => {
                if bytes.len() as u64 > DISCORD_FREE_UPLOAD_LIMIT_BYTES {
                    tracing::warn!(
                        path = %local_path.display(),
                        bytes = bytes.len(),
                        limit = DISCORD_FREE_UPLOAD_LIMIT_BYTES,
                        "media exceeds Discord free-tier upload limit; skipping attachment"
                    );
                    continue;
                }
                let filename = local_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
                    .unwrap_or(fallback_filename);
                let id = u64::try_from(attachments.len()).unwrap_or(0);
                attachments.push(HttpAttachment::from_bytes(filename, bytes, id));
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    path = %local_path.display(),
                    "failed to read generated media for Discord attach"
                );
            }
        }
    }
    attachments
}

fn extension_from_video_url(url: &str) -> &'static str {
    let no_query = url.split('?').next().unwrap_or(url);
    let ext = no_query
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "mp4" => "mp4",
        "webm" => "webm",
        "mov" => "mov",
        _ => "mp4",
    }
}

/// Post the reply as one or more Discord messages.
///
/// For a new conversation whose total body exceeds the auto-thread
/// threshold, we open a thread off the user's message and post every
/// chunk inside it. Otherwise the first chunk is a native Discord
/// reply to the user; subsequent chunks are plain follow-up messages
/// in the same channel (Discord renders them adjacent without the
/// "Replying to…" header repeating on every line).
///
/// Attachments — generated images / videos — are always placed on the
/// LAST chunk so the user sees the prose first and the media at the
/// end. Returns every posted message so the caller can link them all
/// into `message_links`.
#[allow(clippy::too_many_arguments)]
async fn post_reply_chunks(
    state: &State,
    channel_id: Id<ChannelMarker>,
    user_msg_id: Id<MessageMarker>,
    user_content: &str,
    chunks: &[String],
    is_new: bool,
    generated: &[HttpAttachment],
    conversation: &Conversation,
    turn_id: Uuid,
) -> Result<Vec<Message>, BotError> {
    debug_assert!(!chunks.is_empty(), "must have at least one chunk to post");
    let suppress = MessageFlags::SUPPRESS_EMBEDS;

    let (target_channel, first_chunk_replies_to_user): (Id<ChannelMarker>, bool) =
        if should_open_thread(is_new, chunks) {
            let title = make_thread_title(user_content);
            let thread = state
                .http
                .create_thread_from_message(channel_id, user_msg_id, &title)
                .await?
                .model()
                .await?;
            // Explicitly join the thread we just created. Despite the
            // GUILD_MESSAGES intent technically covering public-thread
            // messages, in practice MESSAGE_CREATE events for
            // bot-created threads don't always flow until the bot is a
            // thread member. Sending the reply below would also
            // auto-add us, but we hit the join endpoint first so
            // subsequent @mentions get delivered without waiting for
            // the first post to land.
            if let Err(err) = state.http.join_thread(thread.id).await {
                tracing::warn!(
                    error = %err,
                    thread = %thread.id,
                    "failed to join newly-created thread; in-thread \
                     follow-ups may not be received"
                );
            }
            (thread.id, false)
        } else {
            (channel_id, true)
        };

    let total = chunks.len();
    let mut posted: Vec<Message> = Vec::with_capacity(total);
    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i + 1 == total;
        let mut builder = state
            .http
            .create_message(target_channel)
            .content(chunk)
            .flags(suppress);
        if i == 0 && first_chunk_replies_to_user {
            builder = builder.reply(user_msg_id);
        }
        if is_last && !generated.is_empty() {
            builder = builder.attachments(generated);
        }
        let posted_msg = builder.await?.model().await?;

        // Link each chunk to the conversation as it's posted, so a
        // partial-failure here still leaves the earlier chunks
        // discoverable for thread continuation. Best-effort: a DB
        // hiccup here doesn't fail the overall post.
        if let Err(err) = state
            .db
            .record_message_link(
                i64::try_from(posted_msg.id.get()).unwrap_or(i64::MAX),
                conversation.discord_guild_id,
                conversation.id,
                turn_id,
                "assistant",
            )
            .await
        {
            tracing::warn!(
                error = %err,
                message_id = %posted_msg.id,
                "failed to link posted chunk into conversation"
            );
        }

        posted.push(posted_msg);
    }
    Ok(posted)
}

/// Hard-cap a string to `max` BYTES, appending a `…` (3 UTF-8 bytes)
/// when truncation occurs. The returned string's `len()` is always
/// `<= max`. The cutoff is walked back to a char boundary so we never
/// slice mid-codepoint.
/// Decide whether a new-conversation reply should open a thread.
/// Threading triggers if EITHER the raw character count or the
/// approximate visual-row count exceeds its threshold (see the
/// constants for rationale). Follow-ups in an existing conversation
/// (`is_new = false`) always reply inline.
fn should_open_thread(is_new: bool, chunks: &[String]) -> bool {
    if !is_new {
        return false;
    }
    let total_chars: usize = chunks.iter().map(|c| c.len()).sum();
    if total_chars > REPLY_LENGTH_THRESHOLD {
        return true;
    }
    let total_rendered_lines: usize = chunks.iter().map(|c| rendered_line_count(c)).sum();
    total_rendered_lines > REPLY_RENDERED_LINES_THRESHOLD
}

/// Approximate the number of visual rows a Discord client would
/// render the given text as. Counts `\n`-separated logical lines and
/// adds extra rows for lines that exceed [`REPLY_WRAP_WIDTH`] chars
/// (which most clients soft-wrap). Used by the auto-thread heuristic
/// so tall replies — like a 10-row numbered list — get threaded even
/// when their raw char count is modest.
fn rendered_line_count(text: &str) -> usize {
    text.split('\n')
        .map(|line| {
            let chars = line.chars().count();
            if chars == 0 {
                1
            } else {
                chars.div_ceil(REPLY_WRAP_WIDTH)
            }
        })
        .sum()
}

/// Normalize a status string for duplicate-detection: lowercase, trim,
/// strip trailing ellipses / dots / spaces. Two strings that differ
/// only in `...` vs `…` vs a trailing space collapse to the same key.
fn normalize_status_for_dedup(s: &str) -> String {
    let lower = s.to_lowercase();
    let trimmed = lower.trim();
    let cleaned = trimmed.trim_end_matches(|c: char| c == '.' || c == '…' || c.is_whitespace());
    cleaned.to_string()
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let ellipsis_bytes = '…'.len_utf8();
    let mut cutoff = max.saturating_sub(ellipsis_bytes);
    while cutoff > 0 && !s.is_char_boundary(cutoff) {
        cutoff -= 1;
    }
    format!("{}…", &s[..cutoff])
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

/// Pick the best display name for a Discord message author. Priority:
/// guild nickname (when present) → global display name → username.
/// The returned slice borrows from `msg`, so callers wanting to store
/// it should `.to_string()`.
fn best_display_name(msg: &Message) -> &str {
    if let Some(member) = &msg.member
        && let Some(nick) = &member.nick
    {
        return nick;
    }
    if let Some(gn) = &msg.author.global_name {
        return gn;
    }
    &msg.author.name
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

/// Best display name for a mentioned user: guild nickname → username.
/// (Unlike [`best_display_name`], a `Mention` carries no `global_name`.)
fn mention_display_name(m: &Mention) -> &str {
    if let Some(member) = &m.member
        && let Some(nick) = &member.nick
    {
        return nick;
    }
    &m.name
}

/// Rewrite raw `<@ID>` / `<@!ID>` user mentions into `Name (<@ID>)` so the
/// model both learns who is referenced *and* keeps the raw mention token it
/// needs to ping that user back. The bot's own mention is skipped — it has
/// already been removed by [`strip_mentions`].
fn resolve_user_mentions(content: &str, msg: &Message, bot_user_id: Id<UserMarker>) -> String {
    let mut out = content.to_string();
    for m in &msg.mentions {
        if m.id == bot_user_id {
            continue;
        }
        let id = m.id.get();
        let replacement = format!("{} (<@{id}>)", mention_display_name(m));
        out = out
            .replace(&format!("<@{id}>"), &replacement)
            .replace(&format!("<@!{id}>"), &replacement);
    }
    out
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

    fn fake_conversation() -> Conversation {
        Conversation {
            id: uuid::Uuid::nil(),
            created_at: time::OffsetDateTime::now_utc(),
            discord_guild_id: 0,
            discord_channel_id: 0,
            created_by_user_id: 0,
            root_discord_message_id: 0,
            title: None,
            title_generated_at: None,
            model: "test".to_string(),
        }
    }

    fn fake_persona() -> Persona {
        Persona {
            provider: grok_discord_bot_core::LlmProviderKind::Xai,
            model: "grok-4.3".to_string(),
            system_prompt: "You are Chud. Be edgy.".to_string(),
            temperature: None,
            top_p: None,
            xai: None,
            anthropic: None,
            image_provider: None,
            video_provider: None,
        }
    }

    #[test]
    fn system_prompt_keeps_operational_block_first_then_persona() {
        let p = fake_persona();
        let out = compose_system_prompt(&p, &PrivacyMode::ConversationOnly, false, false, 7, None);
        // Operational block leads; persona voice follows it so the
        // non-persona rules can't be displaced by an adversarial persona.
        assert!(out.trim_start().starts_with("— Operational context"));
        let ops_at = out.find("Operational context").unwrap();
        let persona_at = out.find("You are Chud").unwrap();
        assert!(ops_at < persona_at);
        // Dynamic bits are present.
        assert!(out.contains("model `grok-4.3`"));
        assert!(out.contains("via the xai API"));
        assert!(out.contains(crate::VERSION));
        // The ordered version number leads the build line, with the git
        // descriptor in parens.
        assert!(out.contains(&format!("Bot build: v7 ({})", crate::VERSION)));
        // Web search is always advertised.
        assert!(out.contains("- Web search:"));
    }

    #[test]
    fn system_prompt_gates_capabilities_on_what_is_enabled() {
        let p = fake_persona();
        // Assert on the capability *lines* (the gated content), not bare
        // tool names: the always-on conventions block legitimately mentions
        // a tool like `fetch_messages` (for the `<@ID>` mention guidance)
        // regardless of whether its capability is enabled this turn.
        // ConversationOnly + no media providers → only the web-search line.
        let minimal =
            compose_system_prompt(&p, &PrivacyMode::ConversationOnly, false, false, 1, None);
        assert!(minimal.contains("- Web search:"));
        assert!(!minimal.contains("- Image generation & editing:"));
        assert!(!minimal.contains("- Video generation:"));
        assert!(!minimal.contains("- Recent channel messages:"));

        // Open privacy + both media providers → all four capability lines.
        let full = compose_system_prompt(
            &p,
            &PrivacyMode::opt_in_default(),
            true,
            true,
            2,
            Some("  Discord ToS: be nice.  "),
        );
        assert!(full.contains("- Image generation & editing:"));
        assert!(full.contains("- Video generation:"));
        assert!(full.contains("- Recent channel messages:"));
        // Operator addendum is appended (trimmed) under its own header.
        assert!(full.contains("— Operator policy —"));
        assert!(full.contains("Discord ToS: be nice."));
    }

    fn ctx(source: &str, role: &str, content: &str) -> ContextItem {
        ContextItem {
            position: 0,
            source: source.to_string(),
            role: role.to_string(),
            content: content.to_string(),
            discord_message_id: None,
        }
    }

    #[test]
    fn assemble_messages_serves_prior_images_and_skips_live_uploads() {
        // Live gateway path: a `turn:*:image:*` row is served from storage;
        // a `discord:msg:*:image:*` row is skipped here (it rides the live
        // Discord URL via `saved_images`).
        let context = vec![
            ctx("system", "system", "sys"),
            ctx("turn:abc:user", "user", "[bob]: hi"),
            ctx("turn:abc:image:0", "user", "file://images/old.png"),
            ctx("discord:msg:9", "user", "[amy]: draw"),
            ctx("discord:msg:9:image:0", "user", "file://images/up.png"),
        ];
        let msgs = assemble_messages(&context, &[], "https://ex.com");
        // system, user(bob)+served-image, user(amy)+annotation
        let images: usize = msgs
            .iter()
            .flat_map(|m| &m.blocks)
            .filter(|b| matches!(b, TurnBlock::Image { .. }))
            .count();
        // Only the prior `turn:*` image is attached (no live saved_images);
        // the `discord:msg:*` upload is skipped.
        assert_eq!(images, 1);
        // The reference annotation lists BOTH images by their file:// id.
        let all_text: String = msgs
            .iter()
            .flat_map(|m| &m.blocks)
            .filter_map(|b| match b {
                TurnBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(all_text.contains("file://images/old.png"));
        assert!(all_text.contains("file://images/up.png"));
    }

    #[test]
    fn assemble_messages_attaches_reasoning_to_following_assistant_turn() {
        // A `turn:*:reasoning` item decodes its {provider, data} blob and
        // rides as the LEADING block of the assistant turn that follows it
        // (never as its own message).
        let context = vec![
            ctx("turn:abc:user", "user", "[bob]: hi"),
            ctx(
                "turn:abc:reasoning",
                "assistant",
                r#"{"provider":"xai","data":[{"type":"reasoning","id":"rs_1"}]}"#,
            ),
            ctx("turn:abc:assistant", "assistant", "the answer"),
        ];
        let msgs = assemble_messages(&context, &[], "https://ex.com");
        assert_eq!(msgs.len(), 2, "reasoning must not become its own message");
        let assistant = &msgs[1];
        assert_eq!(assistant.role, MessageRole::Assistant);
        match &assistant.blocks[0] {
            TurnBlock::Reasoning {
                provider_name,
                data,
            } => {
                assert_eq!(provider_name, "xai");
                assert_eq!(data[0]["id"], "rs_1");
            }
            other => panic!("expected leading reasoning block, got {other:?}"),
        }
        assert!(matches!(&assistant.blocks[1], TurnBlock::Text(t) if t == "the answer"));
    }

    #[test]
    fn assemble_messages_attaches_uploads_from_saved_images_as_served_url() {
        // Live gateway path with a fresh upload: the `discord:msg:*:image:*`
        // row is skipped in the loop, but its bytes are attached from
        // `saved_images`.
        let context = vec![
            ctx("discord:msg:9", "user", "[amy]: draw"),
            ctx("discord:msg:9:image:0", "user", "file://images/up.png"),
        ];
        let saved = vec![SavedImage {
            stored_uri: "file://images/up.png".to_string(),
            live_url: "https://cdn.discord/up.png".to_string(),
            mime_type: Some("image/png".to_string()),
        }];
        let msgs = assemble_messages(&context, &saved, "https://ex.com");
        let urls: Vec<&str> = msgs
            .iter()
            .flat_map(|m| &m.blocks)
            .filter_map(|b| match b {
                TurnBlock::Image { url, .. } => Some(url.as_str()),
                _ => None,
            })
            .collect();
        // The served storage URL minted from `stored_uri` — same form the
        // `turn:*` replay path produces, not the live Discord link.
        assert_eq!(urls, vec!["https://ex.com/images/up.png"]);
    }

    #[test]
    fn assemble_messages_upload_falls_back_to_live_url_when_unservable() {
        // A `stored_uri` that mints no served URL (e.g. `s3://`, which
        // `to_public_url` doesn't handle yet) degrades to the live Discord
        // URL so the model still sees the image this turn.
        let context = vec![
            ctx("discord:msg:9", "user", "[amy]: draw"),
            ctx("discord:msg:9:image:0", "user", "s3://bucket/up.png"),
        ];
        let saved = vec![SavedImage {
            stored_uri: "s3://bucket/up.png".to_string(),
            live_url: "https://cdn.discord/up.png".to_string(),
            mime_type: Some("image/png".to_string()),
        }];
        let msgs = assemble_messages(&context, &saved, "https://ex.com");
        let urls: Vec<&str> = msgs
            .iter()
            .flat_map(|m| &m.blocks)
            .filter_map(|b| match b {
                TurnBlock::Image { url, .. } => Some(url.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(urls, vec!["https://cdn.discord/up.png"]);
    }

    #[test]
    fn system_prompt_omits_operator_policy_when_blank() {
        let p = fake_persona();
        let out = compose_system_prompt(
            &p,
            &PrivacyMode::ConversationOnly,
            false,
            false,
            1,
            Some("   "),
        );
        assert!(!out.contains("Operator policy"));
    }

    #[test]
    fn split_short_input_yields_single_chunk() {
        let out = split_into_messages("just a sentence.", 100);
        assert_eq!(out, vec!["just a sentence.".to_string()]);
    }

    #[test]
    fn split_prefers_paragraph_breaks() {
        let body = format!(
            "{}\n\n{}\n\n{}",
            "a".repeat(40),
            "b".repeat(40),
            "c".repeat(40)
        );
        let out = split_into_messages(&body, 60);
        // Each "a"/"b"/"c" block is 40 chars; budget 60; paragraph
        // breaks must keep each block intact in its own chunk.
        assert_eq!(out.len(), 3);
        assert!(out[0].chars().all(|c| c == 'a'));
        assert!(out[1].chars().all(|c| c == 'b'));
        assert!(out[2].chars().all(|c| c == 'c'));
    }

    #[test]
    fn split_falls_through_to_word_boundary() {
        // No paragraph/sentence breaks; the splitter should still
        // break on spaces rather than mid-word.
        let body = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let out = split_into_messages(body, 20);
        for chunk in &out {
            assert!(chunk.len() <= 20);
        }
        // Each chunk's whitespace-separated tokens must all be real
        // input words — no half-tokens from cutting mid-word.
        let input_words: std::collections::HashSet<&str> = body.split_whitespace().collect();
        for chunk in &out {
            for word in chunk.split_whitespace() {
                assert!(input_words.contains(word), "broken word: {word}");
            }
        }
        // Round-trip: chunks rejoined preserve all input words in order.
        let rejoined = out.join(" ");
        assert_eq!(
            rejoined.split_whitespace().collect::<Vec<_>>(),
            body.split_whitespace().collect::<Vec<_>>()
        );
    }

    #[test]
    fn assemble_attaches_footer_to_last_chunk_when_it_fits() {
        let reply = FormattedReply {
            body: "first paragraph.\n\nsecond paragraph.".to_string(),
            footer: Some("\n\n-# trace".to_string()),
        };
        let chunks = assemble_chunks(&reply);
        assert!(chunks.last().unwrap().contains("trace"));
    }

    #[test]
    fn assemble_keeps_trace_link_when_body_is_very_long() {
        // Body alone exceeds one Discord message; the splitter
        // produces multiple chunks and the footer survives at the end.
        let body = "Lorem ipsum dolor sit amet.\n\n".repeat(120);
        let conversation = fake_conversation();
        let reply = format_reply(&body, true, &conversation, "https://example.com");
        let chunks = assemble_chunks(&reply);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.len() <= DISCORD_MESSAGE_BUDGET, "chunk over budget");
        }
        assert!(chunks.last().unwrap().contains("full trace"));
        assert!(chunks.last().unwrap().contains("https://example.com/c/"));
    }

    #[test]
    fn thread_title_falls_back_when_only_mentions() {
        assert_eq!(make_thread_title("<@123>"), "Grok");
        assert_eq!(make_thread_title("<@123> what is rust"), "what is rust");
    }

    #[test]
    fn fix_mentions_wraps_bare_snowflakes() {
        let id = "238508325464047627"; // 18 digits
        assert_eq!(
            fix_bare_mentions(&format!("hey @{id} watch out")),
            format!("hey <@{id}> watch out")
        );
    }

    #[test]
    fn fix_mentions_leaves_existing_mentions_alone() {
        let id = "238508325464047627";
        let input = format!("<@{id}> already wrapped");
        assert_eq!(fix_bare_mentions(&input), input);
    }

    #[test]
    fn fix_mentions_ignores_short_digits_and_words() {
        assert_eq!(fix_bare_mentions("@everyone"), "@everyone");
        assert_eq!(fix_bare_mentions("foo@123 bar"), "foo@123 bar");
        assert_eq!(fix_bare_mentions("user@example.com"), "user@example.com");
    }

    #[test]
    fn safety_refusal_detected_in_xai_403_body() {
        assert!(body_indicates_safety_refusal(
            r#"{"error":"Content violates usage guidelines. Failed check: SAFETY_CHECK_TYPE_CSAM"}"#
        ));
        assert!(body_indicates_safety_refusal(
            r#"{"error":"Content violates Usage Guidelines"}"#
        ));
        assert!(!body_indicates_safety_refusal(
            r#"{"error":"rate limited"}"#
        ));
        assert!(!body_indicates_safety_refusal(r#"unauthorized"#));
    }

    #[test]
    fn upstream_safety_refusal_matches_403_with_safety_body() {
        let err = grok_discord_bot_core::LlmError::Api {
            status: 403,
            body: "Content violates usage guidelines".to_string(),
        };
        assert!(is_upstream_safety_refusal(&err));
        // A 403 without the safety language is NOT a safety refusal.
        let other = grok_discord_bot_core::LlmError::Api {
            status: 403,
            body: "forbidden".to_string(),
        };
        assert!(!is_upstream_safety_refusal(&other));
        // Safety-ish text on a non-403 status is also not (wrong status).
        let five = grok_discord_bot_core::LlmError::Api {
            status: 500,
            body: "safety_check".to_string(),
        };
        assert!(!is_upstream_safety_refusal(&five));
    }

    #[test]
    fn fix_mentions_handles_multiple() {
        let a = "238508325464047627";
        let b = "1335037364980023356";
        assert_eq!(
            fix_bare_mentions(&format!("@{a} and @{b} both")),
            format!("<@{a}> and <@{b}> both")
        );
    }

    #[test]
    fn tool_definitions_gated_by_mode_and_media_keys() {
        let names = |mode: &PrivacyMode, image: bool, video: bool| -> Vec<String> {
            build_tool_definitions(mode, image, video)
                .into_iter()
                .map(|t| t.name)
                .collect()
        };

        // OptIn, no media keys: fetch_messages + post_status_message.
        assert_eq!(
            names(&PrivacyMode::OptIn, false, false),
            vec!["fetch_messages", "post_status_message"]
        );

        // OptIn + image + video keys: full client toolset.
        assert_eq!(
            names(&PrivacyMode::OptIn, true, true),
            vec![
                "fetch_messages",
                "generate_image",
                "generate_video",
                "post_status_message",
            ]
        );

        // ConversationOnly drops fetch_messages but post_status_message
        // is always available.
        assert_eq!(
            names(&PrivacyMode::ConversationOnly, false, false),
            vec!["post_status_message"]
        );
        assert_eq!(
            names(&PrivacyMode::ConversationOnly, true, false),
            vec!["generate_image", "post_status_message"]
        );
    }

    #[test]
    fn rendered_lines_counts_short_lines_individually() {
        let body = "a\nb\nc\nd\ne";
        assert_eq!(rendered_line_count(body), 5);
    }

    #[test]
    fn rendered_lines_adds_rows_for_wrap() {
        let line = "x".repeat(REPLY_WRAP_WIDTH * 3); // wraps to 3 rows
        assert_eq!(rendered_line_count(&line), 3);
        // mixed: 1 short + 1 that wraps 2x
        let mixed = format!("hi\n{}", "y".repeat(REPLY_WRAP_WIDTH + 1));
        assert_eq!(rendered_line_count(&mixed), 1 + 2);
    }

    #[test]
    fn rendered_lines_counts_blank_lines() {
        assert_eq!(rendered_line_count("a\n\nb"), 3);
    }

    #[test]
    fn should_open_thread_respects_both_signals() {
        // Follow-up: never threads.
        let big = vec!["x".repeat(REPLY_LENGTH_THRESHOLD + 100)];
        assert!(!should_open_thread(false, &big));
        // New + big char count: threads.
        assert!(should_open_thread(true, &big));
        // New + tall but small: threads on line count.
        let tall = vec!["a\n".repeat(REPLY_RENDERED_LINES_THRESHOLD + 5)];
        assert!(tall[0].len() < REPLY_LENGTH_THRESHOLD);
        assert!(should_open_thread(true, &tall));
        // New + small + short: inline.
        assert!(!should_open_thread(true, &["hi".to_string()]));
    }

    #[test]
    fn top10_style_reply_exceeds_line_threshold_but_not_char_threshold() {
        // Approximation of the screenshot: 3 numbered lists of 10
        // short entries plus a few headers/blank lines. Char count
        // sits well under REPLY_LENGTH_THRESHOLD, but the line
        // count alone should be enough to thread.
        let body = "**GDP:**\n\n".to_string()
            + &(1..=10)
                .map(|i| format!("{i}. Country\n"))
                .collect::<String>()
            + "\n**Population:**\n\n"
            + &(1..=10)
                .map(|i| format!("{i}. Country\n"))
                .collect::<String>()
            + "\n**Best:**\n\n"
            + &(1..=5)
                .map(|i| format!("{i}. Country\n"))
                .collect::<String>();
        assert!(body.len() < REPLY_LENGTH_THRESHOLD);
        assert!(rendered_line_count(&body) > REPLY_RENDERED_LINES_THRESHOLD);
    }
}
