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
    AgentRun, AnyProvider, ChatTurn, ContextItem, Conversation, Db, LlmProvider, LlmProviderKind,
    MessageRole, NoopObserver, Persona, PrivacyMode, StepRequest, StepResponse, StorageConfig,
    ToolDefinition, ToolError, ToolExecutor, TurnBlock,
    imagegen::{ImageGenRequest, ImageGenerator, ImageQuality},
    run_agent, storage,
    videogen::{VideoGenRequest, VideoGenerator, VideoResolution},
};
use uuid::Uuid;
use serde::Serialize;
use serde_json::{Value, json};
use thiserror::Error;
use twilight_gateway::{EventTypeFlags, Intents, Shard, ShardId, StreamExt};
use twilight_http::Client as HttpClient;
use twilight_http::request::channel::reaction::RequestReactionType;
use twilight_model::channel::Message;
use twilight_model::channel::message::MessageFlags;
use twilight_model::gateway::event::Event;
use twilight_model::gateway::payload::incoming::GuildCreate;
use twilight_model::http::attachment::Attachment as HttpAttachment;
use twilight_model::id::Id;
use twilight_model::id::marker::{
    ApplicationMarker, ChannelMarker, GuildMarker, MessageMarker, UserMarker,
};

use crate::commands;

/// Auto-thread when the new-conversation reply is heavy enough that
/// inlining it would dominate the channel. We trigger threading on
/// EITHER signal:
///   - total characters across all chunks > [`REPLY_LENGTH_THRESHOLD`]
///   - "rendered" line count > [`REPLY_RENDERED_LINES_THRESHOLD`]
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

/// Discord free-tier upload size cap. Files larger than this are
/// linked rather than attached (avoids a Discord-side reject).
const DISCORD_FREE_UPLOAD_LIMIT_BYTES: u64 = 25 * 1024 * 1024;

/// System prompt for the pre-flight moderation classifier. The bot
/// runs in **private friends-only servers**, so the default is ALLOW
/// and the refusal list is narrowly scoped to genuinely illegal or
/// near-illegal content. Banter, slurs, dark humor, political shit-
/// talking, and edgy art requests are explicitly *not* refusals.
const MODERATION_PROMPT: &str = "You are a TOS compliance classifier for a \
private friends-only Discord server. The DEFAULT IS ALLOW. Only REFUSE the \
narrowly listed categories below.

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

When in any doubt, ALLOW.

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

/// State shared across all message-handler tasks.
struct State {
    http: Arc<HttpClient>,
    download_http: reqwest::Client,
    db: Db,
    /// One provider per kind we have credentials for. A persona's
    /// `provider` field keys into this map at turn time so xAI and
    /// Anthropic personas can coexist in the same deployment.
    providers: HashMap<LlmProviderKind, AnyProvider>,
    web_base_url: String,
    bot_user_id: Id<UserMarker>,
    app_id: Id<ApplicationMarker>,
    default_privacy: PrivacyMode,
    /// Personas keyed by name. Each pairs a system prompt with a
    /// (provider, model) pair and optional sampling knobs.
    personas: HashMap<String, Persona>,
    /// Floor fallback when no `persona_selections` row matches the
    /// resolution chain. Always a valid key in `personas`.
    default_persona: String,
    images_dir: PathBuf,
    videos_dir: PathBuf,
    /// Present only when an xAI API key is configured; gates the
    /// `generate_image` tool exposure.
    image_gen: Option<Arc<ImageGenerator>>,
    /// Same gating; xAI's video endpoints share the chat key.
    video_gen: Option<Arc<VideoGenerator>>,
}

/// Entry point for the `grok bot` subcommand.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    discord_token: String,
    dev_guild_id: Option<u64>,
    db: Db,
    providers: HashMap<LlmProviderKind, AnyProvider>,
    personas: HashMap<String, Persona>,
    default_persona: String,
    web_base_url: String,
    default_privacy: PrivacyMode,
    storage_config: StorageConfig,
    image_gen: Option<Arc<ImageGenerator>>,
    video_gen: Option<Arc<VideoGenerator>>,
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

    if let Err(err) = commands::register(&http, application.id, dev_guild_id).await {
        tracing::error!(error = %err, "failed to register slash commands; continuing without them");
    }

    let state = Arc::new(State {
        http,
        download_http: reqwest::Client::new(),
        db,
        providers,
        web_base_url,
        bot_user_id: current.id,
        app_id: application.id,
        default_privacy,
        personas,
        default_persona,
        images_dir: storage_config.images_dir,
        videos_dir: storage_config.videos_dir,
        image_gen,
        video_gen,
    });

    let mut shard = Shard::new(ShardId::ONE, discord_token, intents);
    let watched = EventTypeFlags::MESSAGE_CREATE
        | EventTypeFlags::INTERACTION_CREATE
        | EventTypeFlags::GUILD_CREATE;

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
                        state.personas.clone(),
                        state.default_persona.clone(),
                        state.app_id,
                        boxed.0,
                    )
                    .await;
                });
            }
            Event::GuildCreate(boxed) => log_guild_create(&boxed),
            _ => {}
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
    };

    match provider.step(request).await {
        Ok(StepResponse::Final { content: verdict, .. }) => {
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

/// Synthesize an [`LlmError`] representing a safety refusal that
/// happened inside a client-side tool call, so the existing
/// [`is_upstream_safety_refusal`] dispatch in `handle_message` lights up
/// and reacts ❓.
fn synthesize_safety_refusal(reason: &str) -> grok_discord_bot_core::LlmError {
    grok_discord_bot_core::LlmError::Api {
        status: 403,
        body: format!("SAFETY_CHECK refusal: {reason}"),
    }
}

/// Top-level handler for one mention. Resolves the privacy mode, gates
/// on ChannelOnly, sets the 👀 reaction, calls [`process`], then
/// transitions the reaction to ✅ or ❌.
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
            channel = %msg.channel_id,
            guild = ?msg.guild_id.map(|g| g.get()),
            author = %msg.author.name,
            mentioned_user_ids = ?msg.mentions.iter().map(|u| u.id.get()).collect::<Vec<_>>(),
            bot_id = %state.bot_user_id,
            content_preview = %msg.content.chars().take(80).collect::<String>(),
            "ignoring message (bot not @-mentioned)"
        );
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
        let in_allowed_channel = msg.channel_id.get() == *channel_id;
        // A message inside a Grok-owned thread is by definition rooted
        // in a channel where the bot already accepted a turn, so it
        // shouldn't be filtered out even if `msg.channel_id` (which
        // for a thread message is the thread's own id, not the
        // parent's) doesn't match the configured allowed channel.
        let in_grok_thread = if in_allowed_channel {
            false
        } else {
            let channel_as_msg =
                i64::try_from(msg.channel_id.get()).unwrap_or(i64::MAX);
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
    // message clearly violates Discord TOS.
    let stripped = strip_mentions(&msg.content, state.bot_user_id);
    if !stripped.is_empty() && !moderation_allows(&state, &stripped).await {
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
        Ok(()) => {
            let _ = state
                .http
                .create_reaction(msg.channel_id, msg.id, &done)
                .await;
        }
        Err(BotError::Llm(ref llm_err)) if is_upstream_safety_refusal(llm_err) => {
            tracing::info!(
                error = %llm_err,
                "message refused by upstream safety check; reacting ❓"
            );
            let _ = state
                .http
                .create_reaction(msg.channel_id, msg.id, &refused)
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
    let user_id_i64 = i64::try_from(msg.author.id.get()).unwrap_or(i64::MAX);

    let resolved_persona_name = state
        .db
        .resolve_persona(
            conversation_id_for_persona,
            guild_id_for_persona,
            channel_id_i64,
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

    let user_content = strip_mentions(&msg.content, state.bot_user_id);

    // Persist any image attachments before recording context items so
    // every image gets its own `discord:msg:<id>:image:<i>` context row
    // for the viewer trace. Keep the original Discord CDN URL in memory
    // to pass to the LLM (it's still fresh; cheaper than base64).
    let saved_images = save_image_attachments(state, msg).await;

    let mut initial_context = build_context(
        state,
        msg,
        &conversation,
        is_new,
        &user_content,
        privacy_mode,
        &persona.system_prompt,
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
        )
        .await?;
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
    // attachments. The system prompt is constant (lives in the bot
    // config) and prior-turn user/assistant content is already stored
    // verbatim in the `turns` table, so re-stamping them into
    // `context_items` every turn would just duplicate data and grow
    // the table quadratically with conversation length.
    for item in initial_context
        .iter()
        .filter(|i| i.source.starts_with("discord:msg:"))
    {
        state.db.record_context_item(turn.id, item).await?;
    }

    // Build the LLM-facing chat history from the initial context items.
    // Image rows are skipped here and re-attached below as proper
    // ToolBlock::Image blocks on the user's current turn, using the
    // original Discord URLs (fresh, no base64 overhead).
    let mut messages: Vec<ChatTurn> = initial_context
        .iter()
        .filter(|c| !c.source.contains(":image:"))
        .map(|c| ChatTurn::text(MessageRole::from_str_lossy(&c.role), c.content.clone()))
        .collect();
    if !saved_images.is_empty() {
        if let Some(last) = messages.last_mut() {
            // Inject the attachment URLs as TEXT into the user's turn
            // so the model can pass them verbatim to tools like
            // generate_image. Without this hint, the model sees the
            // images via the structured input_image content block but
            // doesn't treat the URL as a quotable string and tends to
            // invent paths instead.
            let url_list = saved_images
                .iter()
                .map(|i| format!("- {}", i.live_url))
                .collect::<Vec<_>>()
                .join("\n");
            let annotation = format!(
                "\n\n[Images attached to this message — when calling \
                 generate_image to edit one of them, pass the exact URL \
                 below as a reference_images entry. Do not invent paths.]\n{url_list}"
            );
            match last.blocks.first_mut() {
                Some(TurnBlock::Text(text)) => text.push_str(&annotation),
                _ => last.blocks.insert(0, TurnBlock::Text(annotation)),
            }
            for image in &saved_images {
                last.blocks.push(TurnBlock::Image {
                    url: image.live_url.clone(),
                    mime_type: image.mime_type.clone(),
                });
            }
        }
    }

    // Tools available to the model for this turn:
    //   - fetch_messages: every mode except ConversationOnly
    //   - generate_image / start_video_generation / check_video_status:
    //     only when an xAI key is configured
    //   - post_status_message: always available
    let tools = build_tool_definitions(
        privacy_mode,
        state.image_gen.is_some(),
        state.video_gen.is_some(),
    );

    let executor = BotToolExecutor {
        http: Arc::clone(&state.http),
        db: state.db.clone(),
        bot_user_id: state.bot_user_id,
        default_channel_id: msg.channel_id,
        user_msg_id: msg.id,
        guild_id: conversation.discord_guild_id,
        conversation_id: conversation.id,
        turn_id: turn.id,
        privacy_mode: privacy_mode.clone(),
        image_gen: state.image_gen.clone(),
        video_gen: state.video_gen.clone(),
        images_dir: state.images_dir.clone(),
        videos_dir: state.videos_dir.clone(),
        last_status_text: Mutex::new(None),
    };

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
        MAX_AGENT_ITERATIONS,
    )
    .await;

    // Persist all tool calls (server + client) in execution order.
    for (i, tc) in agent_run.tool_calls.iter().enumerate() {
        state
            .db
            .record_tool_call(turn.id, i32::try_from(i).unwrap_or(0), tc)
            .await?;
    }

    // Collect any media the agent generated this turn for upload as
    // Discord attachments on the outgoing reply.
    let generated_attachments = collect_generated_attachments(
        &state.images_dir,
        &state.videos_dir,
        &agent_run.tool_calls,
    )
    .await;

    // If any generate_image or check_video_status call was refused by
    // xAI's safety layer, treat the whole turn as a TOS refusal and
    // bail out with ❓ — same shape as if the chat call itself had
    // been refused. We do this BEFORE the generic media-failed path so
    // a safety 403 doesn't get a ❌ + warning-text reply.
    let media_safety_refused = agent_run.tool_calls.iter().any(|tc| {
        matches!(tc.tool_name.as_str(), "generate_image" | "generate_video")
            && tc
                .response
                .get("error")
                .and_then(|v| v.as_str())
                .map(body_indicates_safety_refusal)
                .unwrap_or(false)
    });
    if media_safety_refused {
        tracing::info!(
            conversation = %conversation.id,
            turn = %turn.id,
            "media generation refused by upstream safety; surfacing as TOS refusal"
        );
        state
            .db
            .fail_turn(turn.id, "media generation refused by upstream safety")
            .await
            .ok();
        return Err(BotError::Llm(synthesize_safety_refusal(
            "media generation was refused by xAI's safety policy",
        )));
    }

    // Detect "model claimed success but media generation failed" — any
    // generate_image call with no image_uri in its response, or any
    // start_video_generation that never reached a check_video_status
    // result with a video_uri.
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
    // Partial-success handling: if the agent loop errored mid-flight
    // (e.g. xAI returned a transient 5xx on the final-answer step) but
    // we already have generated media or some prior assistant text,
    // surface what we have rather than dumping the raw error on the
    // user.
    let agent_loop_error = agent_run.error.as_deref();
    let have_media = !generated_attachments.is_empty();
    let final_content: String = if !agent_run.content.is_empty() {
        agent_run.content.clone()
    } else if have_media && agent_loop_error.is_some() {
        // Media generated, but the model never got to write the
        // closing line. Post the media with a minimal acknowledgement.
        "Here you go.".to_string()
    } else {
        agent_run.content.clone()
    };

    let answer_text = if media_gen_failed {
        format!("⚠️ {failure_label} failed.\n\n{final_content}")
    } else if let Some(err) = agent_loop_error {
        if have_media {
            // The interesting work succeeded; the failed step was the
            // final-answer LLM call. Keep the reply concise.
            tracing::warn!(
                conversation = %conversation.id,
                turn = %turn.id,
                error = %err,
                "agent loop errored after media generation succeeded; \
                 falling back to short reply + attachment"
            );
            final_content
        } else {
            // Nothing useful happened. Surface the error briefly so the
            // user knows the turn died.
            let snippet = err.chars().take(200).collect::<String>();
            format!("⚠️ {snippet}")
        }
    } else {
        final_content
    };
    let formatted = format_reply(&answer_text, is_new, &conversation, &state.web_base_url);
    let chunks = assemble_chunks(&formatted);
    let total_rendered_lines: usize = chunks.iter().map(|c| rendered_line_count(c)).sum();
    let threaded = should_open_thread(is_new, &chunks);
    let reply_msgs = post_reply_chunks(
        state,
        msg,
        &chunks,
        is_new,
        &generated_attachments,
        &conversation,
        turn.id,
    )
    .await?;
    let reply_msg = reply_msgs
        .last()
        .cloned()
        .expect("post_reply_chunks always returns at least one message");
    if media_gen_failed {
        tracing::warn!(
            conversation = %conversation.id,
            turn = %turn.id,
            label = failure_label,
            "media generation was attempted but produced no output; reply marked as failed"
        );
    }
    tracing::info!(
        conversation = %conversation.id,
        turn = %turn.id,
        reply_msg = %reply_msg.id,
        threaded,
        chunks = reply_msgs.len(),
        reply_chars = agent_run.content.len(),
        rendered_lines = total_rendered_lines,
        tool_calls = agent_run.tool_calls.len(),
        "turn: reply posted"
    );

    state
        .db
        .complete_turn(
            turn.id,
            &agent_run.content,
            i64::try_from(reply_msg.id.get()).unwrap_or(i64::MAX),
        )
        .await?;
    // Note: user + assistant message_links are recorded earlier — the
    // user link right after start_turn (so thread continuation works
    // even on partial failures), and assistant chunks inside
    // post_reply_chunks as each one is posted.

    if media_gen_failed {
        // Bubble the failure up so handle_message reacts ❌ instead of ✅.
        return Err(BotError::Llm(grok_discord_bot_core::LlmError::Transport(
            format!("{failure_label} was attempted but produced no output"),
        )));
    }
    if agent_loop_error.is_some() && !have_media {
        // Bare error, nothing to salvage — ❌.
        return Err(BotError::Llm(grok_discord_bot_core::LlmError::Transport(
            agent_loop_error.unwrap().to_string(),
        )));
    }
    Ok(())
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
        if let Some(conv_id) = state.db.lookup_conversation_by_message(parent_id).await? {
            if let Some(conv) = state.db.get_conversation(conv_id).await? {
                return Ok(Some(conv));
            }
        }
    }

    let channel_id = i64::try_from(msg.channel_id.get()).unwrap_or(i64::MAX);
    if let Some(conv_id) = state.db.lookup_conversation_by_message(channel_id).await? {
        if let Some(conv) = state.db.get_conversation(conv_id).await? {
            return Ok(Some(conv));
        }
    }

    Ok(None)
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

/// A single image attachment that's been persisted to local disk plus
/// the live Discord URL we'll hand to the LLM on this turn.
struct SavedImage {
    /// `file://images/<uuid>.<ext>` — recorded in `context_items`.
    stored_uri: String,
    /// Original Discord CDN URL — used for the in-memory LLM call only.
    /// Don't store; signed URLs expire after ~24h.
    live_url: String,
    /// `content_type` from the Discord attachment metadata, if any.
    mime_type: Option<String>,
}

/// Download every image-typed attachment on `msg` to `state.images_dir`.
/// Failures are logged and skipped — a broken attachment shouldn't fail
/// the whole reply.
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
            &state.images_dir,
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
    if let Some(ct) = content_type {
        if ct.starts_with("image/") {
            return true;
        }
    }
    let ext = filename.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
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

To edit/restyle an image the user attached, pass its EXACT URL string in \
`reference_images`. The URL appears in the user's turn (look for lines \
starting with `https://cdn.discordapp.com/...` or `file://images/...`). \
NEVER invent or guess a path — `reference_images` must contain only URLs \
that appear verbatim in the conversation. If you can't find a real URL, \
omit `reference_images` and generate from text alone.

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
                    "description": "Optional list of 0-3 image URIs to use as references. https:// URLs and file:// URIs from prior turns both work. Pass a user's attached image URL here to edit/restyle it.",
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
    image_gen: Option<Arc<ImageGenerator>>,
    video_gen: Option<Arc<VideoGenerator>>,
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

    async fn generate_image(&self, input: Value) -> Result<Value, ToolError> {
        let Some(generator) = self.image_gen.as_ref() else {
            return Err(ToolError::Execution(
                "image generation isn't configured on this bot".to_string(),
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
        let quality = match input.get("quality").and_then(Value::as_str) {
            Some("quality") => ImageQuality::Quality,
            _ => ImageQuality::Standard,
        };

        let req = ImageGenRequest {
            prompt: prompt.clone(),
            references,
            aspect_ratio,
            quality,
            images_dir: self.images_dir.clone(),
        };

        let generated = generator
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
    async fn generate_video(&self, input: Value) -> Result<Value, ToolError> {
        let Some(generator) = self.video_gen.as_ref() else {
            return Err(ToolError::Execution(
                "video generation isn't configured on this bot".to_string(),
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
        let resolution = match input.get("resolution").and_then(Value::as_str) {
            Some("720p") => VideoResolution::P720,
            _ => VideoResolution::P480,
        };

        let req = VideoGenRequest {
            prompt: prompt.clone(),
            image_url,
            duration_seconds: duration,
            aspect_ratio,
            resolution,
        };

        // Submit, record the row, then poll-and-download in one
        // generator.generate() call. State is persisted: the
        // video_jobs row exists from the moment xAI returns
        // request_id, so a bot crash mid-poll leaves the request
        // discoverable for a future restart-resume.
        let request_id = generator
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
            turn = %self.turn_id,
            "videogen: job submitted and persisted; polling inline"
        );

        // Poll-until-done using the lower-level primitives so we can
        // update the DB row on terminal transitions without
        // rebuilding the whole loop.
        let video_meta = loop {
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            let status = generator
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

        let bytes = generator
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
                    turn = %self.turn_id,
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
            turn = %self.turn_id,
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

/// Walk the agent's tool-call trace and load any generated media (images
/// + videos) from disk, in order. Each becomes a Discord file
/// attachment on the outgoing reply, skipping anything that exceeds
/// Discord's free-tier upload size limit.
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
    let ext = no_query.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
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
    user_msg: &Message,
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
            let title = make_thread_title(&user_msg.content);
            let thread = state
                .http
                .create_thread_from_message(user_msg.channel_id, user_msg.id, &title)
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
            (user_msg.channel_id, true)
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
            builder = builder.reply(user_msg.id);
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
    let cleaned = trimmed.trim_end_matches(|c: char| {
        c == '.' || c == '…' || c.is_whitespace()
    });
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

    fn fake_conversation() -> Conversation {
        Conversation {
            id: uuid::Uuid::nil(),
            created_at: time::OffsetDateTime::now_utc(),
            discord_guild_id: 0,
            discord_channel_id: 0,
            created_by_user_id: 0,
            root_discord_message_id: 0,
            title: None,
            model: "test".to_string(),
        }
    }

    #[test]
    fn split_short_input_yields_single_chunk() {
        let out = split_into_messages("just a sentence.", 100);
        assert_eq!(out, vec!["just a sentence.".to_string()]);
    }

    #[test]
    fn split_prefers_paragraph_breaks() {
        let body = format!("{}\n\n{}\n\n{}", "a".repeat(40), "b".repeat(40), "c".repeat(40));
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
        let input_words: std::collections::HashSet<&str> =
            body.split_whitespace().collect();
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
    fn synthetic_safety_error_round_trips_through_detector() {
        let err = synthesize_safety_refusal("generate_image was refused");
        assert!(is_upstream_safety_refusal(&err));
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
            + &(1..=10).map(|i| format!("{i}. Country\n")).collect::<String>()
            + "\n**Population:**\n\n"
            + &(1..=10).map(|i| format!("{i}. Country\n")).collect::<String>()
            + "\n**Best:**\n\n"
            + &(1..=5).map(|i| format!("{i}. Country\n")).collect::<String>();
        assert!(body.len() < REPLY_LENGTH_THRESHOLD);
        assert!(rendered_line_count(&body) > REPLY_RENDERED_LINES_THRESHOLD);
    }
}
