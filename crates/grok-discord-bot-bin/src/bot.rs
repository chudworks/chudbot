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

use std::path::PathBuf;
use std::sync::Arc;

use grok_discord_bot_core::{
    AgentRun, AnyProvider, BotConfig, ChatTurn, ContextItem, Conversation, Db, LlmProvider,
    MessageRole, PrivacyMode, StepRequest, StepResponse, StorageConfig, ToolDefinition,
    ToolError, ToolExecutor, TurnBlock,
    imagegen::{ImageGenRequest, ImageGenerator, ImageQuality},
    run_agent, storage,
};
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

/// System prompt for the pre-flight moderation classifier. Kept tight
/// so a fast/cheap call resolves to one of two tokens. Categories
/// loosely follow Discord's Community Guidelines; we deliberately
/// leave "edgy / vulgar / opinionated" out of the refusal list — Grok
/// is supposed to be a bit unhinged and Chud wants that flavor.
const MODERATION_PROMPT: &str = "You are a Discord TOS compliance classifier. \
Decide whether the user message violates Discord's Community Guidelines.

Categories that DO violate:
- CSAM or any sexualization of minors
- Doxxing (sharing non-public personal info to harm)
- Credible threats of violence
- Encouragement of self-harm or suicide
- Illegal-content arrangements (drug/weapon sales, trafficking)
- Targeted hate speech against protected groups
- Malware, phishing, large-scale spam

NOT violations: profanity, edgy jokes, political opinions, criticism, \
sarcasm, requests to generate edgy art, NSFW jokes without minors, asking \
about news or current events.

Respond with EXACTLY one token: ALLOW or REFUSE. No punctuation, no \
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
    llm: AnyProvider,
    web_base_url: String,
    bot_user_id: Id<UserMarker>,
    app_id: Id<ApplicationMarker>,
    default_privacy: PrivacyMode,
    bot_config: BotConfig,
    images_dir: PathBuf,
    /// Present only when an xAI API key is configured; gates the
    /// `generate_image` tool exposure.
    image_gen: Option<Arc<ImageGenerator>>,
}

/// Entry point for the `grok bot` subcommand.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    discord_token: String,
    dev_guild_id: Option<u64>,
    db: Db,
    llm: AnyProvider,
    web_base_url: String,
    default_privacy: PrivacyMode,
    bot_config: BotConfig,
    storage_config: StorageConfig,
    image_gen: Option<Arc<ImageGenerator>>,
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
        llm,
        web_base_url,
        bot_user_id: current.id,
        app_id: application.id,
        default_privacy,
        bot_config,
        images_dir: storage_config.images_dir,
        image_gen,
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
    let request = StepRequest {
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

    match state.llm.step(request).await {
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
///
/// xAI returns HTTP 403 with a body like:
/// `{"code":"...","error":"Content violates usage guidelines. ... \
/// Failed check: SAFETY_CHECK_TYPE_CSAM"}` when its server-side safety
/// classifiers refuse a prompt. We treat any 403 whose body mentions
/// either the safety-check label or the "violates usage guidelines"
/// phrase as a real refusal — distinct from a transient transport
/// error, and worth surfacing as ❓ rather than ❌.
fn is_upstream_safety_refusal(err: &grok_discord_bot_core::LlmError) -> bool {
    match err {
        grok_discord_bot_core::LlmError::Api { status, body } => {
            if *status != 403 {
                return false;
            }
            let lower = body.to_ascii_lowercase();
            lower.contains("safety_check") || lower.contains("violates usage guidelines")
        }
        _ => false,
    }
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

    let (conversation, is_new) = resolve_conversation(state, msg).await?;
    tracing::info!(
        conversation = %conversation.id,
        is_new,
        model = %conversation.model,
        "turn: conversation resolved"
    );

    let user_content = strip_mentions(&msg.content, state.bot_user_id);

    // Persist any image attachments before recording context items so
    // every image gets its own `discord:msg:<id>:image:<i>` context row
    // for the viewer trace. Keep the original Discord CDN URL in memory
    // to pass to the LLM (it's still fresh; cheaper than base64).
    let saved_images = save_image_attachments(state, msg).await;

    let mut initial_context =
        build_context(state, msg, &conversation, is_new, &user_content, privacy_mode).await?;

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
    tracing::info!(
        conversation = %conversation.id,
        turn = %turn.id,
        turn_index = turn.turn_index,
        context_items = initial_context.len(),
        images = saved_images.len(),
        "turn: started"
    );

    for item in &initial_context {
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

    // Tools available to the model for this turn. fetch_messages is
    // exposed in every mode except ConversationOnly; generate_image is
    // exposed only when an xAI key is configured.
    let tools = build_tool_definitions(privacy_mode, state.image_gen.is_some());

    let executor = BotToolExecutor {
        http: Arc::clone(&state.http),
        db: state.db.clone(),
        bot_user_id: state.bot_user_id,
        default_channel_id: msg.channel_id,
        guild_id: conversation.discord_guild_id,
        privacy_mode: privacy_mode.clone(),
        image_gen: state.image_gen.clone(),
        images_dir: state.images_dir.clone(),
    };

    let agent_result = run_agent(
        &state.llm,
        messages,
        tools,
        &executor,
        true, // server-side web search always enabled
        MAX_OUTPUT_TOKENS,
        state.bot_config.temperature,
        state.bot_config.top_p,
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

    // Collect any images the agent generated this turn for upload as
    // Discord attachments on the outgoing reply.
    let generated_attachments =
        collect_generated_attachments(&state.images_dir, &agent_run.tool_calls).await;

    // Detect "model claimed success but generate_image failed" — every
    // generate_image call produced no usable image_uri. We want to NOT
    // pretend things worked, so prepend a clear warning to the reply
    // and surface ❌ instead of ✅ at the caller.
    let attempted_image_gen = agent_run
        .tool_calls
        .iter()
        .any(|tc| tc.tool_name == "generate_image");
    let image_gen_failed = attempted_image_gen && generated_attachments.is_empty();
    let answer_text = if image_gen_failed {
        let error_snippet = agent_run
            .tool_calls
            .iter()
            .find(|tc| tc.tool_name == "generate_image")
            .and_then(|tc| tc.response.get("error"))
            .and_then(|v| v.as_str())
            .unwrap_or("(no error message)")
            .chars()
            .take(200)
            .collect::<String>();
        format!(
            "⚠️ Image generation failed: `{}`. (Model's claimed text follows.)\n\n{}",
            error_snippet, agent_run.content
        )
    } else {
        agent_run.content.clone()
    };
    let reply_text =
        format_reply(&answer_text, is_new, &conversation, &state.web_base_url);

    let reply_msg =
        post_reply(state, msg, &reply_text, is_new, &generated_attachments).await?;
    if image_gen_failed {
        tracing::warn!(
            conversation = %conversation.id,
            turn = %turn.id,
            "image generation was attempted but produced no images; reply marked as failed"
        );
    }
    let threaded = is_new && reply_text.len() > REPLY_LENGTH_THRESHOLD;
    tracing::info!(
        conversation = %conversation.id,
        turn = %turn.id,
        reply_msg = %reply_msg.id,
        threaded,
        reply_chars = agent_run.content.len(),
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

    if image_gen_failed {
        // Bubble the failure up so handle_message reacts ❌ instead of ✅.
        return Err(BotError::Llm(grok_discord_bot_core::LlmError::Transport(
            "generate_image was attempted but produced no images".to_string(),
        )));
    }
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
        state.bot_config.system_prompt.clone(),
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
fn build_tool_definitions(mode: &PrivacyMode, image_gen_available: bool) -> Vec<ToolDefinition> {
    let mut tools = Vec::new();
    if !matches!(mode, PrivacyMode::ConversationOnly) {
        tools.push(fetch_messages_tool());
    }
    if image_gen_available {
        tools.push(generate_image_tool());
    }
    tools
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

fn generate_image_tool() -> ToolDefinition {
    ToolDefinition {
        name: "generate_image".to_string(),
        description: "Generate an image with xAI's Grok Imagine model. Use \
this whenever the user asks for an image, picture, drawing, illustration, or \
visual.

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
/// it can capture the channel + guild + image-gen context.
struct BotToolExecutor {
    http: Arc<HttpClient>,
    db: Db,
    bot_user_id: Id<UserMarker>,
    default_channel_id: Id<ChannelMarker>,
    guild_id: i64,
    privacy_mode: PrivacyMode,
    image_gen: Option<Arc<ImageGenerator>>,
    images_dir: PathBuf,
}

impl ToolExecutor for BotToolExecutor {
    async fn execute(&self, name: &str, input: Value) -> Result<Value, ToolError> {
        match name {
            "fetch_messages" => self.fetch_messages(input).await,
            "generate_image" => self.generate_image(input).await,
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

/// Walk the agent's tool-call trace and load any images the model
/// generated this turn from disk, in order. Each becomes a Discord file
/// attachment on the outgoing reply.
async fn collect_generated_attachments(
    images_dir: &std::path::Path,
    tool_calls: &[grok_discord_bot_core::ToolCallRecord],
) -> Vec<HttpAttachment> {
    let mut attachments: Vec<HttpAttachment> = Vec::new();
    for (i, call) in tool_calls.iter().enumerate() {
        if call.tool_name != "generate_image" {
            continue;
        }
        let Some(uri) = call.response.get("image_uri").and_then(Value::as_str) else {
            continue;
        };
        let Some(local_path) = storage::file_uri_to_local_path(uri, images_dir) else {
            continue;
        };
        match tokio::fs::read(&local_path).await {
            Ok(bytes) => {
                let filename = local_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("image-{i}.png"));
                let id = u64::try_from(attachments.len()).unwrap_or(0);
                attachments.push(HttpAttachment::from_bytes(filename, bytes, id));
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    path = %local_path.display(),
                    "failed to read generated image for Discord attach"
                );
            }
        }
    }
    attachments
}

/// Send the reply. For a new conversation with a long answer we open a
/// public thread off the user's message and post inside it; otherwise
/// we reply inline. Generated images ride along as file attachments.
async fn post_reply(
    state: &State,
    user_msg: &Message,
    body: &str,
    is_new: bool,
    generated: &[HttpAttachment],
) -> Result<Message, BotError> {
    // Suppress Discord's link-preview embeds on every bot reply. The
    // model's answers frequently include citation URLs (the trace link
    // + tool-call result links + sources), and an unfurled embed per
    // URL drowns the channel. Equivalent to a user clicking the little
    // X on each preview, or wrapping URLs in <…>.
    let suppress = MessageFlags::SUPPRESS_EMBEDS;

    if is_new && body.len() > REPLY_LENGTH_THRESHOLD {
        let title = make_thread_title(&user_msg.content);
        let thread = state
            .http
            .create_thread_from_message(user_msg.channel_id, user_msg.id, &title)
            .await?
            .model()
            .await?;
        let trimmed = truncate(body, 1990);
        let mut builder = state
            .http
            .create_message(thread.id)
            .content(&trimmed)
            .flags(suppress);
        if !generated.is_empty() {
            builder = builder.attachments(generated);
        }
        let reply = builder.await?.model().await?;
        Ok(reply)
    } else {
        let trimmed = truncate(body, 1990);
        let mut builder = state
            .http
            .create_message(user_msg.channel_id)
            .content(&trimmed)
            .reply(user_msg.id)
            .flags(suppress);
        if !generated.is_empty() {
            builder = builder.attachments(generated);
        }
        let reply = builder.await?.model().await?;
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
        let names = |mode: &PrivacyMode, image_gen: bool| -> Vec<String> {
            build_tool_definitions(mode, image_gen)
                .into_iter()
                .map(|t| t.name)
                .collect()
        };
        assert_eq!(names(&PrivacyMode::OptIn, false), vec!["fetch_messages"]);
        assert_eq!(
            names(&PrivacyMode::OptIn, true),
            vec!["fetch_messages", "generate_image"]
        );
        assert!(names(&PrivacyMode::ConversationOnly, false).is_empty());
        assert_eq!(
            names(&PrivacyMode::ConversationOnly, true),
            vec!["generate_image"]
        );
    }
}
