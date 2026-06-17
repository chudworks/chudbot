//! Message-to-turn orchestration: privacy checks, context preparation, execution, and retries.
//!
//! This module is the platform-neutral path from an incoming platform event to
//! a stored conversation turn. It decides whether a message should wake the
//! bot, resolves privacy and agent configuration, records the durable turn
//! input, runs the selected agent, and posts the terminal platform reply.
//!
//! Reaction handlers live here too because retry and stop/resume requests share
//! the same invariants: retries replay the original turn instead of creating a
//! sibling turn, while stop requests update the conversation and cancel any
//! matching in-flight execution through the runtime cancellation registry.

use crate::prelude::*;
use crate::*;

/// Fully prepared state needed to execute one model-backed turn.
pub(crate) struct TurnExecution {
    /// Conversation that owns the turn and receives live viewer events.
    pub(crate) conversation: Conversation,
    /// Durable turn row that will be driven to one terminal status.
    pub(crate) turn: Turn,
    /// Resolved agent name recorded with the turn input and trace span.
    pub(crate) agent_name: String,
    /// Agent configuration used to build providers, tools, and model request shape.
    pub(crate) agent_config: AgentConfig,
    /// Final system prompt, including any conversation-specific guidance.
    pub(crate) system_prompt: String,
    /// Model transcript prepared from stored conversation state and current context.
    pub(crate) transcript: Transcript,
    /// Runtime privacy/opt-in settings captured for this turn.
    pub(crate) settings: RuntimeSettings,
    /// Platform message that assistant output should reply to.
    pub(crate) reply_to: MessageRef,
    /// Whether this turn opened the conversation and should include first-reply behavior.
    pub(crate) is_new: bool,
    /// Tool traces produced before agent execution, such as audio transcription preflight.
    pub(crate) preflight_tool_traces: Vec<ToolTrace>,
    /// Usage records produced before agent execution.
    pub(crate) preflight_usage: Vec<UsageRecord>,
}

impl TurnExecution {
    /// Return turn usage with preflight usage prepended in execution order.
    pub(crate) fn usage_with_preflight(&self, mut usage: Vec<UsageRecord>) -> Vec<UsageRecord> {
        if self.preflight_usage.is_empty() {
            return usage;
        }
        let mut out = self.preflight_usage.clone();
        out.append(&mut usage);
        out
    }
}

/// Existing conversation found for a new platform message.
#[derive(Debug, Clone)]
pub(crate) struct ExistingConversation {
    /// Loaded conversation snapshot used for continuation and transcript assembly.
    pub(crate) snapshot: ConversationSnapshot,
    /// Lookup route that found the snapshot.
    pub(crate) source: ConversationLookupSource,
}

/// Where a conversation lookup matched, used for audio wake-up decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConversationLookupSource {
    /// The message arrived in a channel already linked to a conversation.
    Channel,
    /// The message explicitly replied to a message linked to a conversation.
    ReferencedMessage,
    /// The current message was already linked to a conversation.
    Message,
}

impl ConversationLookupSource {
    /// Return whether this lookup source should let audio continue a conversation.
    ///
    /// Already-linked current messages are treated as idempotent reprocessing,
    /// not as a fresh audio mention.
    pub(crate) fn counts_as_audio_mention(self) -> bool {
        matches!(self, Self::Channel | Self::ReferencedMessage)
    }
}

/// Context items prepared for the model transcript and persisted trace.
#[derive(Debug, Clone)]
pub(crate) struct PreparedTurnContext {
    /// Ordered context entries saved with the turn input.
    pub(crate) items: Vec<chudbot_api::ContextItem>,
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Handle one platform message and, when eligible, turn it into model work.
    ///
    /// The handler performs all cheap wake-up gates before mutating storage or
    /// showing user-visible reactions. Once a message passes those gates, it is
    /// normalized into the text shape used by transcripts, the author profile is
    /// cached, and `process_mentioned_message` owns the durable turn lifecycle.
    pub async fn handle_message(
        &self,
        mut message: PlatformMessage,
    ) -> Result<BotAction, BotError> {
        let referenced = message.referenced_message_id();
        tracing::debug!(
            author_is_bot = message.author.is_bot,
            mentions = message.mentions.len(),
            mention_profiles = message.mention_profiles.len(),
            attachments = message.attachments.len(),
            content_chars = message.content.chars().count(),
            reference_kind = platform_message_reference_kind(&message.reference),
            reference_platform = ?referenced.map(|message| message.platform.as_str()),
            reference_guild = ?referenced
                .and_then(|message| message.guild_id.as_ref().map(ExternalId::as_str)),
            reference_channel = ?referenced.map(|message| message.channel_id.as_str()),
            reference_message = ?referenced.map(|message| message.message_id.as_str()),
            has_hydrated_reference = message.referenced_message().is_some(),
            "received platform message"
        );
        let bot_user = self
            .platforms
            .bot_user(&message.id.platform)
            .await
            .map_err(platform_error)?;
        let mentioned_bot = message
            .mentions
            .iter()
            .any(|user| same_platform_user(user, &bot_user.id));
        let has_audio_attachments = message_has_audio_attachments(&message);
        if message.author.is_bot || (!mentioned_bot && !has_audio_attachments) {
            tracing::debug!(
                author_is_bot = message.author.is_bot,
                mentioned_bot,
                has_audio_attachments,
                "ignoring message"
            );
            return Ok(BotAction::Ignored);
        }

        // Load runtime policy before deciding whether this message may bind to
        // a conversation or expose surrounding channel context.
        let settings = self.runtime_settings(&message).await?;
        tracing::debug!(
            privacy = privacy_mode_kind(&settings.privacy),
            user_opted_in = settings.user_opted_in,
            "loaded runtime settings"
        );

        // Existing conversation state is part of the privacy decision: a
        // thread or quoted reply can be valid even when the literal event
        // channel differs from the configured channel-only scope.
        let existing = self.lookup_existing_conversation(&message).await?;
        if !self
            .privacy_allows_message_channel(
                &settings.privacy,
                &message.id,
                existing.as_ref().map(|existing| &existing.snapshot),
            )
            .await?
        {
            tracing::debug!(
                privacy = privacy_mode_kind(&settings.privacy),
                "privacy mode rejected message channel"
            );
            return Ok(BotAction::Ignored);
        }
        if let Some(snapshot) = &existing
            && snapshot.snapshot.conversation.stopped_at.is_some()
        {
            tracing::debug!(
                conversation = %snapshot.snapshot.conversation.id,
                stopped_at = ?snapshot.snapshot.conversation.stopped_at,
                "ignoring message because conversation is stopped"
            );
            return Ok(BotAction::Ignored);
        }

        // Audio-only messages can wake the bot only after transcription shows a
        // wake word or the message continues an existing conversation.
        let needs_audio_preflight = has_audio_attachments && !mentioned_bot;
        let resolved_agent = if needs_audio_preflight {
            Some(
                self.resolve_turn_agent(&message, existing.as_ref().map(|e| &e.snapshot))
                    .await?,
            )
        } else {
            None
        };
        let audio_preflight_continues_conversation = needs_audio_preflight
            && existing
                .as_ref()
                .is_some_and(|existing| existing.source.counts_as_audio_mention());
        let should_preflight_audio = needs_audio_preflight
            && no_mention_audio_preflight_enabled(
                resolved_agent.as_ref().map(|(_, agent)| agent),
                audio_preflight_continues_conversation,
            );
        let incoming_audio = if should_preflight_audio {
            self.prepare_incoming_audio_context(
                &message,
                resolved_agent.as_ref().map(|(_, agent)| agent),
            )
            .await?
        } else {
            IncomingAudioContext::default()
        };
        let audio_mentions_wake_word = resolved_agent
            .as_ref()
            .and_then(|(_, agent)| agent.audio_transcription.as_ref())
            .and_then(TranscriptionBinding::wake_word)
            .is_some_and(|wake_word| incoming_audio_mentions_wake_word(&incoming_audio, wake_word));
        let audio_continues_conversation = audio_preflight_continues_conversation
            && !incoming_audio.transcriptions.is_empty()
            && should_preflight_audio;
        if !mentioned_bot && !audio_continues_conversation && !audio_mentions_wake_word {
            tracing::debug!(
                mentioned_bot,
                has_audio_attachments,
                audio_transcriptions = incoming_audio.transcriptions.len(),
                audio_mentions_wake_word,
                audio_continues_conversation,
                should_preflight_audio,
                "ignoring message"
            );
            return Ok(BotAction::Ignored);
        }

        // From here onward the model sees plain message text with normalized
        // mentions and any automatic audio transcription appended.
        append_audio_transcriptions_to_message_content(
            &mut message.content,
            &incoming_audio.transcriptions,
        );
        message.content = normalize_mention_content(
            &message.content,
            &bot_user.id,
            &message.mentions,
            &message.mention_profiles,
        );

        self.storage
            .upsert_user(message.author.clone())
            .await
            .map_err(storage_error)?;
        self.spawn_avatar_download(message.author.clone());
        self.publish_user(message.author.id.clone());

        // From this point the bot has accepted responsibility for a turn, so
        // the user gets a working reaction until the terminal action is known.
        let user_message = message.id.clone();
        self.add_unicode_reaction(&user_message, WORKING_REACTION, "turn_working")
            .await;
        let action = self
            .process_mentioned_message(
                message,
                existing.map(|existing| existing.snapshot),
                settings,
                resolved_agent,
                incoming_audio,
            )
            .await;
        self.remove_own_unicode_reaction(&user_message, WORKING_REACTION, "turn_working")
            .await;
        self.react_for_action(&user_message, &action).await;
        action
    }

    /// Resolve the agent configuration that should handle this message.
    ///
    /// Stored per-conversation, guild, channel, or user overrides are resolved
    /// first. The runtime config then supplies the concrete agent and validates
    /// that its referenced provider services exist before turn construction
    /// proceeds.
    pub(crate) async fn resolve_turn_agent(
        &self,
        message: &PlatformMessage,
        existing: Option<&ConversationSnapshot>,
    ) -> Result<(String, AgentConfig), BotError> {
        let resolved_agent = self
            .storage
            .resolve_agent(ResolveAgent {
                message_provider: message.id.platform.clone(),
                conversation_id: existing.map(|s| s.conversation.id),
                guild_key: guild_key(&message.id),
                channel_key: self
                    .agent_scope_channel(&message.id)
                    .await
                    .channel_id
                    .as_str()
                    .to_string(),
                user_key: message.author.id.user_id.as_str().to_string(),
            })
            .await
            .map_err(storage_error)?;
        let (agent_name, agent_config) = self
            .config
            .agent_or_platform_default(resolved_agent.as_deref(), &message.id.platform)?;
        let agent_config = agent_config.clone();
        self.ensure_agent_services_exist(&agent_name, &agent_config)?;
        tracing::debug!(
            resolved_agent = %agent_name,
            storage_agent = ?resolved_agent,
            provider = %agent_config.provider,
            model = %agent_config.model.id,
            "resolved agent for turn"
        );
        Ok((agent_name, agent_config))
    }

    /// Create or continue a conversation and persist the complete turn input.
    ///
    /// This is the durable boundary for a message that has already passed the
    /// wake-up and privacy gates. It runs moderation, opens a conversation when
    /// needed, begins the user turn, records platform context and transcript
    /// input, then delegates terminal execution to `execute_turn`.
    pub(crate) async fn process_mentioned_message(
        &self,
        message: PlatformMessage,
        existing: Option<ConversationSnapshot>,
        settings: RuntimeSettings,
        resolved_agent: Option<(String, AgentConfig)>,
        incoming_audio: IncomingAudioContext,
    ) -> Result<BotAction, BotError> {
        // Moderation happens before a turn exists so refused messages do not
        // create empty conversations or persisted user-turn rows.
        let user_display_name = display_name(&message);
        if !self.moderation_allows(&message, &user_display_name).await? {
            tracing::info!("message refused by moderation preflight");
            return Ok(BotAction::RefusedMessage);
        }

        // Audio preflight may already have resolved the agent so transcription
        // wake-word settings and the final model turn cannot diverge.
        let (agent_name, agent_config) = match resolved_agent {
            Some(resolved) => resolved,
            None => self.resolve_turn_agent(&message, existing.as_ref()).await?,
        };
        tracing::Span::current().record("agent", tracing::field::display(&agent_name));
        tracing::Span::current()
            .record("provider", tracing::field::display(&agent_config.provider));
        tracing::Span::current().record("model", tracing::field::display(&agent_config.model.id));
        tracing::debug!(
            resolved_agent = %agent_name,
            provider = %agent_config.provider,
            model = %agent_config.model.id,
            "resolved agent for turn"
        );

        // Conversation identity must exist before the final system prompt is
        // composed, because new conversations get a concrete trace URL only
        // after storage assigns the UUID.
        let (snapshot, is_new) = match existing {
            Some(snapshot) => (snapshot, false),
            None => {
                let system_instructions =
                    self.compose_system_prompt(&agent_config, &settings.privacy, None);
                let snapshot = self
                    .storage
                    .open_conversation(OpenConversation {
                        channel: channel_from_message(&message.id),
                        created_by: message.author.id.clone(),
                        root_message: message.id.clone(),
                        initial_model: agent_config.model.id.clone(),
                        agent_name: agent_name.clone(),
                        provider: agent_config.provider.clone(),
                        system_instructions: system_instructions.clone(),
                        title: None,
                    })
                    .await
                    .map_err(storage_error)?;
                self.publish_conversation(snapshot.conversation.id, ConversationEventKind::Created);
                tracing::info!(
                    conversation = %snapshot.conversation.id,
                    "opened new conversation"
                );
                (snapshot, true)
            }
        };
        tracing::Span::current().record(
            "conversation",
            tracing::field::display(snapshot.conversation.id),
        );
        tracing::Span::current().record("is_new", is_new);
        let system_instructions = self.compose_system_prompt(
            &agent_config,
            &settings.privacy,
            Some(snapshot.conversation.id),
        );

        // Begin the durable turn before assembling context so every saved
        // context item, transcript, trace, and platform reply has one owner.
        let turn = self
            .storage
            .begin_turn(BeginTurn {
                conversation_id: snapshot.conversation.id,
                user_message: message.id.clone(),
                user_message_created_at: message.created_at,
                user: message.author.id.clone(),
                user_display_name: display_name(&message),
                user_content: message.content.clone(),
            })
            .await
            .map_err(storage_error)?;
        tracing::Span::current().record("turn", tracing::field::display(turn.id));
        self.publish_conversation(snapshot.conversation.id, ConversationEventKind::TurnStarted);
        tracing::info!(
            conversation = %snapshot.conversation.id,
            turn = %turn.id,
            turn_ordinal = turn.ordinal,
            history_cutoff = ?turn.history_cutoff,
            is_new,
            "started turn"
        );

        self.storage
            .link_message(MessageLink {
                message: message.id.clone(),
                conversation_id: snapshot.conversation.id,
                turn_id: turn.id,
                role: "user".to_string(),
            })
            .await
            .map_err(storage_error)?;
        tracing::debug!("linked user message to turn");

        // Prepare model-visible context from quoted/current messages, including
        // any audio work already performed during wake-up preflight.
        let preflight_tool_traces = incoming_audio.tool_traces();
        let preflight_usage = incoming_audio.usage_records();
        let turn_context = self
            .prepare_turn_context(&message, &settings, &snapshot.conversation, incoming_audio)
            .await?;
        let prompt_snapshot = self
            .storage
            .load_conversation(ConversationLookup::Id {
                id: snapshot.conversation.id,
            })
            .await
            .map_err(storage_error)?
            .ok_or(BotError::MissingConversation {
                conversation_id: snapshot.conversation.id,
            })?;
        let transcript = self
            .transcript_for_turn(&prompt_snapshot, &turn, &turn_context.items)
            .await?;
        tracing::debug!(
            transcript_turns = transcript.turns.len(),
            system_instructions_chars = system_instructions.chars().count(),
            "assembled model transcript"
        );
        // Persist the exact prompt input before model execution so the trace
        // viewer can inspect failed, cancelled, and successful turns uniformly.
        self.storage
            .save_turn_input(SaveTurnInput {
                turn_id: turn.id,
                agent_name: agent_name.clone(),
                provider: agent_config.provider.clone(),
                model: agent_config.model.id.clone(),
                system_instructions: system_instructions.clone(),
                context: turn_context.items,
                transcript: Some(transcript.clone()),
            })
            .await
            .map_err(storage_error)?;
        self.publish_conversation(
            snapshot.conversation.id,
            ConversationEventKind::ContextRecorded,
        );
        tracing::debug!("saved turn input");

        self.execute_turn(TurnExecution {
            conversation: prompt_snapshot.conversation,
            turn,
            agent_name,
            agent_config,
            system_prompt: system_instructions,
            transcript,
            settings,
            reply_to: message.id,
            is_new,
            preflight_tool_traces,
            preflight_usage,
        })
        .await
    }

    #[tracing::instrument(
        name = "bot.handle_reaction",
        skip_all,
        fields(
            platform = %reaction.message.platform,
            guild = ?reaction.message.guild_id,
            channel = %reaction.message.channel_id,
            message = %reaction.message.message_id,
            user = %reaction.user.user_id,
            removed,
        )
    )]
    /// Handle one platform reaction that may request retry or conversation stop.
    ///
    /// The reaction glyph is a user-facing control surface, but only retry and
    /// stop are interpreted here. Stop/resume is admin-only; retry is resolved
    /// through the stored message link so either a failed reply or its turn can
    /// locate the correct conversation.
    pub(crate) async fn handle_reaction(
        &self,
        reaction: PlatformReaction,
        removed: bool,
    ) -> Result<BotAction, BotError> {
        let bot_user = self
            .platforms
            .bot_user(&reaction.message.platform)
            .await
            .map_err(platform_error)?;
        if same_platform_user(&reaction.user, &bot_user.id) {
            tracing::debug!("ignoring bot's own reaction");
            return Ok(BotAction::Ignored);
        }
        let ReactionKind::Unicode { name } = &reaction.reaction else {
            tracing::debug!("ignoring non-unicode reaction");
            return Ok(BotAction::Ignored);
        };
        tracing::debug!(reaction = %name, "handling unicode reaction");

        match (name.as_str(), removed) {
            (RETRY_REACTION, false) => self.retry_from_message(reaction.message).await,
            (STOP_REACTION, _) => {
                if !self.is_admin(&reaction.user) {
                    tracing::debug!("stop reaction ignored because user is not configured admin");
                    return Ok(BotAction::Ignored);
                }
                self.set_stop(reaction.message, reaction.user, !removed)
                    .await
            }
            _ => {
                tracing::debug!(reaction = %name, "ignoring reaction");
                Ok(BotAction::Ignored)
            }
        }
    }

    #[tracing::instrument(
        name = "bot.retry_from_message",
        skip_all,
        fields(
            platform = %message.platform,
            guild = ?message.guild_id,
            channel = %message.channel_id,
            message = %message.message_id,
            conversation = tracing::field::Empty,
            turn = tracing::field::Empty,
            agent = tracing::field::Empty,
        )
    )]
    /// Replay an eligible failed turn from a platform message link.
    ///
    /// A retry reuses the original turn id and stored conversation history
    /// instead of creating a new turn. Stored context is replayed when present,
    /// prior assistant/error messages for that turn are best-effort deleted, and
    /// the turn then follows the same `execute_turn` terminal path as a fresh
    /// message.
    pub(crate) async fn retry_from_message(
        &self,
        message: MessageRef,
    ) -> Result<BotAction, BotError> {
        let Some(link) = self
            .storage
            .load_message_link(message)
            .await
            .map_err(storage_error)?
        else {
            tracing::debug!("retry ignored because message has no link");
            return Ok(BotAction::Ignored);
        };
        tracing::Span::current().record(
            "conversation",
            tracing::field::display(link.conversation_id),
        );
        tracing::Span::current().record("turn", tracing::field::display(link.turn_id));
        // Existing assistant links are captured before prepare_retry mutates
        // turn state so failed platform replies can be removed after the retry
        // is known to be eligible.
        let prior_links = self
            .storage
            .load_message_links_for_turn(link.turn_id)
            .await
            .map_err(storage_error)?;
        if self
            .storage
            .load_conversation(ConversationLookup::Id {
                id: link.conversation_id,
            })
            .await
            .map_err(storage_error)?
            .as_ref()
            .is_some_and(|snapshot| snapshot.conversation.stopped_at.is_some())
        {
            tracing::info!("retry ignored because conversation is stopped");
            return Ok(BotAction::Ignored);
        }
        // Storage owns retry eligibility and state reset. If it declines, the
        // reaction was valid but the turn should not be run again.
        let Some(retry) = self
            .storage
            .prepare_retry(link.turn_id)
            .await
            .map_err(storage_error)?
        else {
            tracing::debug!("retry ignored because turn is not eligible");
            return Ok(BotAction::Ignored);
        };
        let Some(turn_snapshot) = retry
            .conversation
            .turns
            .iter()
            .find(|snapshot| snapshot.turn.id == retry.turn_id)
        else {
            return Err(BotError::MissingRetryTurn {
                turn_id: retry.turn_id,
            });
        };
        let turn = turn_snapshot.turn.clone();
        if retry.conversation.conversation.stopped_at.is_some() {
            tracing::info!("retry ignored because conversation is stopped");
            return Ok(BotAction::Ignored);
        }
        let (agent_name, agent_config) = self
            .config
            .agent_or_platform_default(turn.agent_name.as_deref(), &turn.user_message.platform)?;
        let agent_config = agent_config.clone();
        self.ensure_agent_services_exist(&agent_name, &agent_config)?;
        tracing::Span::current().record("agent", tracing::field::display(&agent_name));
        tracing::debug!(
            provider = %agent_config.provider,
            model = %agent_config.model.id,
            "prepared turn retry"
        );
        // Retry transcripts come from the stored conversation, not a new fetch
        // of channel history, so privacy is narrowed to the conversation itself.
        let settings = RuntimeSettings {
            privacy: PrivacyMode::ConversationOnly,
            user_opted_in: true,
        };
        let system_instructions = turn_snapshot
            .system_instructions
            .clone()
            .unwrap_or_else(|| {
                self.compose_system_prompt(
                    &agent_config,
                    &settings.privacy,
                    Some(retry.conversation.conversation.id),
                )
            });
        let stored_context = replayable_context_items(&turn_snapshot.context);
        let has_stored_context = !stored_context.is_empty();
        let transcript = self
            .transcript_for_retry(
                &retry.conversation,
                turn_snapshot,
                &stored_context,
                has_stored_context,
            )
            .await?;
        // Save the replayed prompt input before deleting visible failure
        // messages; even a retry that later fails should have inspectable input.
        self.storage
            .save_turn_input(SaveTurnInput {
                turn_id: turn.id,
                agent_name: agent_name.clone(),
                provider: agent_config.provider.clone(),
                model: agent_config.model.id.clone(),
                system_instructions: system_instructions.clone(),
                context: stored_context,
                transcript: Some(transcript.clone()),
            })
            .await
            .map_err(storage_error)?;
        self.publish_conversation(
            retry.conversation.conversation.id,
            ConversationEventKind::ContextRecorded,
        );

        // Platform cleanup is best-effort. Storage state is already prepared for
        // retry, so inability to delete an old error message must not block it.
        for link in prior_links
            .iter()
            .filter(|link| link.role.starts_with("assistant"))
        {
            if let Err(error) = self.platforms.delete_message(link.message.clone()).await {
                tracing::warn!(
                    error = %error,
                    message = %link.message.message_id,
                    "failed to delete prior failed reply during retry"
                );
            }
        }

        let retry_user_message = turn.user_message.clone();
        self.remove_own_unicode_reaction(&retry_user_message, ERROR_REACTION, "retry_error_clear")
            .await;
        self.add_unicode_reaction(&retry_user_message, WORKING_REACTION, "retry_working")
            .await;
        let action = self
            .execute_turn(TurnExecution {
                conversation: retry.conversation.conversation,
                turn,
                agent_name,
                agent_config,
                system_prompt: system_instructions,
                transcript,
                settings,
                reply_to: retry_user_message.clone(),
                is_new: false,
                preflight_tool_traces: Vec::new(),
                preflight_usage: Vec::new(),
            })
            .await;
        self.remove_own_unicode_reaction(&retry_user_message, WORKING_REACTION, "retry_working")
            .await;
        self.react_for_action(&retry_user_message, &action).await;
        action
    }

    #[tracing::instrument(
        name = "bot.set_stop",
        skip_all,
        fields(
            platform = %message.platform,
            guild = ?message.guild_id,
            channel = %message.channel_id,
            message = %message.message_id,
            user = %user.user_id,
            stop,
            conversation = tracing::field::Empty,
        )
    )]
    /// Stop or resume the conversation associated with a message or channel.
    ///
    /// Stopping a conversation prevents new work from starting and cooperatively
    /// cancels matching in-flight turns. Resuming only clears the stored stop
    /// marker; it does not restart cancelled work.
    pub(crate) async fn set_stop(
        &self,
        message: MessageRef,
        user: chudbot_api::UserRef,
        stop: bool,
    ) -> Result<BotAction, BotError> {
        let snapshot = self
            .storage
            .load_conversation(ConversationLookup::Message {
                message: message.clone(),
            })
            .await
            .map_err(storage_error)?;
        let snapshot = match snapshot {
            Some(snapshot) => Some(snapshot),
            None => self
                .storage
                .load_conversation(ConversationLookup::Channel {
                    channel: channel_from_message(&message),
                })
                .await
                .map_err(storage_error)?,
        };
        let Some(snapshot) = snapshot else {
            tracing::debug!("stop/resume ignored because message maps to no conversation");
            return Ok(BotAction::Ignored);
        };
        let conversation_id = snapshot.conversation.id;
        tracing::Span::current().record("conversation", tracing::field::display(conversation_id));
        // The durable stop flag is the source of truth. In-memory cancellation
        // is only a follow-up signal for turns currently running in this process.
        let changed = self
            .storage
            .set_conversation_stop(if stop {
                ConversationStop::Stop {
                    conversation_id,
                    stopped_by: user,
                }
            } else {
                ConversationStop::Resume { conversation_id }
            })
            .await
            .map_err(storage_error)?;
        if changed {
            self.publish_conversation(conversation_id, ConversationEventKind::ConversationUpdated);
            if stop {
                let cancelled = self.turn_cancellations.cancel_conversation(conversation_id);
                if cancelled > 0 {
                    tracing::info!(
                        cancelled,
                        "cancelled in-flight turn(s) for stopped conversation"
                    );
                }
            }
            tracing::info!(changed, "conversation stop state updated");
        } else {
            tracing::debug!("conversation stop state was unchanged");
        }
        Ok(if stop {
            BotAction::StoppedConversation { conversation_id }
        } else {
            BotAction::ResumedConversation { conversation_id }
        })
    }

    #[tracing::instrument(
        name = "bot.execute_turn",
        skip_all,
        fields(
            conversation = %execution.conversation.id,
            turn = %execution.turn.id,
            turn_ordinal = execution.turn.ordinal,
            history_cutoff = ?execution.turn.history_cutoff,
            response_ordinal = ?execution.turn.response_ordinal,
            agent = %execution.agent_name,
            provider = %execution.agent_config.provider,
            model = %execution.agent_config.model.id,
            transcript_turns = execution.transcript.turns.len(),
            is_new = execution.is_new,
        )
    )]
    /// Run the agent for a prepared turn and drive it to one terminal state.
    ///
    /// Execution appends preflight traces, rechecks conversation stop state,
    /// registers cooperative cancellation, records model/tool traces, posts the
    /// platform reply or error message, and finally marks the turn completed,
    /// failed, refused, or cancelled in storage.
    pub(crate) async fn execute_turn(
        &self,
        mut execution: TurnExecution,
    ) -> Result<BotAction, BotError> {
        // Audio preflight traces belong at the front of the turn trace because
        // they happened before the model transcript was assembled.
        for (ordinal, trace) in execution.preflight_tool_traces.iter().cloned().enumerate() {
            let trace_kind = tool_trace_kind(&trace);
            self.storage
                .append_tool_trace(
                    execution.turn.id,
                    i32::try_from(ordinal).unwrap_or(i32::MAX),
                    trace,
                )
                .await
                .map_err(storage_error)?;
            self.publish_conversation(
                execution.conversation.id,
                ConversationEventKind::ToolTraceRecorded,
            );
            tracing::trace!(ordinal, trace_kind, "recorded preflight tool trace");
        }
        let preflight_trace_count = execution.preflight_tool_traces.len();
        // Stop can be requested after the turn was prepared but before the
        // agent is built. Recheck storage to avoid launching new provider work.
        if self
            .storage
            .load_conversation(ConversationLookup::Id {
                id: execution.conversation.id,
            })
            .await
            .map_err(storage_error)?
            .as_ref()
            .is_some_and(|snapshot| snapshot.conversation.stopped_at.is_some())
        {
            tracing::info!("turn cancelled because conversation is stopped before execution");
            self.storage
                .finish_turn(FinishTurn::Cancelled {
                    turn_id: execution.turn.id,
                    reason: "cancelled by admin stop reaction".to_string(),
                    usage: execution.usage_with_preflight(Vec::new()),
                })
                .await
                .map_err(storage_error)?;
            self.publish_conversation(
                execution.conversation.id,
                ConversationEventKind::TurnUpdated,
            );
            return Ok(BotAction::CancelledTurn {
                conversation_id: execution.conversation.id,
                turn_id: execution.turn.id,
            });
        }
        tracing::debug!("building agent for turn");
        let agent = self.build_agent(
            &execution.agent_name,
            &execution.agent_config,
            execution.system_prompt.clone(),
            &execution.settings,
            &execution.reply_to,
            &execution.turn.user,
            &execution.turn.user_display_name,
            execution.conversation.id,
            execution.turn.id,
            true,
            &mut Vec::new(),
        )?;
        let transcript = std::mem::take(&mut execution.transcript);
        let replayed_media_refs = media_reply_refs_from_transcript(&transcript).await;
        tracing::info!("running agent");
        let cancel_guard = self
            .turn_cancellations
            .register(execution.conversation.id, execution.turn.id);
        let cancel_token = cancel_guard.token();
        let typing = self.spawn_typing_indicator(channel_from_message(&execution.reply_to));
        // Registration ties this concrete turn to admin stop requests. Dropping
        // the guard unregisters the turn once the provider run is done.
        let run = tokio::select! {
            biased;
            () = cancel_token.cancelled() => {
                tracing::info!("turn cancelled before agent completed");
                None
            }
            run = agent.run(transcript) => Some(run),
        };
        typing.stop().await;
        let Some(run) = run else {
            self.storage
                .finish_turn(FinishTurn::Cancelled {
                    turn_id: execution.turn.id,
                    reason: "cancelled by admin stop reaction".to_string(),
                    usage: execution.usage_with_preflight(Vec::new()),
                })
                .await
                .map_err(storage_error)?;
            self.publish_conversation(
                execution.conversation.id,
                ConversationEventKind::TurnUpdated,
            );
            return Ok(BotAction::CancelledTurn {
                conversation_id: execution.conversation.id,
                turn_id: execution.turn.id,
            });
        };
        // Provider errors before an `AgentRun` still need a terminal storage
        // state so retry controls and the trace viewer remain consistent.
        let run = match run {
            Ok(run) => run,
            Err(error) => {
                tracing::warn!(error = %error, "agent failed before producing run output");
                let message = error.to_string();
                if error_indicates_safety_refusal(&message) {
                    return self
                        .refuse_turn(&execution, "refused by upstream safety")
                        .await;
                }
                return self
                    .fail_turn(&execution, format!("model failed: {message}"))
                    .await;
            }
        };
        drop(cancel_guard);
        tracing::debug!(
            outcome = agent_outcome_kind(&run.outcome),
            model_steps = run.model_steps.len(),
            trace_events = run.trace.len(),
            usage_records = run.all_usage().len(),
            last_model = ?run.last_model_id,
            has_continuation = run.final_continuation.is_some(),
            "agent run completed"
        );

        // Model-step and tool traces are persisted before outcome handling so
        // failed or refused turns still expose the full execution trail.
        for step in run.model_steps.iter().cloned() {
            self.storage
                .append_model_step_trace(execution.turn.id, step)
                .await
                .map_err(storage_error)?;
        }

        for (ordinal, trace) in run.trace.iter().cloned().enumerate() {
            let trace_kind = tool_trace_kind(&trace);
            let storage_ordinal = ordinal.saturating_add(preflight_trace_count);
            self.storage
                .append_tool_trace(
                    execution.turn.id,
                    i32::try_from(storage_ordinal).unwrap_or(i32::MAX),
                    trace,
                )
                .await
                .map_err(storage_error)?;
            self.publish_conversation(
                execution.conversation.id,
                ConversationEventKind::ToolTraceRecorded,
            );
            tracing::trace!(ordinal = storage_ordinal, trace_kind, "recorded tool trace");
        }

        if safety_refusal_in_tool_trace(&run.trace) {
            tracing::info!("turn refused by upstream safety in a client tool");
            return self
                .refuse_turn(&execution, "refused by upstream safety")
                .await;
        }

        // Media emitted in this run and media replayed from prior assistant
        // messages share the same de-duplication path before final reply text is
        // cleaned and attachments are loaded.
        let mut generated_media_refs = generated_media_reply_refs(&run.trace);
        for reference in replayed_media_refs {
            if !generated_media_refs.iter().any(|seen| seen == &reference) {
                generated_media_refs.push(reference);
            }
        }
        let generated_media = generated_reply_media(&self.media_store, &run.trace).await;

        // Only this match writes terminal turn state. Every branch publishes a
        // viewer update after storage reaches its final status for the turn.
        match &run.outcome {
            AgentOutcome::Completed { answer } => {
                let text = strip_generated_media_refs(&answer.text, &generated_media_refs);
                let text = if text.trim().is_empty() {
                    "Done.".to_string()
                } else {
                    text
                };
                let text = append_generated_media_public_urls(text, &generated_media.public_urls);
                let content = self.format_reply(&text, execution.is_new, execution.conversation.id);
                let rendered_lines = rendered_line_count(&content);
                let open_thread = should_thread(
                    execution.is_new,
                    &content,
                    self.config.thread_threshold_chars,
                    self.config.thread_threshold_lines,
                )
                .then(|| ThreadRequest {
                    title: thread_title(&execution),
                });
                let posted = self
                    .platforms
                    .send_message(SendMessage {
                        channel: channel_from_message(&execution.reply_to),
                        reply_to: Some(execution.reply_to.clone()),
                        content: content.clone(),
                        attachments: generated_media.attachments,
                        suppress_embeds: true,
                        open_thread,
                    })
                    .await
                    .map_err(platform_error)?;
                tracing::info!(
                    reply_message = %posted.id.message_id,
                    reply_channel = %posted.channel.channel_id,
                    answer_chars = content.chars().count(),
                    rendered_lines,
                    thread_threshold_chars = self.config.thread_threshold_chars,
                    thread_threshold_lines = self.config.thread_threshold_lines,
                    opened_thread = posted.channel != channel_from_message(&execution.reply_to),
                    "posted assistant reply"
                );
                self.storage
                    .link_message(MessageLink {
                        message: posted.id.clone(),
                        conversation_id: execution.conversation.id,
                        turn_id: execution.turn.id,
                        role: "assistant".to_string(),
                    })
                    .await
                    .map_err(storage_error)?;
                for message in &posted.extra_messages {
                    self.storage
                        .link_message(MessageLink {
                            message: message.clone(),
                            conversation_id: execution.conversation.id,
                            turn_id: execution.turn.id,
                            role: "assistant".to_string(),
                        })
                        .await
                        .map_err(storage_error)?;
                }
                if posted.channel != channel_from_message(&execution.reply_to) {
                    self.storage
                        .link_channel(ChannelLink {
                            channel: posted.channel.clone(),
                            conversation_id: execution.conversation.id,
                            turn_id: execution.turn.id,
                            role: "thread".to_string(),
                        })
                        .await
                        .map_err(storage_error)?;
                    tracing::debug!(
                        thread_channel = %posted.channel.channel_id,
                        "linked thread channel to conversation"
                    );
                }
                self.storage
                    .finish_turn(FinishTurn::Completed {
                        turn_id: execution.turn.id,
                        assistant_content: content,
                        assistant_message: posted.id,
                        usage: execution.usage_with_preflight(run.all_usage()),
                    })
                    .await
                    .map_err(storage_error)?;
                self.publish_conversation(
                    execution.conversation.id,
                    ConversationEventKind::TurnUpdated,
                );
                if execution.turn.ordinal == 0 && execution.conversation.title.is_none() {
                    self.spawn_title_generation(
                        execution.conversation.id,
                        execution.agent_name.clone(),
                    );
                }
                tracing::info!("turn completed");
                Ok(BotAction::CompletedTurn {
                    conversation_id: execution.conversation.id,
                    turn_id: execution.turn.id,
                })
            }
            AgentOutcome::IterationLimit { max_iterations } => {
                tracing::warn!(max_iterations, "turn hit agent iteration limit");
                self.fail_turn(
                    &execution,
                    format!("model hit iteration limit ({max_iterations})"),
                )
                .await
            }
            AgentOutcome::Failed { error, partial } => {
                tracing::warn!(
                    error = %error,
                    has_partial = partial.is_some(),
                    "agent reported failed outcome"
                );
                let mut message = error.to_string();
                if let Some(partial) = partial
                    && !partial.text.trim().is_empty()
                {
                    message.push_str("\n\nPartial answer:\n");
                    message.push_str(&partial.text);
                }
                self.fail_turn(&execution, message).await
            }
            AgentOutcome::Cancelled { reason } => {
                tracing::info!(reason = %reason, "turn cancelled");
                self.storage
                    .finish_turn(FinishTurn::Cancelled {
                        turn_id: execution.turn.id,
                        reason: reason.clone(),
                        usage: execution.usage_with_preflight(run.all_usage()),
                    })
                    .await
                    .map_err(storage_error)?;
                self.publish_conversation(
                    execution.conversation.id,
                    ConversationEventKind::TurnUpdated,
                );
                Ok(BotAction::CancelledTurn {
                    conversation_id: execution.conversation.id,
                    turn_id: execution.turn.id,
                })
            }
        }
    }

    /// Mark a turn failed, post a visible error reply, and expose retry affordance.
    ///
    /// The failure reply is linked with an `assistant_error` role so a later
    /// retry can remove it without confusing it for successful assistant output.
    pub(crate) async fn fail_turn(
        &self,
        execution: &TurnExecution,
        error: String,
    ) -> Result<BotAction, BotError> {
        let content = format!("Warning: {error}");
        let posted = self
            .platforms
            .send_message(SendMessage {
                channel: channel_from_message(&execution.reply_to),
                reply_to: Some(execution.reply_to.clone()),
                content,
                attachments: Vec::new(),
                suppress_embeds: true,
                open_thread: None,
            })
            .await
            .map_err(platform_error)?;
        tracing::info!(
            error_message = %posted.id.message_id,
            channel = %posted.channel.channel_id,
            "posted turn failure reply"
        );
        self.storage
            .finish_turn(FinishTurn::Failed {
                turn_id: execution.turn.id,
                error,
                assistant_content: None,
                assistant_message: Some(posted.id.clone()),
                usage: execution.usage_with_preflight(Vec::new()),
            })
            .await
            .map_err(storage_error)?;
        self.storage
            .link_message(MessageLink {
                message: posted.id.clone(),
                conversation_id: execution.conversation.id,
                turn_id: execution.turn.id,
                role: "assistant_error".to_string(),
            })
            .await
            .map_err(storage_error)?;
        for message in &posted.extra_messages {
            self.storage
                .link_message(MessageLink {
                    message: message.clone(),
                    conversation_id: execution.conversation.id,
                    turn_id: execution.turn.id,
                    role: "assistant_error".to_string(),
                })
                .await
                .map_err(storage_error)?;
        }
        if let Err(error) = self
            .platforms
            .add_reaction(
                posted.id,
                ReactionKind::Unicode {
                    name: RETRY_REACTION.to_string(),
                },
            )
            .await
        {
            tracing::warn!(error = %error, "failed to add retry reaction to failed reply");
        }
        self.publish_conversation(
            execution.conversation.id,
            ConversationEventKind::TurnUpdated,
        );
        tracing::warn!("turn marked failed");
        Ok(BotAction::FailedTurn {
            conversation_id: execution.conversation.id,
            turn_id: execution.turn.id,
        })
    }

    #[tracing::instrument(
        name = "bot.refuse_turn",
        skip_all,
        fields(
            conversation = %execution.conversation.id,
            turn = %execution.turn.id,
            agent = %execution.agent_name,
        )
    )]
    /// Mark a turn as refused without posting an assistant error message.
    ///
    /// Safety refusals can occur after a durable turn exists. They are stored as
    /// failed turns for trace visibility, but return the same high-level action
    /// as moderation refusals so reaction handling uses the refusal status.
    pub(crate) async fn refuse_turn(
        &self,
        execution: &TurnExecution,
        reason: &str,
    ) -> Result<BotAction, BotError> {
        self.storage
            .finish_turn(FinishTurn::Failed {
                turn_id: execution.turn.id,
                error: reason.to_string(),
                assistant_content: None,
                assistant_message: None,
                usage: execution.usage_with_preflight(Vec::new()),
            })
            .await
            .map_err(storage_error)?;
        self.publish_conversation(
            execution.conversation.id,
            ConversationEventKind::TurnUpdated,
        );
        Ok(BotAction::RefusedMessage)
    }

    /// Start a background typing indicator that refreshes until explicitly stopped.
    pub(crate) fn spawn_typing_indicator(&self, channel: ChannelRef) -> TypingIndicator {
        let platforms = self.platforms.clone();
        let stop = CancellationToken::new();
        let stopped = stop.clone();
        let task = tokio::spawn(async move {
            loop {
                if let Err(error) = platforms.typing(channel.clone()).await {
                    tracing::warn!(
                        error = %error,
                        channel = %channel.channel_id,
                        "failed to send typing indicator"
                    );
                }
                tokio::select! {
                    biased;
                    () = stopped.cancelled() => break,
                    () = tokio::time::sleep(TYPING_REFRESH_INTERVAL) => {}
                }
            }
        });
        TypingIndicator { stop, task }
    }

    /// Load the privacy mode and opt-in flag that apply to this message author.
    pub(crate) async fn runtime_settings(
        &self,
        message: &PlatformMessage,
    ) -> Result<RuntimeSettings, BotError> {
        let settings = self
            .storage
            .runtime_settings(
                message.id.platform.clone(),
                guild_key(&message.id),
                message.author.id.user_id.as_str().to_string(),
            )
            .await
            .map_err(storage_error)?;
        tracing::trace!(
            platform = %message.id.platform,
            guild = ?message.id.guild_id,
            user = %message.author.id.user_id,
            privacy = privacy_mode_kind(&settings.privacy),
            opted_in = settings.user_opted_in,
            "runtime settings loaded"
        );
        Ok(settings)
    }

    /// Resolve the channel key used by channel-scoped agent overrides.
    ///
    /// Platform threads inherit their parent channel's agent setting. If the
    /// platform lookup fails, the event channel remains the conservative scope.
    pub(crate) async fn agent_scope_channel(&self, message: &MessageRef) -> ChannelRef {
        let channel = channel_from_message(message);
        match self.platforms.parent_channel(channel.clone()).await {
            Ok(parent) => parent,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    platform = %message.platform,
                    channel = %message.channel_id,
                    "failed to resolve parent channel for agent scope; using event channel"
                );
                channel
            }
        }
    }

    /// Check whether the message channel is allowed under the active privacy mode.
    ///
    /// Only `channel_only` constrains the channel here. Existing conversations
    /// are allowed to continue from linked thread/reply surfaces, while new work
    /// must match either the configured channel or its platform parent.
    pub(crate) async fn privacy_allows_message_channel(
        &self,
        mode: &PrivacyMode,
        message: &MessageRef,
        existing: Option<&ConversationSnapshot>,
    ) -> Result<bool, BotError> {
        let PrivacyMode::ChannelOnly {
            channel: allowed, ..
        } = mode
        else {
            return Ok(true);
        };
        let actual = channel_from_message(message);
        if &actual == allowed {
            return Ok(true);
        }
        if existing.is_some() {
            tracing::debug!(
                actual_channel = %actual.channel_id,
                allowed_channel = %allowed.channel_id,
                "allowing channel_only message because it continues an existing conversation"
            );
            return Ok(true);
        }
        let parent = self
            .platforms
            .parent_channel(actual)
            .await
            .map_err(platform_error)?;
        Ok(&parent == allowed)
    }

    #[tracing::instrument(
        name = "bot.lookup_conversation",
        skip_all,
        fields(
            platform = %message.id.platform,
            guild = ?message.id.guild_id,
            channel = %message.id.channel_id,
            message = %message.id.message_id,
            has_reference = message.referenced_message_id().is_some(),
        )
    )]
    /// Locate the conversation, if any, that an incoming message should continue.
    ///
    /// Lookup order is intentional: current channel first for thread-style
    /// continuations, referenced message second for replies, and the current
    /// message link last for idempotent reprocessing of an already-linked event.
    pub(crate) async fn lookup_existing_conversation(
        &self,
        message: &PlatformMessage,
    ) -> Result<Option<ExistingConversation>, BotError> {
        // Channel lookup catches active thread/channel continuations before any
        // quoted-message relationship is considered.
        let channel = channel_from_message(&message.id);
        tracing::debug!(
            lookup = "channel",
            lookup_platform = %channel.platform,
            lookup_guild = ?channel.guild_id.as_ref().map(ExternalId::as_str),
            lookup_channel = %channel.channel_id,
            "looking up existing conversation by channel"
        );
        if let Some(snapshot) = self
            .storage
            .load_conversation(ConversationLookup::Channel {
                channel: channel.clone(),
            })
            .await
            .map_err(storage_error)?
        {
            tracing::debug!(
                conversation = %snapshot.conversation.id,
                source = "channel",
                "found existing conversation"
            );
            return Ok(Some(ExistingConversation {
                snapshot,
                source: ConversationLookupSource::Channel,
            }));
        }
        tracing::debug!(
            lookup = "channel",
            lookup_platform = %channel.platform,
            lookup_guild = ?channel.guild_id.as_ref().map(ExternalId::as_str),
            lookup_channel = %channel.channel_id,
            "no existing conversation found by channel"
        );

        // Referenced-message lookup lets an explicit reply join the conversation
        // that produced or linked the replied-to platform message.
        if let Some(referenced) = message.referenced_message_id().cloned() {
            tracing::debug!(
                lookup = "referenced_message",
                reference_kind = platform_message_reference_kind(&message.reference),
                lookup_platform = %referenced.platform,
                lookup_guild = ?referenced.guild_id.as_ref().map(ExternalId::as_str),
                lookup_channel = %referenced.channel_id,
                lookup_message = %referenced.message_id,
                "looking up existing conversation by referenced message"
            );
            if let Some(snapshot) = self
                .storage
                .load_conversation(ConversationLookup::Message {
                    message: referenced.clone(),
                })
                .await
                .map_err(storage_error)?
            {
                tracing::debug!(
                    conversation = %snapshot.conversation.id,
                    source = "referenced_message",
                    lookup_platform = %referenced.platform,
                    lookup_guild = ?referenced.guild_id.as_ref().map(ExternalId::as_str),
                    lookup_channel = %referenced.channel_id,
                    lookup_message = %referenced.message_id,
                    "found existing conversation"
                );
                return Ok(Some(ExistingConversation {
                    snapshot,
                    source: ConversationLookupSource::ReferencedMessage,
                }));
            }
            tracing::debug!(
                lookup = "referenced_message",
                reference_kind = platform_message_reference_kind(&message.reference),
                lookup_platform = %referenced.platform,
                lookup_guild = ?referenced.guild_id.as_ref().map(ExternalId::as_str),
                lookup_channel = %referenced.channel_id,
                lookup_message = %referenced.message_id,
                "no existing conversation found by referenced message"
            );
        } else {
            tracing::debug!(
                reference_kind = platform_message_reference_kind(&message.reference),
                "skipping referenced-message lookup because no referenced message id was available"
            );
        }

        // Current-message lookup handles repeated delivery after the message has
        // already been linked to a turn.
        tracing::debug!(
            lookup = "message",
            lookup_platform = %message.id.platform,
            lookup_guild = ?message.id.guild_id.as_ref().map(ExternalId::as_str),
            lookup_channel = %message.id.channel_id,
            lookup_message = %message.id.message_id,
            "looking up existing conversation by current message"
        );
        let snapshot = self
            .storage
            .load_conversation(ConversationLookup::Message {
                message: message.id.clone(),
            })
            .await
            .map_err(storage_error)?;
        if let Some(snapshot) = &snapshot {
            tracing::debug!(
                conversation = %snapshot.conversation.id,
                source = "message",
                lookup_platform = %message.id.platform,
                lookup_guild = ?message.id.guild_id.as_ref().map(ExternalId::as_str),
                lookup_channel = %message.id.channel_id,
                lookup_message = %message.id.message_id,
                "found existing conversation"
            );
        } else {
            tracing::debug!(
                lookup = "message",
                lookup_platform = %message.id.platform,
                lookup_guild = ?message.id.guild_id.as_ref().map(ExternalId::as_str),
                lookup_channel = %message.id.channel_id,
                lookup_message = %message.id.message_id,
                "no existing conversation found by current message"
            );
        }
        Ok(snapshot.map(|snapshot| ExistingConversation {
            snapshot,
            source: ConversationLookupSource::Message,
        }))
    }

    /// Build persisted context items for the quoted message and current message.
    ///
    /// Quoted user messages are included only when privacy allows them. Quoted
    /// assistant replies from the same conversation are skipped because the
    /// transcript already replays those assistant turns.
    pub(crate) async fn prepare_turn_context(
        &self,
        message: &PlatformMessage,
        settings: &RuntimeSettings,
        conversation: &Conversation,
        incoming_audio: IncomingAudioContext,
    ) -> Result<PreparedTurnContext, BotError> {
        let mut items = Vec::new();
        let mut position = 0;

        // Quoted context is useful for reply semantics, but must not duplicate
        // an assistant answer already present in the conversation transcript.
        if let Some(referenced) = message.referenced_message()
            && self
                .quoted_message_allowed(referenced, settings, conversation)
                .await?
            && !self
                .quoted_assistant_message_already_replays(referenced, conversation)
                .await?
        {
            self.push_message_context(
                &mut items,
                &mut position,
                MessageContextInput {
                    kind: "quoted",
                    message: referenced,
                    relationship: PlatformMessageRelationship::Referenced,
                    saved_audio: None,
                    audio_transcriptions: &[],
                },
            )
            .await?;
        }

        // Audio may have been saved during wake-up preflight. Empty media marks
        // "already handled but not exposed" so push_message_context does not
        // save the same attachments a second time.
        let current_audio_media = match incoming_audio.saved_audio {
            Some(audio_media) if incoming_audio.expose_audio_to_model => Some(audio_media),
            Some(_) => Some(Vec::new()),
            None => None,
        };

        self.push_message_context(
            &mut items,
            &mut position,
            MessageContextInput {
                kind: "message",
                message,
                relationship: PlatformMessageRelationship::Current,
                saved_audio: current_audio_media,
                audio_transcriptions: &incoming_audio.transcriptions,
            },
        )
        .await?;

        Ok(PreparedTurnContext { items })
    }

    /// Save message media and append one platform message context block.
    ///
    /// The first item is the platform adapter's structured message context.
    /// Follow-up items expose stored image/audio URIs and a concise image-ref
    /// hint so model tools can address attachments by stable media ids.
    pub(crate) async fn push_message_context(
        &self,
        items: &mut Vec<chudbot_api::ContextItem>,
        position: &mut i32,
        input: MessageContextInput<'_>,
    ) -> Result<(), BotError> {
        let MessageContextInput {
            kind,
            message,
            relationship,
            saved_audio,
            audio_transcriptions,
        } = input;

        // Attachments are copied into the media store before the platform JSON
        // is serialized so stable URIs can be injected into the context value.
        let image_media = self
            .save_matching_attachments(message, MediaCategory::Image, "image", looks_like_image_ref)
            .await;
        let audio_media = match saved_audio {
            Some(audio_media) => audio_media,
            None => {
                self.save_matching_attachments(
                    message,
                    MediaCategory::Audio,
                    "audio",
                    looks_like_audio_ref,
                )
                .await
            }
        };

        let mut value = self
            .platforms
            .message_context(message, relationship)
            .await
            .map_err(platform_error)?;
        inject_audio_attachment_refs(&mut value, &audio_media);
        inject_audio_transcriptions(&mut value, audio_transcriptions);
        let content = serde_json::to_string_pretty(&value).unwrap_or_else(|_| value.to_string());
        items.push(chudbot_api::ContextItem {
            position: *position,
            source: format!("platform:{kind}:{}", message.id.message_id.as_str()),
            role: "user".to_string(),
            content,
            message: Some(message.id.clone()),
        });
        *position += 1;

        // Media URI items follow the message JSON in stable attachment order,
        // preserving the position sequence used by transcript assembly.
        let mut image_refs = Vec::new();
        for saved in image_media {
            let uri = saved.media.uri().to_string();
            image_refs.push(uri.clone());
            items.push(chudbot_api::ContextItem {
                position: *position,
                source: format!(
                    "platform:{kind}:{}:image:{}",
                    message.id.message_id.as_str(),
                    saved.attachment_index
                ),
                role: "user".to_string(),
                content: uri,
                message: Some(message.id.clone()),
            });
            *position += 1;
        }
        for saved in &audio_media {
            items.push(chudbot_api::ContextItem {
                position: *position,
                source: format!(
                    "platform:{kind}:{}:audio:{}",
                    message.id.message_id.as_str(),
                    saved.attachment_index
                ),
                role: "user".to_string(),
                content: saved.media.uri().to_string(),
                message: Some(message.id.clone()),
            });
            *position += 1;
        }
        if !image_refs.is_empty() {
            items.push(chudbot_api::ContextItem {
                position: *position,
                source: format!(
                    "platform:{kind}:{}:image_refs",
                    message.id.message_id.as_str()
                ),
                role: "user".to_string(),
                content: format!(
                    "Image attachment reference IDs available for tool calls: {}",
                    image_refs.join(", ")
                ),
                message: Some(message.id.clone()),
            });
            *position += 1;
        }
        Ok(())
    }

    /// Decide whether a referenced message may be included as quoted context.
    ///
    /// Open and channel-only modes allow quoted context. Conversation-only mode
    /// suppresses it. Opt-in mode allows bot messages, messages already linked
    /// to this conversation, or opted-in guild users.
    pub(crate) async fn quoted_message_allowed(
        &self,
        referenced: &PlatformMessage,
        settings: &RuntimeSettings,
        conversation: &Conversation,
    ) -> Result<bool, BotError> {
        match &settings.privacy {
            PrivacyMode::Open { .. } | PrivacyMode::ChannelOnly { .. } => Ok(true),
            PrivacyMode::ConversationOnly => Ok(false),
            PrivacyMode::OptIn => {
                if referenced.author.is_bot {
                    return Ok(true);
                }
                if self
                    .storage
                    .load_conversation(ConversationLookup::Channel {
                        channel: channel_from_message(&referenced.id),
                    })
                    .await
                    .map_err(storage_error)?
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.conversation.id == conversation.id)
                {
                    return Ok(true);
                }
                let Some(guild) = referenced.id.guild_id.as_ref() else {
                    return Ok(false);
                };
                self.storage
                    .user_privacy(
                        referenced.id.platform.clone(),
                        guild.as_str().to_string(),
                        referenced.author.id.user_id.as_str().to_string(),
                    )
                    .await
                    .map_err(storage_error)
                    .map(|opted_in| opted_in.unwrap_or(false))
            }
        }
    }

    /// Check whether a quoted assistant message is already present in transcript history.
    pub(crate) async fn quoted_assistant_message_already_replays(
        &self,
        referenced: &PlatformMessage,
        conversation: &Conversation,
    ) -> Result<bool, BotError> {
        let Some(link) = self
            .storage
            .load_message_link(referenced.id.clone())
            .await
            .map_err(storage_error)?
        else {
            return Ok(false);
        };
        let already_replays = message_link_replays_as_assistant(&link, conversation.id);
        if already_replays {
            tracing::trace!(
                conversation = %conversation.id,
                message = %referenced.id.message_id,
                "skipping quoted assistant message already present in transcript"
            );
        }
        Ok(already_replays)
    }
}
