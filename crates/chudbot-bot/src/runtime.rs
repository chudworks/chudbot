//! Bot runtime construction, event loop plumbing, and task lifecycle helpers.
//!
//! The runtime is the platform-neutral owner of Chudbot's live services. It
//! accepts concrete registries from `chudbot-bin`, receives platform events,
//! fans those events out to per-event tasks, and coordinates orderly shutdown
//! for memory jobs, event handlers, and background work.

use std::ops::Deref;

use dashmap::DashMap;

use crate::prelude::*;
use crate::*;

/// Compile-time service types for a bot runtime.
///
/// `chudbot-bot` stays generic over concrete platform, storage, provider, and
/// event implementations. The binary crate supplies those concrete types while
/// this crate keeps orchestration static and provider-neutral.
pub trait BotRuntimeTypes {
    /// Registry that performs cloneable platform I/O such as replies and history fetches.
    type Platforms: MessagePlatformRegistry + Clone + Send + Sync + 'static;
    /// Durable storage implementation for conversations, turns, settings, and jobs.
    type Storage: BotStorage + Clone + Send + Sync + 'static;
    /// Media store used for incoming attachments and generated assets.
    type Media: MediaStore + Clone + Send + Sync + 'static;
    /// LLM provider registry keyed by runtime provider name.
    type Llms: LlmProviderRegistry + Clone + Send + Sync + 'static;
    /// Image provider registry keyed by runtime provider name.
    type Images: ImageGeneratorRegistry + Clone + Send + Sync + 'static;
    /// Video provider registry keyed by runtime provider name.
    type Videos: VideoGeneratorRegistry + Clone + Send + Sync + 'static;
    /// Audio transcription provider registry keyed by runtime provider name.
    type Audio: AudioTranscriberRegistry + Clone + Send + Sync + 'static;
    /// Live event sink for the trace viewer and other subscribers.
    type Events: EventSink + Clone + Send + Sync + 'static;
}

/// Platform-neutral bot runtime and shared service handle.
pub struct BotRuntime<R: BotRuntimeTypes> {
    inner: Arc<BotRuntimeInner<R>>,
}

impl<R> std::fmt::Debug for BotRuntime<R>
where
    R: BotRuntimeTypes,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BotRuntime").finish_non_exhaustive()
    }
}

impl<R> Clone for BotRuntime<R>
where
    R: BotRuntimeTypes,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<R> Deref for BotRuntime<R>
where
    R: BotRuntimeTypes,
{
    type Target = BotRuntimeInner<R>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Shared runtime state behind [`BotRuntime`].
#[doc(hidden)]
#[derive(Debug)]
pub struct BotRuntimeInner<R: BotRuntimeTypes> {
    /// Cloneable platform registry used for platform-side effects.
    pub(crate) platforms: R::Platforms,
    /// Durable storage used by turn handling, commands, and background jobs.
    pub(crate) storage: R::Storage,
    /// Media store for attachments, generated media, and model-visible assets.
    pub(crate) media_store: R::Media,
    /// Runtime-routed LLM providers.
    pub(crate) llms: R::Llms,
    /// Runtime-routed image generators.
    pub(crate) images: R::Images,
    /// Runtime-routed video generators.
    pub(crate) videos: R::Videos,
    /// Runtime-routed audio transcribers.
    pub(crate) audio: R::Audio,
    /// Publisher for live trace-viewer events.
    pub(crate) events: R::Events,
    /// Background tasks that outlive a single platform event.
    pub(crate) background: TaskTracker,
    /// Active turn cancellation tokens, keyed by conversation and turn.
    pub(crate) turn_cancellations: TurnCancellations,
    /// Single-node quota locks for video generation submit/check sections.
    pub(crate) video_rate_limit_locks: VideoRateLimitLocks,
    /// Shared HTTP client for runtime downloads such as avatars and media.
    pub(crate) download_http: reqwest::Client,
    /// Resolved bot configuration used by all runtime paths.
    pub(crate) config: BotConfig,
    /// User-memory runtime configuration passed to memory workers and tools.
    pub(crate) memory_config: memory::MemoryConfig,
    /// Reserved agent configs cached at startup to avoid per-turn resolution.
    pub(crate) system_agents: RuntimeSystemAgents,
}

/// Runtime service implementations supplied by the binary crate.
///
/// This is the construction boundary between concrete service setup in
/// `chudbot-bin` and platform-neutral orchestration in `chudbot-bot`.
#[derive(Debug)]
pub struct BotRuntimeParts<R: BotRuntimeTypes> {
    /// Message platform registry that performs cloneable platform I/O.
    pub platforms: R::Platforms,
    /// Durable bot storage for conversations, turns, settings, and jobs.
    pub storage: R::Storage,
    /// Media storage for incoming and generated assets.
    pub media_store: R::Media,
    /// LLM provider registry available to configured agents.
    pub llms: R::Llms,
    /// Image generation registry available to client tools.
    pub images: R::Images,
    /// Video generation registry available to client tools.
    pub videos: R::Videos,
    /// Audio transcription registry available to audio ingestion and tools.
    pub audio: R::Audio,
    /// Live event sink used to update trace-viewer subscribers.
    pub events: R::Events,
    /// User-memory runtime configuration supplied by the config loader.
    pub memory: memory::MemoryConfig,
}

/// In-memory cancellation registry for currently running turns.
///
/// The registry is intentionally process-local. It lets commands or reactions
/// cancel active work in this process without changing the durable conversation
/// model. Each `register` call must be paired with `unregister`; callers get
/// that guarantee by holding the returned [`TurnCancellationGuard`].
#[derive(Debug, Clone, Default)]
pub(crate) struct TurnCancellations {
    inner: Arc<DashMap<(ConversationId, TurnId), CancellationToken>>,
}

impl TurnCancellations {
    /// Register a turn and return the guard that removes it on drop.
    pub(crate) fn register(
        &self,
        conversation_id: ConversationId,
        turn_id: TurnId,
    ) -> TurnCancellationGuard {
        let token = CancellationToken::new();
        self.inner.insert((conversation_id, turn_id), token.clone());
        TurnCancellationGuard {
            registry: self.clone(),
            conversation_id,
            turn_id,
            token,
        }
    }

    /// Remove a turn from the registry after it finishes or is abandoned.
    pub(crate) fn unregister(&self, conversation_id: ConversationId, turn_id: TurnId) {
        self.inner.remove(&(conversation_id, turn_id));
    }

    /// Cancel every currently registered turn in a conversation.
    pub(crate) fn cancel_conversation(&self, conversation_id: ConversationId) -> usize {
        // Clone tokens before cancelling so waking waiters never extends the
        // registry critical section held by DashMap shard guards.
        let tokens = self
            .inner
            .iter()
            .filter_map(|entry| {
                let (entry_conversation_id, _) = *entry.key();
                (entry_conversation_id == conversation_id).then(|| entry.value().clone())
            })
            .collect::<Vec<_>>();
        let count = tokens.len();
        for token in tokens {
            token.cancel();
        }
        count
    }
}

/// RAII registration for one cancellable turn.
///
/// Dropping this guard is the only cleanup path the caller needs: completed,
/// failed, and cancelled turns all leave the registry through `Drop`.
#[derive(Debug)]
pub(crate) struct TurnCancellationGuard {
    registry: TurnCancellations,
    conversation_id: ConversationId,
    turn_id: TurnId,
    token: CancellationToken,
}

impl TurnCancellationGuard {
    /// Return the cancellation token that turn execution should poll.
    pub(crate) fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

impl Drop for TurnCancellationGuard {
    fn drop(&mut self) {
        self.registry.unregister(self.conversation_id, self.turn_id);
    }
}

/// Background typing indicator task for a single platform channel.
///
/// The task is deliberately owned separately from the global `TaskTracker`
/// because the corresponding turn path needs to stop it at a precise point.
#[derive(Debug)]
pub(crate) struct TypingIndicator {
    /// Token observed by the task loop.
    pub(crate) stop: CancellationToken,
    /// Spawned task that sends periodic typing notifications.
    pub(crate) task: JoinHandle<()>,
}

impl TypingIndicator {
    /// Request the typing task to stop and log any join failure.
    pub(crate) async fn stop(self) {
        self.stop.cancel();
        if let Err(error) = self.task.await {
            log_task_join_error("typing indicator", &error);
        }
    }
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes,
{
    /// Construct a bot runtime from concrete service parts and resolved config.
    ///
    /// This is the startup boundary for process-local state. It creates fresh
    /// task/cancellation trackers and caches reserved system-agent resolution so
    /// event handling can use the effective config without recomputing it for
    /// every message.
    pub fn new(parts: BotRuntimeParts<R>, config: BotConfig) -> Self {
        tracing::debug!(
            agents = config.agents.len(),
            platforms = config.platforms.len(),
            default_agent = %config.default_agent,
            "constructing bot runtime"
        );
        // Reserved agents inherit from normal agent config, but the resolved
        // view is immutable for this process and belongs at runtime startup.
        let system_agents = RuntimeSystemAgents::from_config(&config);
        Self {
            inner: Arc::new(BotRuntimeInner {
                platforms: parts.platforms,
                storage: parts.storage,
                media_store: parts.media_store,
                llms: parts.llms,
                images: parts.images,
                videos: parts.videos,
                audio: parts.audio,
                events: parts.events,
                background: TaskTracker::new(),
                turn_cancellations: TurnCancellations::default(),
                video_rate_limit_locks: VideoRateLimitLocks::default(),
                download_http: reqwest::Client::new(),
                config,
                memory_config: parts.memory,
                system_agents,
            }),
        }
    }

    /// Borrow the resolved bot config used by this runtime.
    pub fn config(&self) -> &BotConfig {
        &self.config
    }
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Run the platform event loop with explicit shutdown behavior.
    ///
    /// The loop registers commands, starts the memory runtime, then repeatedly
    /// accepts platform events. Each event is handled in its own task so slow
    /// turns do not block gateway intake. Shutdown stops intake first, then
    /// drains in-flight event work, background work, and finally the platform.
    #[tracing::instrument(
        name = "bot.run_with_options",
        skip_all,
        fields(
            agents = self.config.agents.len(),
            platforms = self.config.platforms.len(),
            default_agent = %self.config.default_agent,
            drain_timeout_ms = options.drain_timeout.as_millis(),
        )
    )]
    pub async fn run_with_options<E>(
        self,
        mut platform_events: E,
        shutdown: CancellationToken,
        options: BotRunOptions,
    ) -> Result<(), BotError>
    where
        E: MessagePlatformEvents<Error = <R::Platforms as MessagePlatformRegistry>::Error>
            + Send
            + 'static,
    {
        // 1. Register command definitions before accepting user events so new
        // platform sessions expose the expected slash-command surface.
        self.platforms
            .register_commands(command_definitions())
            .await
            .map_err(platform_error)?;
        // 2. Memory jobs share the parent shutdown signal but can be cancelled
        // early on platform-stream failure before returning the error.
        let memory_shutdown = shutdown.child_token();
        self.spawn_memory_runtime(memory_shutdown.clone());
        tracing::info!("bot event loop starting");
        let mut tasks = JoinSet::new();
        loop {
            tokio::select! {
                biased;
                // 3. External shutdown wins over new intake and completed task
                // logging, which keeps service termination responsive.
                _ = shutdown.cancelled() => {
                    tracing::info!("bot shutdown requested; stopping platform event intake");
                    break;
                }
                // 4. Opportunistically log completed event tasks while intake
                // continues; any still running tasks are drained after break.
                Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                    log_event_task_result(result);
                }
                // 5. Platform errors stop auxiliary memory work and surface as
                // runtime errors; normal shutdown events just break intake.
                event = platform_events.next_event() => {
                    let event = match event {
                        Ok(event) => event,
                        Err(error) => {
                            memory_shutdown.cancel();
                            return Err(platform_error(error));
                        }
                    };
                    tracing::trace!(
                        event = platform_event_kind(&event),
                        "received platform event"
                    );
                    if matches!(event, PlatformEvent::Shutdown) {
                        tracing::info!("platform event stream requested shutdown");
                        break;
                    }
                    // 6. Event work uses a cheap runtime clone and reports its
                    // result through the JoinSet for centralized logging.
                    let runtime = self.clone();
                    tasks.spawn(async move {
                        let event_name = platform_event_kind(&event);
                        let result = runtime.handle_event(event).await;
                        (event_name, result)
                    });
                }
            }
        }

        // 7. Shutdown order matters: stop memory producers, let event handlers
        // finish, give background jobs a window, then close platform resources.
        memory_shutdown.cancel();
        drain_event_tasks(&mut tasks, options.drain_timeout).await;
        drain_background_tasks(&self.background, options.drain_timeout).await;
        platform_events.shutdown().await.map_err(platform_error)?;
        tracing::info!("bot event loop stopped");
        Ok(())
    }

    /// Dispatch one platform event to the appropriate runtime handler.
    ///
    /// The event loop owns concurrency and shutdown; this method only translates
    /// an event into a bot action and records the outcome for diagnostics.
    #[tracing::instrument(
        name = "bot.handle_event",
        skip_all,
        fields(event = platform_event_kind(&event))
    )]
    pub async fn handle_event(&self, event: PlatformEvent) -> Result<BotAction, BotError> {
        let action = match event {
            PlatformEvent::Ready { .. } => Ok(BotAction::Ignored),
            PlatformEvent::GuildProfileUpdated { guild } => self.handle_guild_profile(guild).await,
            PlatformEvent::MessageCreated { message } => self.handle_message(*message).await,
            PlatformEvent::ReactionAdded { reaction } => {
                self.handle_reaction(reaction, false).await
            }
            PlatformEvent::ReactionRemoved { reaction } => {
                self.handle_reaction(reaction, true).await
            }
            PlatformEvent::Command { command } => self.handle_command(command).await,
            PlatformEvent::Shutdown => Ok(BotAction::Shutdown),
        };
        match &action {
            Ok(action) => {
                tracing::debug!(action = bot_action_kind(action), "platform event handled")
            }
            Err(error) => tracing::warn!(error = %error, "platform event handling failed"),
        }
        action
    }

    /// Publish an updated user profile to live subscribers.
    pub(crate) fn publish_user(&self, user: chudbot_api::UserRef) {
        self.events.publish(LiveEvent::UserProfileUpdated { user });
    }

    /// Publish a conversation lifecycle update to live subscribers.
    pub(crate) fn publish_conversation(
        &self,
        conversation_id: ConversationId,
        kind: ConversationEventKind,
    ) {
        tracing::trace!(
            conversation = %conversation_id,
            event = conversation_event_kind(kind),
            "publishing conversation event"
        );
        self.events.publish(LiveEvent::Conversation {
            conversation_id,
            kind,
        });
    }

    /// Check whether a user matches any configured admin identity.
    pub(crate) fn is_admin(&self, user: &chudbot_api::UserRef) -> bool {
        self.config.admins.iter().any(|admin| {
            admin.platform == user.platform
                && admin.user_id == user.user_id
                && admin
                    .guild_id
                    .as_ref()
                    .is_none_or(|guild| user.guild_id.as_ref() == Some(guild))
        })
    }

    /// Add a status reaction, logging failures as non-fatal platform issues.
    pub(crate) async fn add_unicode_reaction(&self, message: &MessageRef, name: &str, label: &str) {
        if let Err(error) = self
            .platforms
            .add_reaction(
                message.clone(),
                ReactionKind::Unicode {
                    name: name.to_string(),
                },
            )
            .await
        {
            tracing::warn!(error = %error, reaction = name, label, "failed to add reaction");
        }
    }

    /// Remove one of the bot's own status reactions if the platform allows it.
    pub(crate) async fn remove_own_unicode_reaction(
        &self,
        message: &MessageRef,
        name: &str,
        label: &str,
    ) {
        if let Err(error) = self
            .platforms
            .remove_own_reaction(
                message.clone(),
                ReactionKind::Unicode {
                    name: name.to_string(),
                },
            )
            .await
        {
            tracing::warn!(error = %error, reaction = name, label, "failed to remove reaction");
        }
    }

    /// Reflect a completed event action back onto the triggering message.
    pub(crate) async fn react_for_action(
        &self,
        message: &MessageRef,
        action: &Result<BotAction, BotError>,
    ) {
        match action {
            Ok(BotAction::CompletedTurn { .. }) => {
                self.add_unicode_reaction(message, SUCCESS_REACTION, "turn_success")
                    .await;
            }
            Ok(BotAction::FailedTurn { .. }) | Err(_) => {
                self.add_unicode_reaction(message, ERROR_REACTION, "turn_error")
                    .await;
            }
            Ok(BotAction::RefusedMessage) => {
                self.add_unicode_reaction(message, REFUSED_REACTION, "turn_refused")
                    .await;
            }
            Ok(BotAction::CancelledTurn { .. }) => {
                tracing::info!("turn cancelled; leaving only the stop reaction as status");
            }
            Ok(_) => {}
        }
    }
}

/// Log a spawned task join failure at a severity that matches the failure mode.
pub(crate) fn log_task_join_error(task: &'static str, error: &JoinError) {
    if error.is_cancelled() {
        tracing::debug!(task, error = %error, "task was cancelled");
    } else if error.is_panic() {
        tracing::error!(task, error = %error, "task panicked");
    } else {
        tracing::warn!(task, error = %error, "task join failed");
    }
}

/// Spawn a named background task and route its join result through runtime logging.
///
/// The `TaskTracker` observes the outer wrapper, while the inner Tokio task
/// keeps panic/cancellation classification consistent with directly joined
/// runtime tasks.
pub(crate) fn spawn_background_task<F>(tracker: &TaskTracker, task: &'static str, future: F)
where
    F: Future<Output = ()> + Send + 'static,
{
    tracker.spawn(async move {
        if let Err(error) = tokio::spawn(future).await {
            log_task_join_error(task, &error);
        }
    });
}

/// Log the result produced by one platform-event task.
pub(crate) fn log_event_task_result(
    result: Result<(&'static str, Result<BotAction, BotError>), JoinError>,
) {
    match result {
        Ok((event, Ok(action))) => {
            tracing::debug!(
                event,
                action = bot_action_kind(&action),
                "event task completed"
            )
        }
        Ok((event, Err(error))) => {
            tracing::warn!(event, error = %error, "event task failed")
        }
        Err(error) if error.is_cancelled() => {
            tracing::debug!("event task was cancelled during shutdown")
        }
        Err(error) if error.is_panic() => tracing::error!(error = %error, "event task panicked"),
        Err(error) => tracing::warn!(error = %error, "event task join failed"),
    }
}

/// Drain in-flight platform-event tasks during shutdown.
///
/// Event tasks are allowed to complete within the configured timeout so normal
/// replies, trace events, and logs can finish. Tasks that overrun the shutdown
/// window are aborted and then joined to record their final status.
pub(crate) async fn drain_event_tasks(
    tasks: &mut JoinSet<(&'static str, Result<BotAction, BotError>)>,
    timeout: Duration,
) {
    if tasks.is_empty() {
        tracing::debug!("no in-flight event tasks to drain");
        return;
    }

    tracing::info!(
        in_flight = tasks.len(),
        timeout_ms = timeout.as_millis(),
        "draining in-flight event tasks"
    );
    // Join first so already-finished event work can publish/log its result before
    // shutdown falls back to aborting the remaining tasks.
    let drained = tokio::time::timeout(timeout, async {
        while let Some(result) = tasks.join_next().await {
            log_event_task_result(result);
        }
    })
    .await;
    if drained.is_ok() {
        tracing::info!("in-flight event tasks drained");
        return;
    }

    let remaining = tasks.len();
    tracing::warn!(
        remaining,
        timeout_ms = timeout.as_millis(),
        "event task drain timed out; aborting remaining tasks"
    );
    tasks.abort_all();
    while let Some(result) = tasks.join_next().await {
        log_event_task_result(result);
    }
}

/// Close the global background tracker and wait briefly for best-effort cleanup.
pub(crate) async fn drain_background_tasks(tracker: &TaskTracker, timeout: Duration) {
    if tracker.is_empty() {
        tracing::debug!("no background tasks to drain");
        tracker.close();
        return;
    }

    tracing::info!(
        in_flight = tracker.len(),
        timeout_ms = timeout.as_millis(),
        "draining background tasks"
    );
    // Background jobs are best-effort cleanup; close the tracker and wait only
    // for the configured shutdown window.
    tracker.close();

    if tokio::time::timeout(timeout, tracker.wait()).await.is_ok() {
        tracing::info!("background tasks drained");
        return;
    }

    tracing::warn!(
        remaining = tracker.len(),
        timeout_ms = timeout.as_millis(),
        "background task drain timed out"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_cancellations_cancel_only_matching_conversation() {
        let registry = TurnCancellations::default();
        let conversation_id = ConversationId::new();
        let other_conversation_id = ConversationId::new();

        let first = registry.register(conversation_id, TurnId::new());
        let second = registry.register(conversation_id, TurnId::new());
        let other = registry.register(other_conversation_id, TurnId::new());
        let first_token = first.token();
        let second_token = second.token();
        let other_token = other.token();

        assert_eq!(registry.cancel_conversation(conversation_id), 2);
        assert!(first_token.is_cancelled());
        assert!(second_token.is_cancelled());
        assert!(!other_token.is_cancelled());
    }

    #[test]
    fn turn_cancellation_guard_unregisters_on_drop() {
        let registry = TurnCancellations::default();
        let conversation_id = ConversationId::new();
        let token = {
            let guard = registry.register(conversation_id, TurnId::new());
            guard.token()
        };

        assert_eq!(registry.cancel_conversation(conversation_id), 0);
        assert!(!token.is_cancelled());
    }
}
