//! Bot runtime construction, event loop plumbing, and task lifecycle helpers.

use crate::prelude::*;
use crate::*;

/// Compile-time service types for a bot runtime.
pub trait BotRuntimeTypes {
    type Platforms: MessagePlatformRegistry + Clone + Send + Sync + 'static;
    type Storage: BotStorage + Clone + Send + Sync + 'static;
    type Media: MediaStore + Clone + Send + Sync + 'static;
    type Llms: LlmProviderRegistry + Clone + Send + Sync + 'static;
    type Images: ImageGeneratorRegistry + Clone + Send + Sync + 'static;
    type Videos: VideoGeneratorRegistry + Clone + Send + Sync + 'static;
    type Audio: AudioTranscriberRegistry + Clone + Send + Sync + 'static;
    type Events: EventSink + Clone + Send + Sync + 'static;
}

/// Platform-neutral bot runtime.
#[derive(Debug)]
pub struct BotRuntime<R: BotRuntimeTypes> {
    pub(crate) platforms: R::Platforms,
    pub(crate) storage: R::Storage,
    pub(crate) media_store: R::Media,
    pub(crate) llms: R::Llms,
    pub(crate) images: R::Images,
    pub(crate) videos: R::Videos,
    pub(crate) audio: R::Audio,
    pub(crate) events: R::Events,
    pub(crate) background: TaskTracker,
    pub(crate) turn_cancellations: TurnCancellations,
    pub(crate) video_rate_limit_locks: VideoRateLimitLocks,
    pub(crate) download_http: reqwest::Client,
    pub(crate) config: BotConfig,
    pub(crate) memory_config: memory::MemoryConfig,
    pub(crate) system_agents: RuntimeSystemAgents,
}

impl<R> Clone for BotRuntime<R>
where
    R: BotRuntimeTypes,
{
    fn clone(&self) -> Self {
        Self {
            platforms: self.platforms.clone(),
            storage: self.storage.clone(),
            media_store: self.media_store.clone(),
            llms: self.llms.clone(),
            images: self.images.clone(),
            videos: self.videos.clone(),
            audio: self.audio.clone(),
            events: self.events.clone(),
            background: self.background.clone(),
            turn_cancellations: self.turn_cancellations.clone(),
            video_rate_limit_locks: self.video_rate_limit_locks.clone(),
            download_http: self.download_http.clone(),
            config: self.config.clone(),
            memory_config: self.memory_config.clone(),
            system_agents: self.system_agents.clone(),
        }
    }
}

/// Runtime service implementations supplied by the binary crate.
#[derive(Debug, Clone)]
pub struct BotRuntimeParts<R: BotRuntimeTypes> {
    /// Message platform registry.
    pub platforms: R::Platforms,
    /// Durable bot storage.
    pub storage: R::Storage,
    /// Media storage.
    pub media_store: R::Media,
    /// LLM provider registry.
    pub llms: R::Llms,
    /// Image generation registry.
    pub images: R::Images,
    /// Video generation registry.
    pub videos: R::Videos,
    /// Audio transcription registry.
    pub audio: R::Audio,
    /// Live event sink.
    pub events: R::Events,
    /// User-memory runtime configuration.
    pub memory: memory::MemoryConfig,
}

/// In-memory cancellation registry for currently running turns.
#[derive(Debug, Clone, Default)]
pub(crate) struct TurnCancellations {
    inner: Arc<Mutex<BTreeMap<ConversationId, BTreeMap<TurnId, CancellationToken>>>>,
}

impl TurnCancellations {
    pub(crate) fn register(
        &self,
        conversation_id: ConversationId,
        turn_id: TurnId,
    ) -> TurnCancellationGuard {
        let token = CancellationToken::new();
        self.inner
            .lock()
            .expect("turn cancellation mutex poisoned")
            .entry(conversation_id)
            .or_default()
            .insert(turn_id, token.clone());
        TurnCancellationGuard {
            registry: self.clone(),
            conversation_id,
            turn_id,
            token,
        }
    }

    pub(crate) fn unregister(&self, conversation_id: ConversationId, turn_id: TurnId) {
        let mut inner = self.inner.lock().expect("turn cancellation mutex poisoned");
        if let Some(turns) = inner.get_mut(&conversation_id) {
            turns.remove(&turn_id);
            if turns.is_empty() {
                inner.remove(&conversation_id);
            }
        }
    }

    pub(crate) fn cancel_conversation(&self, conversation_id: ConversationId) -> usize {
        let tokens = self
            .inner
            .lock()
            .expect("turn cancellation mutex poisoned")
            .get(&conversation_id)
            .map(|turns| turns.values().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let count = tokens.len();
        for token in tokens {
            token.cancel();
        }
        count
    }
}

/// RAII registration for one cancellable turn.
#[derive(Debug)]
pub(crate) struct TurnCancellationGuard {
    registry: TurnCancellations,
    conversation_id: ConversationId,
    turn_id: TurnId,
    token: CancellationToken,
}

impl TurnCancellationGuard {
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
#[derive(Debug)]
pub(crate) struct TypingIndicator {
    pub(crate) stop: CancellationToken,
    pub(crate) task: JoinHandle<()>,
}

impl TypingIndicator {
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
    /// Construct a bot runtime.
    pub fn new(parts: BotRuntimeParts<R>, config: BotConfig) -> Self {
        tracing::debug!(
            agents = config.agents.len(),
            platforms = config.platforms.len(),
            default_agent = %config.default_agent,
            "constructing bot runtime"
        );
        let system_agents = RuntimeSystemAgents::from_config(&config);
        Self {
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
        }
    }

    /// Borrow the bot config.
    pub fn config(&self) -> &BotConfig {
        &self.config
    }
}

impl<R> BotRuntime<R>
where
    R: BotRuntimeTypes + 'static,
{
    /// Run the platform event loop with explicit shutdown behavior.
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
    pub async fn run_with_options(
        &self,
        shutdown: CancellationToken,
        options: BotRunOptions,
    ) -> Result<(), BotError> {
        self.platforms
            .register_commands(command_definitions())
            .await
            .map_err(platform_error)?;
        let memory_shutdown = shutdown.child_token();
        self.spawn_memory_runtime(memory_shutdown.clone());
        tracing::info!("bot event loop starting");
        let mut tasks = JoinSet::new();
        loop {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {
                    tracing::info!("bot shutdown requested; stopping platform event intake");
                    break;
                }
                Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                    log_event_task_result(result);
                }
                event = self.platforms.next_event() => {
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
                    let runtime = (*self).clone();
                    tasks.spawn(async move {
                        let event_name = platform_event_kind(&event);
                        let result = runtime.handle_event(event).await;
                        (event_name, result)
                    });
                }
            }
        }

        memory_shutdown.cancel();
        drain_event_tasks(&mut tasks, options.drain_timeout).await;
        drain_background_tasks(&self.background, options.drain_timeout).await;
        self.platforms.shutdown().await.map_err(platform_error)?;
        tracing::info!("bot event loop stopped");
        Ok(())
    }

    /// Handle one platform event.
    #[tracing::instrument(
        name = "bot.handle_event",
        skip_all,
        fields(event = platform_event_kind(&event))
    )]
    pub async fn handle_event(&self, event: PlatformEvent) -> Result<BotAction, BotError> {
        let action = match event {
            PlatformEvent::Ready { .. } => Ok(BotAction::Ignored),
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

    pub(crate) fn publish_user(&self, user: chudbot_api::UserRef) {
        self.events.publish(LiveEvent::UserProfileUpdated { user });
    }

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

pub(crate) fn log_task_join_error(task: &'static str, error: &JoinError) {
    if error.is_cancelled() {
        tracing::debug!(task, error = %error, "task was cancelled");
    } else if error.is_panic() {
        tracing::error!(task, error = %error, "task panicked");
    } else {
        tracing::warn!(task, error = %error, "task join failed");
    }
}

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
