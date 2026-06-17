//! Background user-memory scheduler and job worker.
//!
//! The runtime is intentionally small and storage-driven: each tick asks
//! storage to enqueue due diary/compaction work, leases a bounded batch of
//! runnable jobs, executes the claimed jobs in-process, and records a terminal
//! or retry completion for every claimed row. Cross-process ownership,
//! duplicate-window suppression, and persisted retry state live behind
//! [`BotStorage`]; this module keeps the in-process worker loop and model-run
//! lifecycle explicit.

use std::collections::{BTreeSet, VecDeque};

use chudbot_api::{
    Agent, AgentOutcome, AgentRun, BotStorage, LlmProviderRegistry, MediaStore,
    MemoryJobCompletion, MemoryJobKind, MemoryJobSchedule, MemoryTurnWindow, Model, ModelId,
    NewUserMemoryDiaryEntry, NewUserMemoryDocumentRevision, NoClientTools, Transcript, UsageRecord,
    UserMemoryJob,
};
use thiserror::Error;
use time::OffsetDateTime;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::RoutedLlmBackend;
use crate::config::SystemAgentConfig;

use super::compact::compact_input;
use super::config::{MemoryAgentSet, MemoryConfig, MemoryConfigError};
use super::diary::diary_transcript;
use super::{memory_guild_id, memory_profile_display_name, memory_scope_id, memory_user_ref};

/// In-process memory scheduler and worker.
///
/// A `MemoryRuntime` owns no durable state beyond its configured services.
/// Scheduling, lease ownership, attempt counts, and source cutoffs are stored
/// in `BotStorage`, which lets multiple bot processes poll concurrently
/// without sharing memory.
#[derive(Debug, Clone)]
pub struct MemoryRuntime<S, L, M> {
    storage: S,
    llms: L,
    media_store: M,
    config: MemoryConfig,
    agents: MemoryAgentSet,
}

impl<S, L, M> MemoryRuntime<S, L, M> {
    /// Construct a memory runtime from storage, provider registries, and the
    /// resolved memory-agent configuration.
    pub(crate) fn new(
        storage: S,
        llms: L,
        media_store: M,
        config: MemoryConfig,
        agents: MemoryAgentSet,
    ) -> Self {
        Self {
            storage,
            llms,
            media_store,
            config,
            agents,
        }
    }
}

/// Errors from the memory runtime.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Configuration is invalid.
    #[error(transparent)]
    Config(#[from] MemoryConfigError),
    /// Storage operation failed.
    #[error("storage error: {0}")]
    Storage(String),
    /// Model operation failed.
    #[error("model error: {0}")]
    Model(String),
}

impl<S, L, M> MemoryRuntime<S, L, M>
where
    S: BotStorage + Clone + Send + Sync + 'static,
    L: LlmProviderRegistry + Clone + Send + Sync + 'static,
    M: MediaStore + Clone + Send + Sync + 'static,
{
    /// Run the memory scheduler until shutdown.
    ///
    /// The loop runs one storage-backed tick at a time, then sleeps for the
    /// configured poll interval. Shutdown is biased in both waits so a process
    /// stop can interrupt idle polling promptly. If cancellation interrupts an
    /// in-flight tick, any unfinished claimed jobs rely on their storage lease
    /// expiring before another worker can claim them.
    pub async fn run_until_shutdown(&self, shutdown: CancellationToken) -> Result<(), MemoryError> {
        if !self.config.enabled {
            tracing::debug!("memory runtime disabled");
            return Ok(());
        }
        self.config.compaction_interval_seconds()?;
        self.config.diary_backfill_window_seconds()?;
        self.config.diary_interval_seconds()?;
        tracing::info!(
            diary_agent = %self.agents.diary.name,
            diary_provider = %self.agents.diary.provider,
            diary_model = %self.agents.diary.model.id,
            compact_agent = %self.agents.compact.name,
            compact_provider = %self.agents.compact.provider,
            compact_model = %self.agents.compact.model.id,
            poll_interval_seconds = self.config.poll_interval_seconds,
            diary_backfill_window = %self.config.diary_backfill_window,
            diary_interval = %self.config.diary_interval,
            max_jobs_per_tick = self.config.max_jobs_per_tick,
            max_concurrent_jobs = self.config.max_concurrent_jobs,
            "memory runtime starting"
        );
        loop {
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break,
                result = self.run_tick() => {
                    if let Err(error) = result {
                        tracing::warn!(error = %error, "memory scheduler tick failed");
                    }
                }
            }
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break,
                () = tokio::time::sleep(self.config.poll_interval()) => {}
            }
        }
        tracing::info!("memory runtime stopped");
        Ok(())
    }

    /// Enqueue due work, claim a bounded batch, and run that claimed batch.
    ///
    /// `enqueue_due_memory_jobs` computes durable diary/compaction work from
    /// source cutoffs and retry state. `claim_memory_jobs` then leases at most
    /// `max_jobs_per_tick` rows until `lease_until`; the worker id only needs to
    /// be unique enough to identify this process tick in storage.
    async fn run_tick(&self) -> Result<(), MemoryError> {
        let now = OffsetDateTime::now_utc();
        let compaction_interval = self.config.compaction_interval_seconds()?;
        let diary_backfill_window = self.config.diary_backfill_window_seconds()?;
        let diary_interval = self.config.diary_interval_seconds()?;
        let compact_due_before =
            now - time::Duration::seconds(i64::try_from(compaction_interval).unwrap_or(i64::MAX));
        let diary_cutoff =
            now - time::Duration::seconds(i64::try_from(diary_backfill_window).unwrap_or(i64::MAX));
        let diary_due_before =
            now - time::Duration::seconds(i64::try_from(diary_interval).unwrap_or(i64::MAX));
        let enqueued = self
            .storage
            .enqueue_due_memory_jobs(MemoryJobSchedule {
                now,
                diary_cutoff,
                diary_due_before,
                diary_window_seconds: diary_interval,
                compact_due_before,
            })
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let lease_until = now
            + time::Duration::seconds(
                i64::try_from(self.config.lease_duration().as_secs()).unwrap_or(i64::MAX),
            );
        let worker_id = format!(
            "memory:{}:{}",
            std::process::id(),
            now.unix_timestamp_nanos()
        );
        let jobs = self
            .storage
            .claim_memory_jobs(worker_id, self.config.max_jobs_per_tick.max(1), lease_until)
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        tracing::debug!(
            enqueued,
            claimed = jobs.len(),
            "memory scheduler tick claimed work"
        );
        self.run_claimed_jobs(jobs).await
    }

    /// Execute a claimed batch with global and per-memory-key concurrency
    /// limits.
    ///
    /// At most `max_concurrent_jobs` tasks run in this process. Within that
    /// window, only one job for a given `memory_key` is active at a time so a
    /// diary write and compaction for the same subject cannot race each other
    /// locally. Other processes still rely on the storage lease/claim contract
    /// for cross-process exclusion.
    async fn run_claimed_jobs(&self, jobs: Vec<UserMemoryJob>) -> Result<(), MemoryError> {
        let mut pending = VecDeque::from(jobs);
        let mut active_keys = BTreeSet::new();
        let mut running = JoinSet::new();
        let max_concurrent = self.config.max_concurrent_jobs.max(1) as usize;

        while !pending.is_empty() || !running.is_empty() {
            // Fill available worker slots without starting two tasks for the
            // same memory key; blocked same-key jobs stay pending until the
            // active task reports completion.
            while running.len() < max_concurrent {
                let Some(index) = pending
                    .iter()
                    .position(|job| !active_keys.contains(&job.memory_key))
                else {
                    break;
                };
                let job = pending.remove(index).expect("pending index exists");
                active_keys.insert(job.memory_key.clone());
                let runtime = (*self).clone();
                running.spawn(async move {
                    let memory_key = job.memory_key.clone();
                    let target_user_name = runtime.load_memory_job_user_name(&job).await;
                    let agent = runtime.agent_config(job.kind);
                    let span = memory_job_span(&job, agent, target_user_name.as_deref());
                    let result = runtime.run_job_with_completion(job).instrument(span).await;
                    (memory_key, result)
                });
            }

            let Some(result) = running.join_next().await else {
                break;
            };
            match result {
                Ok((memory_key, result)) => {
                    // Releasing the key happens after durable completion is
                    // attempted, so a following same-key job sees the latest
                    // diary/document state whenever completion succeeded.
                    active_keys.remove(&memory_key);
                    if let Err(error) = result {
                        tracing::warn!(memory_key, error = %error, "memory job failed");
                    }
                }
                Err(error) => {
                    tracing::warn!(error = %error, "memory job task join failed");
                }
            }
        }
        Ok(())
    }

    /// Run one claimed job and persist its completion state.
    ///
    /// Successful jobs are marked completed. Failed jobs become terminal once
    /// their persisted attempt count has reached `max_job_attempts`; otherwise
    /// they are rescheduled after `retry_backoff(attempts)`. The backoff is
    /// computed only after the model/storage failure, which keeps retry timing
    /// tied to the actual worker outcome rather than to claim time.
    async fn run_job_with_completion(&self, job: UserMemoryJob) -> Result<(), MemoryError> {
        let result = self.run_job(&job).await;
        let completion = match result {
            Ok(()) => MemoryJobCompletion::Completed { job_id: job.id },
            Err(error) if job.attempts >= self.config.max_job_attempts.max(1) => {
                MemoryJobCompletion::Failed {
                    job_id: job.id,
                    error: error.to_string(),
                }
            }
            Err(error) => MemoryJobCompletion::Retry {
                job_id: job.id,
                error: error.to_string(),
                next_run_at: OffsetDateTime::now_utc() + self.config.retry_backoff(job.attempts),
            },
        };
        self.storage
            .finish_memory_job(completion)
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))
    }

    /// Dispatch a claimed job to the diary or compaction worker.
    async fn run_job(&self, job: &UserMemoryJob) -> Result<(), MemoryError> {
        tracing::debug!(
            job = %job.id,
            kind = ?job.kind,
            memory_key = %job.memory_key,
            attempts = job.attempts,
            "running memory job"
        );
        match job.kind {
            MemoryJobKind::Diary => self.run_diary_job(job).await,
            MemoryJobKind::Compact => self.run_compact_job(job).await,
        }
    }

    /// Build and store a diary entry for one scheduled conversation window.
    ///
    /// Empty or malformed windows are treated as successful no-ops so the
    /// scheduler can advance past them. Non-empty windows load the current
    /// memory document, ask the diary agent for a markdown entry, and persist
    /// both the text and provider usage as the durable job side effect.
    async fn run_diary_job(&self, job: &UserMemoryJob) -> Result<(), MemoryError> {
        let (Some(window_start), Some(window_end)) = (job.window_start, job.window_end) else {
            tracing::warn!(job = %job.id, "diary job has no window");
            return Ok(());
        };
        let turns = self
            .storage
            .load_memory_turn_window(MemoryTurnWindow {
                key: job.key.clone(),
                window_start,
                window_end,
                max_turns: self.config.max_transcript_turns_per_diary_job.max(1),
            })
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        if turns.is_empty() {
            tracing::debug!(job = %job.id, "diary job window had no turns");
            return Ok(());
        }
        let document = self
            .storage
            .load_user_memory_document(job.key.clone())
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let transcript =
            diary_transcript(&job.key, document.as_ref(), &turns, &self.media_store).await;
        let agent_config = self.agent_config(MemoryJobKind::Diary).clone();
        let output = self.run_memory_model(&agent_config, transcript).await?;
        self.storage
            .save_user_memory_diary_entry(NewUserMemoryDiaryEntry {
                key: job.key.clone(),
                window_start,
                window_end,
                source_turn_ids: turns.iter().map(|turn| turn.turn_id).collect(),
                markdown: output.text,
                agent_name: agent_config.name.clone(),
                llm_provider: agent_config.provider.clone(),
                llm_model: output.model_id,
                usage: output.usage,
            })
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        Ok(())
    }

    /// Compact pending memory events and diary entries into the user's memory
    /// document.
    ///
    /// The source cutoffs are derived from the newest material included in the
    /// revision, or preserved from the existing document when a source class is
    /// absent. That makes a completed compaction idempotent from the
    /// scheduler's point of view: the next enqueue pass can skip everything
    /// already folded into the saved revision.
    async fn run_compact_job(&self, job: &UserMemoryJob) -> Result<(), MemoryError> {
        let document = self
            .storage
            .load_user_memory_document(job.key.clone())
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let events = self
            .storage
            .list_pending_memory_events(
                job.key.clone(),
                document
                    .as_ref()
                    .and_then(|document| document.source_event_cutoff),
            )
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        let diaries = self
            .storage
            .list_pending_memory_diary_entries(
                job.key.clone(),
                document
                    .as_ref()
                    .and_then(|document| document.source_diary_cutoff),
            )
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        if events.is_empty() && diaries.is_empty() {
            tracing::debug!(job = %job.id, "compact job had no source material");
            return Ok(());
        }

        let input = compact_input(&job.key, document.as_ref(), &events, &diaries);
        let agent_config = self.agent_config(MemoryJobKind::Compact).clone();
        let output = self
            .run_memory_model(&agent_config, Transcript::from_user_text(input))
            .await?;
        let MemoryModelOutput {
            text: markdown,
            model_id: llm_model,
            usage,
        } = output;
        let source_event_cutoff = events
            .iter()
            .map(|event| event.created_at)
            .max()
            .or_else(|| {
                document
                    .as_ref()
                    .and_then(|document| document.source_event_cutoff)
            });
        let source_diary_cutoff =
            diaries
                .iter()
                .map(|entry| entry.created_at)
                .max()
                .or_else(|| {
                    document
                        .as_ref()
                        .and_then(|document| document.source_diary_cutoff)
                });
        tracing::debug!(
            job = %job.id,
            model = %llm_model,
            events = events.len(),
            diaries = diaries.len(),
            markdown_chars = markdown.chars().count(),
            usage_records = usage.len(),
            "saving compact memory profile"
        );
        self.storage
            .save_user_memory_document_revision(NewUserMemoryDocumentRevision {
                key: job.key.clone(),
                markdown,
                source_event_ids: events.iter().map(|event| event.id).collect(),
                source_diary_entry_ids: diaries.iter().map(|entry| entry.id).collect(),
                source_event_cutoff,
                source_diary_cutoff,
                agent_name: agent_config.name.clone(),
                llm_provider: agent_config.provider.clone(),
                llm_model,
                usage,
            })
            .await
            .map_err(|error| MemoryError::Storage(error.to_string()))?;
        Ok(())
    }

    /// Run a memory system agent and normalize the provider result for storage.
    ///
    /// Memory agents do not expose client tools. The routed LLM backend records
    /// the configured provider/model, while `memory_model_output` validates the
    /// agent outcome, captures usage, and falls back to the configured model id
    /// when the provider did not report a concrete model.
    async fn run_memory_model(
        &self,
        agent_config: &SystemAgentConfig,
        transcript: Transcript,
    ) -> Result<MemoryModelOutput, MemoryError> {
        let agent = Agent::new(
            Model {
                backend: RoutedLlmBackend::new(self.llms.clone(), agent_config.provider.clone()),
                spec: agent_config.model.clone(),
            },
            agent_config.spec.clone(),
            NoClientTools,
        );
        let run = agent
            .run(transcript)
            .await
            .map_err(|error| MemoryError::Model(error.to_string()))?;
        memory_model_output(run, &agent_config.model.id)
    }

    fn agent_config(&self, kind: MemoryJobKind) -> &SystemAgentConfig {
        match kind {
            MemoryJobKind::Diary => &self.agents.diary,
            MemoryJobKind::Compact => &self.agents.compact,
        }
    }

    /// Best-effort lookup for a readable memory subject label on the outer job
    /// span.
    async fn load_memory_job_user_name(&self, job: &UserMemoryJob) -> Option<String> {
        let user = memory_user_ref(&job.key);
        let profiles = match self.storage.load_user_profiles(vec![user]).await {
            Ok(profiles) => profiles,
            Err(error) => {
                tracing::warn!(
                    job = %job.id,
                    memory_key = %job.memory_key,
                    message_provider = %job.key.platform,
                    scope_key = %job.key.scope_key,
                    target_user_id = %job.key.user_key,
                    error = %error,
                    "failed to load memory subject profile for tracing"
                );
                return None;
            }
        };
        profiles
            .first()
            .and_then(|profile| memory_profile_display_name(&profile.profile, &job.key.user_key))
    }
}

/// Build the parent tracing span for a claimed memory job.
///
/// Provider and agent logs inherit the subject/scope fields from this span, so
/// nested model spans do not need to repeat the same identifiers.
fn memory_job_span(
    job: &UserMemoryJob,
    agent: &SystemAgentConfig,
    target_user_name: Option<&str>,
) -> tracing::Span {
    let span = tracing::info_span!(
        "memory.job",
        job = %job.id,
        kind = ?job.kind,
        memory_agent = %agent.name,
        provider = %agent.provider,
        model = %agent.model.id,
        memory_key = %job.memory_key,
        message_provider = %job.key.platform,
        scope_key = %job.key.scope_key,
        scope_id = %memory_scope_id(&job.key.scope_key),
        guild_id = tracing::field::Empty,
        user_id = %job.key.user_key,
        target_user_id = %job.key.user_key,
        target_user_name = tracing::field::Empty,
        attempts = job.attempts,
    );
    if let Some(guild_id) = memory_guild_id(&job.key.scope_key) {
        span.record("guild_id", tracing::field::display(guild_id));
    }
    if let Some(name) = target_user_name {
        span.record("target_user_name", tracing::field::display(name));
    }
    span
}

/// Text, model identity, and usage extracted from a completed memory-agent run.
#[derive(Debug, Clone)]
struct MemoryModelOutput {
    text: String,
    model_id: ModelId,
    usage: Vec<UsageRecord>,
}

/// Convert an `AgentRun` into the durable output shape expected by memory
/// storage.
///
/// Only a non-empty completed answer is a successful memory result. Iteration
/// limits, provider failures, and cancellations become model errors so the job
/// completion path can either retry with backoff or mark the job terminal.
fn memory_model_output(
    run: AgentRun,
    fallback_model_id: &ModelId,
) -> Result<MemoryModelOutput, MemoryError> {
    let usage = run.all_usage();
    let model_id = run
        .last_model_id
        .unwrap_or_else(|| fallback_model_id.clone());
    match run.outcome {
        AgentOutcome::Completed { answer } => {
            let text = answer.text.trim().to_string();
            if text.is_empty() {
                return Err(MemoryError::Model(
                    "memory model returned empty text".to_string(),
                ));
            }
            Ok(MemoryModelOutput {
                text,
                model_id,
                usage,
            })
        }
        AgentOutcome::IterationLimit { max_iterations } => Err(MemoryError::Model(format!(
            "memory model hit iteration limit ({max_iterations})"
        ))),
        AgentOutcome::Failed { error, partial } => {
            let mut message = error.to_string();
            if let Some(partial) = partial
                && !partial.text.trim().is_empty()
            {
                message.push_str("\n\nPartial answer:\n");
                message.push_str(&partial.text);
            }
            Err(MemoryError::Model(message))
        }
        AgentOutcome::Cancelled { reason } => Err(MemoryError::Model(format!(
            "memory model cancelled: {reason}"
        ))),
    }
}
