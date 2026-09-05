//! User memory documents, diary entries, events, and background jobs.

use std::collections::BTreeMap;

use chudbot_api::{
    ConversationId, MediaUri, MemoryJobCompletion, MemoryJobKind, MemoryJobSchedule,
    MemoryTurnWindow, ModelId, NewUserMemoryDiaryEntry, NewUserMemoryDocumentRevision,
    NewUserMemoryEvent, PlatformName, ProviderName, TurnId, UserMemoryAudioTranscription,
    UserMemoryDiaryEntry, UserMemoryDocument, UserMemoryEvent, UserMemoryEventKind,
    UserMemoryImageContext, UserMemoryJob, UserMemoryKey, UserMemoryTurn,
};
use serde_json::Value;
use sqlx::{PgPool, Row};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::bot::canonical_media_uri_string;
use crate::{SqlxStorage, SqlxStorageError};

impl SqlxStorage {
    pub(super) async fn load_user_memory_document(
        &self,
        key: UserMemoryKey,
    ) -> Result<Option<UserMemoryDocument>, SqlxStorageError> {
        let row = sqlx::query(
            "SELECT message_provider, scope_key, subject_user_key, revision, markdown, \
                    last_compacted_at, source_event_cutoff, source_diary_cutoff, \
                    created_at, created_at AS updated_at \
               FROM user_memory_document_versions \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
              ORDER BY revision DESC \
              LIMIT 1",
        )
        .bind(key.platform.as_str())
        .bind(&key.scope_key)
        .bind(&key.user_key)
        .fetch_optional(&self.pool)
        .await?;
        row.map(document_from_row).transpose()
    }

    pub(super) async fn append_user_memory_event(
        &self,
        event: NewUserMemoryEvent,
    ) -> Result<UserMemoryEvent, SqlxStorageError> {
        let id = Uuid::new_v4();
        let row = sqlx::query(
            "INSERT INTO user_memory_events \
               (id, message_provider, scope_key, subject_user_key, actor_user_key, kind, body, \
                tags, confidence, source_conversation_id, source_turn_id, source_tool_trace_id, \
                supersedes_event_id) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13) \
             RETURNING id, message_provider, scope_key, subject_user_key, actor_user_key, kind, \
                       body, tags, confidence, source_conversation_id, source_turn_id, \
                       source_tool_trace_id, supersedes_event_id, created_at, updated_at",
        )
        .bind(id)
        .bind(event.key.platform.as_str())
        .bind(&event.key.scope_key)
        .bind(&event.key.user_key)
        .bind(&event.actor_user_key)
        .bind(memory_event_kind_as_str(event.kind))
        .bind(&event.body)
        .bind(serde_json::to_value(&event.tags)?)
        .bind(event.confidence)
        .bind(event.source_conversation_id.map(|id| id.0))
        .bind(event.source_turn_id.map(|id| id.0))
        .bind(event.source_tool_trace_id)
        .bind(event.supersedes_event_id)
        .fetch_one(&self.pool)
        .await?;
        memory_event_from_row(row)
    }

    pub(super) async fn list_pending_memory_events(
        &self,
        key: UserMemoryKey,
        since: Option<OffsetDateTime>,
    ) -> Result<Vec<UserMemoryEvent>, SqlxStorageError> {
        let rows = sqlx::query(
            "SELECT id, message_provider, scope_key, subject_user_key, actor_user_key, kind, \
                    body, tags, confidence, source_conversation_id, source_turn_id, \
                    source_tool_trace_id, supersedes_event_id, created_at, updated_at \
               FROM user_memory_events \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
                AND ($4::timestamptz IS NULL OR created_at > $4) \
              ORDER BY created_at, id",
        )
        .bind(key.platform.as_str())
        .bind(&key.scope_key)
        .bind(&key.user_key)
        .bind(since)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(memory_event_from_row).collect()
    }

    pub(super) async fn list_pending_memory_diary_entries(
        &self,
        key: UserMemoryKey,
        since: Option<OffsetDateTime>,
    ) -> Result<Vec<UserMemoryDiaryEntry>, SqlxStorageError> {
        let rows = sqlx::query(
            "SELECT id, message_provider, scope_key, subject_user_key, window_start, window_end, \
                    source_turn_ids, markdown, agent_name, llm_provider, llm_model, usage, \
                    created_at, updated_at \
               FROM user_memory_diary_entries \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
                AND ($4::timestamptz IS NULL OR created_at > $4) \
              ORDER BY created_at, id",
        )
        .bind(key.platform.as_str())
        .bind(&key.scope_key)
        .bind(&key.user_key)
        .bind(since)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(diary_entry_from_row).collect()
    }

    pub(super) async fn list_recent_memory_diary_entries(
        &self,
        key: UserMemoryKey,
        limit: u32,
    ) -> Result<Vec<UserMemoryDiaryEntry>, SqlxStorageError> {
        let mut rows = sqlx::query(
            "SELECT id, message_provider, scope_key, subject_user_key, window_start, window_end, \
                    source_turn_ids, markdown, agent_name, llm_provider, llm_model, usage, \
                    created_at, updated_at \
               FROM user_memory_diary_entries \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
              ORDER BY created_at DESC, id DESC \
              LIMIT $4",
        )
        .bind(key.platform.as_str())
        .bind(&key.scope_key)
        .bind(&key.user_key)
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await?;
        rows.reverse();
        rows.into_iter().map(diary_entry_from_row).collect()
    }

    pub(super) async fn save_user_memory_diary_entry(
        &self,
        entry: NewUserMemoryDiaryEntry,
    ) -> Result<UserMemoryDiaryEntry, SqlxStorageError> {
        let id = Uuid::new_v4();
        let source_turn_ids = entry
            .source_turn_ids
            .iter()
            .map(|turn_id| turn_id.0)
            .collect::<Vec<_>>();
        let row = sqlx::query(
            "INSERT INTO user_memory_diary_entries \
               (id, message_provider, scope_key, subject_user_key, window_start, window_end, \
                source_turn_ids, markdown, agent_name, llm_provider, llm_model, usage) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) \
             RETURNING id, message_provider, scope_key, subject_user_key, window_start, \
                       window_end, source_turn_ids, markdown, agent_name, llm_provider, \
                       llm_model, usage, created_at, updated_at",
        )
        .bind(id)
        .bind(entry.key.platform.as_str())
        .bind(&entry.key.scope_key)
        .bind(&entry.key.user_key)
        .bind(entry.window_start)
        .bind(entry.window_end)
        .bind(source_turn_ids)
        .bind(&entry.markdown)
        .bind(&entry.agent_name)
        .bind(entry.llm_provider.as_str())
        .bind(entry.llm_model.as_str())
        .bind(serde_json::to_value(&entry.usage)?)
        .fetch_one(&self.pool)
        .await?;
        diary_entry_from_row(row)
    }

    pub(super) async fn save_user_memory_document_revision(
        &self,
        document: NewUserMemoryDocumentRevision,
    ) -> Result<UserMemoryDocument, SqlxStorageError> {
        let mut tx = self.pool.begin().await?;
        let existing_revision: Option<i64> = sqlx::query_scalar(
            "SELECT revision FROM user_memory_document_versions \
              WHERE message_provider = $1 AND scope_key = $2 AND subject_user_key = $3 \
              ORDER BY revision DESC \
              LIMIT 1 \
              FOR UPDATE",
        )
        .bind(document.key.platform.as_str())
        .bind(&document.key.scope_key)
        .bind(&document.key.user_key)
        .fetch_optional(&mut *tx)
        .await?;
        let revision = existing_revision.unwrap_or(0) + 1;
        let version_id = Uuid::new_v4();
        let row = sqlx::query(
            "INSERT INTO user_memory_document_versions \
               (id, message_provider, scope_key, subject_user_key, revision, markdown, \
                source_event_ids, source_diary_entry_ids, agent_name, llm_provider, llm_model, \
                usage, last_compacted_at, source_event_cutoff, source_diary_cutoff) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, now(), $13, $14) \
             RETURNING message_provider, scope_key, subject_user_key, revision, markdown, \
                       last_compacted_at, source_event_cutoff, source_diary_cutoff, \
                       created_at, created_at AS updated_at",
        )
        .bind(version_id)
        .bind(document.key.platform.as_str())
        .bind(&document.key.scope_key)
        .bind(&document.key.user_key)
        .bind(revision)
        .bind(&document.markdown)
        .bind(&document.source_event_ids)
        .bind(&document.source_diary_entry_ids)
        .bind(&document.agent_name)
        .bind(document.llm_provider.as_str())
        .bind(document.llm_model.as_str())
        .bind(serde_json::to_value(&document.usage)?)
        .bind(document.source_event_cutoff)
        .bind(document.source_diary_cutoff)
        .fetch_one(&mut *tx)
        .await?;
        tx.commit().await?;
        document_from_row(row)
    }

    pub(super) async fn enqueue_due_memory_jobs(
        &self,
        schedule: MemoryJobSchedule,
    ) -> Result<u64, SqlxStorageError> {
        let mut inserted = 0u64;
        let diary_window_seconds =
            i64::try_from(schedule.diary_window_seconds.max(1)).unwrap_or(i64::MAX);
        sqlx::query(
            "UPDATE user_memory_jobs active_jobs \
                SET status = 'failed', completed_at = COALESCE(completed_at, $1), \
                    leased_by = NULL, leased_until = NULL, \
                    error = COALESCE(error, 'diary window already reached terminal state') \
              WHERE active_jobs.kind = 'diary' \
                AND (active_jobs.status = 'pending' \
                     OR (active_jobs.status = 'running' AND active_jobs.leased_until < $1)) \
                AND active_jobs.window_start IS NOT NULL \
                AND active_jobs.window_end IS NOT NULL \
                AND ( \
                     EXISTS ( \
                         SELECT 1 \
                           FROM user_memory_diary_entries diary_entry \
                          WHERE diary_entry.message_provider = active_jobs.message_provider \
                            AND diary_entry.scope_key = active_jobs.scope_key \
                            AND diary_entry.subject_user_key = active_jobs.subject_user_key \
                            AND diary_entry.window_start = active_jobs.window_start \
                            AND diary_entry.window_end = active_jobs.window_end \
                     ) \
                     OR EXISTS ( \
                         SELECT 1 \
                           FROM user_memory_jobs terminal_jobs \
                          WHERE terminal_jobs.kind = 'diary' \
                            AND terminal_jobs.status IN ('completed', 'failed') \
                            AND terminal_jobs.id <> active_jobs.id \
                            AND terminal_jobs.message_provider = active_jobs.message_provider \
                            AND terminal_jobs.scope_key = active_jobs.scope_key \
                            AND terminal_jobs.subject_user_key = active_jobs.subject_user_key \
                            AND terminal_jobs.window_start = active_jobs.window_start \
                            AND terminal_jobs.window_end = active_jobs.window_end \
                     ) \
                )",
        )
        .bind(schedule.now)
        .execute(&self.pool)
        .await?;
        let diary_rows = sqlx::query(
            "SELECT diary_windows.message_provider, diary_windows.scope_key, \
                    diary_windows.subject_user_key, diary_windows.window_start, \
                    diary_windows.window_start + ($2::double precision * INTERVAL '1 second') AS window_end \
               FROM ( \
                    SELECT candidate_turns.message_provider, candidate_turns.scope_key, \
                           candidate_turns.subject_user_key, \
                           CASE \
                             WHEN candidate_turns.latest_window_end IS NULL \
                               OR candidate_turns.latest_window_end < $1 \
                             THEN candidate_turns.first_completed_at \
                             ELSE candidate_turns.latest_window_end + ( \
                               GREATEST( \
                                 ceil(EXTRACT(EPOCH FROM (candidate_turns.first_completed_at - candidate_turns.latest_window_end))::double precision / $2::double precision)::bigint - 1, \
                                 0::bigint \
                               )::double precision * $2::double precision * INTERVAL '1 second' \
                             ) \
                           END AS window_start \
                      FROM ( \
                           SELECT t.user_message_provider AS message_provider, \
                                  CASE \
                                    WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                    THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                    ELSE 'global' \
                                  END AS scope_key, \
                                  t.user_key AS subject_user_key, \
                                  latest_diary.window_end AS latest_window_end, \
                                  MIN(t.completed_at) AS first_completed_at \
                             FROM turns t \
                             LEFT JOIN ( \
                                  SELECT message_provider, scope_key, subject_user_key, \
                                         MAX(window_end) AS window_end \
                                    FROM ( \
                                         SELECT message_provider, scope_key, subject_user_key, \
                                                window_end \
                                           FROM user_memory_diary_entries \
                                          UNION ALL \
                                         SELECT message_provider, scope_key, subject_user_key, \
                                                window_end \
                                           FROM user_memory_jobs \
                                          WHERE kind = 'diary' \
                                            AND status IN ('completed', 'failed') \
                                            AND window_end IS NOT NULL \
                                    ) processed_diary_windows \
                                   GROUP BY message_provider, scope_key, subject_user_key \
                             ) latest_diary \
                               ON latest_diary.message_provider = t.user_message_provider \
                              AND latest_diary.scope_key = CASE \
                                    WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                    THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                    ELSE 'global' \
                                  END \
                              AND latest_diary.subject_user_key = t.user_key \
                            WHERE t.status = 'completed' \
                              AND t.completed_at IS NOT NULL \
                              AND t.completed_at >= $1 \
                              AND (latest_diary.window_end IS NULL \
                                   OR latest_diary.window_end < $1 \
                                   OR t.completed_at >= latest_diary.window_end) \
                            GROUP BY t.user_message_provider, \
                                  CASE \
                                    WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                    THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                    ELSE 'global' \
                                  END, \
                                  t.user_key, \
                                  latest_diary.window_end \
                      ) candidate_turns \
               ) diary_windows \
              WHERE diary_windows.window_start <= $3",
        )
        .bind(schedule.diary_cutoff)
        .bind(diary_window_seconds)
        .bind(schedule.diary_due_before)
        .fetch_all(&self.pool)
        .await?;
        for row in diary_rows {
            let key = UserMemoryKey {
                platform: PlatformName::new(row.get::<String, _>("message_provider")),
                scope_key: row.get("scope_key"),
                user_key: row.get("subject_user_key"),
            };
            let memory_key = key.memory_key();
            let result = sqlx::query(
                "INSERT INTO user_memory_jobs \
                   (id, kind, message_provider, scope_key, subject_user_key, memory_key, \
                    window_start, window_end, status, next_run_at, dedupe_key) \
                 VALUES ($1, 'diary', $2, $3, $4, $5, $6, $7, 'pending', $8, $9) \
                 ON CONFLICT DO NOTHING",
            )
            .bind(Uuid::new_v4())
            .bind(key.platform.as_str())
            .bind(&key.scope_key)
            .bind(&key.user_key)
            .bind(&memory_key)
            .bind(row.get::<OffsetDateTime, _>("window_start"))
            .bind(row.get::<OffsetDateTime, _>("window_end"))
            .bind(schedule.now)
            .bind(format!("diary:{memory_key}"))
            .execute(&self.pool)
            .await?;
            inserted += result.rows_affected();
        }

        let compact_rows = sqlx::query(
            "SELECT pending_sources.message_provider, pending_sources.scope_key, \
                    pending_sources.subject_user_key \
               FROM ( \
                    SELECT source.message_provider, source.scope_key, source.subject_user_key, \
                           latest_document.last_compacted_at \
                      FROM ( \
                           SELECT e.message_provider, e.scope_key, e.subject_user_key, \
                                  e.created_at, TRUE AS is_event \
                             FROM user_memory_events e \
                            UNION ALL \
                           SELECT de.message_provider, de.scope_key, de.subject_user_key, \
                                  de.created_at, FALSE AS is_event \
                             FROM user_memory_diary_entries de \
                      ) source \
                      LEFT JOIN ( \
                           SELECT DISTINCT ON (message_provider, scope_key, subject_user_key) \
                                  message_provider, scope_key, subject_user_key, last_compacted_at, \
                                  source_event_cutoff, source_diary_cutoff \
                             FROM user_memory_document_versions \
                            ORDER BY message_provider, scope_key, subject_user_key, revision DESC \
                      ) latest_document \
                        ON latest_document.message_provider = source.message_provider \
                       AND latest_document.scope_key = source.scope_key \
                       AND latest_document.subject_user_key = source.subject_user_key \
                     WHERE (source.is_event \
                            AND source.created_at > COALESCE(latest_document.source_event_cutoff, '-infinity'::timestamptz)) \
                        OR (NOT source.is_event \
                            AND source.created_at > COALESCE(latest_document.source_diary_cutoff, '-infinity'::timestamptz)) \
               ) pending_sources \
               LEFT JOIN ( \
                    SELECT DISTINCT message_provider, scope_key, subject_user_key \
                      FROM user_memory_jobs \
                     WHERE kind = 'diary' \
                       AND status IN ('pending', 'running') \
               ) active_diary \
                 ON active_diary.message_provider = pending_sources.message_provider \
                AND active_diary.scope_key = pending_sources.scope_key \
                AND active_diary.subject_user_key = pending_sources.subject_user_key \
               LEFT JOIN ( \
                    SELECT diary_windows.message_provider, diary_windows.scope_key, \
                           diary_windows.subject_user_key \
                      FROM ( \
                           SELECT candidate_turns.message_provider, candidate_turns.scope_key, \
                                  candidate_turns.subject_user_key, \
                                  CASE \
                                    WHEN candidate_turns.latest_window_end IS NULL \
                                      OR candidate_turns.latest_window_end < $2 \
                                    THEN candidate_turns.first_completed_at \
                                    ELSE candidate_turns.latest_window_end + ( \
                                      GREATEST( \
                                        ceil(EXTRACT(EPOCH FROM (candidate_turns.first_completed_at - candidate_turns.latest_window_end))::double precision / $4::double precision)::bigint - 1, \
                                        0::bigint \
                                      )::double precision * $4::double precision * INTERVAL '1 second' \
                                    ) \
                                  END AS window_start \
                             FROM ( \
                                  SELECT t.user_message_provider AS message_provider, \
                                         CASE \
                                           WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                           THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                           ELSE 'global' \
                                         END AS scope_key, \
                                         t.user_key AS subject_user_key, \
                                         latest_diary.window_end AS latest_window_end, \
                                         MIN(t.completed_at) AS first_completed_at \
                                    FROM turns t \
                                    LEFT JOIN ( \
                                         SELECT message_provider, scope_key, subject_user_key, \
                                                MAX(window_end) AS window_end \
                                           FROM ( \
                                                SELECT message_provider, scope_key, subject_user_key, \
                                                       window_end \
                                                  FROM user_memory_diary_entries \
                                                 UNION ALL \
                                                SELECT message_provider, scope_key, subject_user_key, \
                                                       window_end \
                                                  FROM user_memory_jobs \
                                                 WHERE kind = 'diary' \
                                                   AND status IN ('completed', 'failed') \
                                                   AND window_end IS NOT NULL \
                                           ) processed_diary_windows \
                                          GROUP BY message_provider, scope_key, subject_user_key \
                                    ) latest_diary \
                                      ON latest_diary.message_provider = t.user_message_provider \
                                     AND latest_diary.scope_key = CASE \
                                           WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                           THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                           ELSE 'global' \
                                         END \
                                     AND latest_diary.subject_user_key = t.user_key \
                                   WHERE t.status = 'completed' \
                                     AND t.completed_at IS NOT NULL \
                                     AND t.completed_at >= $2 \
                                     AND (latest_diary.window_end IS NULL \
                                          OR latest_diary.window_end < $2 \
                                          OR t.completed_at >= latest_diary.window_end) \
                                   GROUP BY t.user_message_provider, \
                                         CASE \
                                           WHEN t.user_message_channel LIKE 'guild:%:channel:%' \
                                           THEN 'guild:' || split_part(t.user_message_channel, ':', 2) \
                                           ELSE 'global' \
                                         END, \
                                         t.user_key, \
                                         latest_diary.window_end \
                             ) candidate_turns \
                      ) diary_windows \
                     WHERE diary_windows.window_start <= $3 \
               ) due_diary \
                 ON due_diary.message_provider = pending_sources.message_provider \
                AND due_diary.scope_key = pending_sources.scope_key \
                AND due_diary.subject_user_key = pending_sources.subject_user_key \
              WHERE active_diary.message_provider IS NULL \
                AND due_diary.message_provider IS NULL \
              GROUP BY pending_sources.message_provider, pending_sources.scope_key, \
                       pending_sources.subject_user_key, pending_sources.last_compacted_at \
             HAVING COALESCE(pending_sources.last_compacted_at, '-infinity'::timestamptz) <= $1",
        )
        .bind(schedule.compact_due_before)
        .bind(schedule.diary_cutoff)
        .bind(schedule.diary_due_before)
        .bind(diary_window_seconds)
        .fetch_all(&self.pool)
        .await?;
        for row in compact_rows {
            let key = UserMemoryKey {
                platform: PlatformName::new(row.get::<String, _>("message_provider")),
                scope_key: row.get("scope_key"),
                user_key: row.get("subject_user_key"),
            };
            let memory_key = key.memory_key();
            let result = sqlx::query(
                "INSERT INTO user_memory_jobs \
                   (id, kind, message_provider, scope_key, subject_user_key, memory_key, \
                    status, next_run_at, dedupe_key) \
                 VALUES ($1, 'compact', $2, $3, $4, $5, 'pending', $6, $7) \
                 ON CONFLICT DO NOTHING",
            )
            .bind(Uuid::new_v4())
            .bind(key.platform.as_str())
            .bind(&key.scope_key)
            .bind(&key.user_key)
            .bind(&memory_key)
            .bind(schedule.now)
            .bind(format!("compact:{memory_key}"))
            .execute(&self.pool)
            .await?;
            inserted += result.rows_affected();
        }
        Ok(inserted)
    }

    pub(super) async fn claim_memory_jobs(
        &self,
        worker_id: String,
        limit: u32,
        lease_until: OffsetDateTime,
    ) -> Result<Vec<UserMemoryJob>, SqlxStorageError> {
        let rows = sqlx::query(
            "UPDATE user_memory_jobs j \
                SET status = 'running', attempts = attempts + 1, leased_by = $2, \
                    leased_until = $3, started_at = COALESCE(started_at, now()), \
                    completed_at = NULL, error = NULL \
               FROM ( \
                    SELECT picked.id \
                      FROM user_memory_jobs picked \
                      JOIN ( \
                           SELECT candidate.id, \
                                  row_number() OVER ( \
                                      PARTITION BY candidate.memory_key \
                                      ORDER BY candidate.next_run_at, candidate.created_at \
                                  ) AS rn \
                             FROM user_memory_jobs candidate \
                            WHERE candidate.next_run_at <= now() \
                              AND (candidate.status = 'pending' \
                                   OR (candidate.status = 'running' AND candidate.leased_until < now())) \
                              AND NOT EXISTS ( \
                                   SELECT 1 FROM user_memory_jobs active \
                                    WHERE active.memory_key = candidate.memory_key \
                                      AND active.status = 'running' \
                                      AND active.leased_until >= now() \
                                      AND active.id <> candidate.id \
                              ) \
                      ) candidates ON candidates.id = picked.id \
                     WHERE candidates.rn = 1 \
                     ORDER BY picked.next_run_at, picked.created_at \
                     LIMIT $1 \
                     FOR UPDATE OF picked SKIP LOCKED \
               ) picked_jobs \
              WHERE j.id = picked_jobs.id \
              RETURNING j.id, j.kind, j.message_provider, j.scope_key, j.subject_user_key, \
                        j.memory_key, j.window_start, j.window_end, j.attempts, \
                        j.leased_by, j.leased_until, j.dedupe_key",
        )
        .bind(i64::from(limit))
        .bind(&worker_id)
        .bind(lease_until)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(memory_job_from_row).collect()
    }

    pub(super) async fn finish_memory_job(
        &self,
        completion: MemoryJobCompletion,
    ) -> Result<(), SqlxStorageError> {
        match completion {
            MemoryJobCompletion::Completed { job_id } => {
                sqlx::query(
                    "UPDATE user_memory_jobs \
                        SET status = 'completed', completed_at = now(), leased_by = NULL, \
                            leased_until = NULL, error = NULL \
                      WHERE id = $1",
                )
                .bind(job_id)
                .execute(&self.pool)
                .await?;
            }
            MemoryJobCompletion::Retry {
                job_id,
                error,
                next_run_at,
            } => {
                sqlx::query(
                    "UPDATE user_memory_jobs \
                        SET status = 'pending', next_run_at = $2, leased_by = NULL, \
                            leased_until = NULL, error = $3 \
                      WHERE id = $1",
                )
                .bind(job_id)
                .bind(next_run_at)
                .bind(&error)
                .execute(&self.pool)
                .await?;
            }
            MemoryJobCompletion::Failed { job_id, error } => {
                sqlx::query(
                    "UPDATE user_memory_jobs \
                        SET status = 'failed', completed_at = now(), leased_by = NULL, \
                            leased_until = NULL, error = $2 \
                      WHERE id = $1",
                )
                .bind(job_id)
                .bind(&error)
                .execute(&self.pool)
                .await?;
            }
        }
        Ok(())
    }

    pub(super) async fn load_memory_turn_window(
        &self,
        window: MemoryTurnWindow,
    ) -> Result<Vec<UserMemoryTurn>, SqlxStorageError> {
        let rows = sqlx::query(
            "SELECT conversation_id, id AS turn_id, completed_at, user_display_name, \
                    user_content, assistant_content \
               FROM turns \
              WHERE user_message_provider = $1 \
                AND user_key = $2 \
                AND CASE \
                      WHEN user_message_channel LIKE 'guild:%:channel:%' \
                      THEN 'guild:' || split_part(user_message_channel, ':', 2) \
                      ELSE 'global' \
                    END = $3 \
                AND status = 'completed' \
                AND completed_at IS NOT NULL \
                AND completed_at >= $4 \
                AND completed_at < $5 \
              ORDER BY completed_at, ordinal \
              LIMIT $6",
        )
        .bind(window.key.platform.as_str())
        .bind(&window.key.user_key)
        .bind(&window.key.scope_key)
        .bind(window.window_start)
        .bind(window.window_end)
        .bind(i64::from(window.max_turns))
        .fetch_all(&self.pool)
        .await?;
        let mut turns = rows
            .into_iter()
            .map(|row| UserMemoryTurn {
                conversation_id: ConversationId(row.get("conversation_id")),
                turn_id: TurnId(row.get("turn_id")),
                completed_at: row.get("completed_at"),
                user_display_name: row.get("user_display_name"),
                user_content: row.get("user_content"),
                assistant_content: row.get("assistant_content"),
                image_context: Vec::new(),
                audio_transcriptions: Vec::new(),
            })
            .collect::<Vec<_>>();
        let turn_ids = turns.iter().map(|turn| turn.turn_id.0).collect::<Vec<_>>();
        let mut images_by_turn = load_memory_image_context(&self.pool, &turn_ids).await?;
        let mut audio_by_turn = load_memory_audio_transcriptions(&self.pool, &turn_ids).await?;
        for turn in &mut turns {
            turn.image_context = images_by_turn.remove(&turn.turn_id).unwrap_or_default();
            turn.audio_transcriptions = audio_by_turn.remove(&turn.turn_id).unwrap_or_default();
        }
        Ok(turns)
    }
}

async fn load_memory_image_context(
    pool: &PgPool,
    turn_ids: &[Uuid],
) -> Result<BTreeMap<TurnId, Vec<UserMemoryImageContext>>, SqlxStorageError> {
    if turn_ids.is_empty() {
        return Ok(BTreeMap::new());
    }
    let rows = sqlx::query(
        "SELECT a.turn_id, a.media_uri, a.source, m.mime_type \
           FROM turn_assets a \
           LEFT JOIN media_assets m ON m.uri = a.media_uri \
          WHERE a.turn_id = ANY($1) \
            AND a.replayable \
            AND (m.category = 'image' OR m.mime_type LIKE 'image/%' \
                 OR a.media_uri LIKE 'media://images/%' OR a.media_uri LIKE 'file://images/%') \
          ORDER BY a.turn_id, a.ordinal, a.id",
    )
    .bind(turn_ids)
    .fetch_all(pool)
    .await?;
    let mut out = BTreeMap::<TurnId, Vec<UserMemoryImageContext>>::new();
    for row in rows {
        let turn_id = TurnId(row.get("turn_id"));
        out.entry(turn_id)
            .or_default()
            .push(UserMemoryImageContext {
                image_uri: MediaUri::new(row.get::<String, _>("media_uri")),
                source: row.get("source"),
                mime_type: row.get("mime_type"),
            });
    }
    Ok(out)
}

async fn load_memory_audio_transcriptions(
    pool: &PgPool,
    turn_ids: &[Uuid],
) -> Result<BTreeMap<TurnId, Vec<UserMemoryAudioTranscription>>, SqlxStorageError> {
    if turn_ids.is_empty() {
        return Ok(BTreeMap::new());
    }
    let rows = sqlx::query(
        "SELECT ta.turn_id, tt.id AS tool_trace_id, tt.request, tt.response \
           FROM turn_attempt_tool_traces tt \
           JOIN turn_attempts ta ON ta.id = tt.attempt_id \
          WHERE ta.turn_id = ANY($1) \
            AND ta.status = 'completed' \
            AND tt.trace_kind = 'client' \
            AND tt.tool_name = 'transcribe_audio' \
            AND COALESCE(tt.is_error, false) = false \
          ORDER BY ta.turn_id, tt.ordinal",
    )
    .bind(turn_ids)
    .fetch_all(pool)
    .await?;
    let mut out = BTreeMap::<TurnId, Vec<UserMemoryAudioTranscription>>::new();
    for row in rows {
        let turn_id = TurnId(row.get("turn_id"));
        let Some(transcription) = memory_audio_transcription_from_tool_row(&row) else {
            continue;
        };
        out.entry(turn_id).or_default().push(transcription);
    }
    Ok(out)
}

fn memory_audio_transcription_from_tool_row(
    row: &sqlx::postgres::PgRow,
) -> Option<UserMemoryAudioTranscription> {
    let response = row.get::<Option<Value>, _>("response")?;
    let request = row.get::<Option<Value>, _>("request");
    memory_audio_transcription_from_values(row.get("tool_trace_id"), request.as_ref(), &response)
}

fn memory_audio_transcription_from_values(
    tool_trace_id: i64,
    request: Option<&Value>,
    response: &Value,
) -> Option<UserMemoryAudioTranscription> {
    let text = response
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())?
        .to_string();
    Some(UserMemoryAudioTranscription {
        tool_trace_id,
        audio_uri: request.and_then(audio_uri_from_tool_request),
        text,
        language: optional_non_empty_string(response.get("language")),
        duration_seconds: response
            .get("duration_seconds")
            .and_then(Value::as_f64)
            .filter(|duration| duration.is_finite()),
    })
}

fn audio_uri_from_tool_request(request: &Value) -> Option<String> {
    request
        .get("input")
        .and_then(|input| input.get("audio_uri").or_else(|| input.get("audio")))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|uri| !uri.is_empty())
        .and_then(|uri| canonical_media_uri_string(uri).ok())
}

fn optional_non_empty_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}
fn memory_event_kind_as_str(kind: UserMemoryEventKind) -> &'static str {
    match kind {
        UserMemoryEventKind::Remember => "remember",
        UserMemoryEventKind::Correction => "correction",
        UserMemoryEventKind::Forget => "forget",
        UserMemoryEventKind::DiaryObservation => "diary_observation",
        UserMemoryEventKind::OperatorNote => "operator_note",
    }
}

fn memory_event_kind_from_str(kind: &str) -> Result<UserMemoryEventKind, SqlxStorageError> {
    match kind {
        "remember" => Ok(UserMemoryEventKind::Remember),
        "correction" => Ok(UserMemoryEventKind::Correction),
        "forget" => Ok(UserMemoryEventKind::Forget),
        "diary_observation" => Ok(UserMemoryEventKind::DiaryObservation),
        "operator_note" => Ok(UserMemoryEventKind::OperatorNote),
        other => Err(SqlxStorageError::InvalidReference(format!(
            "unknown user memory event kind `{other}`"
        ))),
    }
}

fn memory_job_kind_from_str(kind: &str) -> Result<MemoryJobKind, SqlxStorageError> {
    match kind {
        "diary" => Ok(MemoryJobKind::Diary),
        "compact" => Ok(MemoryJobKind::Compact),
        other => Err(SqlxStorageError::InvalidReference(format!(
            "unknown user memory job kind `{other}`"
        ))),
    }
}

fn user_memory_key(provider: String, scope_key: String, user_key: String) -> UserMemoryKey {
    UserMemoryKey {
        platform: PlatformName::new(provider),
        scope_key,
        user_key,
    }
}

fn memory_event_from_row(row: sqlx::postgres::PgRow) -> Result<UserMemoryEvent, SqlxStorageError> {
    let tags = serde_json::from_value(row.get::<Value, _>("tags"))?;
    let kind = memory_event_kind_from_str(&row.get::<String, _>("kind"))?;
    Ok(UserMemoryEvent {
        id: row.get("id"),
        key: user_memory_key(
            row.get("message_provider"),
            row.get("scope_key"),
            row.get("subject_user_key"),
        ),
        actor_user_key: row.get("actor_user_key"),
        kind,
        body: row.get("body"),
        tags,
        confidence: row.get("confidence"),
        source_conversation_id: row
            .get::<Option<Uuid>, _>("source_conversation_id")
            .map(ConversationId),
        source_turn_id: row.get::<Option<Uuid>, _>("source_turn_id").map(TurnId),
        source_tool_trace_id: row.get("source_tool_trace_id"),
        supersedes_event_id: row.get("supersedes_event_id"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}

fn diary_entry_from_row(
    row: sqlx::postgres::PgRow,
) -> Result<UserMemoryDiaryEntry, SqlxStorageError> {
    let source_turn_ids = row
        .get::<Vec<Uuid>, _>("source_turn_ids")
        .into_iter()
        .map(TurnId)
        .collect();
    let usage = serde_json::from_value(row.get::<Value, _>("usage"))?;
    Ok(UserMemoryDiaryEntry {
        id: row.get("id"),
        key: user_memory_key(
            row.get("message_provider"),
            row.get("scope_key"),
            row.get("subject_user_key"),
        ),
        window_start: row.get("window_start"),
        window_end: row.get("window_end"),
        source_turn_ids,
        markdown: row.get("markdown"),
        agent_name: row.get("agent_name"),
        llm_provider: ProviderName::new(row.get::<String, _>("llm_provider")),
        llm_model: ModelId::new(row.get::<String, _>("llm_model")),
        usage,
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}

fn document_from_row(row: sqlx::postgres::PgRow) -> Result<UserMemoryDocument, SqlxStorageError> {
    Ok(UserMemoryDocument {
        key: user_memory_key(
            row.get("message_provider"),
            row.get("scope_key"),
            row.get("subject_user_key"),
        ),
        revision: row.get("revision"),
        markdown: row.get("markdown"),
        last_compacted_at: row.get("last_compacted_at"),
        source_event_cutoff: row.get("source_event_cutoff"),
        source_diary_cutoff: row.get("source_diary_cutoff"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    })
}

fn memory_job_from_row(row: sqlx::postgres::PgRow) -> Result<UserMemoryJob, SqlxStorageError> {
    let kind = memory_job_kind_from_str(&row.get::<String, _>("kind"))?;
    Ok(UserMemoryJob {
        id: row.get("id"),
        kind,
        key: user_memory_key(
            row.get("message_provider"),
            row.get("scope_key"),
            row.get("subject_user_key"),
        ),
        memory_key: row.get("memory_key"),
        window_start: row.get("window_start"),
        window_end: row.get("window_end"),
        attempts: row.get("attempts"),
        leased_by: row.get("leased_by"),
        leased_until: row.get("leased_until"),
        dedupe_key: row.get("dedupe_key"),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn memory_audio_transcription_parses_tool_request_and_response() {
        let request = json!({
            "id": "call-1",
            "name": "transcribe_audio",
            "input": {
                "audio_uri": "file://audio/voice.ogg"
            }
        });
        let response = json!({
            "text": "I am allergic to coconut.",
            "language": "en",
            "duration_seconds": 3.25,
            "words": [
                {
                    "text": "Ignore",
                    "start": 0.0,
                    "end": 0.25
                }
            ]
        });

        let transcription =
            memory_audio_transcription_from_values(42, Some(&request), &response).unwrap();

        assert_eq!(transcription.tool_trace_id, 42);
        assert_eq!(
            transcription.audio_uri.as_deref(),
            Some("media://audio/voice.ogg")
        );
        assert_eq!(transcription.text, "I am allergic to coconut.");
        assert_eq!(transcription.language.as_deref(), Some("en"));
        assert_eq!(transcription.duration_seconds, Some(3.25));
    }
}
