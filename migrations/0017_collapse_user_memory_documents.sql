-- Store current compact profile state directly on immutable profile revisions.

ALTER TABLE user_memory_document_versions
    ADD COLUMN last_compacted_at TIMESTAMPTZ,
    ADD COLUMN source_event_cutoff TIMESTAMPTZ,
    ADD COLUMN source_diary_cutoff TIMESTAMPTZ;

WITH source_cutoffs AS (
    SELECT v.id,
           MAX(e.created_at) AS event_cutoff,
           MAX(de.created_at) AS diary_cutoff
      FROM user_memory_document_versions v
      LEFT JOIN user_memory_events e
        ON e.id = ANY(v.source_event_ids)
      LEFT JOIN user_memory_diary_entries de
        ON de.id = ANY(v.source_diary_entry_ids)
     GROUP BY v.id
), cumulative_cutoffs AS (
    SELECT v.id,
           MAX(s.event_cutoff) OVER (
               PARTITION BY v.message_provider, v.scope_key, v.subject_user_key
               ORDER BY v.revision
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS source_event_cutoff,
           MAX(s.diary_cutoff) OVER (
               PARTITION BY v.message_provider, v.scope_key, v.subject_user_key
               ORDER BY v.revision
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS source_diary_cutoff
      FROM user_memory_document_versions v
      JOIN source_cutoffs s ON s.id = v.id
)
UPDATE user_memory_document_versions v
   SET last_compacted_at = v.created_at,
       source_event_cutoff = c.source_event_cutoff,
       source_diary_cutoff = c.source_diary_cutoff
  FROM cumulative_cutoffs c
 WHERE c.id = v.id;

WITH latest_version AS (
    SELECT DISTINCT ON (message_provider, scope_key, subject_user_key)
           id, message_provider, scope_key, subject_user_key
      FROM user_memory_document_versions
     ORDER BY message_provider, scope_key, subject_user_key, revision DESC
)
UPDATE user_memory_document_versions v
   SET last_compacted_at = d.last_compacted_at,
       source_event_cutoff = d.source_event_cutoff,
       source_diary_cutoff = d.source_diary_cutoff
  FROM latest_version latest
  JOIN user_memory_documents d
    ON d.message_provider = latest.message_provider
   AND d.scope_key = latest.scope_key
   AND d.subject_user_key = latest.subject_user_key
 WHERE v.id = latest.id;

ALTER TABLE user_memory_document_versions
    ALTER COLUMN last_compacted_at SET NOT NULL;

CREATE INDEX user_memory_document_versions_latest_idx
    ON user_memory_document_versions (message_provider, scope_key, subject_user_key, revision DESC);

DROP TABLE user_memory_documents;
