-- Speed compact scheduling guards for diary backfill completion.

CREATE INDEX user_memory_jobs_active_diary_subject_idx
    ON user_memory_jobs (message_provider, scope_key, subject_user_key)
    WHERE kind = 'diary' AND status IN ('pending', 'running');

CREATE INDEX user_memory_diary_subject_window_end_idx
    ON user_memory_diary_entries (
        message_provider,
        scope_key,
        subject_user_key,
        window_end
    );
