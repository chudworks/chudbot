-- User memory ledger, compact profiles, generated diary artifacts, and durable
-- background jobs.

CREATE TABLE user_memory_events (
    id                     UUID PRIMARY KEY,
    message_provider       TEXT NOT NULL,
    scope_key              TEXT NOT NULL,
    subject_user_key       TEXT NOT NULL,
    actor_user_key         TEXT,
    kind                   TEXT NOT NULL
        CHECK (kind IN ('remember', 'correction', 'forget', 'diary_observation', 'operator_note')),
    body                   TEXT NOT NULL,
    tags                   JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence             REAL,
    source_conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    source_turn_id         UUID REFERENCES turns(id) ON DELETE SET NULL,
    source_tool_trace_id   BIGINT REFERENCES turn_attempt_tool_traces(id) ON DELETE SET NULL,
    supersedes_event_id    UUID REFERENCES user_memory_events(id) ON DELETE SET NULL,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX user_memory_events_subject_created_idx
    ON user_memory_events (message_provider, scope_key, subject_user_key, created_at);

CREATE TABLE user_memory_diary_entries (
    id                    UUID PRIMARY KEY,
    message_provider      TEXT NOT NULL,
    scope_key             TEXT NOT NULL,
    subject_user_key      TEXT NOT NULL,
    window_start          TIMESTAMPTZ NOT NULL,
    window_end            TIMESTAMPTZ NOT NULL,
    source_turn_ids       UUID[] NOT NULL,
    markdown              TEXT NOT NULL,
    agent_name            TEXT NOT NULL,
    llm_provider          TEXT NOT NULL,
    llm_model             TEXT NOT NULL,
    usage                 JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX user_memory_diary_subject_created_idx
    ON user_memory_diary_entries (message_provider, scope_key, subject_user_key, created_at);

CREATE TABLE user_memory_documents (
    message_provider     TEXT NOT NULL,
    scope_key            TEXT NOT NULL,
    subject_user_key     TEXT NOT NULL,
    revision             BIGINT NOT NULL,
    markdown             TEXT NOT NULL,
    last_compacted_at    TIMESTAMPTZ NOT NULL,
    source_event_cutoff  TIMESTAMPTZ,
    source_diary_cutoff  TIMESTAMPTZ,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, scope_key, subject_user_key)
);

CREATE INDEX user_memory_documents_compacted_idx
    ON user_memory_documents (last_compacted_at);

CREATE TABLE user_memory_document_versions (
    id                     UUID PRIMARY KEY,
    message_provider       TEXT NOT NULL,
    scope_key              TEXT NOT NULL,
    subject_user_key       TEXT NOT NULL,
    revision               BIGINT NOT NULL,
    markdown               TEXT NOT NULL,
    source_event_ids       UUID[] NOT NULL,
    source_diary_entry_ids UUID[] NOT NULL,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (message_provider, scope_key, subject_user_key, revision)
);

CREATE TABLE user_memory_jobs (
    id                  UUID PRIMARY KEY,
    kind                TEXT NOT NULL CHECK (kind IN ('diary', 'compact')),
    message_provider    TEXT NOT NULL,
    scope_key           TEXT NOT NULL,
    subject_user_key    TEXT NOT NULL,
    memory_key          TEXT NOT NULL,
    window_start        TIMESTAMPTZ,
    window_end          TIMESTAMPTZ,
    status              TEXT NOT NULL
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    attempts            INTEGER NOT NULL DEFAULT 0,
    next_run_at         TIMESTAMPTZ NOT NULL,
    leased_by           TEXT,
    leased_until        TIMESTAMPTZ,
    dedupe_key          TEXT NOT NULL,
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    error               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX user_memory_jobs_active_dedupe_idx
    ON user_memory_jobs (dedupe_key)
    WHERE status IN ('pending', 'running');

CREATE INDEX user_memory_jobs_due_idx
    ON user_memory_jobs (status, next_run_at, leased_until);

CREATE INDEX user_memory_jobs_active_memory_key_idx
    ON user_memory_jobs (memory_key, leased_until)
    WHERE status = 'running';

CREATE TRIGGER user_memory_events_touch_updated_at
    BEFORE UPDATE ON user_memory_events
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();

CREATE TRIGGER user_memory_diary_entries_touch_updated_at
    BEFORE UPDATE ON user_memory_diary_entries
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();

CREATE TRIGGER user_memory_documents_touch_updated_at
    BEFORE UPDATE ON user_memory_documents
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();

CREATE TRIGGER user_memory_jobs_touch_updated_at
    BEFORE UPDATE ON user_memory_jobs
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
