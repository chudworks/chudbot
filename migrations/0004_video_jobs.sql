-- Video generation requests we've submitted to xAI but may still be
-- polling on. One row per `start_video_generation` tool call.
--
-- Stored separately from `tool_calls` because the lifecycle is
-- different: the same request_id is touched repeatedly (once per
-- `check_video_status` call) and we want a single canonical row per
-- job rather than N tool-call records with stale snapshots.
--
-- On bot startup we can later scan WHERE status = 'pending' and
-- reconstruct in-flight turns to resume polling.

CREATE TABLE video_jobs (
    id              UUID PRIMARY KEY,
    turn_id         UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    request_id      TEXT NOT NULL UNIQUE,
    prompt          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    video_uri       TEXT,
    submitted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    error           TEXT
);

CREATE INDEX video_jobs_turn_idx ON video_jobs (turn_id);
CREATE INDEX video_jobs_status_idx ON video_jobs (status);
