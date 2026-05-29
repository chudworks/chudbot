-- Admin "stop sign" (🛑) — lets a hard-coded admin (configured in
-- config.toml's top-level `admins` list) halt the bot in a single
-- conversation by reacting with the 🛑 (`:octagonal_sign:`) emoji on any
-- message the bot already tracks for that conversation. Removing the
-- reaction resumes it.
--
-- Modeled as nullable columns on `conversations` rather than a separate
-- table: the stop is one bit of per-conversation state read on the hot
-- path (every incoming mention checks it), so keeping it inline with the
-- conversation row it gates avoids an extra lookup. `stopped_at` doubles
-- as the flag (NULL = active) and an audit timestamp; `stopped_by_user_id`
-- records which admin paused it. Both clear back to NULL on resume.
ALTER TABLE conversations
    ADD COLUMN stopped_at          TIMESTAMPTZ,
    ADD COLUMN stopped_by_user_id  BIGINT;
