-- Capture per-message user identity and per-conversation titles so the
-- web viewer can render conversations with real Discord profiles and
-- meaningful headers instead of "user" + "Untitled conversation".
--
-- Two concerns rolled into one migration since they're both about
-- "make the viewer not look anonymous":
--
--   1. Pin which Discord user authored each turn (id + name copied
--      verbatim onto the turn row — names can change, but we want the
--      historical name to stick to the historical turn).
--   2. Cache Discord avatars on disk and remember which hash we have so
--      we only re-fetch on change. Username/display_name live here too
--      as the *current* known values; per-turn rows are the historical
--      record.
--
-- Titles for conversations live on the `conversations` row directly:
-- the column already exists (0001), so this migration only adds
-- `title_generated_at` to record when the background titler ran.

ALTER TABLE conversations
    ADD COLUMN title_generated_at TIMESTAMPTZ;

ALTER TABLE turns
    ADD COLUMN discord_user_id BIGINT,
    ADD COLUMN discord_user_name TEXT;

CREATE INDEX turns_user_idx ON turns (discord_user_id) WHERE discord_user_id IS NOT NULL;

CREATE TABLE discord_users (
    id                          BIGINT PRIMARY KEY,
    username                    TEXT NOT NULL,
    display_name                TEXT,
    avatar_hash                 TEXT,
    avatar_local_path           TEXT,
    last_avatar_fetched_at      TIMESTAMPTZ,
    last_seen_at                TIMESTAMPTZ NOT NULL DEFAULT now()
);
