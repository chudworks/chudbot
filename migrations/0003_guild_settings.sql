-- Per-guild settings. Currently only stores the privacy / context-
-- gathering mode (one of the four "designs" the team discussed):
--   - open                — Design 1
--   - channel_only        — Design 2 (also stores channel_id)
--   - opt_in              — Design 3 (default)
--   - conversation_only   — Design 4
--
-- Stored as JSONB because the variants have different shapes (e.g.
-- channel_only carries a channel_id, open/channel_only carry a
-- history_size). The Rust side uses serde with `#[serde(tag = "mode")]`
-- to keep DB and config encoding identical, so the same row format is
-- used in config.toml's optional `[privacy]` block (which serves as
-- the bootstrap default for guilds with no row here).
--
-- Missing row → fall back to whatever the config's default_privacy
-- setting is (which itself defaults to opt_in).

CREATE TABLE guild_settings (
    discord_guild_id    BIGINT PRIMARY KEY,
    privacy_mode        JSONB NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
