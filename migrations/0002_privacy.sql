-- Per-user, per-guild privacy opt-in for "design 3" (opt-in /
-- privacy-maxxing). Multi-tenant: a user may opt in on one server
-- without opting in on others, so the primary key is the composite
-- (guild_id, user_id).
--
-- The bot's runtime rule:
--   - A missing row → not opted in (the privacy-preserving default).
--   - The bot ALWAYS sees the user's own `@<bot>` mention and any
--     message inside a Grok-owned thread, regardless of this table.
--   - Quoted (Discord-reply) messages are only included as context
--     when their author has opted in for the current guild.

CREATE TABLE user_privacy (
    discord_guild_id    BIGINT NOT NULL,
    discord_user_id     BIGINT NOT NULL,
    opted_in            BOOLEAN NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (discord_guild_id, discord_user_id)
);
