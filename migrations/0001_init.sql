-- Initial schema for grok-discord-bot.
--
-- A *conversation* is the unit surfaced in the web viewer. It is created
-- when a user mentions the bot outside any existing context. Subsequent
-- replies to the bot's messages — or @mentions inside a bot-owned thread
-- — continue that same conversation via the message_links table.
--
-- Each *turn* is one user→assistant exchange. We record the exact context
-- snapshot fed to the model (context_items) and every server-side tool
-- call (tool_calls), so the viewer can show "what did the model see and
-- do" with full fidelity.

CREATE TABLE conversations (
    id                          UUID PRIMARY KEY,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    discord_guild_id            BIGINT NOT NULL,
    discord_channel_id          BIGINT NOT NULL,
    created_by_user_id          BIGINT NOT NULL,
    root_discord_message_id     BIGINT NOT NULL,
    title                       TEXT,
    model                       TEXT NOT NULL
);

CREATE INDEX conversations_created_at_idx ON conversations (created_at DESC);

CREATE TABLE turns (
    id                              UUID PRIMARY KEY,
    conversation_id                 UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_index                      INTEGER NOT NULL,
    created_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at                    TIMESTAMPTZ,
    user_discord_message_id         BIGINT NOT NULL,
    user_content                    TEXT NOT NULL,
    assistant_discord_message_id    BIGINT,
    assistant_content               TEXT,
    status                          TEXT NOT NULL DEFAULT 'pending',
    error                           TEXT,
    UNIQUE (conversation_id, turn_index)
);

CREATE INDEX turns_conversation_idx ON turns (conversation_id, turn_index);

CREATE TABLE context_items (
    id                  BIGSERIAL PRIMARY KEY,
    turn_id             UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    position            INTEGER NOT NULL,
    source              TEXT NOT NULL,
    role                TEXT NOT NULL,
    content             TEXT NOT NULL,
    discord_message_id  BIGINT,
    UNIQUE (turn_id, position)
);

CREATE INDEX context_items_turn_idx ON context_items (turn_id, position);

CREATE TABLE tool_calls (
    id              BIGSERIAL PRIMARY KEY,
    turn_id         UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    ordinal         INTEGER NOT NULL,
    tool_name       TEXT NOT NULL,
    request         JSONB NOT NULL,
    response        JSONB NOT NULL,
    UNIQUE (turn_id, ordinal)
);

CREATE INDEX tool_calls_turn_idx ON tool_calls (turn_id, ordinal);

-- Lookup table mapping every Discord message the bot is aware of (both
-- user prompts and bot replies) to its owning conversation. This is how
-- we resolve "is this @mention continuing an existing conversation, or
-- starting a new one?" without paging Discord history.
CREATE TABLE message_links (
    discord_message_id  BIGINT PRIMARY KEY,
    conversation_id     UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_id             UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    role                TEXT NOT NULL
);

CREATE INDEX message_links_conversation_idx ON message_links (conversation_id);
