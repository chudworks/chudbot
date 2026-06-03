-- chudbot 2.0 storage model.
--
-- The existing 0001-0010 migrations remain as the deployed v1 history. This
-- migration creates the v2 tables under temporary names, copies legacy rows,
-- drops the old tables, then renames the v2 tables into the canonical names.
-- Core bot tables use provider-owned opaque platform keys:
--
--   message_provider = "discord"
--   channel          = "guild:<guild_id>:channel:<channel_id>"
--   message          = "<platform message id>"
--   user_key         = "<platform user id>"
--
-- Discord-specific decomposition is optional enrichment in discord_channels;
-- bot routing and lookup logic must not depend on it.

CREATE OR REPLACE FUNCTION chudbot_touch_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TABLE v2_app_versions (
    id              SERIAL PRIMARY KEY,
    git_version     TEXT NOT NULL UNIQUE,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE v2_media_assets (
    uri             TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    name            TEXT NOT NULL,
    mime_type       TEXT NOT NULL,
    size_bytes      BIGINT NOT NULL DEFAULT 0,
    storage_backend TEXT NOT NULL DEFAULT 'local',
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (category, name)
);

CREATE TABLE v2_platform_channels (
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    parent_channel   TEXT,
    channel_kind     TEXT NOT NULL DEFAULT 'channel',
    display_name     TEXT,
    raw              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, channel),
    FOREIGN KEY (message_provider, parent_channel)
        REFERENCES v2_platform_channels (message_provider, channel)
);

CREATE TABLE v2_platform_users (
    message_provider TEXT NOT NULL,
    user_key         TEXT NOT NULL,
    username         TEXT NOT NULL,
    display_name     TEXT,
    avatar_url       TEXT,
    avatar_media_uri TEXT REFERENCES v2_media_assets(uri),
    is_bot           BOOLEAN NOT NULL DEFAULT false,
    raw              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, user_key)
);

CREATE TABLE v2_platform_messages (
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    message          TEXT NOT NULL,
    author_user_key  TEXT,
    content          TEXT,
    platform_created_at TIMESTAMPTZ,
    raw              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, channel, message),
    FOREIGN KEY (message_provider, channel)
        REFERENCES v2_platform_channels (message_provider, channel),
    FOREIGN KEY (message_provider, author_user_key)
        REFERENCES v2_platform_users (message_provider, user_key)
);

CREATE TABLE v2_platform_message_attachments (
    id               BIGSERIAL PRIMARY KEY,
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    message          TEXT NOT NULL,
    ordinal          INTEGER NOT NULL,
    attachment_key   TEXT,
    url              TEXT NOT NULL,
    filename         TEXT NOT NULL,
    content_type     TEXT,
    size_bytes       BIGINT,
    media_uri        TEXT REFERENCES v2_media_assets(uri),
    raw              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (message_provider, channel, message, ordinal),
    FOREIGN KEY (message_provider, channel, message)
        REFERENCES v2_platform_messages (message_provider, channel, message)
        ON DELETE CASCADE
);

CREATE TABLE v2_conversations (
    id                    UUID PRIMARY KEY,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    message_provider       TEXT NOT NULL,
    channel                TEXT NOT NULL,
    created_by_user_key    TEXT NOT NULL,
    root_message_provider  TEXT NOT NULL,
    root_message_channel   TEXT NOT NULL,
    root_message           TEXT NOT NULL,
    agent_name             TEXT NOT NULL,
    llm_provider           TEXT NOT NULL,
    llm_model              TEXT NOT NULL,
    sampling               JSONB NOT NULL DEFAULT '{}'::jsonb,
    provider_options       JSONB NOT NULL DEFAULT '{}'::jsonb,
    system_instructions    TEXT NOT NULL,
    title                  TEXT,
    title_generated_at     TIMESTAMPTZ,
    created_app_version_id INTEGER REFERENCES v2_app_versions(id),
    stopped_at             TIMESTAMPTZ,
    stopped_by_provider    TEXT,
    stopped_by_user_key    TEXT,
    next_turn_ordinal      BIGINT NOT NULL DEFAULT 0,
    next_response_ordinal  BIGINT NOT NULL DEFAULT 0,
    metadata               JSONB NOT NULL DEFAULT '{}'::jsonb,
    FOREIGN KEY (message_provider, channel)
        REFERENCES v2_platform_channels (message_provider, channel),
    FOREIGN KEY (message_provider, created_by_user_key)
        REFERENCES v2_platform_users (message_provider, user_key)
);

CREATE TABLE v2_turns (
    id                         UUID PRIMARY KEY,
    conversation_id            UUID NOT NULL REFERENCES v2_conversations(id) ON DELETE CASCADE,
    ordinal                    BIGINT NOT NULL,
    history_cutoff             BIGINT,
    response_ordinal           BIGINT,
    status                     TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'completed', 'failed', 'cancelled')),
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_message_created_at    TIMESTAMPTZ NOT NULL,
    completed_at               TIMESTAMPTZ,
    user_message_provider      TEXT NOT NULL,
    user_message_channel       TEXT NOT NULL,
    user_message               TEXT NOT NULL,
    user_key                   TEXT NOT NULL,
    user_display_name          TEXT NOT NULL,
    user_content               TEXT NOT NULL,
    assistant_message_provider TEXT,
    assistant_message_channel  TEXT,
    assistant_message          TEXT,
    assistant_content          TEXT,
    error                      TEXT,
    continuation               JSONB,
    app_version_id             INTEGER REFERENCES v2_app_versions(id),
    metadata                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (conversation_id, ordinal),
    UNIQUE (conversation_id, response_ordinal)
);

CREATE TABLE v2_turn_attempts (
    id                         UUID PRIMARY KEY,
    turn_id                    UUID NOT NULL REFERENCES v2_turns(id) ON DELETE CASCADE,
    attempt_ordinal            INTEGER NOT NULL,
    status                     TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'completed', 'failed', 'cancelled')),
    started_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at               TIMESTAMPTZ,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    agent_name                 TEXT NOT NULL,
    llm_provider               TEXT NOT NULL,
    llm_model                  TEXT NOT NULL,
    system_instructions        TEXT NOT NULL,
    assistant_message_provider TEXT,
    assistant_message_channel  TEXT,
    assistant_message          TEXT,
    assistant_content          TEXT,
    error                      TEXT,
    continuation               JSONB,
    app_version_id             INTEGER REFERENCES v2_app_versions(id),
    metadata                   JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (turn_id, attempt_ordinal)
);

CREATE TABLE v2_turn_attempt_context_items (
    id               BIGSERIAL PRIMARY KEY,
    attempt_id       UUID NOT NULL REFERENCES v2_turn_attempts(id) ON DELETE CASCADE,
    ordinal          INTEGER NOT NULL,
    source           TEXT NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    message_provider TEXT,
    channel          TEXT,
    message          TEXT,
    media_uri        TEXT REFERENCES v2_media_assets(uri),
    raw              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (attempt_id, ordinal)
);

CREATE TABLE v2_turn_attempt_input_messages (
    id          BIGSERIAL PRIMARY KEY,
    attempt_id  UUID NOT NULL REFERENCES v2_turn_attempts(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (attempt_id, ordinal)
);

CREATE TABLE v2_turn_attempt_input_blocks (
    id               BIGSERIAL PRIMARY KEY,
    input_message_id BIGINT NOT NULL REFERENCES v2_turn_attempt_input_messages(id) ON DELETE CASCADE,
    ordinal          INTEGER NOT NULL,
    block_kind       TEXT NOT NULL,
    text_content     TEXT,
    media_uri        TEXT REFERENCES v2_media_assets(uri),
    payload          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (input_message_id, ordinal)
);

CREATE TABLE v2_turn_attempt_tool_traces (
    id             BIGSERIAL PRIMARY KEY,
    attempt_id     UUID NOT NULL REFERENCES v2_turn_attempts(id) ON DELETE CASCADE,
    ordinal        INTEGER NOT NULL,
    trace_kind     TEXT NOT NULL CHECK (trace_kind IN ('client', 'server', 'grounding')),
    tool_name      TEXT,
    provider       TEXT,
    tool_use_id    TEXT,
    is_error       BOOLEAN,
    request        JSONB,
    response       JSONB,
    trace          JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (attempt_id, ordinal)
);

CREATE TABLE v2_turn_assets (
    id              BIGSERIAL PRIMARY KEY,
    turn_id          UUID NOT NULL REFERENCES v2_turns(id) ON DELETE CASCADE,
    attempt_id       UUID REFERENCES v2_turn_attempts(id) ON DELETE SET NULL,
    media_uri        TEXT NOT NULL REFERENCES v2_media_assets(uri),
    source           TEXT NOT NULL,
    replayable       BOOLEAN NOT NULL DEFAULT true,
    context_item_id  BIGINT REFERENCES v2_turn_attempt_context_items(id) ON DELETE SET NULL,
    tool_trace_id    BIGINT REFERENCES v2_turn_attempt_tool_traces(id) ON DELETE SET NULL,
    ordinal          INTEGER NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (turn_id, media_uri, source)
);

CREATE TABLE v2_usage_records (
    id               BIGSERIAL PRIMARY KEY,
    conversation_id  UUID NOT NULL REFERENCES v2_conversations(id) ON DELETE CASCADE,
    turn_id          UUID REFERENCES v2_turns(id) ON DELETE CASCADE,
    attempt_id       UUID REFERENCES v2_turn_attempts(id) ON DELETE CASCADE,
    tool_trace_id    BIGINT REFERENCES v2_turn_attempt_tool_traces(id) ON DELETE SET NULL,
    media_uri        TEXT REFERENCES v2_media_assets(uri),
    provider         TEXT NOT NULL,
    model            TEXT,
    subject_kind     TEXT NOT NULL,
    subject_name     TEXT,
    input_tokens     BIGINT,
    cached_tokens    BIGINT,
    output_tokens    BIGINT,
    reasoning_tokens BIGINT,
    total_tokens     BIGINT,
    cost_amount      TEXT,
    cost_unit        TEXT,
    cost_estimated   BOOLEAN,
    raw              JSONB,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE v2_message_links (
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    message          TEXT NOT NULL,
    conversation_id  UUID NOT NULL REFERENCES v2_conversations(id) ON DELETE CASCADE,
    turn_id          UUID NOT NULL REFERENCES v2_turns(id) ON DELETE CASCADE,
    attempt_id       UUID REFERENCES v2_turn_attempts(id) ON DELETE SET NULL,
    role             TEXT NOT NULL,
    linked_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, channel, message)
);

CREATE TABLE v2_channel_links (
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    conversation_id  UUID NOT NULL REFERENCES v2_conversations(id) ON DELETE CASCADE,
    turn_id          UUID NOT NULL REFERENCES v2_turns(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    linked_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, channel, role)
);

CREATE TABLE v2_video_jobs (
    id              UUID PRIMARY KEY,
    turn_id          UUID NOT NULL REFERENCES v2_turns(id) ON DELETE CASCADE,
    attempt_id       UUID REFERENCES v2_turn_attempts(id) ON DELETE SET NULL,
    tool_trace_id    BIGINT REFERENCES v2_turn_attempt_tool_traces(id) ON DELETE SET NULL,
    video_provider   TEXT NOT NULL,
    provider_job_id  TEXT NOT NULL,
    model            TEXT,
    prompt           TEXT NOT NULL,
    request          JSONB NOT NULL DEFAULT '{}'::jsonb,
    status           TEXT NOT NULL DEFAULT 'pending',
    output_uri       TEXT REFERENCES v2_media_assets(uri),
    submitted_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at     TIMESTAMPTZ,
    error            TEXT,
    raw              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_provider, provider_job_id)
);

CREATE TABLE v2_privacy_settings (
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    privacy_mode     JSONB NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, channel)
);

CREATE TABLE v2_user_privacy (
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    user_key         TEXT NOT NULL,
    opted_in         BOOLEAN NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, channel, user_key)
);

CREATE TABLE v2_conversation_agent_selections (
    conversation_id UUID PRIMARY KEY REFERENCES v2_conversations(id) ON DELETE CASCADE,
    agent_name      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE v2_channel_agent_selections (
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    agent_name       TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, channel)
);

CREATE TABLE v2_user_agent_selections (
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    user_key         TEXT NOT NULL,
    agent_name       TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, channel, user_key)
);

CREATE TABLE v2_provider_agent_selections (
    message_provider TEXT PRIMARY KEY,
    agent_name       TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE v2_discord_channels (
    message_provider TEXT NOT NULL,
    channel          TEXT NOT NULL,
    guild_id         TEXT,
    channel_id       TEXT,
    thread_id        TEXT,
    name             TEXT,
    kind             TEXT,
    raw              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (message_provider, channel),
    FOREIGN KEY (message_provider, channel)
        REFERENCES v2_platform_channels (message_provider, channel)
        ON DELETE CASCADE
);

-- Legacy data transfer.

INSERT INTO v2_app_versions (id, git_version, first_seen_at, created_at, updated_at)
SELECT id, git_version, first_seen, first_seen, first_seen
FROM app_versions;

SELECT setval(
    pg_get_serial_sequence('v2_app_versions', 'id'),
    GREATEST((SELECT COALESCE(MAX(id), 0) FROM v2_app_versions), 1),
    (SELECT EXISTS (SELECT 1 FROM v2_app_versions))
);

INSERT INTO v2_platform_channels (
    message_provider, channel, channel_kind, created_at, updated_at,
    first_seen_at, last_seen_at, raw
)
SELECT DISTINCT
    'discord',
    'guild:' || c.discord_guild_id::text || ':channel:' || c.discord_channel_id::text,
    'channel',
    MIN(c.created_at),
    now(),
    MIN(c.created_at),
    now(),
    jsonb_build_object(
        'legacy_discord_guild_id', c.discord_guild_id::text,
        'legacy_discord_channel_id', c.discord_channel_id::text
    )
FROM conversations c
GROUP BY c.discord_guild_id, c.discord_channel_id
ON CONFLICT DO NOTHING;

INSERT INTO v2_platform_channels (message_provider, channel, channel_kind, raw)
SELECT DISTINCT 'discord', 'guild:' || discord_guild_id::text, 'workspace',
       jsonb_build_object('legacy_discord_guild_id', discord_guild_id::text)
FROM conversations
ON CONFLICT DO NOTHING;

INSERT INTO v2_platform_channels (message_provider, channel, channel_kind, raw)
SELECT DISTINCT 'discord', 'guild:' || discord_guild_id::text, 'workspace',
       jsonb_build_object('legacy_discord_guild_id', discord_guild_id::text)
FROM guild_settings
ON CONFLICT DO NOTHING;

INSERT INTO v2_platform_channels (message_provider, channel, channel_kind, raw)
SELECT DISTINCT 'discord', 'guild:' || discord_guild_id::text, 'workspace',
       jsonb_build_object('legacy_discord_guild_id', discord_guild_id::text)
FROM user_privacy
ON CONFLICT DO NOTHING;

INSERT INTO v2_platform_channels (message_provider, channel, channel_kind, raw)
SELECT DISTINCT 'discord', 'channel:' || key, 'channel',
       jsonb_build_object('legacy_persona_scope', scope, 'legacy_key', key)
FROM persona_selections
WHERE scope = 'channel'
ON CONFLICT DO NOTHING;

INSERT INTO v2_platform_users (
    message_provider, user_key, username, display_name, avatar_media_uri,
    created_at, updated_at, first_seen_at, last_seen_at, raw
)
SELECT
    'discord',
    u.id::text,
    u.username,
    u.display_name,
    NULL,
    u.last_seen_at,
    now(),
    u.last_seen_at,
    u.last_seen_at,
    jsonb_build_object(
        'legacy_discord_user_id', u.id::text,
        'avatar_hash', u.avatar_hash,
        'avatar_local_path', u.avatar_local_path,
        'last_avatar_fetched_at', u.last_avatar_fetched_at
    )
FROM discord_users u
ON CONFLICT DO NOTHING;

INSERT INTO v2_platform_users (message_provider, user_key, username, display_name, raw)
SELECT DISTINCT
    'discord',
    t.discord_user_id::text,
    COALESCE(t.discord_user_name, 'user'),
    t.discord_user_name,
    jsonb_build_object('legacy_source', 'turns.discord_user_id')
FROM turns t
WHERE t.discord_user_id IS NOT NULL
ON CONFLICT DO NOTHING;

INSERT INTO v2_platform_users (message_provider, user_key, username, raw)
SELECT DISTINCT
    'discord',
    c.created_by_user_id::text,
    c.created_by_user_id::text,
    jsonb_build_object('legacy_source', 'conversations.created_by_user_id')
FROM conversations c
ON CONFLICT DO NOTHING;

INSERT INTO v2_platform_users (message_provider, user_key, username, raw)
SELECT DISTINCT
    'discord',
    c.stopped_by_user_id::text,
    c.stopped_by_user_id::text,
    jsonb_build_object('legacy_source', 'conversations.stopped_by_user_id')
FROM conversations c
WHERE c.stopped_by_user_id IS NOT NULL
ON CONFLICT DO NOTHING;

INSERT INTO v2_media_assets (uri, category, name, mime_type, size_bytes, metadata)
SELECT DISTINCT
    uri,
    CASE
        WHEN uri LIKE 'file://images/%' THEN 'image'
        WHEN uri LIKE 'file://videos/%' THEN 'video'
        WHEN uri LIKE 'file://avatars/%' THEN 'avatar'
        ELSE 'other'
    END,
    regexp_replace(uri, '^file://[^/]+/', ''),
    CASE
        WHEN uri LIKE 'file://images/%' THEN 'image/png'
        WHEN uri LIKE 'file://videos/%' THEN 'video/mp4'
        WHEN uri LIKE 'file://avatars/%' THEN 'image/png'
        ELSE 'application/octet-stream'
    END,
    0,
    jsonb_build_object('legacy_placeholder', true)
FROM (
    SELECT content AS uri FROM context_items WHERE content LIKE 'file://%'
    UNION
    SELECT media.media_uri AS uri
    FROM tool_calls tc
    CROSS JOIN LATERAL (
        SELECT COALESCE(
            tc.response->>'uri',
            tc.response->>'image_uri',
            tc.response->>'video_uri',
            tc.response #>> '{content,value,uri}',
            tc.response #>> '{content,value,image_uri}',
            tc.response #>> '{content,value,video_uri}',
            tc.response #>> '{result,content,value,uri}',
            tc.response #>> '{result,content,value,image_uri}',
            tc.response #>> '{result,content,value,video_uri}',
            tc.response #>> '{trace_response,uri}',
            tc.response #>> '{trace_response,image_uri}',
            tc.response #>> '{trace_response,video_uri}'
        ) AS media_uri
    ) media
    WHERE media.media_uri LIKE 'file://%'
    UNION
    SELECT video_uri AS uri
    FROM video_jobs
    WHERE video_uri IS NOT NULL
    UNION
    SELECT 'file://avatars/' || avatar_local_path AS uri
    FROM discord_users
    WHERE avatar_local_path IS NOT NULL
) media
WHERE uri IS NOT NULL
ON CONFLICT DO NOTHING;

UPDATE v2_platform_users
SET avatar_media_uri = 'file://avatars/' || (raw->>'avatar_local_path')
WHERE message_provider = 'discord'
  AND raw ? 'avatar_local_path'
  AND raw->>'avatar_local_path' IS NOT NULL;

INSERT INTO v2_platform_messages (
    message_provider, channel, message, author_user_key, content,
    platform_created_at, created_at, updated_at, first_seen_at, raw
)
SELECT
    'discord',
    'guild:' || c.discord_guild_id::text || ':channel:' || c.discord_channel_id::text,
    t.user_discord_message_id::text,
    COALESCE(t.discord_user_id, c.created_by_user_id)::text,
    t.user_content,
    t.created_at,
    t.created_at,
    t.created_at,
    t.created_at,
    jsonb_build_object('legacy_source', 'turns.user_discord_message_id')
FROM turns t
JOIN conversations c ON c.id = t.conversation_id
ON CONFLICT DO NOTHING;

INSERT INTO v2_platform_messages (
    message_provider, channel, message, author_user_key, content,
    platform_created_at, created_at, updated_at, first_seen_at, raw
)
SELECT
    'discord',
    'guild:' || c.discord_guild_id::text || ':channel:' || c.discord_channel_id::text,
    t.assistant_discord_message_id::text,
    NULL,
    t.assistant_content,
    COALESCE(t.completed_at, t.created_at),
    COALESCE(t.completed_at, t.created_at),
    COALESCE(t.completed_at, t.created_at),
    COALESCE(t.completed_at, t.created_at),
    jsonb_build_object('legacy_source', 'turns.assistant_discord_message_id')
FROM turns t
JOIN conversations c ON c.id = t.conversation_id
WHERE t.assistant_discord_message_id IS NOT NULL
ON CONFLICT DO NOTHING;

INSERT INTO v2_conversations (
    id, created_at, updated_at, message_provider, channel, created_by_user_key,
    root_message_provider, root_message_channel, root_message, agent_name,
    llm_provider, llm_model, system_instructions, title, title_generated_at,
    created_app_version_id, stopped_at, stopped_by_provider, stopped_by_user_key,
    next_turn_ordinal, next_response_ordinal, metadata
)
SELECT
    c.id,
    c.created_at,
    now(),
    'discord',
    'guild:' || c.discord_guild_id::text || ':channel:' || c.discord_channel_id::text,
    c.created_by_user_id::text,
    'discord',
    'guild:' || c.discord_guild_id::text || ':channel:' || c.discord_channel_id::text,
    c.root_discord_message_id::text,
    COALESCE((
        SELECT t.persona_name
        FROM turns t
        WHERE t.conversation_id = c.id AND t.persona_name IS NOT NULL
        ORDER BY t.turn_index
        LIMIT 1
    ), 'legacy'),
    'legacy',
    c.model,
    COALESCE((
        SELECT tsp.content
        FROM turn_system_prompts tsp
        JOIN turns t ON t.id = tsp.turn_id
        WHERE t.conversation_id = c.id
        ORDER BY t.turn_index
        LIMIT 1
    ), ''),
    c.title,
    c.title_generated_at,
    NULL,
    c.stopped_at,
    CASE WHEN c.stopped_by_user_id IS NULL THEN NULL ELSE 'discord' END,
    c.stopped_by_user_id::text,
    COALESCE((SELECT MAX(t.turn_index) + 1 FROM turns t WHERE t.conversation_id = c.id), 0),
    COALESCE((
        SELECT COUNT(*)
        FROM turns t
        WHERE t.conversation_id = c.id AND t.status = 'completed'
    ), 0),
    jsonb_build_object(
        'legacy_discord_guild_id', c.discord_guild_id::text,
        'legacy_discord_channel_id', c.discord_channel_id::text,
        'legacy_model', c.model
    )
FROM conversations c;

WITH completed AS (
    SELECT
        t.id,
        t.conversation_id,
        t.turn_index,
        row_number() OVER (PARTITION BY t.conversation_id ORDER BY t.turn_index) - 1 AS response_ordinal
    FROM turns t
    WHERE t.status = 'completed'
)
INSERT INTO v2_turns (
    id, conversation_id, ordinal, history_cutoff, response_ordinal, status,
    created_at, updated_at, user_message_created_at, completed_at,
    user_message_provider, user_message_channel, user_message, user_key,
    user_display_name, user_content, assistant_message_provider,
    assistant_message_channel, assistant_message, assistant_content, error,
    continuation, app_version_id, metadata
)
SELECT
    t.id,
    t.conversation_id,
    t.turn_index,
    (
        SELECT MAX(c2.response_ordinal)
        FROM completed c2
        WHERE c2.conversation_id = t.conversation_id
          AND c2.turn_index < t.turn_index
    ),
    c.response_ordinal,
    CASE WHEN t.status IN ('pending', 'completed', 'failed', 'cancelled') THEN t.status ELSE 'failed' END,
    t.created_at,
    now(),
    t.created_at,
    t.completed_at,
    'discord',
    'guild:' || conv.discord_guild_id::text || ':channel:' || conv.discord_channel_id::text,
    t.user_discord_message_id::text,
    COALESCE(t.discord_user_id, conv.created_by_user_id)::text,
    COALESCE(t.discord_user_name, COALESCE(t.discord_user_id, conv.created_by_user_id)::text),
    t.user_content,
    CASE WHEN t.assistant_discord_message_id IS NULL THEN NULL ELSE 'discord' END,
    CASE
        WHEN t.assistant_discord_message_id IS NULL THEN NULL
        ELSE 'guild:' || conv.discord_guild_id::text || ':channel:' || conv.discord_channel_id::text
    END,
    t.assistant_discord_message_id::text,
    t.assistant_content,
    t.error,
    CASE
        WHEN t.provider_state IS NULL THEN NULL
        ELSE jsonb_build_object('provider', COALESCE(t.provider_state->>'provider', 'legacy'), 'data', t.provider_state->'data')
    END,
    t.version_id,
    jsonb_build_object('legacy_turn_index', t.turn_index, 'legacy_persona_name', t.persona_name)
FROM turns t
JOIN conversations conv ON conv.id = t.conversation_id
LEFT JOIN completed c ON c.id = t.id;

INSERT INTO v2_turn_attempts (
    id, turn_id, attempt_ordinal, status, started_at, completed_at, created_at,
    updated_at, agent_name, llm_provider, llm_model, system_instructions,
    assistant_message_provider, assistant_message_channel, assistant_message,
    assistant_content, error, continuation, app_version_id, metadata
)
SELECT
    vt.id,
    vt.id,
    0,
    vt.status,
    vt.created_at,
    vt.completed_at,
    vt.created_at,
    now(),
    COALESCE((vt.metadata->>'legacy_persona_name'), vc.agent_name),
    vc.llm_provider,
    vc.llm_model,
    COALESCE(tsp.content, vc.system_instructions),
    vt.assistant_message_provider,
    vt.assistant_message_channel,
    vt.assistant_message,
    vt.assistant_content,
    vt.error,
    vt.continuation,
    vt.app_version_id,
    jsonb_build_object('legacy_attempt', true)
FROM v2_turns vt
JOIN v2_conversations vc ON vc.id = vt.conversation_id
LEFT JOIN turn_system_prompts tsp ON tsp.turn_id = vt.id;

INSERT INTO v2_turn_attempt_context_items (
    attempt_id, ordinal, source, role, content, message_provider,
    channel, message, media_uri, raw, created_at, updated_at
)
SELECT
    ci.turn_id,
    ci.position,
    ci.source,
    ci.role,
    ci.content,
    CASE WHEN ci.discord_message_id IS NULL THEN NULL ELSE 'discord' END,
    CASE
        WHEN ci.discord_message_id IS NULL THEN NULL
        ELSE vt.user_message_channel
    END,
    ci.discord_message_id::text,
    CASE WHEN ci.content LIKE 'file://%' THEN ci.content ELSE NULL END,
    jsonb_build_object('legacy_source', 'context_items'),
    vt.created_at,
    now()
FROM context_items ci
JOIN v2_turns vt ON vt.id = ci.turn_id;

INSERT INTO v2_turn_attempt_input_messages (attempt_id, ordinal, role, metadata, created_at, updated_at)
SELECT
    ci.turn_id,
    ci.position,
    CASE WHEN ci.role = 'assistant' THEN 'assistant' ELSE 'user' END,
    jsonb_build_object('legacy_source', ci.source),
    vt.created_at,
    now()
FROM context_items ci
JOIN v2_turns vt ON vt.id = ci.turn_id;

INSERT INTO v2_turn_attempt_input_blocks (
    input_message_id, ordinal, block_kind, text_content, media_uri, payload,
    created_at, updated_at
)
SELECT
    im.id,
    0,
    CASE WHEN ci.content LIKE 'file://%' THEN 'media' ELSE 'text' END,
    CASE WHEN ci.content LIKE 'file://%' THEN NULL ELSE ci.content END,
    CASE WHEN ci.content LIKE 'file://%' THEN ci.content ELSE NULL END,
    '{}'::jsonb,
    im.created_at,
    im.updated_at
FROM v2_turn_attempt_input_messages im
JOIN context_items ci
  ON ci.turn_id = im.attempt_id
 AND ci.position = im.ordinal;

INSERT INTO v2_turn_attempt_tool_traces (
    attempt_id, ordinal, trace_kind, tool_name, provider, tool_use_id,
    is_error, request, response, trace, created_at, updated_at
)
SELECT
    tc.turn_id,
    tc.ordinal,
    'client',
    tc.tool_name,
    NULL,
    tc.request->>'id',
    (tc.response ? 'error'),
    tc.request,
    tc.response,
    jsonb_build_object(
        'kind', 'client',
        'trace', jsonb_build_object(
            'call', jsonb_build_object(
                'id', COALESCE(tc.request->>'id', tc.ordinal::text),
                'name', tc.tool_name,
                'input', tc.request
            ),
            'result', jsonb_build_object(
                'tool_use_id', COALESCE(tc.request->>'id', tc.ordinal::text),
                'content', jsonb_build_object('kind', 'json', 'value', tc.response),
                'is_error', (tc.response ? 'error')
            ),
            'trace_response', tc.response,
            'usage', '[]'::jsonb
        )
    ),
    vt.created_at,
    now()
FROM tool_calls tc
JOIN v2_turns vt ON vt.id = tc.turn_id;

INSERT INTO v2_turn_assets (
    turn_id, attempt_id, media_uri, source, replayable, context_item_id,
    tool_trace_id, ordinal, created_at, updated_at
)
SELECT
    vt.id,
    vt.id,
    ci.media_uri,
    ci.source,
    true,
    ci.id,
    NULL,
    ci.ordinal,
    ci.created_at,
    ci.updated_at
FROM v2_turn_attempt_context_items ci
JOIN v2_turn_attempts ta ON ta.id = ci.attempt_id
JOIN v2_turns vt ON vt.id = ta.turn_id
WHERE ci.media_uri IS NOT NULL
ON CONFLICT DO NOTHING;

INSERT INTO v2_turn_assets (
    turn_id, attempt_id, media_uri, source, replayable, context_item_id,
    tool_trace_id, ordinal, created_at, updated_at
)
SELECT
    vt.id,
    vt.id,
    media.media_uri,
    tt.tool_name,
    true,
    NULL,
    tt.id,
    tt.ordinal,
    tt.created_at,
    tt.updated_at
FROM v2_turn_attempt_tool_traces tt
JOIN v2_turn_attempts ta ON ta.id = tt.attempt_id
JOIN v2_turns vt ON vt.id = ta.turn_id
CROSS JOIN LATERAL (
    SELECT COALESCE(
        tt.response->>'uri',
        tt.response->>'image_uri',
        tt.response->>'video_uri',
        tt.response #>> '{content,value,uri}',
        tt.response #>> '{content,value,image_uri}',
        tt.response #>> '{content,value,video_uri}',
        tt.response #>> '{result,content,value,uri}',
        tt.response #>> '{result,content,value,image_uri}',
        tt.response #>> '{result,content,value,video_uri}',
        tt.trace #>> '{trace,trace_response,uri}',
        tt.trace #>> '{trace,trace_response,image_uri}',
        tt.trace #>> '{trace,trace_response,video_uri}',
        tt.trace #>> '{trace,result,content,value,uri}',
        tt.trace #>> '{trace,result,content,value,image_uri}',
        tt.trace #>> '{trace,result,content,value,video_uri}'
    ) AS media_uri
) media
WHERE media.media_uri LIKE 'file://%'
ON CONFLICT DO NOTHING;

INSERT INTO v2_message_links (
    message_provider, channel, message, conversation_id, turn_id, attempt_id,
    role, linked_at, created_at, updated_at
)
SELECT
    'discord',
    'guild:' || c.discord_guild_id::text || ':channel:' || c.discord_channel_id::text,
    ml.discord_message_id::text,
    ml.conversation_id,
    ml.turn_id,
    ml.turn_id,
    ml.role,
    COALESCE(t.created_at, c.created_at),
    COALESCE(t.created_at, c.created_at),
    now()
FROM message_links ml
JOIN conversations c ON c.id = ml.conversation_id
LEFT JOIN turns t ON t.id = ml.turn_id
ON CONFLICT DO NOTHING;

INSERT INTO v2_video_jobs (
    id, turn_id, attempt_id, tool_trace_id, video_provider, provider_job_id,
    model, prompt, request, status, output_uri, submitted_at, completed_at,
    error, raw, created_at, updated_at
)
SELECT
    vj.id,
    vj.turn_id,
    vj.turn_id,
    NULL,
    'xai',
    vj.request_id,
    NULL,
    vj.prompt,
    '{}'::jsonb,
    vj.status,
    vj.video_uri,
    vj.submitted_at,
    vj.completed_at,
    vj.error,
    jsonb_build_object('legacy_request_id', vj.request_id),
    vj.submitted_at,
    now()
FROM video_jobs vj;

INSERT INTO v2_privacy_settings (
    message_provider, channel, privacy_mode, created_at, updated_at
)
SELECT
    'discord',
    'guild:' || discord_guild_id::text,
    CASE
        WHEN privacy_mode->>'mode' = 'channel_only' AND privacy_mode ? 'channel_id' THEN
            jsonb_build_object(
                'mode', 'channel_only',
                'channel', jsonb_build_object(
                    'platform', 'discord',
                    'guild_id', discord_guild_id::text,
                    'channel_id', privacy_mode->>'channel_id'
                ),
                'history_size', COALESCE(NULLIF(privacy_mode->>'history_size', '')::int, 20)
            )
        ELSE privacy_mode
    END,
    updated_at,
    updated_at
FROM guild_settings;

INSERT INTO v2_user_privacy (
    message_provider, channel, user_key, opted_in, created_at, updated_at
)
SELECT
    'discord',
    'guild:' || discord_guild_id::text,
    discord_user_id::text,
    opted_in,
    updated_at,
    updated_at
FROM user_privacy;

INSERT INTO v2_conversation_agent_selections (conversation_id, agent_name, created_at, updated_at)
SELECT key::uuid, persona_name, updated_at, updated_at
FROM persona_selections
WHERE scope = 'conversation';

INSERT INTO v2_channel_agent_selections (message_provider, channel, agent_name, created_at, updated_at)
SELECT
    'discord',
    CASE
        WHEN scope = 'guild' THEN 'guild:' || key
        ELSE 'channel:' || key
    END,
    persona_name,
    updated_at,
    updated_at
FROM persona_selections
WHERE scope IN ('guild', 'channel');

INSERT INTO v2_user_agent_selections (message_provider, channel, user_key, agent_name, created_at, updated_at)
SELECT
    'discord',
    'guild:' || split_part(key, ':', 1),
    split_part(key, ':', 2),
    persona_name,
    updated_at,
    updated_at
FROM persona_selections
WHERE scope = 'user';

INSERT INTO v2_discord_channels (
    message_provider, channel, guild_id, channel_id, kind, raw, created_at, updated_at
)
SELECT
    'discord',
    'guild:' || c.discord_guild_id::text || ':channel:' || c.discord_channel_id::text,
    c.discord_guild_id::text,
    c.discord_channel_id::text,
    'channel',
    jsonb_build_object('legacy_source', 'conversations'),
    MIN(c.created_at),
    now()
FROM conversations c
GROUP BY c.discord_guild_id, c.discord_channel_id;

-- Replace old canonical tables with v2 canonical tables.

DROP TABLE IF EXISTS
    tool_calls,
    context_items,
    turn_system_prompts,
    video_jobs,
    message_links,
    turns,
    conversations,
    user_privacy,
    guild_settings,
    persona_selections,
    discord_users,
    app_versions
CASCADE;

ALTER TABLE v2_app_versions RENAME TO app_versions;
ALTER TABLE v2_media_assets RENAME TO media_assets;
ALTER TABLE v2_platform_channels RENAME TO platform_channels;
ALTER TABLE v2_platform_users RENAME TO platform_users;
ALTER TABLE v2_platform_messages RENAME TO platform_messages;
ALTER TABLE v2_platform_message_attachments RENAME TO platform_message_attachments;
ALTER TABLE v2_conversations RENAME TO conversations;
ALTER TABLE v2_turns RENAME TO turns;
ALTER TABLE v2_turn_attempts RENAME TO turn_attempts;
ALTER TABLE v2_turn_attempt_context_items RENAME TO turn_attempt_context_items;
ALTER TABLE v2_turn_attempt_input_messages RENAME TO turn_attempt_input_messages;
ALTER TABLE v2_turn_attempt_input_blocks RENAME TO turn_attempt_input_blocks;
ALTER TABLE v2_turn_attempt_tool_traces RENAME TO turn_attempt_tool_traces;
ALTER TABLE v2_turn_assets RENAME TO turn_assets;
ALTER TABLE v2_usage_records RENAME TO usage_records;
ALTER TABLE v2_message_links RENAME TO message_links;
ALTER TABLE v2_channel_links RENAME TO channel_links;
ALTER TABLE v2_video_jobs RENAME TO video_jobs;
ALTER TABLE v2_privacy_settings RENAME TO privacy_settings;
ALTER TABLE v2_user_privacy RENAME TO user_privacy;
ALTER TABLE v2_conversation_agent_selections RENAME TO conversation_agent_selections;
ALTER TABLE v2_channel_agent_selections RENAME TO channel_agent_selections;
ALTER TABLE v2_user_agent_selections RENAME TO user_agent_selections;
ALTER TABLE v2_provider_agent_selections RENAME TO provider_agent_selections;
ALTER TABLE v2_discord_channels RENAME TO discord_channels;

CREATE INDEX media_assets_category_name_idx
    ON media_assets (category, name);
CREATE INDEX platform_channels_parent_idx
    ON platform_channels (message_provider, parent_channel);
CREATE INDEX platform_messages_author_idx
    ON platform_messages (message_provider, author_user_key);
CREATE INDEX conversations_created_at_idx
    ON conversations (created_at DESC);
CREATE INDEX conversations_channel_idx
    ON conversations (message_provider, channel, created_at DESC);
CREATE UNIQUE INDEX conversations_root_message_idx
    ON conversations (root_message_provider, root_message_channel, root_message);
CREATE INDEX turns_conversation_ordinal_idx
    ON turns (conversation_id, ordinal);
CREATE UNIQUE INDEX turns_user_message_idx
    ON turns (user_message_provider, user_message_channel, user_message);
CREATE INDEX turns_replay_idx
    ON turns (conversation_id, response_ordinal)
    WHERE response_ordinal IS NOT NULL;
CREATE INDEX turns_status_idx
    ON turns (status);
CREATE INDEX turn_attempts_turn_idx
    ON turn_attempts (turn_id, attempt_ordinal);
CREATE INDEX turn_attempt_context_items_attempt_idx
    ON turn_attempt_context_items (attempt_id, ordinal);
CREATE INDEX turn_attempt_input_messages_attempt_idx
    ON turn_attempt_input_messages (attempt_id, ordinal);
CREATE INDEX turn_attempt_tool_traces_attempt_idx
    ON turn_attempt_tool_traces (attempt_id, ordinal);
CREATE INDEX turn_assets_turn_idx
    ON turn_assets (turn_id, ordinal);
CREATE INDEX usage_records_conversation_idx
    ON usage_records (conversation_id, created_at);
CREATE INDEX usage_records_attempt_idx
    ON usage_records (attempt_id);
CREATE INDEX message_links_conversation_idx
    ON message_links (conversation_id);
CREATE INDEX message_links_turn_idx
    ON message_links (turn_id);
CREATE INDEX channel_links_conversation_idx
    ON channel_links (conversation_id);
CREATE INDEX video_jobs_turn_idx
    ON video_jobs (turn_id);
CREATE INDEX video_jobs_status_idx
    ON video_jobs (status);

CREATE TRIGGER app_versions_touch_updated_at
    BEFORE UPDATE ON app_versions
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER media_assets_touch_updated_at
    BEFORE UPDATE ON media_assets
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER platform_channels_touch_updated_at
    BEFORE UPDATE ON platform_channels
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER platform_users_touch_updated_at
    BEFORE UPDATE ON platform_users
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER platform_messages_touch_updated_at
    BEFORE UPDATE ON platform_messages
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER platform_message_attachments_touch_updated_at
    BEFORE UPDATE ON platform_message_attachments
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER conversations_touch_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER turns_touch_updated_at
    BEFORE UPDATE ON turns
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER turn_attempts_touch_updated_at
    BEFORE UPDATE ON turn_attempts
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER turn_attempt_context_items_touch_updated_at
    BEFORE UPDATE ON turn_attempt_context_items
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER turn_attempt_input_messages_touch_updated_at
    BEFORE UPDATE ON turn_attempt_input_messages
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER turn_attempt_input_blocks_touch_updated_at
    BEFORE UPDATE ON turn_attempt_input_blocks
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER turn_attempt_tool_traces_touch_updated_at
    BEFORE UPDATE ON turn_attempt_tool_traces
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER turn_assets_touch_updated_at
    BEFORE UPDATE ON turn_assets
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER usage_records_touch_updated_at
    BEFORE UPDATE ON usage_records
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER message_links_touch_updated_at
    BEFORE UPDATE ON message_links
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER channel_links_touch_updated_at
    BEFORE UPDATE ON channel_links
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER video_jobs_touch_updated_at
    BEFORE UPDATE ON video_jobs
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER privacy_settings_touch_updated_at
    BEFORE UPDATE ON privacy_settings
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER user_privacy_touch_updated_at
    BEFORE UPDATE ON user_privacy
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER conversation_agent_selections_touch_updated_at
    BEFORE UPDATE ON conversation_agent_selections
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER channel_agent_selections_touch_updated_at
    BEFORE UPDATE ON channel_agent_selections
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER user_agent_selections_touch_updated_at
    BEFORE UPDATE ON user_agent_selections
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER provider_agent_selections_touch_updated_at
    BEFORE UPDATE ON provider_agent_selections
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
CREATE TRIGGER discord_channels_touch_updated_at
    BEFORE UPDATE ON discord_channels
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
