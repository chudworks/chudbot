CREATE TYPE vibe_site_status AS ENUM ('creating', 'active', 'archived');
CREATE TYPE vibe_action AS ENUM ('create', 'edit');
CREATE TYPE vibe_job_state AS ENUM (
    'queued', 'coding', 'building', 'repair', 'committing',
    'done', 'no_changes', 'failed', 'cancelled', 'timed_out'
);

CREATE TABLE vibe_sites (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    platform TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status vibe_site_status NOT NULL,
    active_revision_id UUID,
    running_job_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE vibe_jobs (
    id UUID PRIMARY KEY,
    site_id UUID REFERENCES vibe_sites(id) ON DELETE CASCADE,
    site_name TEXT NOT NULL,
    action vibe_action NOT NULL,
    actor_user_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_id UUID NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    tool_use_id TEXT NOT NULL UNIQUE,
    state vibe_job_state NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE vibe_sites
    ADD CONSTRAINT vibe_sites_running_job_fk
    FOREIGN KEY (running_job_id) REFERENCES vibe_jobs(id) ON DELETE SET NULL;

CREATE TABLE vibe_site_editors (
    site_id UUID NOT NULL REFERENCES vibe_sites(id) ON DELETE CASCADE,
    platform TEXT NOT NULL,
    user_id TEXT NOT NULL,
    added_by_user_id TEXT NOT NULL,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (site_id, platform, user_id)
);

CREATE TABLE vibe_revisions (
    id UUID PRIMARY KEY,
    site_id UUID NOT NULL REFERENCES vibe_sites(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    parent_revision_id UUID REFERENCES vibe_revisions(id),
    commit_oid TEXT NOT NULL,
    image_id TEXT NOT NULL,
    message TEXT NOT NULL,
    build_log TEXT NOT NULL,
    actor_user_id TEXT NOT NULL,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
    turn_id UUID NOT NULL REFERENCES turns(id) ON DELETE RESTRICT,
    job_id UUID NOT NULL UNIQUE REFERENCES vibe_jobs(id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (site_id, ordinal)
);

ALTER TABLE vibe_sites
    ADD CONSTRAINT vibe_sites_active_revision_fk
    FOREIGN KEY (active_revision_id) REFERENCES vibe_revisions(id) ON DELETE SET NULL;

CREATE TABLE vibe_sessions (
    token_hash BYTEA PRIMARY KEY,
    platform TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);

CREATE TABLE vibe_oauth_states (
    state_hash BYTEA PRIMARY KEY,
    return_url TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ
);

CREATE INDEX vibe_sites_guild_updated_idx
    ON vibe_sites(platform, guild_id, updated_at DESC);
CREATE INDEX vibe_jobs_running_guild_idx
    ON vibe_jobs(platform, guild_id)
    WHERE state IN ('queued', 'coding', 'building', 'repair', 'committing');
CREATE INDEX vibe_sessions_expiry_idx ON vibe_sessions(expires_at);
CREATE INDEX vibe_oauth_states_expiry_idx ON vibe_oauth_states(expires_at);
