CREATE TABLE vibe_login_links (
    token_hash BYTEA PRIMARY KEY,
    platform TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    requested_by_user_id TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ
);

CREATE INDEX vibe_login_links_expiry_idx ON vibe_login_links(expires_at);
