CREATE TABLE vibe_site_guilds (
    site_id UUID NOT NULL REFERENCES vibe_sites(id) ON DELETE CASCADE,
    platform TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    added_by_user_id TEXT NOT NULL,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (site_id, platform, guild_id)
);

