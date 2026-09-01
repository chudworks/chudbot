CREATE INDEX vibe_site_editors_user_idx
    ON vibe_site_editors(platform, user_id, site_id);

CREATE INDEX vibe_sites_status_updated_idx
    ON vibe_sites(status, updated_at DESC);
