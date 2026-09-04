CREATE TYPE vibe_site_access AS ENUM ('protected', 'public');

ALTER TABLE vibe_sites
    ADD COLUMN access vibe_site_access NOT NULL DEFAULT 'protected';
