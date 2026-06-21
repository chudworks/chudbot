-- Cache current platform guild/workspace icons for management views.

ALTER TABLE platform_channels
    ADD COLUMN icon_hash TEXT,
    ADD COLUMN icon_url TEXT,
    ADD COLUMN icon_media_uri TEXT REFERENCES media_assets(uri);

CREATE INDEX platform_channels_icon_media_idx
    ON platform_channels (icon_media_uri)
    WHERE icon_media_uri IS NOT NULL;
