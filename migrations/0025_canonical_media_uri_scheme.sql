-- Canonicalize stored media identities from legacy file://... to media://...
-- while keeping the same served paths and storage category/name metadata.
--
-- The FK constraints are replaced before updating media_assets.uri so the
-- primary-key rewrite cascades to all relational references. Updating the
-- parent URI before installing ON UPDATE CASCADE would leave child rows
-- pointing at file://... keys and make the replacement constraints fail.

ALTER TABLE platform_users
    DROP CONSTRAINT IF EXISTS v2_platform_users_avatar_media_uri_fkey,
    DROP CONSTRAINT IF EXISTS platform_users_avatar_media_uri_fkey,
    ADD CONSTRAINT platform_users_avatar_media_uri_fkey
        FOREIGN KEY (avatar_media_uri) REFERENCES media_assets(uri) ON UPDATE CASCADE;

ALTER TABLE platform_message_attachments
    DROP CONSTRAINT IF EXISTS v2_platform_message_attachments_media_uri_fkey,
    DROP CONSTRAINT IF EXISTS platform_message_attachments_media_uri_fkey,
    ADD CONSTRAINT platform_message_attachments_media_uri_fkey
        FOREIGN KEY (media_uri) REFERENCES media_assets(uri) ON UPDATE CASCADE;

ALTER TABLE turn_attempt_context_items
    DROP CONSTRAINT IF EXISTS v2_turn_attempt_context_items_media_uri_fkey,
    DROP CONSTRAINT IF EXISTS turn_attempt_context_items_media_uri_fkey,
    ADD CONSTRAINT turn_attempt_context_items_media_uri_fkey
        FOREIGN KEY (media_uri) REFERENCES media_assets(uri) ON UPDATE CASCADE;

ALTER TABLE turn_attempt_input_blocks
    DROP CONSTRAINT IF EXISTS v2_turn_attempt_input_blocks_media_uri_fkey,
    DROP CONSTRAINT IF EXISTS turn_attempt_input_blocks_media_uri_fkey,
    ADD CONSTRAINT turn_attempt_input_blocks_media_uri_fkey
        FOREIGN KEY (media_uri) REFERENCES media_assets(uri) ON UPDATE CASCADE;

ALTER TABLE turn_assets
    DROP CONSTRAINT IF EXISTS v2_turn_assets_media_uri_fkey,
    DROP CONSTRAINT IF EXISTS turn_assets_media_uri_fkey,
    ADD CONSTRAINT turn_assets_media_uri_fkey
        FOREIGN KEY (media_uri) REFERENCES media_assets(uri) ON UPDATE CASCADE;

ALTER TABLE usage_records
    DROP CONSTRAINT IF EXISTS v2_usage_records_media_uri_fkey,
    DROP CONSTRAINT IF EXISTS usage_records_media_uri_fkey,
    ADD CONSTRAINT usage_records_media_uri_fkey
        FOREIGN KEY (media_uri) REFERENCES media_assets(uri) ON UPDATE CASCADE;

ALTER TABLE video_jobs
    DROP CONSTRAINT IF EXISTS v2_video_jobs_output_uri_fkey,
    DROP CONSTRAINT IF EXISTS video_jobs_output_uri_fkey,
    ADD CONSTRAINT video_jobs_output_uri_fkey
        FOREIGN KEY (output_uri) REFERENCES media_assets(uri) ON UPDATE CASCADE;

ALTER TABLE platform_channels
    DROP CONSTRAINT IF EXISTS platform_channels_icon_media_uri_fkey,
    ADD CONSTRAINT platform_channels_icon_media_uri_fkey
        FOREIGN KEY (icon_media_uri) REFERENCES media_assets(uri) ON UPDATE CASCADE;

UPDATE media_assets
   SET uri = regexp_replace(uri, '^file://', 'media://')
 WHERE uri LIKE 'file://%';

UPDATE turn_attempt_context_items
   SET content = regexp_replace(content, '^file://', 'media://')
 WHERE content LIKE 'file://%';

UPDATE turn_attempt_input_blocks
   SET payload = replace(payload::text, 'file://', 'media://')::jsonb
 WHERE payload::text LIKE '%file://%';
