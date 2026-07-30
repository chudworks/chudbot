-- Record who explicitly requested a retry. The platform is inherited from the
-- retried turn's user message, so the platform-scoped user key is sufficient.
-- Initial attempts and provider-internal automatic retries remain NULL.

ALTER TABLE turn_attempts
    ADD COLUMN explicit_retry_user_key TEXT;
