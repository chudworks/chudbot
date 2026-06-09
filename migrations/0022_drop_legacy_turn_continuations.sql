-- Drop legacy final-continuation columns after 0021 has backfilled the
-- authoritative per-step replay state into turn_attempt_model_steps.

ALTER TABLE turns
    DROP COLUMN IF EXISTS continuation;

ALTER TABLE turn_attempts
    DROP COLUMN IF EXISTS continuation;
