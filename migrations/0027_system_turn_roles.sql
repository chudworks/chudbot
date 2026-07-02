-- Stored model-input transcripts now include system turns (runtime-injected
-- user-memory notes), so the role check must accept 'system' alongside the
-- original 'user'/'assistant' pair. The constraint kept its v2_-prefixed
-- creation-time name when migration 0011 renamed the table.
ALTER TABLE turn_attempt_input_messages
    DROP CONSTRAINT v2_turn_attempt_input_messages_role_check;
ALTER TABLE turn_attempt_input_messages
    ADD CONSTRAINT turn_attempt_input_messages_role_check
    CHECK (role IN ('user', 'assistant', 'system'));
