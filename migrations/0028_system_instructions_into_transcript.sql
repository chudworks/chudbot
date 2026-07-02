-- Move persisted system/developer instructions into the per-attempt input
-- transcript. After this migration, the transcript tables are the durable
-- source of truth for the prompt an attempt ran with.

-- Existing input message ordinals start at 0 and the unique
-- (attempt_id, ordinal) constraint is not deferrable, so make room with a
-- two-step offset. Disable the touch trigger while shifting; this is a purely
-- mechanical backfill and should not make every existing input message look
-- updated at migration time.
ALTER TABLE turn_attempt_input_messages
    DISABLE TRIGGER turn_attempt_input_messages_touch_updated_at;

UPDATE turn_attempt_input_messages
   SET ordinal = ordinal + 1000000
  FROM turn_attempts
 WHERE turn_attempts.id = turn_attempt_input_messages.attempt_id
   AND turn_attempts.system_instructions <> '';

UPDATE turn_attempt_input_messages
   SET ordinal = ordinal - 999999
  FROM turn_attempts
 WHERE turn_attempts.id = turn_attempt_input_messages.attempt_id
   AND turn_attempts.system_instructions <> '';

ALTER TABLE turn_attempt_input_messages
    ENABLE TRIGGER turn_attempt_input_messages_touch_updated_at;

INSERT INTO turn_attempt_input_messages
    (attempt_id, ordinal, role, metadata, created_at, updated_at)
SELECT ta.id,
       0,
       'system',
       jsonb_build_object(
           'agent_instructions', true,
           'id', 'chudbot_conversation_' || t.conversation_id || '_system'
       ),
       ta.created_at,
       ta.created_at
  FROM turn_attempts ta
  JOIN turns t ON t.id = ta.turn_id
 WHERE ta.system_instructions <> '';

INSERT INTO turn_attempt_input_blocks
    (input_message_id, ordinal, block_kind, text_content, created_at, updated_at)
SELECT m.id,
       0,
       'text',
       ta.system_instructions,
       ta.created_at,
       ta.created_at
  FROM turn_attempt_input_messages m
  JOIN turn_attempts ta ON ta.id = m.attempt_id
 WHERE m.ordinal = 0
   AND m.role = 'system'
   AND m.metadata->>'agent_instructions' = 'true';

ALTER TABLE turn_attempts
    DROP COLUMN system_instructions;

ALTER TABLE conversations
    DROP COLUMN system_instructions;
