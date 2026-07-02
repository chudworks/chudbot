-- Rewrite historical backfill data to avoid repeating the same marked
-- agent-instructions system turn on every attempt in a conversation.
--
-- This intentionally modifies historical stored transcripts: for each
-- conversation, delete consecutive exact repeats of the marked
-- agent-instructions message. Non-consecutive repeats are kept so an
-- A -> B -> A prompt history still records the later change back to A.
-- Remaining input-message ordinals are compacted per affected attempt.

CREATE TEMP TABLE tmp_duplicate_agent_instruction_messages ON COMMIT DROP AS
WITH marked AS (
    SELECT m.id AS message_id,
           m.attempt_id,
           t.conversation_id,
           t.ordinal AS turn_ordinal,
           ta.attempt_ordinal,
           m.ordinal AS message_ordinal,
           COALESCE(m.metadata->>'agent_instruction_part_key', '__legacy__') AS prompt_identity,
           prompt.prompt_text
      FROM turn_attempt_input_messages m
      JOIN turn_attempts ta ON ta.id = m.attempt_id
      JOIN turns t ON t.id = ta.turn_id
      JOIN LATERAL (
          SELECT b.text_content AS prompt_text
            FROM turn_attempt_input_blocks b
           WHERE b.input_message_id = m.id
             AND b.block_kind = 'text'
             AND b.text_content IS NOT NULL
           ORDER BY b.ordinal
           LIMIT 1
      ) prompt ON prompt.prompt_text IS NOT NULL
     WHERE m.role = 'system'
       AND m.metadata->>'agent_instructions' = 'true'
), ranked AS (
    SELECT message_id,
           attempt_id,
           prompt_identity,
           prompt_text,
           lag(prompt_identity) OVER (
               PARTITION BY conversation_id
               ORDER BY turn_ordinal, attempt_ordinal, message_ordinal, message_id
           ) AS previous_prompt_identity,
           lag(prompt_text) OVER (
               PARTITION BY conversation_id
               ORDER BY turn_ordinal, attempt_ordinal, message_ordinal, message_id
           ) AS previous_prompt_text
      FROM marked
)
SELECT message_id, attempt_id
  FROM ranked
 WHERE prompt_identity = previous_prompt_identity
   AND prompt_text = previous_prompt_text;

CREATE TEMP TABLE tmp_deduplicated_agent_instruction_attempts ON COMMIT DROP AS
SELECT DISTINCT attempt_id
  FROM tmp_duplicate_agent_instruction_messages;

DELETE FROM turn_attempt_input_messages m
 USING tmp_duplicate_agent_instruction_messages d
 WHERE m.id = d.message_id;

-- Deleting the ordinal-0 marked system message leaves most affected attempts
-- starting at 1. Compact only those attempts, using the same offset pattern as
-- migration 0028 because the unique (attempt_id, ordinal) constraint is not
-- deferrable.
ALTER TABLE turn_attempt_input_messages
    DISABLE TRIGGER turn_attempt_input_messages_touch_updated_at;

UPDATE turn_attempt_input_messages m
   SET ordinal = ordinal + 1000000
  FROM tmp_deduplicated_agent_instruction_attempts a
 WHERE m.attempt_id = a.attempt_id;

WITH compacted AS (
    SELECT m.id,
           row_number() OVER (
               PARTITION BY m.attempt_id
               ORDER BY m.ordinal
           ) - 1 AS compacted_ordinal
      FROM turn_attempt_input_messages m
      JOIN tmp_deduplicated_agent_instruction_attempts a
        ON a.attempt_id = m.attempt_id
)
UPDATE turn_attempt_input_messages m
   SET ordinal = compacted.compacted_ordinal
  FROM compacted
 WHERE m.id = compacted.id;

ALTER TABLE turn_attempt_input_messages
    ENABLE TRIGGER turn_attempt_input_messages_touch_updated_at;

DO $$
BEGIN
    IF EXISTS (
        WITH marked AS (
            SELECT m.id AS message_id,
                   t.conversation_id,
                   t.ordinal AS turn_ordinal,
                   ta.attempt_ordinal,
                   m.ordinal AS message_ordinal,
                   COALESCE(m.metadata->>'agent_instruction_part_key', '__legacy__') AS prompt_identity,
                   prompt.prompt_text
              FROM turn_attempt_input_messages m
              JOIN turn_attempts ta ON ta.id = m.attempt_id
              JOIN turns t ON t.id = ta.turn_id
              JOIN LATERAL (
                  SELECT b.text_content AS prompt_text
                    FROM turn_attempt_input_blocks b
                   WHERE b.input_message_id = m.id
                     AND b.block_kind = 'text'
                     AND b.text_content IS NOT NULL
                   ORDER BY b.ordinal
                   LIMIT 1
              ) prompt ON prompt.prompt_text IS NOT NULL
             WHERE m.role = 'system'
               AND m.metadata->>'agent_instructions' = 'true'
        ), ranked AS (
            SELECT prompt_identity,
                   prompt_text,
                   lag(prompt_identity) OVER (
                       PARTITION BY conversation_id
                       ORDER BY turn_ordinal, attempt_ordinal, message_ordinal, message_id
                   ) AS previous_prompt_identity,
                   lag(prompt_text) OVER (
                       PARTITION BY conversation_id
                       ORDER BY turn_ordinal, attempt_ordinal, message_ordinal, message_id
                   ) AS previous_prompt_text
              FROM marked
        )
        SELECT 1
          FROM ranked
         WHERE prompt_identity = previous_prompt_identity
           AND prompt_text = previous_prompt_text
    ) THEN
        RAISE EXCEPTION 'consecutive duplicate agent-instructions messages remain after deduplication';
    END IF;

    IF EXISTS (
        SELECT 1
          FROM (
              SELECT m.attempt_id,
                     count(*) AS messages,
                     min(m.ordinal) AS min_ordinal,
                     max(m.ordinal) AS max_ordinal,
                     count(DISTINCT m.ordinal) AS distinct_ordinals
                FROM turn_attempt_input_messages m
                JOIN tmp_deduplicated_agent_instruction_attempts a
                  ON a.attempt_id = m.attempt_id
               GROUP BY m.attempt_id
          ) attempts
         WHERE messages > 0
           AND (
               min_ordinal <> 0
               OR max_ordinal <> messages - 1
               OR distinct_ordinals <> messages
           )
    ) THEN
        RAISE EXCEPTION 'input-message ordinals were not compacted after agent-instructions deduplication';
    END IF;
END;
$$;
