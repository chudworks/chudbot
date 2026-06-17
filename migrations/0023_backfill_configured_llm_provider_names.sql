-- Provider crates used to stamp model-step usage and continuations with
-- implementation names such as `anthropic` or `openai_compat`. The configured
-- provider key on the turn attempt is the stable runtime identity. Restrict
-- usage rows to the attempt model so subagent model-step usage recorded on the
-- parent turn is not rewritten to the parent provider.

UPDATE usage_records ur
   SET provider = ta.llm_provider,
       raw = CASE
           WHEN ur.raw IS NULL THEN NULL
           ELSE jsonb_set(ur.raw, '{provider}', to_jsonb(ta.llm_provider), true)
       END,
       updated_at = now()
  FROM turn_attempts ta
 WHERE ur.attempt_id = ta.id
   AND ur.subject_kind = 'model_step'
   AND ur.model = ta.llm_model
   AND ur.provider <> ta.llm_provider;

UPDATE turn_attempt_model_steps ms
   SET llm_provider = ta.llm_provider,
       continuation = CASE
           WHEN ms.continuation IS NULL THEN NULL
           ELSE jsonb_set(ms.continuation, '{provider}', to_jsonb(ta.llm_provider), true)
       END,
       updated_at = now()
  FROM turn_attempts ta
 WHERE ms.attempt_id = ta.id
   AND (
       ms.llm_provider <> ta.llm_provider
       OR (
           ms.continuation IS NOT NULL
           AND ms.continuation ->> 'provider' IS DISTINCT FROM ta.llm_provider
       )
   );
