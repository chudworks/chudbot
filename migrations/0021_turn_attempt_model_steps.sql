-- Persist every provider model step emitted during an attempt so later turns
-- can replay provider-native output items in their original order.

CREATE TABLE IF NOT EXISTS turn_attempt_model_steps (
    id             BIGSERIAL PRIMARY KEY,
    attempt_id     UUID NOT NULL REFERENCES turn_attempts(id) ON DELETE CASCADE,
    ordinal        INTEGER NOT NULL,
    step_kind      TEXT NOT NULL CHECK (step_kind IN ('final', 'continue', 'client_tools')),
    llm_provider   TEXT NOT NULL,
    llm_model      TEXT NOT NULL,
    continuation   JSONB,
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (attempt_id, ordinal)
);

CREATE INDEX IF NOT EXISTS turn_attempt_model_steps_attempt_idx
    ON turn_attempt_model_steps (attempt_id, ordinal);

INSERT INTO turn_attempt_model_steps (
    attempt_id, ordinal, step_kind, llm_provider, llm_model, continuation, metadata
)
SELECT ta.id,
       0,
       'final',
       ta.llm_provider,
       ta.llm_model,
       COALESCE(ta.continuation, t.continuation),
       jsonb_build_object(
           'backfilled_from', 'legacy_final_continuation',
           'partial', true
       )
  FROM turn_attempts ta
  JOIN turns t ON t.id = ta.turn_id
 WHERE COALESCE(ta.continuation, t.continuation) IS NOT NULL
ON CONFLICT (attempt_id, ordinal) DO NOTHING;

DROP TRIGGER IF EXISTS turn_attempt_model_steps_touch_updated_at ON turn_attempt_model_steps;
CREATE TRIGGER turn_attempt_model_steps_touch_updated_at
    BEFORE UPDATE ON turn_attempt_model_steps
    FOR EACH ROW EXECUTE FUNCTION chudbot_touch_updated_at();
