-- Persist usage metadata for compact memory profile generations.

ALTER TABLE user_memory_document_versions
    ADD COLUMN agent_name TEXT NOT NULL DEFAULT 'memory_compact',
    ADD COLUMN llm_provider TEXT NOT NULL DEFAULT 'unknown',
    ADD COLUMN llm_model TEXT NOT NULL DEFAULT 'unknown',
    ADD COLUMN usage JSONB NOT NULL DEFAULT '[]'::jsonb;
