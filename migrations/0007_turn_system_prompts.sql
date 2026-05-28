-- Per-turn snapshot of the fully composed system prompt actually sent to
-- the model: the persona's voice + the dynamically-built operational
-- block (build version, model, enabled capabilities, conventions) + any
-- operator policy addendum. The web viewer renders it so a trace shows
-- exactly what the model was instructed with on that turn — which can
-- differ across a conversation when the persona, model, or available
-- tools change.
--
-- Deliberately its own 1:1 table rather than a column on `turns`: the
-- bot's hot path (`load_conversation_history`) selects from `turns` on
-- every turn to rebuild chat history, and we don't want that query to
-- drag this large, viewer-only text. Only the viewer reads it.
CREATE TABLE turn_system_prompts (
    turn_id  UUID PRIMARY KEY REFERENCES turns(id) ON DELETE CASCADE,
    content  TEXT NOT NULL
);
