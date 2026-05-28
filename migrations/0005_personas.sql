-- Persona selection storage. The TOML config defines the set of
-- available personas (system prompt + model + sampling knobs); this
-- table records which persona to use for a given scope at runtime.
--
-- Resolution order (most specific wins): conversation → user-in-guild
-- → channel → guild → config-level default_persona. Each lookup is a
-- single PK probe; the bot does up to four cheap queries per turn and
-- falls back to the config default if nothing matches.
--
-- The `key` column is a free-form string whose meaning depends on
-- `scope`:
--   conversation: conversation UUID (text form)
--   user:         "<guild_id>:<user_id>"  (user prefs are per-guild)
--   channel:      "<channel_id>"          (channels are globally unique)
--   guild:        "<guild_id>"
CREATE TABLE persona_selections (
    scope        TEXT NOT NULL,
    key          TEXT NOT NULL,
    persona_name TEXT NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scope, key)
);

-- Record which persona answered each turn, so the web viewer and any
-- future audit tooling can show "snark answered this one, default the
-- next." Nullable because rows written before this migration won't
-- have a value.
ALTER TABLE turns ADD COLUMN persona_name TEXT;
