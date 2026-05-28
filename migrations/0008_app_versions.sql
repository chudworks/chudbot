-- Ordered, human-friendly build versions ("v1", "v2", …) layered on top
-- of the opaque `git describe` string already baked into the binary via
-- build.rs (env!("GIT_VERSION")).
--
-- One row per distinct build the bot has ever run as. The SERIAL `id`
-- IS the version number surfaced to users and the model ("v{id}"), so
-- it must stay GAP-FREE and monotonic. That rules out the obvious
-- `INSERT ... ON CONFLICT DO NOTHING` UPSERT on startup: Postgres
-- allocates the sequence value BEFORE detecting the conflict, and
-- sequences are non-transactional, so every restart on an
-- already-seen build would burn a number and leave holes. The bot
-- instead SELECTs first and only INSERTs for a genuinely-new build
-- (see `Db::register_app_version`), so a number is consumed exactly
-- once per real version.
--
-- `git_version` stores the full `git describe --tags --always --dirty`
-- descriptor, not a bare SHA — hence the name. Dirty local builds with
-- the same HEAD collapse into one row (acceptable: version numbers only
-- matter for clean prod deploys).
CREATE TABLE app_versions (
    id           SERIAL PRIMARY KEY,
    git_version  TEXT NOT NULL UNIQUE,
    first_seen   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Which build answered each turn. Nullable: legacy turns predate this
-- and stay NULL (the viewer renders nothing for them, same as legacy
-- persona_name). New turns are always stamped at start_turn time. No
-- index needed — app_versions rows are never deleted, so the FK never
-- triggers a scan of turns, and the viewer reads version_id straight
-- off the turn row.
ALTER TABLE turns ADD COLUMN version_id INTEGER REFERENCES app_versions(id);
