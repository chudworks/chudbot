# Chudbot

Chudbot is a Discord bot plus trace viewer. The bot runs model-backed
agents, records the exact turn trace in Postgres, and serves a React viewer
for `/c/<conversation-uuid>`.

## Current Architecture

The `chudbot-*` workspace crates are the source of truth. Historical crate
layouts live in git history, not in the current documentation.

Workspace crates:

- `chudbot-api`: provider-neutral contracts for ids, transcripts, tools,
  agents, media, platform events, storage, live events, usage, and retry.
- `chudbot-bot`: platform-neutral bot orchestration: event handling,
  conversations, turns, commands, agent/tool wiring, title/avatar
  jobs, and user-facing reply behavior.
- `chudbot-discord`: Twilight platform implementation. No Twilight types leak
  into `chudbot-api`.
- `chudbot-web`: Axum JSON API, SSE, media routes, static SPA serving, crawler
  controls, link-preview (OpenGraph) injection, viewer config, and authenticated
  Vibe site/source host routing.
- `chudbot-storage-sqlx`: Postgres `BotStorage` implementation and embedded
  migrations.
- `chudbot-asset-local`: local filesystem `MediaStore`.
- `chudbot-asset-s3`: S3-compatible object storage `MediaStore`.
- `chudbot-vibe`: Vibe names and access policy, bare Git and artifact storage,
  Docker sandbox orchestration, coding tools, and runtime configuration.
- `chudbot-xai`: xAI LLM, image, and video providers.
- `chudbot-gemini`: Google Gemini LLM, Nano Banana image, and Veo video
  providers.
- `chudbot-openai`: OpenAI Responses LLM and image providers.
- `chudbot-openai-compat`: OpenAI-compatible Chat Completions LLM provider
  for local/model-gateway hosts such as vLLM.
- `chudbot-anthropic`: Anthropic Messages LLM provider.
- `chudbot-bin`: thin process launcher and TOML config loader.

## Build And Run

Rust uses nightly, edition 2024.

```sh
cargo build
cargo build --profile distribute -p chudbot-bin
cargo run -p chudbot-bin -- check-config
cargo run -p chudbot-bin -- migrate
cargo run -p chudbot-bin -- serve
cargo test --all-features
```

Frontend development:

```sh
cd frontend
bun install
bun run dev
```

Production helper:

```sh
./serve.sh deploy
```

`serve.sh deploy` pulls with fast-forward only, applies the Vibe firewall,
builds the frontend and pinned sandbox image, installs the Vibe template and
skills, builds `target/distribute/chudbot`, runs `check-config`, stops Chudbot,
runs migrations, atomically installs `$CHUDBOT_DIR/chudbot`, and starts a tmux
session. It does not install systemd units or write helpers into global paths.

The public firewall commands are `firewall-install`, `firewall-check`, and
`firewall-remove`. They own only the `chudbot-sandbox` Docker network,
`chudbot-sbx0` bridge, and `CHUDBOT_SANDBOX` chains. Keep the detailed DGX Spark
procedure in `docs/vibe-operator-runbook.md` rather than duplicating it here.

## Configuration

`config.example.toml` is the config reference. Copy it to `config.toml`.
When changing config schema or semantics, keep `check-config` validation and
diagnostics compatible so invalid configs report rich, spanned, actionable
errors.

The config is agent-first:

- `[bot.agents.<name>]` defines prompt, provider, model, tool exposure, media
  generation bindings, loop limits, and subagents.
- `[bot.skills.<name>]` maps stable skill names to Markdown instruction files.
- `[llm.<name>]`, `[image.<name>]`, `[video.<name>]`, and
  `[platforms.<name>]` define named runtime services.
- `[vibe]` owns rollout, domain, persistent storage, OAuth, sandbox, and limit
  configuration. Discord ids stay strings in TOML and JSON and are parsed at
  startup boundaries.
- `[bot.platforms.<platform>]` binds a platform to its default agent.
- `[logging]` owns tracing setup. Do not add env-only logging controls.

Agent-level `client_tools` is a strict allowlist over the runtime-provided
surface, not an additive list. Omit it for the full configured surface. When an
allowlist is necessary, preserve every required built-in, media, memory, and
subagent tool; adding only a new tool silently disables omitted tools.

Use `agent`, not `persona`, in new code, docs, config, commands, and frontend
text.

## Runtime Behavior

Discord is only the I/O surface. Conversation state lives in Postgres and is
looked up by platform message/channel links.

Chudbot does not maintain a separate context-access policy. Platform permissions
decide what messages and history the bot can see; anything visible to the
configured platform integration is eligible model context.

Slash commands are `/chudbot-agent`.

The web viewer is unauthenticated. Security relies on unguessable UUIDs plus
the web layer's no-index/crawler controls. Do not add route-listing or
guessable conversation discovery.

Vibe is separate from the trace viewer. Site and source hosts require a Discord
OAuth session and a current guild-membership check. Wrong-guild and nonmember
requests must reveal no site data. `/__vibe/` routes take precedence over site
artifacts, and every Vibe response remains private and uncacheable.

Vibe source history lives in local bare Git repositories; served files come
only from immutable per-revision artifact directories. Postgres is authoritative
for the active revision. Treat the database and `$CHUDBOT_DIR/vibe` as one
backup/restore unit.

## Engineering Rules

- Follow [docs/rust-style.md](docs/rust-style.md).
- Never use `serenity`.
- Use native async traits/RPITIT (`impl Future + Send` where crossing spawned
  task or Axum boundaries needs it) for statically dispatched traits.
- Avoid `async-trait` by default. It is allowed for deliberate trait-object
  boundaries where the alternative is hand-written boxed future plumbing.
- Keep `chudbot-api` free of Twilight, SQLx, Reqwest, Axum, and concrete
  provider config.
- Prefer static dispatch and named registries over broad trait-object service
  bags.
- Use `thiserror` for errors, `tracing` for logs, and table-driven tests with
  `test-case`.
- Mock external services in tests; do not hit live Discord or provider APIs.
- Keep frontend changes compatible with the existing trace-viewer design unless
  the task explicitly asks for a redesign.
- Keep Vibe coding agents on the exact `read`, `edit`, and `shell` tool surface.
  The conversation agent owns job state, authorization, Git commits, builds,
  deployment, and user-facing links.
- Use the `vibe` conversation tool only to create or edit websites. Standalone
  image and video requests belong to their configured media-generation tools,
  never to the Vibe coding agent.
- Vibe `read` and `edit` paths may be workspace-relative or rooted at
  `/workspace`. Reject other absolute paths, traversal, `.git`, `node_modules`,
  and `dist`; perform production file I/O in the container namespace so
  symlinks cannot redirect host-side access.
- Resolve host-side Vibe storage and workspace paths before passing them to
  Docker. Bind-mount sources must be absolute; never pass a config-relative
  path directly to the daemon.
- Never mount the Docker socket, config, media, repositories, artifacts, or
  sibling workspaces into a Vibe container. Preserve the non-root user,
  read-only root, dropped capabilities, no-new-privileges, resource limits, and
  dedicated firewalled Docker network.
- Vibe source export and source browsing must omit `node_modules`, `dist`, and
  TypeScript build metadata even for legacy revisions. Do not serve a partial
  artifact or advance the active revision until the clean build and database
  commit both succeed.
