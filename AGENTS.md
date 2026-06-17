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
  conversations, turns, privacy, commands, agent/tool wiring, title/avatar
  jobs, and user-facing reply behavior.
- `chudbot-discord`: Twilight platform implementation. No Twilight types leak
  into `chudbot-api`.
- `chudbot-web`: Axum JSON API, SSE, media routes, static SPA serving, crawler
  controls, link-preview (OpenGraph) injection, and viewer config.
- `chudbot-storage-sqlx`: Postgres `BotStorage` implementation and embedded
  migrations.
- `chudbot-asset-local`: local filesystem `MediaStore`.
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

`serve.sh deploy` builds the frontend with Bun, copies `frontend/dist` to
`$CHUDBOT_DIR/frontend-build`, builds `target/distribute/chudbot`, runs
`check-config`, runs migrations, installs `$CHUDBOT_DIR/chudbot`, and starts a
tmux session.

## Configuration

`config.example.toml` is the config reference. Copy it to `config.toml`.
When changing config schema or semantics, keep `check-config` validation and
diagnostics compatible so invalid configs report rich, spanned, actionable
errors.

The config is agent-first:

- `[bot.agents.<name>]` defines prompt, provider, model, tool exposure, media
  generation bindings, loop limits, and subagents.
- `[llm.<name>]`, `[image.<name>]`, `[video.<name>]`, and
  `[platforms.<name>]` define named runtime services.
- `[bot.platforms.<platform>]` binds a platform to its default agent.
- `[default_privacy]` is the deployment fallback before a guild stores a
  runtime override.
- `[logging]` owns tracing setup. Do not add env-only logging controls.

Use `agent`, not `persona`, in new code, docs, config, commands, and frontend
text.

## Runtime Behavior

Discord is only the I/O surface. Conversation state lives in Postgres and is
looked up by platform message/channel links.

Privacy modes:

- `open`: fetch channel history.
- `channel_only`: fetch only the configured channel.
- `opt_in`: default; non-opted-in fetched messages are redacted.
- `conversation_only`: no history-fetch tool.

Slash commands are `/chudbot-privacy`, `/chudbot-mode`, and `/chudbot-agent`.

The web viewer is unauthenticated. Security relies on unguessable UUIDs plus
the web layer's no-index/crawler controls. Do not add route-listing or
guessable conversation discovery.

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
