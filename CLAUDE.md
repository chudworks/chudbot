# grok-discord-bot

A Discord bot that integrates xAI's Grok API, with a companion web viewer
that shows each conversation's full trace: the messages fed to Grok as
context, every tool call (web search etc.), and the final answer.

## Tech Stack

- **Language**: Rust nightly, edition 2024
- **Discord**: `serenity` + `poise`
- **Web**: `axum` + `maud` (inline HTML, no template files)
- **DB**: Postgres via `sqlx` (compile-time-checked queries)
- **Async runtime**: `tokio`
- **Target platform**: macOS (Chud's Mac Studio), native — no Docker

## Crate Structure

Cargo workspace with two crates under `crates/`:

- `grok-discord-bot-core` — Grok client (behind a `GrokClient` trait so
  the bot/web logic is testable against mocks), domain types
  (`Conversation`, `Turn`, etc.), and the Postgres data layer.
- `grok-discord-bot-bin` — the binary. Contains `bot` and `web` modules
  plus `clap` subcommand parsing. Produces a single binary named `grok`.

## Build & Run

```sh
cargo build                          # debug build
cargo build --profile distribute     # production build
cargo run -- bot                     # run the Discord gateway loop
cargo run -- web                     # run the web viewer
cargo run -- migrate                 # apply Postgres migrations
cargo test --all-features            # run tests (mocks the external APIs)
```

Configuration is via env vars: `DISCORD_TOKEN`, `XAI_API_KEY`,
`POSTGRES_URL`, `WEB_BASE_URL`, `GROK_MODEL`. See README.md.

## Behavior (the architecture in one paragraph)

The bot listens for `@Grok` mentions. A **new conversation** is created
when the mention is *not* a reply to a prior bot message and *not* in a
thread the bot owns; otherwise the existing conversation is continued via
a `message_links(discord_message_id → conversation_id)` lookup table.
Replies are inline by default and auto-spawn a thread when long. The
first reply in a new conversation includes the web viewer URL
(`$WEB_BASE_URL/c/<uuid>`). The web viewer has **no auth** — security
relies on the unguessable UUID. Status is communicated via reaction
emojis: 👀 working, ✅ success, ❌ error.

## Coding Standards

This project follows Chud's Rust style guide in `.claude/rust-style.md`.

Key principles:
- Nightly Rust, minimal dependencies, longevity over convenience
- Static dispatch, iterators over collect, lifetimes over cloning
- `thiserror` for errors, `tracing` for logs, `test-case` for tests
- `where` clauses over inline bounds, `impl Trait` when possible
- Derive `Debug` on all types, use table-based tests
- Block format for dependencies with features

Mock external services (Grok, Discord) in tests rather than hitting the
live APIs. The `GrokClient` trait is the canonical seam.

Reference the full style guide when writing new modules or making
architectural decisions.
