# grok-discord-bot

Discord bot that integrates an LLM (xAI Grok or Anthropic Claude) with
server-side web search, plus a companion web viewer showing each
conversation's full trace: messages fed to the model, every tool call,
and the final answer.

## Tech Stack

- **Language**: Rust nightly, edition 2024
- **Discord**: `twilight` (gateway + http + model + cache + mention).
  Native event-stream API; **never use serenity or any crate requiring
  `async-trait`** (see [[feedback-no-async-trait]]).
- **Web**: `axum` 0.8 + `maud` (inline HTML, no template files)
- **DB**: Postgres via `sqlx` 0.9 with runtime-checked queries
- **LLM**: abstracted behind `LlmProvider::step` in `core::llm`. Two
  implementations: `XaiProvider` and `AnthropicProvider`. Both support
  server-side web search (xAI: `search_parameters`; Anthropic:
  `web_search_20250305`) AND client-side tool calls via the agentic
  harness in `core::agent`.
- **Config**: TOML file (`config.toml` by default). No env vars.
- **Async runtime**: `tokio`
- **Target platform**: macOS (Chud's Mac Studio), native — no Docker

## Crate Structure

Cargo workspace with two crates under `crates/`:

- **`grok-discord-bot-core`** — `LlmProvider` trait + xAI / Anthropic /
  mock impls, conversation domain types, Postgres data layer (`Db`),
  TOML config loader.
- **`grok-discord-bot-bin`** — the binary. Contains `bot` (Discord
  event loop) and `web` (Axum viewer) modules plus `clap` subcommand
  parsing. Produces a single binary named `grok`.

Migrations live at the workspace root in `migrations/` and are baked
into the binary via `sqlx::migrate!`.

## Build & Run

```sh
cargo build                          # debug build
cargo build --profile distribute     # production build
cargo run -- bot                     # run the Discord gateway loop
cargo run -- web                     # run the web viewer
cargo run -- migrate                 # apply Postgres migrations
cargo test --all-features            # run tests (mocks the LLM)
```

Configuration is in `config.toml` (see `config.toml.example`). The
`--config / -c` global flag points at a different path.

## Conversation model

The bot maintains conversations in Postgres, decoupled from Discord
threads. A conversation is created when `@Grok` is mentioned and the
message is *not* a reply to a prior bot message and *not* in a thread
the bot owns. Otherwise, `message_links(discord_message_id →
conversation_id)` resolves the existing conversation to continue.

Replies are inline by default; the bot auto-opens a thread when the
answer would exceed 1500 chars (Discord's hard limit is 2000). The
first reply in a new conversation includes the viewer URL
(`$WEB_BASE_URL/c/<uuid>`). Web viewer auth: **none** — security relies
on the unguessable UUID. Status emojis: 👀 working, ✅ success, ❌ error.

Each turn captures (via `context_items`) the exact snapshot of messages
fed to the model, and (via `tool_calls`) every server-side tool the
model invoked plus its request/response JSON. The viewer renders both
verbatim so traces are auditable.

## Agentic harness

`core::agent::run` drives `LlmProvider::step` in a loop:

1. Send chat history (turns + prior tool uses/results) + tool definitions
   to the provider.
2. If the model returns `StepResponse::Final`, stop.
3. If the model returns `StepResponse::UseTools`, execute each tool via
   the caller-supplied `ToolExecutor`, append both the assistant turn
   (with tool_use blocks) and a user turn (with tool_result blocks) to
   history, then loop.
4. Cap at `MAX_AGENT_ITERATIONS` (6) to prevent runaways.

Every tool call — server-side (web search) and client-side (`fetch_messages`
or any future tool) — is collected in execution order in `AgentRun.tool_calls`
and persisted into the `tool_calls` table. The web viewer renders them
all in order so the conversation trace shows every input and output.

### Client-side tools

The bot's `BotToolExecutor` exposes:

- **`fetch_messages(channel_id?, limit?, before_message_id?)`** — pulls
  recent messages from a Discord channel for context. The model calls
  this when it needs surrounding conversation that wasn't quoted.
- **`generate_image(prompt, reference_images?, aspect_ratio?, quality?)`**
  — calls xAI's Grok Imagine model (`grok-imagine-image` / `-quality`).
  Reference images may be `https://` URLs (passed through) or
  `file://images/…` URIs (base64-encoded from disk before sending).
  The tool saves the result bytes to `images_dir`, returns the
  `file://` URI to the agent, and the bot attaches the bytes to the
  outgoing Discord reply. Exposed only when `[llm.xai]` is configured.

Privacy mode constrains the tool:
- `Open`: returns everything (minus the bot's own messages).
- `ChannelOnly`: rejects fetches against any channel other than the
  configured one.
- `OptIn`: returns messages from opted-in users at full content;
  messages from opted-out users come back with `content =
  "[redacted: ...]"` and `redacted = true`, so the model knows the
  channel has more activity than it can see.
- `ConversationOnly`: the tool is not declared at all — the model can
  only operate from the conversation history it was already given.

## Privacy model

Per-guild, configurable at runtime via slash commands. Four "designs"
(per the group's discussion):

- **`open` (Design 1)** — bulk-fetches recent channel messages on each
  @mention. Best answers, least privacy.
- **`channel_only` (Design 2)** — same as open, but only operates in
  a single configured channel.
- **`opt_in` (Design 3, default)** — only the user's own `@`-mention,
  prior turns of the same conversation, and Discord-reply-quoted
  messages whose author has opted in. The author opt-in is also
  per-guild.
- **`conversation_only` (Design 4)** — strictest: just the user's
  mention and prior turns. Even quoted messages are dropped.

In ALL modes the bot sees the user's `@`-mention itself and the
conversation history reconstructed from the DB. Those are the floor.

Slash commands:
- `/grok-privacy {in|out|status}` — anyone, per-user, per-guild.
- `/grok-mode {show|set}` — admins only, per-guild. The `set`
  subcommand takes `mode`, optional `channel` (required for
  `channel_only`), and optional `history_size`.

Per-guild settings live in `guild_settings.privacy_mode` (JSONB —
serializes the same `PrivacyMode` enum used in config). Missing row
falls back to `config.default_privacy`.

## Multi-tenancy

The bot is multi-tenant at the data layer. `conversations` carry
`discord_guild_id`. `message_links` denormalizes it for guild-scoped
analytics / cleanup. `user_privacy` and `guild_settings` are both
keyed by guild_id. Other tables (turns, context_items, tool_calls)
reach guild via `conversations` transitively.

## Coding Standards

This project follows Chud's Rust style guide in `.claude/rust-style.md`.

Key principles:
- Nightly Rust, minimal dependencies, longevity over convenience.
- Static dispatch, iterators over collect, lifetimes over cloning.
- **No `async-trait`** — use native async fn in traits (RPITIT). When
  picking libraries, prefer ones whose trait dispatch is native-async
  (twilight) over ones that require `#[async_trait]` (serenity).
- `thiserror` for errors, `tracing` for logs, `test-case` for tables.
- `where` clauses over inline bounds, `impl Trait` when possible.
- Derive `Debug` on all types.

Mock external services (LLM, Discord) in tests rather than hitting the
live APIs. The `LlmProvider` trait is the canonical seam, and
`MockProvider` in `core::llm::mock` is the existing impl for tests.
