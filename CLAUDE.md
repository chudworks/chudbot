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
  harness in `core::agent`. Providers are **model-agnostic** — the
  specific model id is supplied per call via `StepRequest::model`, so a
  single provider instance serves any persona that uses that provider.
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

Each turn persists the inputs that are NOVEL to that turn in
`context_items`: the user's `@`-mention, any Discord-reply-quoted
message, and image attachments. The system prompt (constant; in the
bot config) and prior turns' user/assistant text (already columns on
`turns`) are NOT re-stamped per turn — they'd just duplicate data
already on disk and grow the table quadratically with conversation
length. Server-side tool calls and their request/response JSON go
into `tool_calls`. The web viewer renders the stored rows verbatim
plus the prior turns from the `turns` table, so traces stay auditable
without the duplication.

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
- **`start_video_generation(prompt, image_url?, duration_seconds?, aspect_ratio?, resolution?)`**
  / **`check_video_status(request_id, wait_seconds?)`** — agent-driven
  polling pair for `grok-imagine-video`. `start` submits and returns
  immediately with a `request_id` (persisted to `video_jobs`); `check`
  sleeps `wait_seconds` (max 30) and polls once, returning `pending`
  or `done` with the saved `file://videos/…` URI. The model is
  expected to interleave `post_status_message` calls between polls to
  keep the user updated.
- **`post_status_message(text)`** — posts an intermediate Discord
  reply to the user. Always exposed. Used by the model to narrate
  long-running operations without hardcoded boilerplate.

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

## Personas

The TOML config defines named personas under `[personas.*]`. Each
persona ties together a system prompt, a provider (`xai` or
`anthropic`), a model id, and optional `temperature` / `top_p`. The
default fallback is `default_persona = "<name>"` at the top of the
config; that name must be a key in the personas table.

Per-provider knobs live under sub-tables on the persona:
`[personas.<name>.xai]` and `[personas.<name>.anthropic]`. Today the
only field is `xai.reasoning_effort` (`"low"` | `"medium"` |
`"high"`), forwarded as `reasoning: { effort: ... }` on the Responses
API. Reasoning-capable models (grok-4 family) consume it; others
silently ignore it. The slot is typed end-to-end via `ProviderOptions`
in `core::llm`, so adding e.g. Anthropic extended-thinking budget is a
one-field-add on `AnthropicOptions` plus a read in the provider's
`step`.

The provider blocks (`[llm.xai]`, `[llm.anthropic]`) supply only the
api key — the model is no longer there. Include only the provider
blocks the configured personas actually use; validation rejects a
persona that references a provider with no credentials.

At runtime, persona selection lives in `persona_selections(scope, key
→ persona_name)`. Resolution is **most-specific-wins**:
`conversation → user-in-guild → channel → guild → default_persona`.
Each lookup is one PK probe; worst case is four cheap queries per turn.

Slash command:
- `/grok-persona list` — show available personas and their (provider, model).
- `/grok-persona show` — show the full resolution chain at the call site.
- `/grok-persona set name:<persona> scope:<conversation|user|channel|guild>`
  — pin a persona for the given scope. `channel` and `guild` require
  admin; `user` and `conversation` are self-service. `conversation`
  needs to be run inside a Grok-owned thread (so the bot can map the
  channel back to a conversation id).
- `/grok-persona clear scope:<...>` — remove an override.

Each turn stamps `turns.persona_name` with the resolved persona before
the agent runs, so the web viewer can show which persona answered each
turn even when a conversation mixes personas across turns.

Mid-conversation persona switches change *future* turns only — prior
turns are replayed verbatim from `turns.{user,assistant}_content`.

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
- `/grok-persona {set|show|list|clear}` — see the Personas section.

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
