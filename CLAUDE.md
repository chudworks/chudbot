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
- **Web**: `axum` 0.8 JSON API + SSE event stream. Frontend is a
  React 19 + Vite + TypeScript + Zustand + SCSS SPA living in
  `frontend/`. The Rust server serves the built bundle from
  `[web].frontend_dir` (default `./frontend-build`) with an
  `index.html` SPA fallback for client-side routes like `/c/<uuid>`.
  `serve.sh deploy` builds the frontend with `bun` and atomically
  copies `dist/` into `$CHUDBOT_DIR/frontend-build/`.
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
- **`grok-discord-bot-bin`** — the binary. Contains:
  - `app` — shared `AppState`: db, providers, storage paths, the
    broadcast event bus, `CancellationToken`, `TaskTracker`.
  - `bot` — Discord gateway loop (twilight).
  - `web` — Axum JSON API + SSE + static file server.
  - `avatars` — background avatar fetcher.
  - `titles` — background conversation titler.
  - `commands` — slash command dispatchers.
  Produces a single binary named `grok` with two subcommands: `serve`
  (runs bot + web together with graceful shutdown) and `migrate`.

Migrations live at the workspace root in `migrations/` and are baked
into the binary via `sqlx::migrate!`.

## Build & Run

```sh
cargo build                          # debug build
cargo build --profile distribute     # production build
cargo run -- serve                   # run bot + web in one process
cargo run -- migrate                 # apply Postgres migrations
cargo test --all-features            # run tests (mocks the LLM)

# Frontend (separate Vite dev server when iterating on UI)
cd frontend && bun install && bun run dev   # serves on :5173, proxies /api → :1860
```

Ctrl+C on `serve` drains in-flight work (turn handlers, title gen,
avatar fetches) for up to 30 seconds before exiting.

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
on the unguessable UUID. Status emojis: 👀 working, ✅ success, ❌ error,
❓ refused (upstream safety / moderation), 🔄 retry affordance on a failed
turn's reply.

**Admin stop-sign (🛑).** Operator admins (top-level `admins` config — a
list of Discord user ids as strings) can pause the bot in a single
conversation by reacting 🛑 (`:octagonal_sign:`) on any tracked message
(the @mention, a bot reply, or any message inside a Grok-owned thread);
removing the reaction resumes it. State lives on `conversations.
stopped_at` / `stopped_by_user_id` (nullable; `stopped_at` doubles as the
flag). `handle_reaction` routes both `ReactionAdd` and `ReactionRemove`:
🛑 from an admin resolves the message→conversation and calls
`Db::stop_conversation` / `resume_conversation`. While stopped, both the
live-mention path (`handle_message`, gated before the 👀 reaction so a
paused thread shows no sign of life and spends no tokens) and the 🔄-retry
path stay silent. The viewer renders a banner from `stopped_at` and
refetches on the `conversation_updated` SSE event.

**Resilience & failure handling.** Transient upstream blips (HTTP 5xx /
429 / transport) are retried with exponential backoff via `core::retry`,
which wraps every LLM / image / video HTTP call (video `submit` opts out
of network-error retries so a dropped connection can't double-charge a
render). If a turn still fails, `run_turn_and_reply` (in `bot` — the
single seam that owns all user-facing turn output) posts exactly ONE
`⚠️ …` reply, marks the turn `failed` (persisting the error + any salvaged
partial content, both shown in the viewer), and adds a 🔄 reaction to its
own error message. Clicking that 🔄 — or reacting 🔄 on the original
`@`-mention — re-runs the turn: `handle_reaction` maps the message back to
its turn and `Db::reset_turn_for_retry` atomically re-runs ONLY the latest
still-`failed` turn (double-clicks / stale reactions are no-ops; the bot's
own 🔄 is self-ignored via a `user_id == bot` check). Retry reconstructs
the LLM history from the DB — no live gateway message — and reuses the
same `run_turn_and_reply` tail.

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

**Cross-turn images.** Prior turns' *text* is replayed from the `turns`
columns, but images would otherwise be lost (they live only as
`context_items` / `tool_calls` rows, neither of which feeds the model).
So `build_context` calls `Db::load_conversation_image_uris` to gather
both user-uploaded attachments and `generate_image` outputs from earlier
completed turns and re-attaches each as a `TurnBlock::Image` on the user
message it belonged to. These replayed image rows are NOT persisted, so
they can't feed back into the query. Prior-turn images are served from
our own storage via `storage::to_public_url(uri, base_url)` — the single
URL-minting seam (today `{base_url}/images/<name>` via the Axum
`ServeDir`; the `s3://` branch is where a CDN/signed URL goes later) —
because Discord's CDN links expire. The provider fetches that URL
server-side, so cross-turn image vision needs `web.base_url` publicly
reachable (true in prod; local-`localhost` dev can't serve prior-turn
images to the model, though the current turn still works via the live
Discord URL). Replay is capped at `MAX_REPLAYED_IMAGES` (most recent
first; drops are logged).

**Prompt caching.** The Anthropic provider stamps two ephemeral
`cache_control` breakpoints per request — one on the system prompt
(anchoring the stable `tools + system` prefix) and one on the final
message block (extending the cache over the whole history, images
included). Since the agent loop re-sends the full prefix on every
iteration and every later turn re-sends all prior turns + replayed
images, the matched prefix bills at 0.1x instead of full input price.
This is the primary cost control for keeping images in context. xAI
caches automatically (cached-input pricing), so it needs no breakpoints.

## Agentic harness

`core::agent::run` drives `LlmProvider::step` in a loop:

1. Send chat history (turns + prior tool uses/results) + tool definitions
   to the provider.
2. If the model returns `StepResponse::Final`, stop.
3. If the model returns `StepResponse::UseTools`, execute all the
   requested tools **concurrently** (unordered parallelism via a
   `FuturesUnordered`) through the caller-supplied `ToolExecutor`, then
   append both the assistant turn (with tool_use blocks) and a single
   user turn (with one tool_result block per tool_use) to history, then
   loop. The tool-use protocol requires every `tool_result` for an
   assistant turn to come back together in the next user message, so the
   loop fans out, awaits all results, and makes ONE follow-up request —
   the parallelism is pure latency overlap (e.g. a `post_status_message`
   posts while a `generate_image` renders). Each tool is supervised
   independently: a failure becomes an `is_error` tool_result fed back to
   the model, never an abort of its siblings. Completions land in
   arbitrary order but are slotted back into declared order so the trace
   stays deterministic.
4. Cap at `MAX_AGENT_ITERATIONS` (6) to prevent runaways.

Every tool call — server-side (web search) and client-side (`fetch_messages`
or any future tool) — is collected in declared order in `AgentRun.tool_calls`
and persisted into the `tool_calls` table. The web viewer renders them
all in order so the conversation trace shows every input and output.

### Client-side tools

The bot's `BotToolExecutor` exposes:

- **`fetch_messages(channel_id?, limit?, before_message_id?)`** — pulls
  recent messages from a Discord channel for context. The model calls
  this when it needs surrounding conversation that wasn't quoted.
- **`generate_image(prompt, reference_images?, aspect_ratio?, quality?)`**
  — routed through the persona's configured `image_provider`. Reference
  images may be `https://` URLs (passed through) or `file://images/…`
  URIs (base64-encoded from disk before sending if the backend wants
  inline data). The tool saves the result bytes to `images_dir`, returns
  the `file://` URI to the agent, and the bot attaches the bytes to the
  outgoing Discord reply. Exposed only for personas that name a
  backend whose `[image.<kind>]` credentials are configured. The
  `quality` field is a free-form model string — xAI maps `"standard"`
  and `"quality"` to its own model ids; other backends define their own.
- **`generate_video(prompt, image_url?, duration_seconds?, aspect_ratio?, resolution?, model?)`**
  — synchronous submit + poll + download. Routed through the persona's
  `video_provider`. Tool blocks for ~60-120s; the bot persists a
  `video_jobs` row at submit time so a crash mid-poll leaves the
  request discoverable. The model is expected to call
  `post_status_message` in the SAME response so the user sees a status
  line before the wait.
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

**System prompt composition.** The persona's `system_prompt` is only the
*voice*. The actual system prompt sent to the model is built per turn by
`compose_system_prompt` (in `bot`), once the persona is resolved, in this
order: the operator's global `extra_system_prompt` (optional top-level
config scalar — the non-persona slot for deployment-wide rules like the
Discord ToS) + a dynamically-generated **operational block** +
`persona.system_prompt`. The operational block is non-persona and self-updating: the build
version (`env!("GIT_VERSION")`), the model + provider actually answering,
a one-line pointer to each capability whose tool is declared *this* turn
(image/video gen, `fetch_messages`, always-on web search — the HOW stays
in the tool descriptions), and cross-cutting conventions (don't echo the
bracketed context notes we inject or any internal id/URL/`file://` path;
narrate slow ops via `post_status_message`; write for Discord). The
operator policy and operational block come BEFORE the persona, framing it
as a stable non-persona preamble with the persona voice last. Output
is stable within a (deployment, persona, privacy-mode), so it caches
cleanly. Capability lines mirror `build_tool_definitions`, so the prompt
never advertises a tool the model wasn't given. The composed prompt is
snapshotted per turn into `turn_system_prompts(turn_id → content)` and
surfaced in the web viewer (collapsible "System prompt" on each turn) so
a trace shows exactly what the model was instructed with — it can vary
across a conversation as the persona/model/tools change. It's a separate
1:1 table, not a `turns` column, specifically so the hot-path
`load_conversation_history` query never drags the large text; only the
viewer reads it (legacy turns predating the snapshot show `null`).

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

## Image / video provider modularity

Image and video generation use the same trait-+-enum pattern as the LLM
layer:

- `core::imagegen::ImageProvider` (native async fn, no `async-trait`).
  Today's only impl is `XaiImageProvider`; static dispatch via
  `AnyImageProvider` so adding e.g. DALL-E 3, Flux via Fal.ai, or
  Stable Diffusion via Replicate is a one-impl drop-in.
- `core::videogen::VideoProvider` — same shape with `submit` /
  `check_once` / `download_bytes` primitives so the bot can interleave
  status messages between polls. `XaiVideoProvider` today; Runway /
  Pika / Sora drop in by implementing the trait and adding an
  `AnyVideoProvider` variant.

Credentials live in `[image.<kind>]` / `[video.<kind>]` blocks. Personas
opt into a specific backend via `image_provider = "<kind>"` /
`video_provider = "<kind>"`. A persona that doesn't name a backend
simply doesn't expose the corresponding tool. Validation rejects a
persona that names a backend with no matching credentials block.

The per-request `model` field on `ImageGenRequest` / `VideoGenRequest`
is free-form — each backend interprets the string against its own
catalog. xAI's image side accepts `"standard"` / `"quality"`; future
backends are free to expose their own tier names.

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
