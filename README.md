# grok-discord-bot

A Discord bot that integrates an LLM (xAI Grok or Anthropic Claude) with
server-side web search, plus a companion web viewer that shows each
conversation's full trace: the messages fed to the model, every tool
call, and the final answer.

## Run

```sh
grok bot       # start the Discord gateway loop
grok web       # start the Axum web viewer
grok migrate   # apply pending Postgres migrations
```

All three subcommands take `-c / --config <path>` (default `config.toml`).

## Configuration

Copy `config.toml.example` to `config.toml` and fill in your secrets.
The file is gitignored. The bot supports both xAI and Anthropic; pick
one with `[llm].provider`. Both providers have native server-side web
search, enabled automatically.

## What it does

When you `@Grok` (or whatever you name the bot) in your private Discord
server, the bot:

1. Reacts 👀 on your message to show it's working.
2. Figures out whether this is a new conversation or a continuation of an
   existing one (by checking if you replied to one of its past messages,
   or if you're in a thread it owns).
3. Calls the LLM with web search enabled, recording the prompt and every
   tool call into Postgres.
4. Replies inline (or auto-opens a thread for long answers), and
   transitions the reaction to ✅ on success / ❌ on failure.
5. On a new conversation, includes a link to the viewer where you can
   inspect what the model saw and what tools it ran.

## Privacy

Per-guild and configurable at runtime via slash commands:

- `/grok-privacy {in|out|status}` — per-user opt-in (Design 3).
- `/grok-mode {show|set}` — admins choose the design for this guild
  (one of `open`, `channel_only`, `opt_in`, `conversation_only`).

The default for guilds with no explicit mode is `opt_in`. See `CLAUDE.md`
for the four designs in detail.

## Storage

Everything lives in Postgres. Discord is just the I/O surface; no
state is recovered from channel scrolling. The data layer is
multi-tenant — every row reaches a `discord_guild_id`, so the bot can
serve multiple servers from one instance.
