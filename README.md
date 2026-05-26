# grok-discord-bot

Discord bot that integrates xAI Grok, with a companion web viewer for
conversation history. Reply or @mention the bot in Discord; conversations
are stored in Postgres and surfaced at unguessable URLs in the web UI,
including the exact context fed to Grok and every tool call.

## Run

```sh
grok bot       # start the Discord gateway loop
grok web       # start the Axum web viewer
grok migrate   # apply pending Postgres migrations
```

Configuration is via environment variables:

| Var | Purpose |
|-----|---------|
| `DISCORD_TOKEN` | Discord bot token |
| `XAI_API_KEY` | xAI Grok API key |
| `POSTGRES_URL` | Postgres connection URL |
| `WEB_BASE_URL` | Public base URL of the viewer (for links posted in Discord) |
| `GROK_MODEL` | Grok model id (default `grok-4.1-fast`) |
