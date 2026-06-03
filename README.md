# Chudbot

Discord bot + trace viewer for model-backed agents. Chudbot records each
conversation turn in Postgres, including model input, client tool results,
provider-side tool traces, usage records, media references, and final replies.

## Run

```sh
cargo run -p chudbot-bin -- check-config
cargo run -p chudbot-bin -- migrate
cargo run -p chudbot-bin -- serve
```

All subcommands accept `--config <path>`; the default is `config.toml`.

For frontend iteration:

```sh
cd frontend
bun install
bun run dev
```

Vite serves on `:5173` and proxies `/api`, `/images`, `/videos`, and
`/avatars` to the Rust server on `127.0.0.1:1860`.

Production deploy:

```sh
./serve.sh deploy
```

## Configuration

Copy `config.example.toml` to `config.toml`. The example is the reference for
all supported v2 options: logging, database, web serving, storage, default
privacy, named providers, platforms, agents, media generation bindings, and
subagents.

The runtime is agent-first. Agents select named provider services and model
specs; provider credentials live under `[llm.*]`, `[image.*]`, and `[video.*]`.

## Crates

- `chudbot-api`: shared contracts.
- `chudbot-bot`: platform-neutral bot runtime.
- `chudbot-discord`: Twilight platform adapter.
- `chudbot-web`: Axum viewer/API/SSE server.
- `chudbot-storage-sqlx`: Postgres storage.
- `chudbot-asset-local`: local media storage.
- `chudbot-xai`, `chudbot-openai`, `chudbot-anthropic`: provider crates.
- `chudbot-bin`: process launcher.

See `AGENTS.md` and `docs/2.0-api-shapes.md` for migration details.
