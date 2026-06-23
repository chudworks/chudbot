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

Vite serves on `:5173` and proxies `/api`, `/images`, `/videos`, `/avatars`,
and `/guild-icons` to the Rust server on `127.0.0.1:1860`.

Production deploy:

```sh
./serve.sh deploy
```

## Configuration

Copy `config.example.toml` to `config.toml`. The example is the reference for
all supported options: logging, database, web serving, storage, named
providers, platforms, agents, media generation bindings, memory, and subagents.

The runtime is agent-first. Agents select named provider services and model
specs; provider credentials live under `[llm.*]`, `[image.*]`, and `[video.*]`.

## Crates

- `chudbot-api`: shared contracts.
- `chudbot-bot`: platform-neutral bot runtime.
- `chudbot-discord`: Twilight platform adapter.
- `chudbot-web`: Axum viewer/API/SSE server.
- `chudbot-storage-sqlx`: Postgres storage.
- `chudbot-asset-local`, `chudbot-asset-s3`: media storage backends.
- `chudbot-xai`, `chudbot-gemini`, `chudbot-openai`,
  `chudbot-openai-compat`, `chudbot-anthropic`: provider crates.
- `chudbot-bin`: process launcher.

See `AGENTS.md` for repository conventions and maintenance notes.

## License

Copyright (C) 2026  Chud

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.

You should have received a copy of the GNU Affero General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.
