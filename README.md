# Chudbot

Discord bot, trace viewer, and website builder for model-backed agents.
Chudbot records each conversation turn in Postgres, including model input,
client tool results, provider-side tool traces, usage records, media references,
and final replies. Its optional Vibe runtime lets Discord users create and
maintain small React sites through a sandboxed coding agent.

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

The production helper targets Linux. It applies the repository-managed Docker
firewall, builds the pinned sandbox image, installs the Vibe skills and
template, builds Chudbot, validates configuration, runs migrations, and
restarts the tmux service. It does not install a systemd unit or write a helper
into a global system path.

Useful production commands:

```sh
./serve.sh status
./serve.sh firewall-install
./serve.sh firewall-check
./serve.sh firewall-remove
./serve.sh vibe-purge <site-name>
```

## Configuration

Copy `config.example.toml` to `config.toml`. The example is the reference for
all supported options: logging, database, web serving, storage, named
providers, platforms, agents, media generation bindings, memory, and subagents.

The runtime is agent-first. Agents select named provider services and model
specs; provider credentials live under `[llm.*]`, `[image.*]`, `[video.*]`, and
`[audio.*]`.

`[bot.agents.<name>].client_tools` is a strict allowlist over tools registered
by the runtime. It does not add to the default tool set. Omit it when an agent
should receive every configured runtime tool; otherwise include every normal,
media, memory, and subagent tool that agent must retain.

## Vibe Sites

Vibe sites use local bare Git repositories for source history and immutable
built artifacts for serving. Sites are `🔒 protected` by default and authenticate
with Discord OAuth before verifying current guild membership through the bot.
Owners can make deployed sites public through the conversation tool; public
viewers are anonymous, while source/history stays protected. The coding
container receives one workspace bind mount, no Docker socket or Chudbot
secrets, a read-only root filesystem, dropped capabilities, and bounded CPU,
memory, PIDs, output, and runtime.

The default conversation agent loads `vibe_conversation` and exposes a `vibe`
subagent bound to `vibe_coder`. The coding agent loads the `vibe` skill and has
exactly `read`, `edit`, and `shell`; its `read` and `edit` operations execute in
the container namespace. Generated dependency and build paths such as
`node_modules`, `dist`, and TypeScript build metadata are neither committed nor
shown by the source viewer.

See [config.example.toml](config.example.toml) for the complete configuration
and [docs/vibe-operator-runbook.md](docs/vibe-operator-runbook.md) for the DGX
Spark firewall, Docker, Cloudflare, OAuth, backup, and smoke-test procedure.

## Crates

- `chudbot-api`: shared contracts.
- `chudbot-bot`: platform-neutral bot runtime.
- `chudbot-discord`: Twilight platform adapter.
- `chudbot-web`: Axum viewer/API/SSE server.
- `chudbot-storage-sqlx`: Postgres storage.
- `chudbot-asset-local`, `chudbot-asset-s3`: media storage backends.
- `chudbot-vibe`: Vibe access rules, Git/artifact storage, sandbox execution,
  coding tools, names, and site runtime configuration.
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
