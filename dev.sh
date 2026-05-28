#!/usr/bin/env bash
# Convenience wrapper for common dev tasks. Run `./dev.sh` for usage.

set -euo pipefail

cd "$(dirname "$0")"

cmd="${1:-}"
shift || true

case "$cmd" in
    build)
        cargo build "$@"
        ;;
    test)
        cargo test "$@"
        ;;
    run)
        # Forwards to the `grok` binary's subcommands.
        # Examples:
        #   ./dev.sh run serve
        #   ./dev.sh run migrate
        #   ./dev.sh run -- --config /path/to/config.toml serve
        cargo run -- "$@"
        ;;
    lint)
        cargo fmt --all --check
        cargo clippy --all-targets --all-features -- -D warnings -A dead_code -A unused
        # Frontend: ESLint + tsc --noEmit. We require bun here because
        # the frontend toolchain is the source of truth (matches what
        # `serve.sh deploy` runs). If you haven't installed bun, you
        # can skip with FRONTEND_LINT=skip.
        if [[ "${FRONTEND_LINT:-}" == "skip" ]]; then
            echo "frontend lint: skipped (FRONTEND_LINT=skip)"
        elif command -v bun >/dev/null 2>&1; then
            echo "==> frontend lint"
            (cd frontend && bun install --frozen-lockfile && bun run lint && bun run typecheck)
        else
            echo "frontend lint: bun not on PATH -- install from https://bun.sh, or set FRONTEND_LINT=skip" >&2
            exit 1
        fi
        ;;
    fmt)
        cargo fmt --all
        ;;
    *)
        cat >&2 <<USAGE
usage: ./dev.sh <command> [args...]

commands:
  build [...]    cargo build (forwards args)
  test  [...]    cargo test  (forwards args; e.g. ./dev.sh test bot::)
  run   <sub>    cargo run -- <sub>  (sub: serve | migrate)
  lint           cargo fmt --check + cargo clippy + frontend ESLint + tsc
                 (set FRONTEND_LINT=skip to skip the frontend pass)
  fmt            cargo fmt (writes)
USAGE
        exit 1
        ;;
esac
