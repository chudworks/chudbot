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
        #   ./dev.sh run bot
        #   ./dev.sh run web
        #   ./dev.sh run migrate
        #   ./dev.sh run -- --config /path/to/config.toml bot
        cargo run -- "$@"
        ;;
    lint)
        cargo fmt --all --check
        cargo clippy --all-targets --all-features -- -D warnings -A dead_code -A unused
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
  run   <sub>    cargo run -- <sub>  (sub: bot | web | migrate)
  lint           cargo fmt --check && cargo clippy
  fmt            cargo fmt (writes)
USAGE
        exit 1
        ;;
esac
