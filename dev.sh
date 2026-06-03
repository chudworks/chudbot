#!/usr/bin/env bash
# Convenience wrapper for common dev tasks. Run `./dev.sh` for usage.

set -euo pipefail

cd "$(dirname "$0")"

CONFIG="${CONFIG:-config.toml}"
PROFILE="dev"
PROFILE_DIR="debug"
FRONTEND_BUILD_DIR="${FRONTEND_BUILD_DIR:-frontend-build}"

cmd="${1:-}"
shift || true

build_frontend() {
    if ! command -v bun >/dev/null 2>&1; then
        echo "error: bun is not on PATH -- install from https://bun.sh" >&2
        exit 1
    fi
    echo "==> bun install (frontend)"
    (cd frontend && bun install --frozen-lockfile)
    echo "==> bun run build (frontend)"
    (cd frontend && bun run build)
    if [[ ! -f frontend/dist/index.html ]]; then
        echo "error: vite build did not produce frontend/dist/index.html" >&2
        exit 1
    fi

    local stage="${FRONTEND_BUILD_DIR}.new"
    local previous="${FRONTEND_BUILD_DIR}.old"
    rm -rf "$stage" "$previous"
    cp -R frontend/dist "$stage"
    if [[ -d "$FRONTEND_BUILD_DIR" ]]; then
        mv "$FRONTEND_BUILD_DIR" "$previous"
    fi
    mv "$stage" "$FRONTEND_BUILD_DIR"
    rm -rf "$previous"
    echo "==> frontend installed to $FRONTEND_BUILD_DIR"
}

build_dev_binary() {
    echo "==> cargo build --profile $PROFILE -p chudbot-bin"
    cargo build --profile "$PROFILE" -p chudbot-bin
    local built="target/$PROFILE_DIR/chudbot"
    if [[ ! -x "$built" ]]; then
        echo "error: cargo build did not produce $built" >&2
        exit 1
    fi
}

case "$cmd" in
    build)
        cargo build "$@"
        ;;
    serve)
        # Build frontend + dev binary, validate config, then run in the foreground.
        # Override defaults with CONFIG=/path/to/config.toml or FRONTEND_BUILD_DIR=...
        build_frontend
        build_dev_binary
        echo "==> check config ($CONFIG)"
        "target/$PROFILE_DIR/chudbot" --config "$CONFIG" check-config
        echo "==> serve ($CONFIG)"
        exec "target/$PROFILE_DIR/chudbot" --config "$CONFIG" serve "$@"
        ;;
    migrate)
        # Build the dev binary first so migrations use the same artifact as serve.
        build_dev_binary
        echo "==> migrate ($CONFIG)"
        "target/$PROFILE_DIR/chudbot" --config "$CONFIG" migrate "$@"
        ;;
    test)
        cargo test "$@"
        ;;
    run)
        # Forwards to the `chudbot` binary's subcommands.
        # Examples:
        #   ./dev.sh run serve
        #   ./dev.sh run migrate
        #   ./dev.sh run --config /path/to/config.toml serve
        cargo run -p chudbot-bin -- "$@"
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
  serve [...]    build frontend + dev binary, then run chudbot serve
                 env: CONFIG=config.toml FRONTEND_BUILD_DIR=frontend-build
  migrate [...]  build dev binary, then run chudbot migrate
                 env: CONFIG=config.toml
  test  [...]    cargo test  (forwards args; e.g. ./dev.sh test bot::)
  run   [...]    cargo run -p chudbot-bin -- ... (forwards args)
                 e.g. ./dev.sh run --config config.example.toml check-config
  lint           cargo fmt --check + cargo clippy + frontend ESLint + tsc
                 (set FRONTEND_LINT=skip to skip the frontend pass)
  fmt            cargo fmt (writes)
USAGE
        exit 1
        ;;
esac
