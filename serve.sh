#!/usr/bin/env bash
# Production control script for the chudbot Discord bot.
#
# Deployment layout (on the host):
#   $CHUDBOT_DIR/
#     grok-discord-bot/    # this repo (a checkout, kept up to date with `git pull`)
#     grok                 # the installed binary, copied from target/distribute/grok
#     config.toml          # the production config (gitignored in the repo)
#     .env                 # optional; KEY=value lines exported into the bot's tmux session
#     frontend-build/      # built React bundle, copied from frontend/dist on deploy
#     images/, videos/     # media storage (per [storage] in config.toml)
#     avatars/             # cached Discord profile pictures
#     logs/                # tmux pane output, one file per service
#
# Default $CHUDBOT_DIR is $HOME/chudbot. Override with the CHUDBOT_DIR env var.
#
# The bot runs in a tmux session called "chudbot" with one window
# running `grok serve`, which combines the Discord gateway loop and
# the Axum web/API server in a single process. Output is tee'd to
# $CHUDBOT_DIR/logs/grok.log so crash output survives a window close.
#
# The frontend is a React + Vite SPA. `deploy` runs `bun install` and
# `bun run build` inside grok-discord-bot/frontend/, then atomically
# swaps the resulting dist/ into $CHUDBOT_DIR/frontend-build/.
# Axum serves that directory as static files (with index.html as the
# SPA fallback for client-side routes like /c/<uuid>).

set -euo pipefail

CHUDBOT_DIR="${CHUDBOT_DIR:-$HOME/chudbot}"
REPO_DIR="$CHUDBOT_DIR/grok-discord-bot"
FRONTEND_SRC="$REPO_DIR/frontend"
FRONTEND_BUILD="$CHUDBOT_DIR/frontend-build"
BINARY="$CHUDBOT_DIR/grok"
LOG_DIR="$CHUDBOT_DIR/logs"
ENV_FILE="$CHUDBOT_DIR/.env"
SESSION="chudbot"
PROFILE="distribute"

usage() {
    cat <<USAGE
usage: $0 <command>

commands:
  deploy    git pull, build frontend, build binary, stop, migrate, install, start
  restart   restart the tmux session (no rebuild)
  start     start the session if not running
  stop      kill the tmux session
  status    show whether the session is running, with pids per window
  logs      attach to the session (Ctrl-b d to detach)
  migrate   run \`grok migrate\` with the installed binary

env vars:
  CHUDBOT_DIR    deployment root (default: \$HOME/chudbot)

files:
  \$CHUDBOT_DIR/.env   optional; if present, its KEY=value lines are
                      exported into the bot's tmux session on start
USAGE
}

session_alive() {
    tmux has-session -t "$SESSION" 2>/dev/null
}

ensure_binary() {
    if [[ ! -x "$BINARY" ]]; then
        echo "error: binary not found at $BINARY -- run '$0 deploy' first" >&2
        exit 1
    fi
}

start_session() {
    ensure_binary
    if session_alive; then
        echo "session $SESSION already running"
        return
    fi
    mkdir -p "$LOG_DIR"

    # Optionally load $CHUDBOT_DIR/.env into the pane's shell so `grok`
    # inherits its KEY=value pairs. We source it INSIDE the tmux command
    # rather than in this script's shell because tmux's `update-environment`
    # only refreshes a fixed allowlist of vars into new sessions -- arbitrary
    # vars exported out here would not reliably reach the pane. `set -a`
    # (allexport) makes every assignment in the file an exported var, then
    # `set +a` restores normal behavior before `exec`.
    local env_prefix=""
    if [[ -f "$ENV_FILE" ]]; then
        echo "loading environment from $ENV_FILE"
        env_prefix="set -a; . $ENV_FILE; set +a; "
    fi

    # `exec` so `grok` replaces the left side of the pipe -- when it
    # exits, the pipeline (and the single-pane session) ends. `tee -a`
    # keeps output visible in the pane AND persists it to a log file.
    #
    # The `trap '' INT` on the tee side is deliberate: `stop` sends
    # Ctrl-C, which the PTY delivers as SIGINT to the WHOLE foreground
    # group (grok + tee). Without the trap, tee would die instantly and
    # every line grok logs during its 30s graceful drain would vanish
    # into a broken pipe. Ignoring SIGINT on tee (SIG_IGN survives the
    # exec) keeps it alive until grok finishes draining and closes the
    # pipe, so the shutdown is fully captured in the log.
    tmux new-session -d -s "$SESSION" -n grok -c "$CHUDBOT_DIR" \
        "${env_prefix}exec $BINARY serve 2>&1 | { trap '' INT; exec tee -a $LOG_DIR/grok.log; }"
    echo "started session $SESSION (running: grok serve)"
    echo "logs: $LOG_DIR/grok.log"
    echo "attach with: $0 logs"
}

# Seconds to wait for a graceful drain before force-killing. Must be
# comfortably above the binary's own SHUTDOWN_GRACE (30s) so the app
# gets its full drain window plus a margin for teardown.
STOP_TIMEOUT=40

stop_session() {
    if ! session_alive; then
        echo "session $SESSION not running"
        return
    fi
    # `tmux kill-session` tears down the PTY and the processes get
    # SIGHUP, which the binary does NOT treat as a graceful shutdown.
    # Send an actual Ctrl-C instead: the pane's line discipline turns
    # it into SIGINT for the foreground process group (grok + tee),
    # which `grok serve` catches and drains in-flight work for up to
    # 30s before exiting. When grok exits the pipeline ends and the
    # single-pane session closes on its own.
    echo "sending Ctrl-C to $SESSION (graceful drain, up to ${STOP_TIMEOUT}s)..."
    tmux send-keys -t "$SESSION" C-c || true

    local waited=0
    while session_alive; do
        if (( waited >= STOP_TIMEOUT )); then
            echo "still running after ${STOP_TIMEOUT}s; force-killing session"
            tmux kill-session -t "$SESSION" || true
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "stopped session $SESSION"
}

build_frontend() {
    if [[ ! -d "$FRONTEND_SRC" ]]; then
        echo "error: $FRONTEND_SRC not found" >&2
        exit 1
    fi
    if ! command -v bun >/dev/null 2>&1; then
        echo "error: bun is not on PATH -- install from https://bun.sh" >&2
        exit 1
    fi
    echo "==> bun install (frontend)"
    (cd "$FRONTEND_SRC" && bun install --frozen-lockfile)
    echo "==> bun run build (frontend)"
    (cd "$FRONTEND_SRC" && bun run build)
    if [[ ! -f "$FRONTEND_SRC/dist/index.html" ]]; then
        echo "error: vite build did not produce $FRONTEND_SRC/dist/index.html" >&2
        exit 1
    fi

    # Atomic swap: stage the new build under a sibling name, mv the
    # current one out of the way, mv the new one in, then rm the
    # previous. Tracker-and-swap pattern so the bot's ServeDir doesn't
    # see a partially-copied tree at any instant.
    local stage="$CHUDBOT_DIR/.frontend-build.new"
    rm -rf "$stage"
    cp -R "$FRONTEND_SRC/dist" "$stage"

    local previous="$CHUDBOT_DIR/.frontend-build.old"
    rm -rf "$previous"
    if [[ -d "$FRONTEND_BUILD" ]]; then
        mv "$FRONTEND_BUILD" "$previous"
    fi
    mv "$stage" "$FRONTEND_BUILD"
    rm -rf "$previous"
    echo "==> frontend installed to $FRONTEND_BUILD"
}

cmd_deploy() {
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        echo "error: $REPO_DIR is not a git checkout" >&2
        exit 1
    fi

    echo "==> git pull --ff-only"
    git -C "$REPO_DIR" pull --ff-only

    build_frontend

    echo "==> cargo build --profile $PROFILE"
    (cd "$REPO_DIR" && cargo build --profile "$PROFILE")
    local built="$REPO_DIR/target/$PROFILE/grok"
    if [[ ! -x "$built" ]]; then
        echo "error: cargo build did not produce $built" >&2
        exit 1
    fi

    stop_session

    echo "==> migrate"
    (cd "$CHUDBOT_DIR" && "$built" migrate)

    echo "==> install binary -> $BINARY"
    # Stage then rename so the swap is atomic on the same filesystem;
    # avoids ever leaving a half-written binary at $BINARY.
    cp "$built" "$BINARY.new"
    chmod 755 "$BINARY.new"
    mv "$BINARY.new" "$BINARY"

    start_session
    echo "==> deploy complete"
}

cmd_restart() {
    stop_session
    start_session
}

cmd_status() {
    if session_alive; then
        echo "session $SESSION: running"
        tmux list-windows -t "$SESSION" \
            -F '  window #{window_index} (#{window_name}): pid=#{pane_pid} cmd=#{pane_current_command}'
    else
        echo "session $SESSION: not running"
    fi
}

cmd_logs() {
    if ! session_alive; then
        echo "session $SESSION: not running" >&2
        exit 1
    fi
    tmux attach -t "$SESSION"
}

cmd_migrate() {
    ensure_binary
    (cd "$CHUDBOT_DIR" && "$BINARY" migrate)
}

case "${1:-}" in
    deploy)         cmd_deploy ;;
    restart)        cmd_restart ;;
    start)          start_session ;;
    stop)           stop_session ;;
    status)         cmd_status ;;
    logs)           cmd_logs ;;
    migrate)        cmd_migrate ;;
    -h|--help|help|"") usage ;;
    *)              echo "unknown command: $1" >&2; usage; exit 1 ;;
esac
