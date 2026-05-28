#!/usr/bin/env bash
# Production control script for the chudbot Discord bot.
#
# Deployment layout (on the host):
#   $CHUDBOT_DIR/
#     grok-discord-bot/    # this repo (a checkout, kept up to date with `git pull`)
#     grok                 # the installed binary, copied from target/distribute/grok
#     config.toml          # the production config (gitignored in the repo)
#     images/, videos/     # media storage (per [storage] in config.toml)
#     logs/                # tmux pane output, one file per service
#
# Default $CHUDBOT_DIR is $HOME/chudbot. Override with the CHUDBOT_DIR env var.
#
# The bot runs in a tmux session called "chudbot" with two windows:
#   - "bot": `grok bot` (Discord gateway loop)
#   - "web": `grok web` (Axum conversation viewer)
# Both windows run with cwd $CHUDBOT_DIR so the binary picks up config.toml
# and storage dirs by relative path. Output of each window is tee'd to
# $CHUDBOT_DIR/logs/{bot,web}.log so crash output survives a window close.

set -euo pipefail

CHUDBOT_DIR="${CHUDBOT_DIR:-$HOME/chudbot}"
REPO_DIR="$CHUDBOT_DIR/grok-discord-bot"
BINARY="$CHUDBOT_DIR/grok"
LOG_DIR="$CHUDBOT_DIR/logs"
SESSION="chudbot"
PROFILE="distribute"

usage() {
    cat <<USAGE
usage: $0 <command>

commands:
  deploy    git pull, build, stop, migrate, install binary, start
  restart   restart the tmux session (no rebuild)
  start     start the session if not running
  stop      kill the tmux session
  status    show whether the session is running, with pids per window
  logs      attach to the session (Ctrl-b d to detach, Ctrl-b n to switch)
  migrate   run \`grok migrate\` with the installed binary

env vars:
  CHUDBOT_DIR    deployment root (default: \$HOME/chudbot)
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
    # `exec` so the bot replaces the shell -- when the bot exits, the
    # pane exits with the bot's status (no zombie shell). `tee -a` keeps
    # output visible in the pane AND persists it to a log file for
    # post-mortem after a crash closes the window.
    tmux new-session -d -s "$SESSION" -n bot -c "$CHUDBOT_DIR" \
        "exec $BINARY bot 2>&1 | tee -a $LOG_DIR/bot.log"
    tmux new-window -t "$SESSION" -n web -c "$CHUDBOT_DIR" \
        "exec $BINARY web 2>&1 | tee -a $LOG_DIR/web.log"
    echo "started session $SESSION (windows: bot, web)"
    echo "logs: $LOG_DIR/{bot,web}.log"
    echo "attach with: $0 logs"
}

stop_session() {
    if session_alive; then
        tmux kill-session -t "$SESSION"
        echo "stopped session $SESSION"
    else
        echo "session $SESSION not running"
    fi
}

cmd_deploy() {
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        echo "error: $REPO_DIR is not a git checkout" >&2
        exit 1
    fi

    echo "==> git pull --ff-only"
    git -C "$REPO_DIR" pull --ff-only

    echo "==> cargo build --profile $PROFILE"
    (cd "$REPO_DIR" && cargo build --profile "$PROFILE")
    local built="$REPO_DIR/target/$PROFILE/grok"
    if [[ ! -x "$built" ]]; then
        echo "error: build did not produce $built" >&2
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
