#!/usr/bin/env bash
# Production control script for the chudbot Discord bot.
#
# Deployment layout (on the host):
#   $CHUDBOT_DIR/
#     grok-discord-bot/    # this repo (a checkout, kept up to date with `git pull`)
#     chudbot              # the installed binary, copied from target/distribute/chudbot
#     config.toml          # the production config (gitignored in the repo)
#     skills/              # installed Vibe instruction skills referenced by config
#     frontend-build/      # built React bundle, copied from frontend/dist on deploy
#     vibe/                 # persistent Vibe bare repositories and build artifacts
#     images/, videos/     # media storage (per [storage] in config.toml)
#     avatars/             # cached Discord profile pictures
#     logs/                # tmux pane output, one file per service
#
# Default $CHUDBOT_DIR is $HOME/chudbot. Override with the CHUDBOT_DIR env var.
#
# The bot runs in a tmux session called "chudbot" with one window
# running `chudbot serve`, which combines the Discord gateway loop and
# the Axum web/API server in a single process. Output is tee'd to
# $CHUDBOT_DIR/logs/chudbot.log so crash output survives a window close.
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
VIBE_DATA="$CHUDBOT_DIR/vibe"
VIBE_SANDBOX_SRC="$REPO_DIR/vibe-sandbox"
VIBE_SKILL_SRC="$REPO_DIR/skills"
SKILL_DIR="$CHUDBOT_DIR/skills"
VIBE_DOCKER_NETWORK="chudbot-vibe"
VIBE_DOCKER_BRIDGE="chudbot-vibe0"
VIBE_FIREWALL_CHAIN="CHUDBOT_VIBE"
BINARY="$CHUDBOT_DIR/chudbot"
LOG_DIR="$CHUDBOT_DIR/logs"
SESSION="chudbot"
PROFILE="distribute"

usage() {
    cat <<USAGE
usage: $0 <command>

commands:
  deploy    git pull, ensure firewall, build, stop, migrate, install, start
  restart   restart the tmux session (no rebuild)
  start     start the session if not running
  stop      kill the tmux session
  status    show whether the session is running, with pids per window
  logs      attach to the session (Ctrl-b d to detach)
  migrate   run \`chudbot migrate\` with the installed binary
  vibe-purge <name>  permanently purge one Vibe site (operator-only)
  firewall-install   idempotently apply the Vibe Docker firewall rules
  firewall-check     verify the Vibe Docker network and firewall rules
  firewall-remove    stop Chudbot and remove the Vibe firewall rules/network

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

ensure_firewall_tools() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "error: Vibe firewall management supports Linux only" >&2
        exit 1
    fi
    local command
    for command in docker iptables sudo; do
        if ! command -v "$command" >/dev/null 2>&1; then
            echo "error: required command '$command' is not installed" >&2
            exit 1
        fi
    done
}

ensure_vibe_docker_network() {
    if docker network inspect "$VIBE_DOCKER_NETWORK" >/dev/null 2>&1; then
        local bridge
        bridge="$(docker network inspect --format '{{ index .Options "com.docker.network.bridge.name" }}' "$VIBE_DOCKER_NETWORK")"
        if [[ "$bridge" != "$VIBE_DOCKER_BRIDGE" ]]; then
            echo "error: Docker network '$VIBE_DOCKER_NETWORK' uses bridge '$bridge', expected '$VIBE_DOCKER_BRIDGE'" >&2
            exit 1
        fi
        return
    fi
    docker network create \
        --driver bridge \
        --opt "com.docker.network.bridge.name=$VIBE_DOCKER_BRIDGE" \
        "$VIBE_DOCKER_NETWORK" >/dev/null
}

remove_firewall_hook() {
    local tool="$1"
    while sudo "$tool" -C DOCKER-USER -i "$VIBE_DOCKER_BRIDGE" -j "$VIBE_FIREWALL_CHAIN" >/dev/null 2>&1; do
        sudo "$tool" -D DOCKER-USER -i "$VIBE_DOCKER_BRIDGE" -j "$VIBE_FIREWALL_CHAIN"
    done
}

apply_ipv4_firewall() {
    if ! sudo iptables -nL DOCKER-USER >/dev/null 2>&1; then
        echo "error: iptables has no DOCKER-USER chain; enable Docker's iptables firewall backend" >&2
        exit 1
    fi
    sudo iptables -N "$VIBE_FIREWALL_CHAIN" 2>/dev/null || true
    sudo iptables -F "$VIBE_FIREWALL_CHAIN"
    sudo iptables -A "$VIBE_FIREWALL_CHAIN" -d 127.0.0.0/8 -j REJECT
    sudo iptables -A "$VIBE_FIREWALL_CHAIN" -d 169.254.0.0/16 -j REJECT
    sudo iptables -A "$VIBE_FIREWALL_CHAIN" -d 100.64.0.0/10 -j REJECT
    sudo iptables -A "$VIBE_FIREWALL_CHAIN" -d 10.0.0.0/8 -j REJECT
    sudo iptables -A "$VIBE_FIREWALL_CHAIN" -d 172.16.0.0/12 -j REJECT
    sudo iptables -A "$VIBE_FIREWALL_CHAIN" -d 192.168.0.0/16 -j REJECT
    sudo iptables -A "$VIBE_FIREWALL_CHAIN" -j RETURN
    remove_firewall_hook iptables
    sudo iptables -I DOCKER-USER 1 -i "$VIBE_DOCKER_BRIDGE" -j "$VIBE_FIREWALL_CHAIN"
}

apply_ipv6_firewall_if_enabled() {
    if ! command -v ip6tables >/dev/null 2>&1 \
        || ! sudo ip6tables -nL DOCKER-USER >/dev/null 2>&1; then
        return
    fi
    sudo ip6tables -N "$VIBE_FIREWALL_CHAIN" 2>/dev/null || true
    sudo ip6tables -F "$VIBE_FIREWALL_CHAIN"
    sudo ip6tables -A "$VIBE_FIREWALL_CHAIN" -d ::1/128 -j REJECT
    sudo ip6tables -A "$VIBE_FIREWALL_CHAIN" -d fe80::/10 -j REJECT
    sudo ip6tables -A "$VIBE_FIREWALL_CHAIN" -d fc00::/7 -j REJECT
    sudo ip6tables -A "$VIBE_FIREWALL_CHAIN" -j RETURN
    remove_firewall_hook ip6tables
    sudo ip6tables -I DOCKER-USER 1 -i "$VIBE_DOCKER_BRIDGE" -j "$VIBE_FIREWALL_CHAIN"
}

check_firewall_rule() {
    local tool="$1"
    shift
    if ! sudo "$tool" -C "$VIBE_FIREWALL_CHAIN" "$@" >/dev/null 2>&1; then
        echo "error: missing Vibe firewall rule: $tool $VIBE_FIREWALL_CHAIN $*" >&2
        exit 1
    fi
}

check_ipv4_firewall() {
    if ! sudo iptables -nL DOCKER-USER >/dev/null 2>&1; then
        echo "error: iptables has no DOCKER-USER chain; enable Docker's iptables firewall backend" >&2
        exit 1
    fi
    sudo iptables -C DOCKER-USER -i "$VIBE_DOCKER_BRIDGE" -j "$VIBE_FIREWALL_CHAIN" >/dev/null
    check_firewall_rule iptables -d 127.0.0.0/8 -j REJECT
    check_firewall_rule iptables -d 169.254.0.0/16 -j REJECT
    check_firewall_rule iptables -d 100.64.0.0/10 -j REJECT
    check_firewall_rule iptables -d 10.0.0.0/8 -j REJECT
    check_firewall_rule iptables -d 172.16.0.0/12 -j REJECT
    check_firewall_rule iptables -d 192.168.0.0/16 -j REJECT
    check_firewall_rule iptables -j RETURN
}

check_ipv6_firewall_if_enabled() {
    if ! command -v ip6tables >/dev/null 2>&1 \
        || ! sudo ip6tables -nL DOCKER-USER >/dev/null 2>&1; then
        return
    fi
    sudo ip6tables -C DOCKER-USER -i "$VIBE_DOCKER_BRIDGE" -j "$VIBE_FIREWALL_CHAIN" >/dev/null
    check_firewall_rule ip6tables -d ::1/128 -j REJECT
    check_firewall_rule ip6tables -d fe80::/10 -j REJECT
    check_firewall_rule ip6tables -d fc00::/7 -j REJECT
    check_firewall_rule ip6tables -j RETURN
}

remove_firewall_family() {
    local tool="$1"
    if ! command -v "$tool" >/dev/null 2>&1; then
        return
    fi
    if sudo "$tool" -nL DOCKER-USER >/dev/null 2>&1; then
        remove_firewall_hook "$tool"
    fi
    if sudo "$tool" -nL "$VIBE_FIREWALL_CHAIN" >/dev/null 2>&1; then
        sudo "$tool" -F "$VIBE_FIREWALL_CHAIN"
        sudo "$tool" -X "$VIBE_FIREWALL_CHAIN"
    fi
}

cmd_firewall_install() {
    ensure_firewall_tools
    ensure_vibe_docker_network
    echo "==> apply Vibe firewall to $VIBE_DOCKER_NETWORK ($VIBE_DOCKER_BRIDGE)"
    apply_ipv4_firewall
    apply_ipv6_firewall_if_enabled
    cmd_firewall_check
}

cmd_firewall_check() {
    ensure_firewall_tools
    if ! docker network inspect "$VIBE_DOCKER_NETWORK" >/dev/null 2>&1; then
        echo "error: Docker network '$VIBE_DOCKER_NETWORK' is missing" >&2
        exit 1
    fi
    local bridge
    bridge="$(docker network inspect --format '{{ index .Options "com.docker.network.bridge.name" }}' "$VIBE_DOCKER_NETWORK")"
    if [[ "$bridge" != "$VIBE_DOCKER_BRIDGE" ]]; then
        echo "error: Docker network '$VIBE_DOCKER_NETWORK' uses bridge '$bridge', expected '$VIBE_DOCKER_BRIDGE'" >&2
        exit 1
    fi
    check_ipv4_firewall
    check_ipv6_firewall_if_enabled
    echo "Vibe firewall is active on $VIBE_DOCKER_NETWORK ($VIBE_DOCKER_BRIDGE)"
}

cmd_firewall_remove() {
    ensure_firewall_tools
    echo "==> stop Chudbot before removing its Vibe firewall"
    stop_session
    if docker network inspect "$VIBE_DOCKER_NETWORK" >/dev/null 2>&1; then
        local attached
        attached="$(docker network inspect --format '{{ len .Containers }}' "$VIBE_DOCKER_NETWORK")"
        if [[ "$attached" != "0" ]]; then
            echo "error: $attached container(s) remain attached to $VIBE_DOCKER_NETWORK; firewall was not removed" >&2
            exit 1
        fi
    fi
    remove_firewall_family iptables
    remove_firewall_family ip6tables
    if docker network inspect "$VIBE_DOCKER_NETWORK" >/dev/null 2>&1; then
        docker network rm "$VIBE_DOCKER_NETWORK" >/dev/null
    fi
    echo "==> Vibe firewall removed"
}

start_session() {
    ensure_binary
    if session_alive; then
        echo "session $SESSION already running"
        return
    fi
    mkdir -p "$LOG_DIR"

    # `exec` so `chudbot` replaces the left side of the pipe -- when it
    # exits, the pipeline (and the single-pane session) ends. `tee -a`
    # keeps output visible in the pane AND persists it to a log file.
    #
    # The `trap '' INT` on the tee side is deliberate: `stop` sends
    # Ctrl-C, which the PTY delivers as SIGINT to the WHOLE foreground
    # group (chudbot + tee). Without the trap, tee would die instantly and
    # every line chudbot logs during its 30s graceful drain would vanish
    # into a broken pipe. Ignoring SIGINT on tee (SIG_IGN survives the
    # exec) keeps it alive until chudbot finishes draining and closes the
    # pipe, so the shutdown is fully captured in the log.
    tmux new-session -d -s "$SESSION" -n chudbot -c "$CHUDBOT_DIR" \
        "exec $BINARY --config $CHUDBOT_DIR/config.toml serve 2>&1 | { trap '' INT; exec tee -a $LOG_DIR/chudbot.log; }"
    echo "started session $SESSION (running: chudbot serve)"
    echo "logs: $LOG_DIR/chudbot.log"
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
    # it into SIGINT for the foreground process group (chudbot + tee),
    # which `chudbot serve` catches and drains in-flight work for up to
    # 30s before exiting. When chudbot exits the pipeline ends and the
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

build_vibe_sandbox() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "error: docker is not on PATH -- Vibe requires the system Docker daemon" >&2
        exit 1
    fi
    if [[ ! -S /var/run/docker.sock ]]; then
        echo "error: Docker socket not found at /var/run/docker.sock" >&2
        exit 1
    fi
    echo "==> build pinned Vibe sandbox image"
    docker build --pull --tag chudbot-vibe-sandbox:latest "$VIBE_SANDBOX_SRC"
    local image_id
    image_id="$(docker image inspect --format '{{.Id}}' chudbot-vibe-sandbox:latest)"
    if [[ -z "$image_id" ]]; then
        echo "error: Vibe sandbox image build produced no inspectable image id" >&2
        exit 1
    fi
    echo "==> Vibe sandbox image: $image_id"
    mkdir -p "$VIBE_DATA/repos" "$VIBE_DATA/workspaces" "$VIBE_DATA/artifacts"
    local template_stage="$VIBE_DATA/.template.new"
    rm -rf "$template_stage"
    cp -R "$VIBE_SANDBOX_SRC/template" "$template_stage"
    local template_previous="$VIBE_DATA/.template.old"
    rm -rf "$template_previous"
    if [[ -d "$VIBE_DATA/template" ]]; then
        mv "$VIBE_DATA/template" "$template_previous"
    fi
    mv "$template_stage" "$VIBE_DATA/template"
    rm -rf "$template_previous"

    # Config lives one directory above the repository and resolves skill paths
    # relative to itself. Install the two versioned Vibe skills alongside that
    # config without replacing unrelated operator-managed skills.
    mkdir -p "$SKILL_DIR"
    local skill
    for skill in vibe.md vibe-conversation.md; do
        if [[ ! -f "$VIBE_SKILL_SRC/$skill" ]]; then
            echo "error: required Vibe skill not found at $VIBE_SKILL_SRC/$skill" >&2
            exit 1
        fi
        cp "$VIBE_SKILL_SRC/$skill" "$SKILL_DIR/$skill.new"
        chmod 644 "$SKILL_DIR/$skill.new"
        mv "$SKILL_DIR/$skill.new" "$SKILL_DIR/$skill"
    done
    echo "==> Vibe skills installed to $SKILL_DIR"
    echo "==> persistent Vibe data remains at $VIBE_DATA"
}

cmd_deploy() {
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        echo "error: $REPO_DIR is not a git checkout" >&2
        exit 1
    fi

    echo "==> git pull --ff-only"
    git -C "$REPO_DIR" pull --ff-only

    cmd_firewall_install
    build_frontend
    build_vibe_sandbox

    echo "==> cargo build --locked --profile $PROFILE"
    (cd "$REPO_DIR" && cargo build --locked --profile "$PROFILE" -p chudbot-bin)
    local built="$REPO_DIR/target/$PROFILE/chudbot"
    if [[ ! -x "$built" ]]; then
        echo "error: cargo build did not produce $built" >&2
        exit 1
    fi

    echo "==> check config"
    (cd "$CHUDBOT_DIR" && "$built" --config "$CHUDBOT_DIR/config.toml" check-config)

    stop_session

    echo "==> migrate"
    (cd "$CHUDBOT_DIR" && "$built" --config "$CHUDBOT_DIR/config.toml" migrate)

    echo "==> install binary -> $BINARY"
    # Stage then rename so the swap is atomic on the same filesystem;
    # avoids ever leaving a half-written binary at $BINARY.
    cp "$built" "$BINARY.new"
    chmod 755 "$BINARY.new"
    mv "$BINARY.new" "$BINARY"

    start_session
    echo "==> deploy complete"
    echo "==> Vibe remains private only if the DGX firewall and Cloudflare no-cache rule from docs/vibe-operator-runbook.md are active"
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
    (cd "$CHUDBOT_DIR" && "$BINARY" --config "$CHUDBOT_DIR/config.toml" migrate)
}

cmd_vibe_purge() {
    ensure_binary
    local name="${1:-}"
    if [[ -z "$name" ]]; then
        echo "error: vibe-purge requires an exact site name" >&2
        exit 1
    fi
    (cd "$CHUDBOT_DIR" && "$BINARY" --config "$CHUDBOT_DIR/config.toml" vibe purge "$name")
}

case "${1:-}" in
    deploy)         cmd_deploy ;;
    restart)        cmd_restart ;;
    start)          start_session ;;
    stop)           stop_session ;;
    status)         cmd_status ;;
    logs)           cmd_logs ;;
    migrate)        cmd_migrate ;;
    vibe-purge)     cmd_vibe_purge "${2:-}" ;;
    firewall-install) cmd_firewall_install ;;
    firewall-check)   cmd_firewall_check ;;
    firewall-remove)  cmd_firewall_remove ;;
    -h|--help|help|"") usage ;;
    *)              echo "unknown command: $1" >&2; usage; exit 1 ;;
esac
