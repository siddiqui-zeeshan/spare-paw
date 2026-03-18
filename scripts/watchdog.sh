#!/bin/bash
# spare-paw watchdog — monitors heartbeat file freshness
#
# Starts spare-paw, then loops every 30s checking:
#   1. Is the process alive? (PID file)
#   2. Is the heartbeat file fresh? (< MAX_AGE seconds old)
#
# If either check fails, the process is killed and restarted.
# Designed for Termux on Android with termux-wake-lock.

set -euo pipefail

CLAW_DIR="$HOME/.spare-paw"
HEARTBEAT_FILE="$CLAW_DIR/heartbeat"
PID_FILE="$CLAW_DIR/spare-paw.pid"
LOG_DIR="$CLAW_DIR/logs"
LOG_FILE="$LOG_DIR/watchdog.log"
MAX_AGE=90  # seconds before heartbeat is considered stale

mkdir -p "$CLAW_DIR" "$LOG_DIR"

log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $*" >> "$LOG_FILE"
    echo "[$timestamp] $*"
}

start_claw() {
    log "Acquiring Termux wake lock"
    termux-wake-lock 2>/dev/null || true

    log "Starting spare-paw gateway"
    python -m spare_paw gateway >> "$LOG_DIR/gateway-stdout.log" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    log "Started spare-paw with PID $pid"
}

stop_claw() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Killing spare-paw PID $pid"
            kill "$pid" 2>/dev/null || true
            # Wait briefly for graceful shutdown
            for i in 1 2 3 4 5; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            # Force kill if still alive
            if kill -0 "$pid" 2>/dev/null; then
                log "Force killing PID $pid"
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi
}

is_running() {
    if [ ! -f "$PID_FILE" ]; then
        return 1
    fi
    local pid
    pid=$(cat "$PID_FILE")
    if [ -z "$pid" ]; then
        return 1
    fi
    kill -0 "$pid" 2>/dev/null
}

heartbeat_stale() {
    if [ ! -f "$HEARTBEAT_FILE" ]; then
        return 0  # No heartbeat file = stale
    fi
    local file_age
    local now
    now=$(date +%s)
    file_mod=$(date -r "$HEARTBEAT_FILE" +%s 2>/dev/null || stat -c %Y "$HEARTBEAT_FILE" 2>/dev/null)
    if [ -z "$file_mod" ]; then
        return 0
    fi
    file_age=$((now - file_mod))
    if [ "$file_age" -gt "$MAX_AGE" ]; then
        return 0  # Stale
    fi
    return 1  # Fresh
}

# Trap SIGINT/SIGTERM to stop cleanly
cleanup() {
    log "Watchdog shutting down"
    stop_claw
    exit 0
}
trap cleanup SIGINT SIGTERM

log "Watchdog starting"

# Initial start
stop_claw  # Clean up any stale PID
start_claw

while true; do
    sleep 30

    if ! is_running; then
        log "Process not running — restarting"
        stop_claw
        start_claw
        continue
    fi

    if heartbeat_stale; then
        log "Heartbeat stale (>${MAX_AGE}s) — killing and restarting"
        stop_claw
        start_claw
    fi
done
