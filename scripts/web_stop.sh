#!/usr/bin/env bash
#
# Stop the Solidus web app started by scripts/web_start.sh.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="${CASTSIM_WEB_PIDFILE:-$PROJECT_DIR/logs/web.pid}"

if [[ ! -f "$PIDFILE" ]]; then
    echo "No pidfile at $PIDFILE — nothing to stop."
    exit 0
fi

PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "Process $PID is not running; removing stale pidfile."
    rm -f "$PIDFILE"
    exit 0
fi

kill -TERM "$PID"
for _ in $(seq 1 20); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Solidus web app stopped (was pid $PID)."
        rm -f "$PIDFILE"
        exit 0
    fi
    sleep 0.5
done

kill -KILL "$PID" 2>/dev/null || true
rm -f "$PIDFILE"
echo "Solidus web app killed (was pid $PID)."
