#!/usr/bin/env bash
#
# Start the Solidus web app (gunicorn) in the background with a pidfile.
# Wrapper around: python run_web.py
# Stops with: scripts/web_stop.sh
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="${CASTSIM_WEB_PIDFILE:-$PROJECT_DIR/logs/web.pid}"
LOGFILE="${CASTSIM_WEB_LOGFILE:-$PROJECT_DIR/logs/web.log}"
PORT="${CASTSIM_WEB_PORT:-5050}"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Solidus web app already running (pid $(cat "$PIDFILE"))"
    exit 0
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $PORT is already in use; not starting."
    exit 1
fi

rm -f "$PIDFILE"
mkdir -p "$(dirname "$PIDFILE")"

PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

cd "$PROJECT_DIR"
nohup "$PYTHON_BIN" run_web.py --port "$PORT" >>"$LOGFILE" 2>&1 &
echo $! >"$PIDFILE"

for _ in $(seq 1 20); do
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Solidus web app started (pid $(cat "$PIDFILE"), http://localhost:$PORT)"
        echo "Logs: $LOGFILE"
        exit 0
    fi
    sleep 0.5
done

echo "Solidus web app failed to start within 10s — check $LOGFILE"
kill "$(cat "$PIDFILE")" 2>/dev/null || true
rm -f "$PIDFILE"
exit 1
