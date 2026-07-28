#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INSTALL_DEPENDENCIES=false
KEEP_ARTIFACTS="${KEEP_SMOKE_ARTIFACTS:-0}"
for argument in "$@"; do
  case "$argument" in
    --install)
      INSTALL_DEPENDENCIES=true
      ;;
    --keep-artifacts)
      KEEP_ARTIFACTS=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/runtime-smoke.sh [--install] [--keep-artifacts]

Starts an isolated demo-mode API and production dashboard, verifies critical
HTTP, CORS, WebSocket, control, version, and reconciliation paths, then cleans
up all processes and temporary data.

--install         Install the editable Python package and dashboard dependencies.
                  Requires an active virtual environment.
--keep-artifacts  Preserve temporary databases and logs after the run.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $argument" >&2
      exit 2
      ;;
  esac
done

for command in python node npm curl setsid; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 1
  fi
done

python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ is required, found {sys.version.split()[0]}")
PY

node - <<'JS'
const major = Number(process.versions.node.split('.')[0]);
if (major < 22) {
  console.error(`Node.js 22+ is required, found ${process.versions.node}`);
  process.exit(1);
}
JS

if $INSTALL_DEPENDENCIES; then
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "--install requires an active Python virtual environment" >&2
    exit 1
  fi
  python -m pip install -e '.[dev]'
  (
    cd apps/dashboard
    npm install --no-audit --no-fund
  )
fi

python - <<'PY'
import importlib.util
missing = [name for name in ("trading_app", "uvicorn", "websockets") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing) + ". Run with --install.")
PY

if [[ ! -d apps/dashboard/node_modules ]]; then
  echo "Dashboard dependencies are missing. Run with --install." >&2
  exit 1
fi

free_port() {
  python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

API_PORT="${TRADING_SMOKE_API_PORT:-$(free_port)}"
DASHBOARD_PORT="${TRADING_SMOKE_DASHBOARD_PORT:-$(free_port)}"
while [[ "$DASHBOARD_PORT" == "$API_PORT" ]]; do
  DASHBOARD_PORT="$(free_port)"
done

API_URL="http://127.0.0.1:${API_PORT}"
DASHBOARD_URL="http://127.0.0.1:${DASHBOARD_PORT}"
SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/trading-runtime-smoke.XXXXXX")"
API_LOG="$SMOKE_ROOT/api.log"
DASHBOARD_BUILD_LOG="$SMOKE_ROOT/dashboard-build.log"
DASHBOARD_LOG="$SMOKE_ROOT/dashboard.log"
API_PID=""
DASHBOARD_PID=""

terminate_group() {
  local pid="$1"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill -TERM -- "-$pid" >/dev/null 2>&1 || true
    for _ in {1..20}; do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        return
      fi
      sleep 0.1
    done
    kill -KILL -- "-$pid" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  local status=$?
  set +e
  terminate_group "$DASHBOARD_PID"
  terminate_group "$API_PID"
  if [[ "$status" -ne 0 ]]; then
    echo >&2
    echo "Runtime smoke failed. API log:" >&2
    tail -n 120 "$API_LOG" 2>/dev/null >&2 || true
    echo >&2
    echo "Dashboard build log:" >&2
    tail -n 120 "$DASHBOARD_BUILD_LOG" 2>/dev/null >&2 || true
    echo >&2
    echo "Dashboard runtime log:" >&2
    tail -n 120 "$DASHBOARD_LOG" 2>/dev/null >&2 || true
  fi
  if [[ "$KEEP_ARTIFACTS" == "1" ]]; then
    echo "Smoke artifacts: $SMOKE_ROOT"
  else
    rm -rf "$SMOKE_ROOT"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_url() {
  local url="$1"
  local attempts="${2:-120}"
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if curl --silent --show-error --fail --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "Timed out waiting for $url" >&2
  return 1
}

export ALPACA_API_KEY=""
export ALPACA_API_SECRET=""
export SEC_USER_AGENT=""
export TRADING_ENTITY_CATALOG_PATH=""
export TRADING_ENV="runtime-smoke"
export TRADING_DEMO_MODE="true"
export TRADING_EXECUTION_MODE="internal-paper"
export TRADING_STRATEGY_MODE="explainable"
export TRADING_SYMBOLS="AAPL,MSFT,NVDA"
export TRADING_DECISION_INTERVAL_SECONDS="0.05"
export TRADING_MAX_DATA_AGE_SECONDS="5"
export TRADING_CONTROL_API_KEY="runtime-smoke-control-key"
export TRADING_DATABASE_PATH="$SMOKE_ROOT/events.db"
export TRADING_MODEL_REGISTRY_PATH="$SMOKE_ROOT/models"
export TRADING_CORS_ORIGINS="$DASHBOARD_URL"
export NEXT_PUBLIC_API_URL="$API_URL"
export NEXT_PUBLIC_CONTROL_API_KEY="runtime-smoke-control-key"

setsid bash -c 'cd "$1"; shift; exec "$@"' _ "$SMOKE_ROOT" \
  python -m uvicorn trading_app.main:app \
  --host 127.0.0.1 \
  --port "$API_PORT" \
  --log-level info \
  >"$API_LOG" 2>&1 &
API_PID=$!

wait_for_url "$API_URL/health"

API_URL="$API_URL" DASHBOARD_URL="$DASHBOARD_URL" python - <<'PY'
from __future__ import annotations

import json
import os
import time
import urllib.request

from trading_app import __version__

api = os.environ["API_URL"]
dashboard = os.environ["DASHBOARD_URL"]


def read_json(path: str) -> dict[str, object]:
    with urllib.request.urlopen(api + path, timeout=5) as response:
        if response.status != 200:
            raise AssertionError(f"{path} returned {response.status}")
        return json.load(response)


health: dict[str, object] | None = None
for _ in range(80):
    health = read_json("/health")
    if health.get("engine_running") and health.get("status") == "ok":
        break
    time.sleep(0.1)
else:
    raise AssertionError(f"Demo feed never became healthy: {health}")

assert health is not None
assert health["version"] == __version__
assert health["demo_mode"] is True
assert health["execution_mode"] == "internal-paper"
assert health["active_strategy"] == "explainable"
assert health["kill_switch"] is False

openapi = read_json("/openapi.json")
assert openapi["info"]["version"] == __version__

summary = read_json("/v1/dashboard/summary")
assert summary["version"] == __version__
assert summary["engine_running"] is True
assert summary["active_strategy"] == "explainable"
assert summary["symbols"] == ["AAPL", "MSFT", "NVDA"]
assert summary["portfolio"]["equity"] > 0
assert set(summary["recent"]) == {
    "quotes",
    "news",
    "proposals",
    "decisions",
    "orders",
    "fills",
    "system_events",
}

reconciliation = read_json("/v1/reconciliation")
assert reconciliation == {
    "mode": "internal-paper",
    "status": "not_applicable",
    "differences": [],
}

request = urllib.request.Request(
    api + "/v1/dashboard/summary",
    method="OPTIONS",
    headers={
        "Origin": dashboard,
        "Access-Control-Request-Method": "GET",
    },
)
with urllib.request.urlopen(request, timeout=5) as response:
    assert response.headers["access-control-allow-origin"] == dashboard

for enabled in (True, False):
    request = urllib.request.Request(
        api + "/v1/control/kill-switch",
        data=json.dumps({"enabled": enabled}).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Trading-API-Key": "runtime-smoke-control-key",
        },
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = json.load(response)
        assert body["kill_switch"] is enabled
PY

API_PORT="$API_PORT" python - <<'PY'
from __future__ import annotations

import asyncio
import json
import os

import websockets


async def verify() -> None:
    uri = f"ws://127.0.0.1:{os.environ['API_PORT']}/v1/live/events"
    async with websockets.connect(uri, open_timeout=5) as socket:
        message = await asyncio.wait_for(socket.recv(), timeout=5)
        payload = json.loads(message)
        assert payload["type"] in {
            "quote",
            "news",
            "proposal",
            "risk_decision",
            "order",
            "fill",
        }
        assert "created_at" in payload
        assert isinstance(payload["payload"], dict)


asyncio.run(verify())
PY

(
  cd apps/dashboard
  npm run build >"$DASHBOARD_BUILD_LOG" 2>&1
)

setsid bash -c "cd '$ROOT_DIR/apps/dashboard' && exec npm run start -- --hostname 127.0.0.1 --port '$DASHBOARD_PORT'" \
  >"$DASHBOARD_LOG" 2>&1 &
DASHBOARD_PID=$!

wait_for_url "$DASHBOARD_URL/"

DASHBOARD_URL="$DASHBOARD_URL" python - <<'PY'
import os
import urllib.request

with urllib.request.urlopen(os.environ["DASHBOARD_URL"] + "/", timeout=5) as response:
    body = response.read().decode("utf-8")
    assert response.status == 200
    assert "Connecting to trading engine" in body or "AI Trading Dashboard" in body
PY

VERSION="$(python -c 'import trading_app; print(trading_app.__version__)')"
printf '{\n  "status": "ok",\n  "version": "%s",\n  "api": "%s",\n  "dashboard": "%s",\n  "execution_mode": "internal-paper",\n  "database": "temporary"\n}\n' \
  "$VERSION" "$API_URL" "$DASHBOARD_URL"
