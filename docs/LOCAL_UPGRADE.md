# Local runtime upgrade and smoke test

Use this runbook when the checked-out code and installed CLI have been upgraded but the FastAPI backend or Next.js dashboard has not been run recently. The smoke harness is designed for upgrades from older local states such as version 0.4.x.

## Safety boundary

`scripts/runtime-smoke.sh` always overrides runtime configuration with:

- deterministic demo feed
- internal paper broker
- explainable strategy
- temporary SQLite event database
- temporary model registry
- temporary control key
- AAPL, MSFT, and NVDA only
- dynamically allocated loopback ports

It unsets Alpaca credentials and never opens an Alpaca stream or submits a broker order. It does not read or write the configured `.trading/events.db` or model registry. All temporary processes and data are removed after a successful run.

## Requirements

- Linux with `bash`, `curl`, and `setsid`
- Python 3.12 or newer
- Node.js 22 or newer
- an activated Python virtual environment when using `--install`

## Upgrade from an older local runtime

Stop any old backend and dashboard processes first. Then preserve configuration and runtime state before updating:

```bash
cd ~/Desktop/Trading

cp -a .env .env.before-runtime-upgrade 2>/dev/null || true
cp -a .trading/events.db .trading/events.before-runtime-upgrade.db 2>/dev/null || true

git pull origin main
source .venv/bin/activate

bash scripts/runtime-smoke.sh --install
```

A successful run prints a result similar to:

```json
{
  "status": "ok",
  "version": "0.11.0",
  "api": "http://127.0.0.1:...",
  "dashboard": "http://127.0.0.1:...",
  "execution_mode": "internal-paper",
  "database": "temporary"
}
```

The harness verifies:

- the installed package, `/health`, dashboard summary, and OpenAPI document report the same version
- demo mode, internal-paper execution, and explainable strategy are active
- the demo feed becomes healthy
- dashboard summary and portfolio shapes remain compatible
- internal-paper reconciliation is explicitly `not_applicable`
- CORS accepts the smoke dashboard origin
- authenticated kill-switch controls can engage and release
- the live WebSocket emits a structured event
- a production dashboard build succeeds
- the production dashboard server returns its root page

## Keep failure logs

The harness prints recent logs automatically on failure. Preserve the complete temporary directory with:

```bash
KEEP_SMOKE_ARTIFACTS=1 bash scripts/runtime-smoke.sh --install
```

The final line identifies the artifact directory containing:

- `api.log`
- `dashboard-build.log`
- `dashboard.log`
- the disposable smoke database and registry

## Review the real `.env` before normal startup

Do not copy `.env.example` over the existing `.env`. Compare them and review at least:

```dotenv
TRADING_DEMO_MODE=
TRADING_EXECUTION_MODE=
TRADING_DATABASE_PATH=
TRADING_STRATEGY_MODE=
TRADING_MODEL_REGISTRY_PATH=
TRADING_ENTITY_CATALOG_PATH=
TRADING_CONTROL_API_KEY=
TRADING_CORS_ORIGINS=
```

For the first normal startup after a long gap, prefer the safe local defaults:

```dotenv
TRADING_DEMO_MODE=true
TRADING_EXECUTION_MODE=internal-paper
TRADING_STRATEGY_MODE=explainable
```

Start the backend only after the isolated smoke passes:

```bash
source .venv/bin/activate
uvicorn trading_app.main:app --reload --port 8000
```

In another terminal:

```bash
cd ~/Desktop/Trading/apps/dashboard
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Verify:

```bash
curl --fail --silent http://localhost:8000/health | python -m json.tool
curl --fail --silent http://localhost:8000/v1/dashboard/summary | python -m json.tool
```

Only switch back to Alpaca paper mode after the demo/internal runtime is healthy and the actual `.env` has been reviewed. The application does not support live-money execution.
