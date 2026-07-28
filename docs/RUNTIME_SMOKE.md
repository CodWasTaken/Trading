# Runtime smoke verification

Run this gate after upgrading dependencies, changing the API/dashboard contract, or restarting a local stack that has not been used for several releases.

## Preferred isolated gate

The one-command harness is the safest path for an older local runtime such as version 0.4.x:

```bash
cd ~/Desktop/Trading
git pull origin main
source .venv/bin/activate

bash scripts/runtime-smoke.sh --install
```

Requirements:

- Linux with `bash`, `curl`, and `setsid`
- Python 3.12 or newer
- Node.js 22 or newer
- an activated Python virtual environment when using `--install`

The harness forces:

- deterministic demo data
- internal-paper execution
- explainable strategy
- a temporary SQLite event database
- a temporary model registry
- a temporary control key
- dynamically allocated loopback ports

The API process starts from the temporary directory, so the repository's existing `.env` is not loaded. Alpaca credentials are blanked explicitly. The harness cannot open an Alpaca stream, submit a broker order, or mutate the configured `.trading/events.db`.

It then:

1. starts the FastAPI backend;
2. builds and starts the production Next.js dashboard;
3. runs the existing `trading-smoke` contract validator;
4. verifies demo/internal/explainable state and version agreement;
5. verifies internal-paper reconciliation;
6. verifies CORS for the generated dashboard origin;
7. engages and releases the authenticated kill switch;
8. receives a structured live WebSocket event;
9. terminates both process groups and removes temporary state.

A successful run ends with output similar to:

```json
{
  "passed": true,
  "version": "0.11.0",
  "api": "http://127.0.0.1:...",
  "dashboard": "http://127.0.0.1:...",
  "execution_mode": "internal-paper",
  "database": "temporary"
}
```

## Failure logs

Recent API, dashboard-build, and dashboard-runtime logs are printed automatically on failure. Preserve the full temporary directory with:

```bash
KEEP_SMOKE_ARTIFACTS=1 bash scripts/runtime-smoke.sh --install
```

The final line identifies the artifact directory.

## Review the real configuration

The isolated gate deliberately does not validate the real `.env`. Preserve and review it before normal startup:

```bash
cp -a .env .env.before-runtime-upgrade 2>/dev/null || true
cp -a .trading/events.db .trading/events.before-runtime-upgrade.db 2>/dev/null || true
```

Compare `.env` with `.env.example`, paying particular attention to:

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

For the first normal startup after a long gap, prefer:

```dotenv
TRADING_DEMO_MODE=true
TRADING_EXECUTION_MODE=internal-paper
TRADING_STRATEGY_MODE=explainable
```

## Manual running-stack gate

When the backend and dashboard are already running, validate them directly:

```bash
trading-smoke \
  --api-url http://localhost:8000 \
  --dashboard-url http://localhost:3000
```

The command checks API health, engine startup, OpenAPI/package version agreement, dashboard-summary compatibility, and an HTML response from the frontend.

After demo smoke passes, stop both processes before starting Alpaca paper mode so only one engine writes to a given ledger path. The application does not support live-money execution.
