# Runtime smoke verification

Run this gate after upgrading dependencies or changing the API/dashboard contract.

## 1. Start the backend

Use demo/internal-paper mode for the first verification so no provider credentials or broker state are involved:

```bash
cd ~/Desktop/Trading
source .venv/bin/activate

TRADING_DEMO_MODE=true \
TRADING_EXECUTION_MODE=internal-paper \
TRADING_DATABASE_PATH=.trading/smoke-events.db \
uvicorn trading_app.main:app --port 8000
```

The API should remain running in this terminal.

## 2. Start the dashboard

In a second terminal:

```bash
cd ~/Desktop/Trading/apps/dashboard
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

The dashboard should remain running on port 3000.

## 3. Run the automated smoke gate

In a third terminal:

```bash
cd ~/Desktop/Trading
source .venv/bin/activate
trading-smoke
```

The command validates:

- API health and engine startup
- OpenAPI version against the installed package version
- `/health` and dashboard-summary version agreement
- the dashboard-summary response contract
- an HTML response from the frontend

A degraded feed is reported but does not by itself mean the processes failed to start. Version drift, a stopped engine, missing summary fields, HTTP errors, and a non-HTML dashboard response fail the command with exit status 1.

## 4. Inspect manually

- API documentation: `http://localhost:8000/docs`
- Dashboard: `http://localhost:3000`
- Health: `http://localhost:8000/health`

The System state panel should show the same version printed by:

```bash
python -c "import trading_app; print(trading_app.__version__)"
```

After the demo smoke passes, stop both processes before starting Alpaca paper mode so only one engine writes to a given ledger path.
