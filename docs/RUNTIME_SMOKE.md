# Runtime smoke verification

Run this gate after upgrading dependencies or changing the API/dashboard contract.

## Start the backend

Use demo/internal-paper mode first so provider credentials and broker state are not involved:

```bash
cd ~/Desktop/Trading
source .venv/bin/activate

TRADING_DEMO_MODE=true \
TRADING_EXECUTION_MODE=internal-paper \
TRADING_DATABASE_PATH=.trading/smoke-events.db \
uvicorn trading_app.main:app --port 8000
```

## Start the dashboard

In a second terminal:

```bash
cd ~/Desktop/Trading/apps/dashboard
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

## Run the automated gate

In a third terminal:

```bash
cd ~/Desktop/Trading
source .venv/bin/activate
trading-smoke
```

The command validates API health, engine startup, OpenAPI/package version agreement, dashboard-summary compatibility, and an HTML response from the frontend. A degraded feed is reported but does not by itself mean the processes failed to start.

After the demo smoke passes, stop both processes before starting Alpaca paper mode so only one engine writes to a given ledger path.
