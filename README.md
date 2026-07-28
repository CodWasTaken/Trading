# Trading

A private, risk-first AI-assisted paper-trading platform. It ingests market data and breaking stock news, turns those events into explainable signals, applies deterministic portfolio controls, simulates execution, and streams every decision to a live dashboard.

> **Status:** functional MVP foundation. It defaults to a deterministic demo feed and an internal paper broker, so it can run safely without credentials. Alpaca paper trading and real-time market/news adapters are included behind configuration flags.

## What is included

- FastAPI backend with REST and WebSocket APIs
- Deterministic market/news demo stream
- Explainable signal engine and news catalyst scoring
- Non-bypassable risk engine with kill switch, exposure, drawdown, confidence, and stale-data controls
- Conservative internal paper broker with spread and slippage
- Optional Alpaca paper order adapter and official market/news stream configuration
- In-memory event ledger designed around `event_time` and `knowledge_time`
- Next.js live dashboard showing portfolio, decisions, news, positions, orders, and system state
- Unit tests for risk, fills, and end-to-end decision flow
- Docker Compose and GitHub Actions CI
- Architecture, security, and production roadmap documentation

## Safety model

The model never sends an order directly. The flow is always:

```text
market/news -> features -> strategy proposal -> risk engine -> paper broker -> audit ledger
```

Any stale feed, kill switch, daily-loss breach, drawdown breach, oversized position, excessive spread, duplicate signal, or low-confidence proposal fails closed.

## Quick start

### Backend

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
uvicorn trading_app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the API. The demo engine starts automatically unless `TRADING_DEMO_MODE=false`.

### Dashboard

```bash
cd apps/dashboard
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open `http://localhost:3000`.

### Docker

```bash
docker compose up --build
```

## Alpaca paper mode

Create paper credentials and configure:

```dotenv
TRADING_DEMO_MODE=false
TRADING_EXECUTION_MODE=alpaca-paper
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_STREAM_URL=wss://stream.data.alpaca.markets/v2/iex
ALPACA_NEWS_STREAM_URL=wss://stream.data.alpaca.markets/v1beta1/news
```

The application intentionally does not support live-money execution. Adding it requires a separate adapter, explicit configuration, and a production-readiness review.

## Repository layout

```text
apps/api/trading_app/     backend domain and services
apps/dashboard/           live Next.js dashboard
tests/                    backend tests
docs/                     architecture and rollout guidance
.github/workflows/        CI
```

## Current strategy

The included strategy is deliberately simple and explainable. It combines short-term momentum, news sentiment, novelty, source quality, and confidence. It is a vertical-slice reference implementation, not a claim of profitability. Replace it through the `Strategy` protocol after building point-in-time historical datasets and walk-forward validation.

## Next production milestones

1. PostgreSQL/TimescaleDB event persistence and migrations
2. Historical point-in-time dataset builder and replay engine
3. SEC filing ingestion and structured event extraction
4. LightGBM champion/challenger training pipeline with MLflow
5. Broker reconciliation, alerting, and deployment hardening
6. Sixty to ninety live paper-trading days before any discussion of real funds

See [`docs/ROADMAP.md`](docs/ROADMAP.md).
