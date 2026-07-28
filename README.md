# Trading

A private, risk-first AI-assisted paper-trading platform. It ingests market data and breaking stock news, turns those events into explainable signals, applies deterministic portfolio controls, simulates execution, and streams every decision to a live dashboard.

> **Status:** functional paper-trading, data-ingestion, and research foundation. It defaults to a deterministic demo feed and an internal paper broker, so it can run safely without credentials. Alpaca paper trading and real-time market/news adapters are included behind configuration flags.

## What is included

- FastAPI backend with REST and WebSocket APIs
- Deterministic market/news demo stream
- Live Alpaca quotes and news, plus paginated historical bars and news backfills
- Headline deduplication, novelty scoring, source-quality weighting, sentiment, and catalyst classification
- SEC EDGAR recent-filings client and API endpoint
- Point-in-time dataset builder that joins news by `knowledge_time` to prevent look-ahead leakage
- Explainable signal engine and news catalyst scoring
- Non-bypassable risk engine with kill switch, exposure, drawdown, confidence, and stale-data controls
- Conservative internal paper broker with spread and slippage
- Optional Alpaca paper order adapter and official market/news stream configuration
- Live event ledger backed by an append-only SQLite journal using `event_time` and `knowledge_time`
- Cost-aware backtests, walk-forward validation, a trainable return model, and a versioned champion registry
- Next.js live dashboard showing portfolio, decisions, news, positions, orders, and system state
- Tests for risk, fills, persistence, research, data pagination, SEC parsing, news enrichment, model promotion, and leakage boundaries
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

## Alpaca paper and data mode

Create paper credentials and configure:

```dotenv
TRADING_DEMO_MODE=false
TRADING_EXECUTION_MODE=alpaca-paper
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets
ALPACA_DATA_BASE_URL=https://data.alpaca.markets
ALPACA_DATA_STREAM_URL=wss://stream.data.alpaca.markets/v2/iex
ALPACA_NEWS_STREAM_URL=wss://stream.data.alpaca.markets/v1beta1/news
```

The application intentionally does not support live-money execution. Adding it requires a separate adapter, explicit configuration, and a production-readiness review.

### Historical backfills

Download bars or enriched news as JSON Lines:

```bash
trading-research backfill bars \
  --symbols AAPL,MSFT,NVDA \
  --start 2025-01-01T00:00:00Z \
  --end 2026-01-01T00:00:00Z \
  --timeframe 1Hour \
  --output .trading/history/bars.jsonl

trading-research backfill news \
  --symbols AAPL,MSFT,NVDA \
  --start 2025-01-01T00:00:00Z \
  --end 2026-01-01T00:00:00Z \
  --output .trading/history/news.jsonl
```

Historical news providers generally expose source publication time, not the exact time your live system would have received the item. Treat that limitation explicitly in research metadata.

## SEC EDGAR

Set a descriptive application name and contact email:

```dotenv
SEC_USER_AGENT=Trading Research your-email@example.com
```

Then query recent authoritative filings through:

```text
GET /v1/sec/320193/filings?forms=10-K,10-Q,8-K&limit=25
```

## Repository layout

```text
apps/api/trading_app/     backend domain, data, execution, persistence, and research services
apps/dashboard/           live Next.js dashboard
tests/                    backend tests
docs/                     architecture and rollout guidance
.github/workflows/        CI
```

## Research and champion models

The project includes a deterministic end-to-end training check:

```bash
trading-research demo-train --rows 600 --registry .trading/models --promote
```

This exercises feature rows, model fitting, transaction-cost-aware walk-forward evaluation, versioned artefacts, promotion gates, and the `champion` alias. The generated dataset is synthetic and exists only to validate the machinery; its results are never evidence of market profitability.

After a model trained on genuine point-in-time market data passes the gates, select it with:

```dotenv
TRADING_STRATEGY_MODE=champion
TRADING_MODEL_REGISTRY_PATH=.trading/models
```

The champion model still only creates proposals. Every proposal continues through the same non-bypassable risk engine.

## Current strategy

The default strategy is deliberately simple and explainable. It combines short-term momentum, news sentiment, novelty, source quality, and confidence. It is a vertical-slice reference implementation, not a claim of profitability. A promoted champion can replace its signal generation without changing execution or risk boundaries.

## Next production milestones

1. Deterministic replay from the durable event ledger and a full historical quote dataset
2. Issuer/subsidiary/supplier entity linking and higher-capacity financial NLP extraction
3. Higher-capacity champion/challenger models and experiment tracking
4. Broker trade-update reconciliation and more realistic partial-fill/impact models
5. Authentication, alerts, backups, and deployment hardening
6. Sixty to ninety live paper-trading days before any discussion of real funds

See [`docs/ROADMAP.md`](docs/ROADMAP.md).
