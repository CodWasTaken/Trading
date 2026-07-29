# Trading

A private, risk-first AI-assisted paper-trading platform. It ingests market data and breaking stock news, turns those events into explainable signals, applies deterministic portfolio controls, simulates execution, and streams every decision to a live dashboard.

> **Status:** functional paper-trading, data-ingestion, research, replay, and operations foundation. It defaults to a deterministic demo feed and an internal paper broker, so it can run safely without credentials. Alpaca paper trading and real-time market/news adapters are included behind configuration flags.

## What is included

- FastAPI backend with REST and WebSocket APIs
- Deterministic market/news demo stream
- Live Alpaca quotes and news, plus paginated historical bars and news backfills
- Headline deduplication, novelty scoring, source-quality weighting, sentiment, and catalyst classification
- SEC EDGAR recent-filings client and API endpoint
- Point-in-time dataset builders that join news by `knowledge_time` and timestamp bar features when the close is actually known
- A versioned, hashed 48-stock multi-sector universe manifest bound exactly to
  datasets, calibration, holdout, replay, and paper runtime
- Explainable signal engine and news catalyst scoring
- Non-bypassable risk engine with kill switch, signed long/short limits, margin
  buying power, explicit easy-to-borrow checks, drawdown, confidence, and
  stale-data controls
- Conservative internal paper broker with spread and slippage
- Versioned cost manifests separating observed spread, slippage, commissions,
  regulatory sell fees, impact stress, borrow, margin interest, dividend
  replacement, and optional PLN→USD funding conversion
- Separate estimated Polish tax reports that preserve pre-tax strategy results
  and are explicitly labelled as estimates, not tax advice
- Alpaca paper adapter that waits for actual filled quantity and average price instead of booking estimated fills
- Broker-position reconciliation endpoint and execution-error audit events
- Protected state-changing controls through an optional API key
- Live feed-age diagnostics and stale-symbol reporting
- Live event ledger backed by an append-only SQLite journal using `event_time` and `knowledge_time`
- Deterministic ledger replay/integrity audit command
- Shared-engine historical replay with semantic trace hashes and double-run determinism verification
- Daily portfolio-control rollover shared by live paper trading and historical replay
- Cost-aware, multi-symbol backtests with purged walk-forward validation, a trainable return model, and a versioned champion registry
- Stitched non-overlapping out-of-sample evaluation with horizon-correct annualization
- Deterministic generations of exactly 100 calibration-only candidates with
  nested purged validation, resumable bounded execution, composite ranking,
  and three frozen holdout finalists
- Equal-weight benchmark, cash baseline, no-news ablation, and per-symbol diagnostics
- Historical dataset and real-data training CLI with source hashes, metadata, and explicit promotion gates
- Next.js live dashboard showing portfolio, decisions, news, positions, feed health, incidents, and system state
- Tests for risk, fills, persistence, research, provider pagination, SEC parsing, news enrichment, model promotion, reconciliation, replay determinism, and leakage boundaries
- Docker Compose and GitHub Actions CI
- Architecture, security, and production roadmap documentation

## Safety model

The model never sends an order directly. The flow is always:

```text
market/news -> features -> strategy proposal -> risk engine -> paper broker -> audit ledger
```

Any stale feed, kill switch, daily-loss breach, drawdown breach, oversized position, excessive spread, duplicate signal, low-confidence proposal, broker rejection, or fill timeout fails closed.

Paper startup and governed research also fail closed when their symbols do not
exactly match the declared frozen universe. Historical experiments must use the
manifest effective for that experiment; today’s large companies must never be
projected backward into an earlier untouched period.

Short opening also fails closed unless the paper runtime has a current explicit
easy-to-borrow status. A `SELL` first reduces any long inventory; only the
quantity crossing through zero opens a short. The signed ledger records long and
short entry/increase, partial exit, full exit, and reversal separately. Short
sale proceeds increase cash but the negative marked position remains in equity.
Borrow and dividend-replacement assumptions are configurable paper costs.

Execution costs are applied before pre-tax strategy performance is reported.
PLN→USD conversion is an optional funding cash flow and is never multiplied by
trade turnover. Estimated Polish capital-gains tax is reported separately at
`GET /v1/reports/polish-tax-estimate`; it is never subtracted from individual
fills. The estimate is deliberately simplified and is not tax advice.

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

## Operator security

For any deployment beyond a trusted local machine, configure a strong control key:

```dotenv
TRADING_CONTROL_API_KEY=replace-with-a-long-random-secret
```

The kill-switch, pause, and resume endpoints then require `X-Trading-API-Key`. Docker passes the value into the single-user dashboard build so its controls continue to work. Because browser-visible environment variables are not secret from someone who can access that dashboard, internet-facing deployments should put the dashboard behind authentication and use a server-side proxy for control actions.

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
TRADING_ORDER_FILL_TIMEOUT_SECONDS=15
TRADING_ORDER_POLL_INTERVAL_SECONDS=0.25
TRADING_EASY_TO_BORROW_SYMBOLS=AAPL,MSFT
TRADING_MAX_SHORT_POSITION_PCT=0.03
TRADING_MAX_GROSS_SHORT_EXPOSURE_PCT=0.30
```

The paper adapter submits an order, polls until the broker reports an actual fill, uses the broker’s filled quantity and average price, and cancels an unfilled remainder on timeout. Check internal-versus-broker quantities with:

```text
GET /v1/reconciliation
```

The application intentionally does not support live-money execution. Adding it requires a separate adapter, explicit configuration, and a production-readiness review.

The ETB list above is an operator-supplied paper assumption, not inferred
borrow availability. Omit a symbol or leave the list empty and attempts to open
that short are rejected. Broker recalls are represented as auditable forced-cover
instructions; they never bypass the paper-only broker boundary.

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

### Replay the shared strategy-risk-broker path

Replay the historical bars and news through the same `TradingEngine`, strategy, `RiskEngine`, portfolio, and internal paper broker used by the application:

```bash
trading-research replay \
  --bars .trading/history/bars.jsonl \
  --news .trading/history/news.jsonl \
  --strategy explainable \
  --bar-minutes 60 \
  --spread-bps 10 \
  --slippage-bps 2 \
  --verify-determinism \
  --output .trading/replays/explainable-2025.json \
  --trace-output .trading/replays/explainable-2025.trace.jsonl
```

The report includes source hashes, risk configuration, final portfolio state, return, drawdown, event counts, rejection reasons, fill totals, and a canonical semantic trace hash. `--verify-determinism` runs the replay twice with fresh strategy state and fails if the hashes differ.

Historical OHLC bars do not contain executable bid/ask quotes. Replay therefore applies the explicit `--spread-bps` value only to execution and risk simulation; the research model still does not train on a fabricated spread feature. The replay clock advances with historical knowledge time so stale-data checks remain active without comparing historical events to the current wall clock. Daily trade counts and daily P&L controls reset when the market-data date changes.

Champion replay is available with `--strategy champion --registry .trading/models`. A synthetic champion is rejected by default; `--allow-synthetic-champion` exists only for pipeline diagnostics and does not make synthetic results market evidence.

### Build a genuine point-in-time dataset

Convert the backfills into features and forward-return labels. `--bar-minutes` must match the bar timeframe used during backfill:

```bash
trading-research build-dataset \
  --bars .trading/history/bars.jsonl \
  --news .trading/history/news.jsonl \
  --bar-minutes 60 \
  --lookback-bars 19 \
  --forecast-bars 5 \
  --output .trading/datasets/aapl-msft-nvda-1h.jsonl
```

The command writes the feature rows plus an adjacent `.metadata.json` file containing source paths, SHA-256 hashes, feature names, label horizon, time boundaries, and known data limitations. A one-hour bar is timestamped for research at `bar_start + 60 minutes`, because its closing price is not known at the bar start.

Historical OHLC bars contain no bid/ask quotes. The historical model therefore trains on momentum, news score, and volatility instead of fabricating a spread feature. Live spread checks remain mandatory in the risk engine.

### Train and validate a candidate

For a governed search, run exactly 100 predeclared configurations on calibration
data only:

```bash
trading-candidates \
  --dataset .trading/datasets/calibration.jsonl \
  --output .trading/candidates/generation-1 \
  --generation 1 \
  --seed 1729 \
  --max-workers 4
```

The command rejects sealed holdouts, duplicate or non-100 configuration sets,
fewer than eight nested chronological folds, mismatched frozen universes, and
non-five-bar governed experiments. It preserves every configuration, metric,
failure, model artifact, cost manifest, and input hash. Only the top three
calibration-ranked candidates are frozen for the one-time sealed-holdout stage;
the other 97 must never be scored there. See
[`docs/CANDIDATE_GENERATIONS.md`](docs/CANDIDATE_GENERATIONS.md).

First register a candidate without promotion:

```bash
trading-research train \
  --dataset .trading/datasets/aapl-msft-nvda-1h.jsonl \
  --registry .trading/models \
  --minimum-train-rows 500 \
  --test-rows 100 \
  --transaction-cost-bps 5 \
  --cost-config config/costs/conservative-us-paper-v1.json \
  --periods-per-year 1638
```

Validation uses expanding multi-symbol walk-forward folds. Rows whose labels reach into a test fold are purged from that fold’s training set. Out-of-sample predictions from every fold are stitched into one chronological stream before metrics are calculated, so Sharpe is based on the complete return series rather than an average of fold Sharpes.

Forward labels are not compounded as if they were independent every bar. With `forecast_bars=5`, only non-overlapping five-bar horizons are used for portfolio returns. The CLI divides the source bar frequency by the label horizon automatically, so `1638` hourly market periods become `327.6` five-hour evaluation periods per year.

The training output includes:

- candidate metrics and the number of rows actually used after removing overlapping horizons
- an equal-weight long benchmark and a cash baseline
- a no-news model ablation using the same folds and costs
- per-symbol observations, signals, returns, hit rate, and turnover
- excess return over the benchmark and Sharpe improvement attributable to the news feature

Inspect the candidate and registry:

```bash
trading-research registry --registry .trading/models
```

Only request promotion after reviewing the result:

```bash
trading-research train \
  --dataset .trading/datasets/aapl-msft-nvda-1h.jsonl \
  --registry .trading/models \
  --minimum-train-rows 500 \
  --test-rows 100 \
  --transaction-cost-bps 5 \
  --periods-per-year 1638 \
  --promote
```

The versioned default gate manifest requires at least eight calibration folds,
1,000 scored observations, positive cost-adjusted net and benchmark-excess
returns, Sharpe above `0.50`, and drawdown below `15%`. A sealed holdout may
contain at most three finalists and must have at least 300 observations,
positive net and excess returns, positive multiplicity-adjusted lower bounds,
and drawdown below `15%`. Symbol/sector concentration and short borrow/exposure
evidence are mandatory and fail closed when absent. A failed promotion never
replaces the current champion.

Results produced before version `0.6.0` should be rerun before comparison because older evaluation averaged fold Sharpes and compounded overlapping forward labels.

## SEC EDGAR

Set a descriptive application name and contact email:

```dotenv
SEC_USER_AGENT=Trading Research your-email@example.com
```

Then query recent authoritative filings through:

```text
GET /v1/sec/320193/filings?forms=10-K,10-Q,8-K&limit=25
```

## Ledger audit

Replay the append-only journal and validate proposal, risk-decision, order, and fill relationships:

```bash
trading-research audit-ledger --database .trading/events.db
```

The report flags orphan fills, orders without risk decisions, orders attached to rejected proposals, duplicate IDs, and approved proposals that never reached order creation.

## Repository layout

```text
apps/api/trading_app/     backend domain, data, execution, persistence, replay, and research services
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

The champion model still only creates proposals. Every proposal continues through the same non-bypassable risk engine. Historical validation is not sufficient on its own; the candidate still needs sustained live paper evidence before any production review.

## Current strategy

The default strategy is deliberately simple and explainable. It combines short-term momentum, news sentiment, novelty, source quality, and confidence. It is a vertical-slice reference implementation, not a claim of profitability. A promoted champion can replace its signal generation without changing execution or risk boundaries.

## Next production milestones

1. Regime, sector, event-type, and threshold-sensitivity replay reports
2. Issuer/subsidiary/supplier entity linking and higher-capacity financial NLP extraction
3. Higher-capacity champion/challenger models and experiment tracking
4. Streaming broker trade updates, queue/partial-fill simulation, and automatic reconciliation alerts
5. Server-side user authentication, encrypted backups, and deployment runbooks
6. Sixty to ninety live paper-trading days before any discussion of real funds

See [`docs/ROADMAP.md`](docs/ROADMAP.md).
