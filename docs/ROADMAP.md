# Production roadmap

The repository contains a working vertical slice, not a validated profitable strategy. Build outward in this order.

## Milestone 1 — durable data and replay

- PostgreSQL/TimescaleDB schema and Alembic migrations
- Raw provider payload archive in object storage
- Historical bars, quotes, news, corporate actions, and delisted symbols
- Deterministic market clock and point-in-time replay
- Feed latency, duplicate detection, and gap alerts

**Exit gate:** replay produces identical proposals and risk decisions from the same event log.

## Milestone 2 — news intelligence

- SEC EDGAR submissions and filing facts
- Company investor-relations feeds
- Entity linking for issuers, subsidiaries, sectors, suppliers, and competitors
- Event taxonomy, novelty detection, article versioning, source-quality scoring
- Financial sentiment model and structured LLM extraction with schema validation

**Exit gate:** labelled evaluation reports precision/recall by event type and confirms no future information leakage.

## Milestone 3 — research and model registry

- Feature definitions shared by offline and online paths
- LightGBM price/news models by forecast horizon
- Purged walk-forward validation with transaction costs
- Baselines: cash, benchmark, equal weight, momentum, and no-news ablation
- MLflow experiment tracking, champion/challenger aliases, and rollback
- Drift and calibration monitoring

**Exit gate:** the candidate improves risk-adjusted out-of-sample performance across several windows and regimes.

## Milestone 4 — execution realism

- Quote-aware limit orders
- Partial fills, queue position, latency, market impact, and participation limits
- Broker trade-update stream and continuous reconciliation
- Halt, calendar, corporate-action, and retry/idempotency handling
- Conservative internal simulator running beside broker paper trading

**Exit gate:** internal and broker paper ledgers reconcile, with unexplained differences at zero.

## Milestone 5 — operations and security

- Authentication and role-protected control endpoints
- Secret manager and key rotation
- Prometheus/OpenTelemetry, alerting, backups, and restore drills
- Signed model artefacts and immutable deployment history
- Runbooks for stale feeds, broker outage, reconciliation failure, and kill switch

**Exit gate:** fault-injection tests demonstrate fail-closed behaviour.

## Paper promotion policy

A champion remains paper-only until it has, at minimum:

- 60–90 trading days of live paper results
- positive return after conservative costs
- benchmark-beating risk-adjusted performance
- acceptable drawdown and tail loss
- no excessive dependence on one ticker, event, or market regime
- no open data-quality, execution, or reconciliation incidents

Passing these gates is evidence for further review, not a guarantee of future profit.
