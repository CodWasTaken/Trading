# Production roadmap

The repository contains a working vertical slice, not a validated profitable strategy. Build outward in this order.

## Milestone 1 — durable data and replay

Implemented foundation:

- append-only SQLite event journal and integrity audit
- paginated historical bars and news backfills with source hashes
- point-in-time dataset construction
- deterministic historical clock
- shared strategy → risk → paper-broker replay
- semantic trace hashes and double-run determinism verification
- regime, sector, event-type, and threshold-sensitivity diagnostics

Remaining production work:

- PostgreSQL/TimescaleDB schema and Alembic migrations
- raw provider payload archive in object storage
- historical quotes, corporate actions, and delisted symbols
- feed latency, duplicate detection, and gap alerts

**Exit gate:** replay produces identical proposals and risk decisions from the same event log, while data-gap and corporate-action handling are demonstrated on production-grade archives.

## Milestone 2 — news intelligence

Implemented foundation:

- SEC EDGAR submissions and recent-filings API
- deterministic sentiment, novelty, source quality, and article versioning
- weighted primary and secondary financial-event taxonomy
- typed percentage, money, EPS, and impact-direction extraction
- provider-symbol issuer links
- explicit issuer/subsidiary/supplier/customer/competitor/partner relationship catalogs
- immutable historical archive reclassification with before/after audits
- ticker-scoped deduplication that retains multi-symbol articles
- exact requested-universe filtering for historical news backfills
- knowledge-time effective windows for catalog relationships
- deterministic stratified human-label templates with immutable prediction hashes
- primary, secondary, entity-link, relation, and confusion-matrix evaluation reports
- balanced deterministic independent-reviewer packet assignment
- pairwise primary Cohen's kappa plus secondary/entity agreement reports
- unanimous-only consensus generation that keeps disputes pending for adjudication

Remaining production work:

- SEC filing facts and XBRL normalization
- company investor-relations feeds and raw article version archive
- curated production entity catalog with dated relationship provenance
- completed multi-reviewer event and entity-linking label sets
- human adjudication of disputed labels and taxonomy-guideline revision
- untouched validation archives and final independent evaluation
- higher-capacity financial sentiment and structured extraction models
- schema-constrained LLM extraction with deterministic fallback and disagreement review

**Exit gate:** labelled evaluation reports precision/recall by event type and entity relation, confirms no future information leakage, and demonstrates stable improvements on an untouched archive.

## Milestone 3 — research and model registry

Implemented foundation:

- feature definitions shared by offline and online paths
- purged walk-forward validation with transaction costs
- cash, equal-weight, benchmark, and no-news comparisons
- versioned model registry with champion alias and fail-closed promotion gates
- non-overlapping horizon evaluation and per-symbol diagnostics
- automatic challenger alias for newly registered models
- append-only alias history with promotion-gate snapshots
- operator promotion, custom aliases, and champion rollback without artifact deletion
- chronological calibration/holdout splitting with label-boundary purging
- sealed split manifests with source and output hashes
- frozen-model one-time scoring per model-version and holdout hash
- registry-backed holdout evaluation history
- promotion gates requiring both calibration and untouched-holdout evidence
- predeclared candidate-family budgets and Bonferroni adjustment
- deterministic circular moving-block bootstrap confidence intervals
- promotion gates on adjusted holdout net-return and benchmark-excess lower bounds

Remaining production work:

- higher-capacity price/news models by forecast horizon
- independent experiment manifests and comparison reports
- drift and calibration monitoring
- several genuinely fresh holdout windows across regimes

**Exit gate:** the candidate improves risk-adjusted out-of-sample performance across several untouched windows and regimes.

## Milestone 4 — execution realism

- quote-aware limit orders
- partial fills, queue position, latency, market impact, and participation limits
- broker trade-update stream and continuous reconciliation
- halt, calendar, corporate-action, and retry/idempotency handling
- conservative internal simulator running beside broker paper trading

**Exit gate:** internal and broker paper ledgers reconcile, with unexplained differences at zero.

## Milestone 5 — operations and security

- authentication and role-protected control endpoints
- secret manager and key rotation
- Prometheus/OpenTelemetry, alerting, backups, and restore drills
- signed model artefacts and immutable deployment history
- runbooks for stale feeds, broker outage, reconciliation failure, and kill switch

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
