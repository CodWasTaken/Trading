# Production roadmap

The repository contains a working vertical slice, not a validated profitable strategy. Build outward in this order.

## Signed long/short accounting

Implemented in version 0.17.0:

- signed position quantities and direction-aware average entry prices
- short entry/increase, partial/full cover, and two-way reversal accounting
- long/short realized and unrealized PnL attribution
- short-sale cash proceeds with negative marked inventory included in equity
- gross long, gross short, net exposure, initial buying power, and maintenance margin
- separate long/short position and aggregate limits
- fail-closed easy-to-borrow validation only on short-opening quantities
- configurable borrow and dividend-replacement carrying costs
- auditable borrow-recall and forced-cover abstractions
- deterministic replay coverage for short opening, covering, reversal, and rejection

This is accounting and safety infrastructure, not evidence that a short strategy
is profitable. All execution remains internal paper or Alpaca paper.

## Explicit cost model

Implemented in version 0.18.0:

- versioned cost manifests with stable hashes in experiment and registry metadata
- observed spread, slippage, commission, regulatory sell fee, and impact stress
- borrow, margin-interest, and dividend-replacement financing attribution
- explicit optional PLN→USD funding conversion, never an automatic trade fee
- pre-tax performance after execution/financing costs
- separate estimated Polish tax and after-tax summaries labelled as non-advice
- identical frozen cost manifests in calibration, holdout, monitoring, and replay

The bundled conservative manifest is an initial paper assumption. It must be
reviewed against observed paper fills and current broker schedules; it is not a
claim about future achievable costs.

## Frozen major-stock universe

Implemented in version 0.19.0:

- 48 U.S.-listed common stocks across nine sectors
- effective date, inclusion rules, liquidity/price/coverage thresholds, and
  shortability requirements
- hashed source declaration and canonical manifest hash
- exact binding for datasets, calibration, holdout, replay, diagnostics, and
  paper runtime
- fail-closed symbol, binding, source-hash, manifest-hash, and effective-date checks

The declaration source is not a substitute for archived point-in-time vendor
evidence. A later list must receive a new universe ID and effective date; it
cannot retroactively redefine an earlier experiment or untouched holdout.

## Deterministic candidate generations

Implemented in version 0.20.0:

- exactly 100 unique configurations, generated or supplied, per generation
- deterministic seeds and configuration fingerprints
- calibration-only nested, purged, chronological validation with at least
  eight outer folds
- bounded parallel execution, per-candidate artifacts and failures, and safe
  resumption against a hashed immutable generation manifest
- predeclared composite ranking across return, excess return, risk, stability,
  concentration, turnover, long/short balance, news ablation, threshold
  robustness, and borrow dependence
- at most ten generations with a three-stale-generation stopping rule
- exactly three frozen finalists and an explicit `candidate_family_size=3`
  boundary for the later sealed holdout

The generator has no holdout input and rejects any dataset marked sealed or
`untouched_holdout`. Candidate rankings are calibration evidence, not promotion
evidence or a profitability claim.

## Fail-closed promotion gates

Implemented in version 0.21.0:

- a versioned, hashed strict promotion-gate manifest
- calibration minimums of eight purged folds, 1,000 scored observations,
  positive cost-adjusted net and benchmark-excess return, Sharpe above 0.50,
  and drawdown below 15%
- a maximum of three sealed-holdout finalists, 300 holdout observations,
  positive point estimates and adjusted lower bounds for net and excess return,
  and drawdown below 15%
- 20% symbol and 35% sector PnL-contribution caps
- zero unborrowable shorts, 30% maximum gross short exposure, and 3% maximum
  single-short exposure
- fail-closed missing attribution, borrow, exposure, statistical-plan, and
  uncertainty evidence
- exact effective thresholds and manifest hashes in registry history
- a non-disableable untouched-holdout requirement for historical models

Passing these gates changes only the historical champion alias. It does not
authorize live-money execution or establish profitability.

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
- immutable later-window feature and prediction drift reports
- realized RMSE, calibration slope, active-signal quality, and frozen paper-simulation monitoring
- fail-closed monitoring recommendations with non-zero unhealthy CLI exit status

Remaining production work:

- higher-capacity price/news models by forecast horizon
- independent experiment manifests and comparison reports
- several genuinely fresh holdout windows across regimes
- automated scheduling and alert delivery for matured monitoring windows

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
