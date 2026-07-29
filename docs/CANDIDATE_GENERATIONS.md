# Governed candidate generations

Candidate search is confined to the calibration dataset. The runner accepts or
deterministically generates exactly 100 unique configurations for one
generation, evaluates them with nested purged chronological validation, and
freezes exactly three finalists. It cannot accept a sealed or untouched-holdout
dataset.

## Inputs and immutable plan

Before generation 1, freeze:

- the point-in-time calibration dataset and its metadata;
- the versioned universe manifest effective for the experiment;
- the conservative cost manifest;
- the seed, score weights, resource limit, and eight-or-more fold count; and
- the meaningful-improvement threshold used by the stopping rule.

The generated manifest hashes the dataset, metadata, all configurations, cost
model, and universe. Resumption fails if any governed input changes. Each
candidate directory retains its configuration, fitted calibration model, metric
report, or failure report. Failed candidates are not silently discarded.

```bash
trading-candidates \
  --dataset .trading/datasets/calibration.jsonl \
  --output .trading/candidates/generation-1 \
  --generation 1 \
  --seed 1729 \
  --cost-config config/costs/conservative-us-paper-v1.json \
  --universe-manifest config/universes/us-liquid-large-cap-v1.json \
  --max-workers 4 \
  --outer-folds 8
```

An externally prepared JSON array may be supplied with
`--candidate-configs`. It must contain exactly 100 distinct configurations.
Parallelism is bounded to 1–16 workers and never changes deterministic ranking.

## Validation and leakage boundary

Every outer test block follows its training block. Training rows whose
five-bar label reaches the test boundary are purged. The last 20% of each outer
training window is an inner chronological validation block, with the same label
boundary purge. Portfolio results use non-overlapping five-bar outcomes.

The five-hour target is the historical realized return over the next five
hourly bars. It is a training label, never a model input or live feature. At a
live decision, the model receives only features known then. News contributes
only when `knowledge_time <= decision_time`.

The runner does not expose a holdout argument. It verifies
`dataset_role=calibration`, rejects `sealed=true`, and records
`holdout_accessed=false` in every completed metric report. A human opening the
holdout can still invalidate it; filesystem controls cannot restore untouched
status after inspection.

## Predeclared composite ranking

Raw return alone does not determine rank. Fixed weights reward net return after
costs, benchmark excess, Sharpe, controlled drawdown, fold/symbol/sector
stability, balanced long and short contribution, news-ablation improvement,
and threshold robustness. They penalize symbol/sector/regime concentration,
turnover, fragile threshold peaks, sparse activity, one-sided performance, and
short-borrow dependence.

The score is a calibration selection device, not proof of profitability.
Conservative execution, borrow, financing, and dividend-replacement assumptions
are preserved in the generation manifest.

## Lifecycle

Run at most ten generations. Stop after three consecutive generations without
the predeclared meaningful improvement. Do not adjust that threshold after
viewing results.

The final generation writes `finalists.json` with three frozen model and metric
hashes and `candidate_family_size=3`. Only these three may be evaluated once on
the sealed holdout under its multiple-comparison plan. Never score all 100 on
the holdout, tune a finalist after seeing a holdout result, or reclassify an
inspected period as untouched. The eventual decision may be “no champion.”
