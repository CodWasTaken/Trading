# Model registry governance

The registry stores immutable model artifacts and mutable aliases. Aliases are operational pointers, not model files. Moving an alias never deletes or rewrites a registered model.

## Registry roles

- `challenger` points to the newest normally registered model.
- `champion` points to the model selected for the application when `TRADING_STRATEGY_MODE=champion`.
- custom aliases may identify experiments, releases, or review candidates.

Every normal registration moves `challenger` and appends an alias-history event. It does not change `champion`.

## Inspect the registry

```bash
trading-registry --registry .trading/models summary
```

Inspect one immutable model record:

```bash
trading-registry --registry .trading/models inspect MODEL_VERSION
```

View all alias changes:

```bash
trading-registry --registry .trading/models history
```

Or only champion changes:

```bash
trading-registry --registry .trading/models history \
  --alias champion
```

View untouched-holdout evaluations:

```bash
trading-registry --registry .trading/models holdouts
```

Each alias event records the previous version, new version, action, reason, timestamp, and action details. Holdout records separately preserve the model version, split ID, dataset and report hashes, frozen configuration, metrics, and diagnostics. Older registries remain readable and are upgraded on their next write.

## Set a non-champion alias

```bash
trading-registry --registry .trading/models set-alias \
  candidate-news-v2 MODEL_VERSION \
  --reason "candidate for labelled-news comparison"
```

The operator CLI intentionally refuses direct `champion` assignment. Use `promote` or `rollback` so champion changes retain their safety semantics and audit details.

## Promote a challenger

A governed promotion requires both registered purged walk-forward metrics and a sealed untouched-holdout evaluation. Create that evidence with the workflow in `docs/HOLDOUT_VALIDATION.md`.

```bash
trading-registry --registry .trading/models promote MODEL_VERSION \
  --reason "passed calibration and sealed holdout gates" \
  --gate-config config/promotion/strict-paper-v1.json
```

The bundled gate manifest requires at least eight calibration folds and 1,000
scored observations, positive cost-adjusted net and benchmark-excess returns,
Sharpe above 0.50, and drawdown below 15%. The holdout plan may contain no more
than three finalists and requires 300 observations, positive net and excess
returns, positive multiplicity-adjusted lower bounds for both, and drawdown
below 15%.

Both calibration and holdout evidence must also show no symbol above 20% or
sector above 35% of total PnL, zero unborrowable short orders, gross short
exposure no higher than 30%, a single short no higher than 3%, and explicit
borrow-status validation. Missing metrics fail closed. CLI threshold overrides
are validated and the effective manifest plus SHA-256 are recorded in champion
history.

Promotion fails closed if the holdout is missing or any calibration, holdout,
concentration, or short-safety gate fails. The historical holdout requirement
cannot be disabled through the Python API.

`trading-research train --promote` no longer provides a one-step shortcut for real historical datasets: the newly trained version cannot already have a holdout record, so the request fails closed after registration. Train, evaluate the frozen challenger once, then promote with `trading-registry`.

Synthetic fixture promotion remains useful only for pipeline diagnostics and is rejected by historical champion replay unless the explicit synthetic override is provided. It is never market evidence.

## Roll back

Undo the most recent champion change:

```bash
trading-registry --registry .trading/models rollback \
  --reason "paper replay regression"
```

Restore a specific registered model:

```bash
trading-registry --registry .trading/models rollback \
  --version MODEL_VERSION \
  --reason "restore last verified paper model"
```

Rollback changes only the champion alias. The failed or superseded model remains in the registry for audit, diagnosis, and reproducibility.

## Operational boundaries

- Restart the API after changing `champion`; the current process loads its strategy at startup.
- Never edit `registry.json` manually while the application is running.
- Never delete model artifacts referenced by aliases, history, or holdout evaluations.
- Keep synthetic fixtures out of market-evidence decisions.
- Do not reuse a holdout result to tune the next model; reserve a new later period.
- A registry alias does not authorize live-money execution; the platform remains paper-only.
