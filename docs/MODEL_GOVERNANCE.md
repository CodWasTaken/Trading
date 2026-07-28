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
  --minimum-folds 5 \
  --minimum-sharpe 0.25 \
  --maximum-drawdown 0.15 \
  --minimum-observations 500 \
  --minimum-excess-return 0.0 \
  --minimum-news-sharpe-delta 0.0 \
  --minimum-holdout-net-return 0.0 \
  --minimum-holdout-sharpe 0.0 \
  --maximum-holdout-drawdown 0.15 \
  --minimum-holdout-observations 100 \
  --minimum-holdout-excess-return 0.0
```

Promotion fails closed if the holdout is missing or any selected calibration or holdout gate fails. A successful event records the complete gate configuration, registered calibration metrics, and exact holdout evaluation used for the decision.

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
