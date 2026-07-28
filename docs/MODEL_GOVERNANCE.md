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

Each event records the previous version, new version, action, reason, timestamp, and action details. Older registries without a schema version or history remain readable and are upgraded on their next write.

## Set a non-champion alias

```bash
trading-registry --registry .trading/models set-alias \
  candidate-news-v2 MODEL_VERSION \
  --reason "candidate for labelled-news comparison"
```

The operator CLI intentionally refuses direct `champion` assignment. Use `promote` or `rollback` so champion changes retain their safety semantics and audit details.

## Promote a challenger

```bash
trading-registry --registry .trading/models promote MODEL_VERSION \
  --reason "passed approved research gates" \
  --minimum-folds 5 \
  --minimum-sharpe 0.25 \
  --maximum-drawdown 0.15 \
  --minimum-observations 500 \
  --minimum-excess-return 0.0 \
  --minimum-news-sharpe-delta 0.0
```

Promotion fails closed if any quantitative gate fails. A successful event records the complete gate configuration and the registered metrics used for the decision.

These gates currently evaluate the model's registered purged walk-forward report. They are necessary but not sufficient. Do not treat a successful promotion as independent validation until an untouched holdout evaluation has also been implemented and completed. All promoted models remain paper-only.

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
- Never delete model artifacts referenced by aliases or history.
- Keep synthetic fixtures out of market-evidence decisions.
- A registry alias does not authorize live-money execution; the platform remains paper-only.
