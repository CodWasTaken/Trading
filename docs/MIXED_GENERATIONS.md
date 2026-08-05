# Executing mixed-model generations

The mixed-generation runner executes exactly 100 calibration-only candidates and freezes exactly three finalists only after every candidate has a complete, hash-bound report.

## Create the plan

```bash
trading-integrations mixed-plan \
  --generation 1 \
  --seed 1729 \
  --output .trading/generations/cycle-01-plan.json
```

The executable default quota is:

- 25 tabular or linear return models;
- 20 sequence return models;
- 15 foundation-feature return models;
- 20 reinforcement-learning portfolio policies;
- 10 news-feature variants;
- 10 feature ablations.

Ensemble model classes remain supported by the model registry, but the mixed quota uses ablations because they can be evaluated independently without fitting ensemble weights on shared out-of-fold predictions.

## Declare resources

Create a resource manifest mapping `base`, each foundation provider, and each news variant to calibration datasets. Candidate-specific dataset overrides are supported. The manifest also declares the conservative cost configuration, frozen universe manifest, outer-fold count, and bounded worker count.

Every dataset is verified as `dataset_role=calibration` and not sealed before any candidate starts. The execution manifest records dataset and metadata hashes.

## Execute and resume

```bash
trading-integrations mixed-execute \
  --plan .trading/generations/cycle-01-plan.json \
  --resources .trading/generations/cycle-01-resources.json \
  --output-dir .trading/generations/cycle-01
```

A candidate writes either `metrics.json` or `failure.json` under its own directory. Re-running preserves successful candidates. To retry only failures:

```bash
trading-integrations mixed-execute \
  --plan .trading/generations/cycle-01-plan.json \
  --resources .trading/generations/cycle-01-resources.json \
  --output-dir .trading/generations/cycle-01 \
  --retry-failed
```

The runner cannot finalize with 99 candidates. Once all 100 succeed, it ranks the common cost-adjusted metrics and writes three immutable finalists for the sealed holdout.

## Reinforcement learning

RL candidates are evaluated on purged chronological outer folds. Each fold trains only on the fold's calibration history and evaluates deterministically on the later fold. Ranking uses out-of-sample return after execution and financing costs, benchmark excess, drawdown, fold stability, long/short contribution, concentration, and turnover. Training reward is never used as the promotion score.

The final full-calibration policy remains `live_compatible=false`. It cannot be selected in the dashboard until a future shared-engine policy adapter is separately validated and promoted.
