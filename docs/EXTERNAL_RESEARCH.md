# External research backends

Qlib, FinRL, and QuantConnect LEAN may provide useful independent predictions, policies, or shadow-backtest evidence. They are never authoritative for model promotion.

`trading-integrations import-external` normalizes an operator-exported JSON artifact together with:

- backend and tool version;
- configuration path and SHA-256;
- frozen universe hash;
- dataset hash;
- model identifier;
- source artifact hash.

Supported input shapes are:

- Qlib: `predictions` list and `metrics` object;
- FinRL: `policy` object and `episodes` list;
- LEAN: `orders` list and `statistics` object.

```bash
trading-integrations import-external \
  --input .trading/external/qlib-run.json \
  --spec .trading/external/qlib-run-spec.json \
  --output .trading/external/qlib-evidence.json
```

Imported evidence cannot move registry aliases. Any model or prediction series that influences a candidate must still pass the platform's own point-in-time checks, cost simulation, native deterministic replay, sealed holdout, promotion gates, and paper monitoring.

## Mixed-family generations

Create the deterministic 100-candidate plan:

```bash
trading-integrations mixed-plan \
  --generation 1 \
  --seed 1729 \
  --output .trading/generations/cycle-01-plan.json
```

The default quota is 25 tabular/linear, 20 sequence, 15 foundation, 20 RL, 10 news variants, and 10 ensemble or ablation candidates. Every worker writes a calibration-only report. Finalization requires all 100 successful, unique reports and rejects any report that accessed the holdout. It freezes exactly three finalists.
