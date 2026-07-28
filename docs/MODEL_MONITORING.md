# Paper-model drift and calibration monitoring

Historical validation is a starting condition, not permanent evidence. After a champion enters paper operation, build later labelled feature windows once their forecast horizons have matured and compare them with the exact calibration dataset used to register the model.

The monitor does not retrain or move aliases. It writes an immutable evidence report and recommends either `continue_paper` or `pause_and_review`.

## Prepare a matured recent dataset

Use the same feature definitions, bar interval, forecast horizon, news window, and symbol semantics as the champion's calibration dataset. Every recent row must have a realized `target_return`, so the newest bars cannot be included until the configured forecast horizon has elapsed.

The recent dataset must start strictly after the final feature timestamp in the calibration reference. Overlapping data is rejected.

## Run the monitor

```bash
trading-monitor evaluate \
  --reference .trading/datasets/fresh-calibration.jsonl \
  --recent .trading/datasets/paper-matured-2026-10.jsonl \
  --registry .trading/models \
  --alias champion \
  --output .trading/monitoring/champion-2026-10.json \
  --minimum-recent-rows 100 \
  --minimum-active-signals 20 \
  --maximum-feature-psi 0.25 \
  --maximum-feature-mean-shift 1.0 \
  --maximum-prediction-mean-shift 1.0 \
  --maximum-rmse-ratio 2.0 \
  --minimum-active-hit-rate 0.50 \
  --minimum-calibration-slope 0.25 \
  --maximum-calibration-slope 1.75
```

`--version MODEL_VERSION` monitors an explicit immutable model instead of an alias.

The command verifies:

- the calibration dataset hash matches the model's registered `dataset_sha256`;
- recent timestamps begin after calibration;
- reference, recent, and model feature schemas agree;
- model weights, threshold, transaction costs, and annualization remain frozen;
- the output path does not already exist.

An unhealthy report is still written, then the CLI exits with status code `2` so scheduled paper operations can fail closed.

## Evidence in the report

### Feature drift

For every feature:

- calibration and recent mean and standard deviation;
- standardized absolute mean shift;
- standard-deviation ratio;
- population stability index using fixed bins anchored to the calibration distribution.

PSI and mean-shift thresholds are governance rules, not laws of nature. Start conservative, review false alerts, and change thresholds only in a documented new policy version rather than after seeing one unfavorable window.

### Prediction drift

The report compares:

- prediction mean and standard deviation;
- standardized prediction-mean shift;
- calibration and recent active-signal rates under the frozen threshold.

### Realized calibration

Once outcomes mature, the report calculates:

- mean absolute error and root mean squared error;
- recent-to-calibration RMSE ratio;
- prediction/target correlation;
- calibration intercept and slope from realized return on prediction;
- active-signal count, hit rate, and average realized return.

A calibration slope near one means realized return changes approximately in proportion to the model's scores on that window. A low, negative, or extreme slope is a review signal, not an automatic instruction to refit.

### Paper simulation

The frozen threshold and transaction costs are applied to both calibration and recent windows with the same non-overlapping-horizon simulator. This provides context but does not replace broker-paper reconciliation.

## Respond to an unhealthy report

When `recommended_action` is `pause_and_review`:

1. Engage the paper kill switch or stop champion-mode services.
2. Preserve the report and recent dataset unchanged.
3. Identify whether the alert reflects feed quality, changed feature construction, universe changes, a market regime shift, or true model degradation.
4. Do not lower thresholds merely to clear the alert.
5. Roll back to a previously verified champion only when its current monitoring evidence is acceptable.
6. Start a new governed calibration/holdout research cycle for any materially changed model.

Monitoring data must not be silently appended to training data. Once used for diagnosis or model selection, it becomes calibration evidence and cannot serve as untouched validation for the replacement model.

## Operational cadence

For hourly models, produce monitoring windows only after enough independent horizons and active signals have matured; a daily or weekly job can evaluate the accumulated labelled window. Tiny rolling windows produce unstable calibration estimates and should fail the minimum-row or minimum-active-signal gates.

A healthy report does not prove profitability. Champion review still requires several genuinely fresh holdouts, deterministic replay, continuous broker-paper reconciliation, and at least 60–90 trading days of live paper evidence.
