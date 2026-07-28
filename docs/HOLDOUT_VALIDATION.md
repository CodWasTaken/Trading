# Untouched holdout validation

Walk-forward validation inside a calibration dataset is used to choose features, thresholds, ridge strength, and other design decisions. It is not an untouched final test once those choices have been influenced by its results.

The holdout workflow creates a later sealed period, purges calibration labels that cross its boundary, scores each frozen model version at most once on the holdout hash, estimates uncertainty with a deterministic moving-block bootstrap, and records that evidence in the model registry.

## Independence rule

A period is untouched only when its results have not influenced the model, features, thresholds, costs, universe, or research decisions. A previously inspected 2025 period cannot be made untouched retroactively by renaming or splitting it.

Select a fresh future interval before looking at its returns. Record the boundary and the maximum number of candidates that may be tested in the split manifest, avoid opening the holdout dataset, and use the tooling only after each challenger is frozen.

## 1. Build one complete point-in-time dataset

Build the full fresh period from immutable bars and news:

```bash
trading-research build-dataset \
  --bars .trading/history/bars-fresh.jsonl \
  --news .trading/history/news-fresh.jsonl \
  --output .trading/datasets/fresh-full.jsonl \
  --lookback-bars 19 \
  --forecast-bars 5 \
  --bar-minutes 60 \
  --news-window-hours 24
```

Do not train on this full file before reserving the holdout.

## 2. Seal the split and statistical plan

Choose the boundary and candidate budget before inspecting holdout outcomes:

```bash
trading-holdout split \
  --dataset .trading/datasets/fresh-full.jsonl \
  --calibration-output .trading/datasets/fresh-calibration.jsonl \
  --holdout-output .trading/datasets/fresh-holdout.jsonl \
  --split-time 2026-10-01T00:00:00Z \
  --candidate-family-size 3 \
  --bootstrap-samples 5000 \
  --confidence-level 0.95 \
  --manifest .trading/datasets/fresh-split-manifest.json
```

The split uses feature timestamps. Rows before the boundary whose `label_end_time` reaches or crosses the boundary are removed from calibration. Holdout rows begin at the boundary. The manifest records source, metadata, output hashes, split ID, boundary, row counts, purge count, candidate-family size, confidence level, bootstrap sample count, and optional fixed block size.

`--candidate-family-size` is the maximum number of frozen candidate versions that the research plan permits on this holdout. The bootstrap interval uses a Bonferroni adjustment across that predeclared family. Declaring one candidate and then testing five makes the evidence invalid even though the software can detect only repeated use of the same model artifact.

The default circular moving-block length is derived from the number of non-overlapping holdout periods. `--bootstrap-block-size` may freeze an explicit length when a dependence assumption was selected before scoring.

Treat the holdout JSONL, metadata, and report as sealed research evidence. The operating system cannot prevent a human from opening them; methodological independence still depends on discipline.

## 3. Train only on calibration

```bash
trading-research train \
  --dataset .trading/datasets/fresh-calibration.jsonl \
  --registry .trading/models \
  --minimum-train-rows 500 \
  --test-rows 100 \
  --transaction-cost-bps 10 \
  --periods-per-year 1638 \
  --threshold 0.0005 \
  --ridge 0.001
```

The registered model becomes `challenger`. Its validation metadata freezes the prediction threshold, transaction-cost assumption, and effective periods per year. Do not use `--promote` as a shortcut: governed promotion requires a recorded holdout evaluation.

Thresholds and hyperparameters may be changed during calibration. Every changed configuration creates a new model version. Test no more than the predeclared candidate-family size on the holdout, and do not tune a later candidate to an earlier holdout result.

## 4. Score the frozen challenger once

```bash
trading-holdout evaluate \
  --dataset .trading/datasets/fresh-holdout.jsonl \
  --registry .trading/models \
  --output .trading/evaluations/fresh-holdout-challenger.json
```

By default, the command scores the current `challenger`; `--version` selects an explicit registered model. It verifies the holdout role and seal, matches the feature schema, loads the frozen model artifact, and reuses the threshold and costs stored at registration.

The registry refuses a second evaluation of the same model version on the same holdout SHA-256 hash. Creating a copied file with identical bytes does not bypass the check.

The report includes candidate metrics, equal-weight benchmark metrics, per-symbol diagnostics, non-overlapping period returns, input hashes, split ID, frozen configuration, and deterministic circular moving-block bootstrap evidence. It reports adjusted lower and upper bounds for candidate net return, benchmark return, and benchmark-excess return, plus the resampled probability that each candidate quantity is positive.

Bootstrap intervals summarize uncertainty conditional on the historical holdout and block-dependence approximation. They are not guarantees and do not repair an invalid or repeatedly inspected holdout.

Inspect recorded evaluations with:

```bash
trading-registry --registry .trading/models holdouts
```

Or for one model:

```bash
trading-registry --registry .trading/models holdouts \
  --version MODEL_VERSION
```

## 5. Promote through calibration, holdout, and uncertainty gates

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
  --minimum-holdout-excess-return 0.0 \
  --minimum-holdout-net-return-lower-bound 0.0 \
  --minimum-holdout-excess-return-lower-bound 0.0
```

The default operator gates require the adjusted bootstrap lower bounds for both net return and benchmark-excess return to be positive. A positive point estimate with a lower bound at or below zero fails.

Promotion fails when the holdout record is missing or any selected calibration, holdout, or uncertainty gate fails. The promotion-history event embeds the exact holdout evaluation and complete gate configuration used.

Passing historical gates does not authorize real-money execution. The model still requires deterministic replay, operational review, and at least 60–90 trading days of live paper evidence.

## 6. Avoid holdout and multiple-comparison overfitting

A one-score software limit prevents accidental repeated evaluation of the same model artifact, but it cannot stop a researcher from manually changing the next model after seeing the result. After holdout scoring:

- accept or reject each predeclared candidate without tuning it to the result;
- do not exceed the sealed candidate-family budget;
- do not change confidence or block assumptions after seeing the interval;
- use a newly accumulated later holdout for the next materially changed research cycle;
- preserve failed results rather than deleting them;
- compare performance across several untouched periods and regimes before trusting stability.

The 2025 AAPL/MSFT/NVDA archive already used for threshold and replay exploration remains calibration or diagnostic data, never untouched evidence.
