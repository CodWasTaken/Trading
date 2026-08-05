# Pretrained model integrations

Pretrained models are research priors, not pretrained profitable strategies. Every model is installed explicitly by an operator, pinned to a revision, hashed locally, and evaluated through the platform's native calibration, sealed-holdout, replay, cost, and paper controls.

## Supported time-series providers

- **TimesFM 2.5** through its PyTorch forecasting API.
- **Chronos / Chronos-2** through the `chronos-forecasting` pipeline API.
- **Moirai / Moirai 2.0** through Uni2TS and GluonTS predictors.

The integration does not download weights during API startup or CI. Create a `PretrainedModelSpec` JSON file with the provider, upstream model identifier, exact revision, local path, recursive SHA-256, licence, context length, and five-bar prediction length. Verify it before use:

```bash
trading-integrations verify-foundation --spec .trading/pretrained/chronos-2.json
```

## Point-in-time feature generation

Foundation models receive only historical closes ending at the completed bar. They do not receive `target_return`, later bars, sealed-holdout outcomes, or future covariates.

```bash
trading-integrations foundation-features \
  --bars .trading/history/cycle-01-bars.jsonl \
  --spec .trading/pretrained/chronos-2.json \
  --output .trading/features/chronos-2.jsonl

trading-integrations join-foundation \
  --dataset .trading/datasets/cycle-01-calibration.jsonl \
  --sidecar .trading/features/chronos-2.jsonl \
  --output .trading/datasets/cycle-01-calibration-chronos.jsonl
```

The sidecar records feature time, history end time, forecast end time, point forecasts, quantiles, expected return, uncertainty, model revision, and weight hash. Joining requires exact symbol and feature-time alignment and fails closed on missing rows unless explicitly configured otherwise.

## FinGPT-compatible news models

Local Transformers or PEFT checkpoints may produce structured research annotations for sentiment, event type, entities, relations, and confidence. The output is immutable and preserves the source event and knowledge timestamps.

```bash
trading-integrations verify-news-model --spec .trading/pretrained/fingpt.json
trading-integrations news-infer \
  --input .trading/history/news-v3-full.jsonl \
  --spec .trading/pretrained/fingpt.json \
  --output .trading/features/fingpt-news.jsonl
```

A news model never receives order authority. Its quality must be measured against the independently completed reviewer packets and against the current news pipeline and no-news ablation.

## Operational limitations

Foundation and language-model dependencies are optional and can require substantial RAM, GPU memory, and disk. Upstream APIs and licences must be reviewed at the pinned revision. Model availability does not make a candidate eligible for paper selection; only compatible return models that pass native governance gates can become `active-paper`.
