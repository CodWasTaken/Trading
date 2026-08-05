# Model assets and news-AI features

Model weights are not committed to this repository. They are never downloaded during API startup, dashboard startup, tests, or CI. Acquisition happens only through an explicit operator command.

## Acquire a pinned snapshot

Create a JSON request containing:

- the upstream repository identifier;
- an exact tag or commit revision;
- the local destination;
- the operator-reviewed licence identifier;
- the asset kind;
- optional file include and exclude patterns.

Run:

```bash
trading-integrations acquire-model \
  --request .trading/pretrained/acquisition-request.json \
  --output .trading/pretrained/acquisition-report.json
```

The command resolves the revision, downloads to the declared local directory, recursively hashes the snapshot, and writes a provenance report. The report never contains access credentials.

Successful acquisition is not validation or promotion evidence. The asset must still be verified, converted into point-in-time features or a governed model artifact, and pass calibration, sealed holdout, replay, and paper gates.

## Join financial-news AI features

After immutable news inference:

```bash
trading-integrations join-news-ai \
  --dataset .trading/datasets/cycle-01-calibration.jsonl \
  --inference .trading/features/fingpt-news.jsonl \
  --output .trading/datasets/cycle-01-calibration-news-ai.jsonl \
  --window-hours 24 \
  --max-items 5
```

For each dataset row, the join uses only inference records whose `knowledge_time` is at or before the feature timestamp. Later annotations are ignored. The appended features are confidence-weighted sentiment, average confidence, event intensity, and average entity count. Parse failures are excluded and recorded in metadata.

The target return is copied unchanged and is never passed into news inference or feature aggregation.
