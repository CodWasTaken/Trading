# Labelled news evaluation

Category counts are not quality metrics. A higher or lower `other` rate can reflect better conservatism, worse recall, a changed source archive, or a different symbol mix. Use a human-labelled sample to measure the classifier directly.

## 1. Create a deterministic sample

```bash
mkdir -p .trading/labels

trading-news sample-labels \
  --input .trading/history/news-v3-full.jsonl \
  --output .trading/labels/news-v3-review.csv \
  --size 400 \
  --minimum-per-stratum 3 \
  --seed news-v3-baseline
```

The sampler first takes up to the requested minimum from every observed `predicted event type × symbol` stratum, then fills the remaining rows by a deterministic SHA-256 rank. Reusing the same archive, seed, size, and minimum produces the same CSV.

A sidecar named `news-v3-review.metadata.json` records the source hash, sampling design, selected strata, allowed labels, and point-in-time controls.

## 2. Review the CSV

Do not edit these source and prediction columns:

- `record_sha256`
- `event_id`, `symbol`, `event_time`, `knowledge_time`
- `source`, `headline`, `summary`
- every `predicted_*` column

Each row hashes those immutable columns. Evaluation fails if they change. Text beginning with spreadsheet formula characters is prefixed with an apostrophe in the review CSV.

Edit only:

- `gold_primary_event_type`
- `gold_secondary_event_types_json`
- `gold_entities_json`
- `review_status`
- `reviewer`
- `notes`

Set `review_status` to:

- `labeled` when the gold primary label is complete
- `skip` when the item cannot be judged from the archived text
- `pending` while unfinished

`gold_secondary_event_types_json` is optional. Use `[]` to say that the item was reviewed and has no secondary events. Leave it blank to exclude the row from secondary-event metrics.

`gold_entities_json` is also optional. Use objects with `relation` plus either `ticker` or `canonical_name`:

```json
[
  {
    "relation": "issuer",
    "ticker": "AAPL",
    "canonical_name": "Apple Inc."
  },
  {
    "relation": "supplier",
    "ticker": "SWKS",
    "canonical_name": "Skyworks Solutions"
  }
]
```

Use `[]` to say the entity field was reviewed and no links should exist. Leave it blank to exclude the row from entity metrics.

## 3. Evaluate completed labels

Partial reports are useful during review:

```bash
trading-news evaluate-labels \
  --labels .trading/labels/news-v3-review.csv \
  --output .trading/labels/news-v3-evaluation.json
```

Require every row to be resolved before producing a final report:

```bash
trading-news evaluate-labels \
  --labels .trading/labels/news-v3-review.csv \
  --output .trading/labels/news-v3-evaluation.json \
  --require-complete
```

The report contains:

- primary-event accuracy, macro and weighted F1
- precision, recall, F1, support, and prediction counts for every event type
- a primary-event confusion matrix
- gold and predicted `other` rates on the same labelled rows
- secondary-event multilabel precision, recall, F1, and exact-match rate
- exact entity-link metrics
- relation-only entity metrics
- review status and reviewer counts
- source archive and sampling metadata

## Interpretation boundaries

A stratified sample deliberately differs from the archive's natural class distribution. Per-class precision and recall are the primary outputs. Overall accuracy and the sample's `other` rate should not be projected directly onto the full archive without weighting.

Do not tune taxonomy rules on this labelled CSV and then call the same report independent validation. Keep a separate untouched archive or label set for the final comparison. Multiple independent reviewers are required to estimate inter-annotator agreement.
