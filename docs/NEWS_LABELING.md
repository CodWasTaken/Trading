# Labelled news evaluation

Category counts are not quality metrics. A higher or lower `other` rate can reflect better conservatism, worse recall, a changed source archive, or a different symbol mix. Use human-labelled samples to measure the classifier directly.

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

## 2. Create independent reviewer packets

Do not have multiple people edit one shared CSV. Create independent, balanced packets from the untouched pending template:

```bash
trading-news assign-reviewers \
  --labels .trading/labels/news-v3-review.csv \
  --output-dir .trading/labels/news-v3-reviewers \
  --reviewers analyst-a,analyst-b,analyst-c \
  --reviews-per-item 2 \
  --seed news-v3-review-assignment
```

Every sampled item is assigned to exactly `--reviews-per-item` reviewers. Assignment uses a deterministic least-loaded algorithm with a hash tie-break, so reviewer workloads differ by at most the unavoidable remainder. The command refuses templates that already contain labels and refuses to overwrite existing packets.

The output directory contains one CSV per reviewer and `review-assignments.json`, which records packet hashes, reviewer workloads, and pairwise overlap.

## 3. Review each packet independently

Do not edit these source and prediction columns:

- `record_sha256`
- `event_id`, `symbol`, `event_time`, `knowledge_time`
- `source`, `headline`, `summary`
- every `predicted_*` column

Each row hashes those immutable columns. Agreement, adjudication, and evaluation fail if they change. Text beginning with spreadsheet formula characters is prefixed with an apostrophe in the review CSV.

Edit only:

- `gold_primary_event_type`
- `gold_secondary_event_types_json`
- `gold_entities_json`
- `review_status`
- `reviewer`
- `notes`

The reviewer name is prefilled by the packet generator and should remain unchanged throughout that file.

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

Reviewers should not compare answers before agreement is measured. Discussion before independent submission inflates agreement and hides ambiguous taxonomy definitions.

## 4. Measure inter-annotator agreement

After reviewers finish their independent packets:

```bash
trading-news review-agreement \
  --labels \
    .trading/labels/news-v3-reviewers/news-v3-review.analyst-a.csv \
    .trading/labels/news-v3-reviewers/news-v3-review.analyst-b.csv \
    .trading/labels/news-v3-reviewers/news-v3-review.analyst-c.csv \
  --output .trading/labels/news-v3-agreement.json \
  --require-complete
```

The agreement report contains:

- pairwise observed primary-label agreement
- expected agreement and Cohen's kappa for every reviewer pair
- directional confusion tables between reviewers
- exact-match, mean-Jaccard, and micro-F1 agreement for secondary labels
- exact entity-link and relation-only agreement
- unanimous and disputed primary-label counts
- overlap and completion coverage

Agreement is not classifier accuracy. It measures whether the labeling policy is sufficiently clear for humans to produce reproducible gold labels.

## 5. Create a unanimous-only consensus file

```bash
trading-news adjudicate-labels \
  --labels \
    .trading/labels/news-v3-reviewers/news-v3-review.analyst-a.csv \
    .trading/labels/news-v3-reviewers/news-v3-review.analyst-b.csv \
    .trading/labels/news-v3-reviewers/news-v3-review.analyst-c.csv \
  --output .trading/labels/news-v3-consensus.csv \
  --minimum-reviewers 2
```

The command automatically accepts a primary gold label only when all available labeled reviewers agree and the minimum reviewer count is met. Primary disagreements remain `pending`, have their gold fields cleared, and include a compact disagreement summary in `notes`.

Optional secondary and entity fields are carried into the consensus file only when every contributing reviewer supplied a nonblank semantically identical value. Otherwise those optional fields remain blank for separate adjudication.

A row marked `skip` by at least the required number of reviewers, with no labeled review, becomes a consensus skip. All other under-reviewed rows remain pending.

A human adjudicator must resolve every pending disagreement without consulting future returns. Preserve the consensus CSV as the final gold artifact and do not edit reviewer packets afterward.

## 6. Evaluate adjudicated labels

Partial reports are useful during adjudication:

```bash
trading-news evaluate-labels \
  --labels .trading/labels/news-v3-consensus.csv \
  --output .trading/labels/news-v3-evaluation.json
```

Require every row to be resolved before producing a final report:

```bash
trading-news evaluate-labels \
  --labels .trading/labels/news-v3-consensus.csv \
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

Do not tune taxonomy rules on this consensus CSV and then call the same report independent validation. Keep a separate untouched archive or separately sampled holdout for final comparison. Agreement should be reviewed before adjudication: low kappa or concentrated confusion usually means the labeling policy needs clarification before model changes.
