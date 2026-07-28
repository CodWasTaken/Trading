# News intelligence and entity linking

Version 0.9.0 adds an auditable enrichment layer for historical and live news.

## What changes

- Event classification uses weighted phrases and keywords instead of first-keyword wins.
- One event can have a primary event type and secondary event types.
- Exact deduplication is scoped by ticker, source, and normalized headline. A multi-symbol article is retained once for every provider-supplied ticker.
- Similar articles receive a deterministic `article_version` while novelty remains point-in-time and symbol-scoped.
- Percentages, currency amounts, EPS values, and impact direction are extracted into typed event attributes.
- Every event receives a provider-symbol issuer link.
- Subsidiary, supplier, customer, competitor, partner, regulator, and person links require an explicit catalog alias match.

The deterministic extractor is a transparent baseline. It does not claim labelled-model accuracy, and relationships are never inferred merely because two names occur in the same article.

## Entity catalog

Start from `examples/entity_catalog.json`. The catalog is deliberately explicit:

```json
{
  "issuers": [
    {
      "ticker": "ACME",
      "canonical_name": "Acme Holdings",
      "aliases": ["Acme"],
      "sector": "Industrials",
      "related_entities": [
        {
          "canonical_name": "Precision Parts Ltd",
          "relation": "supplier",
          "aliases": ["Precision Parts"],
          "ticker": "PPL"
        }
      ]
    }
  ]
}
```

Aliases are matched as complete normalized phrases. Ambiguous aliases within an issuer profile fail validation instead of choosing one relationship silently.

Validate a catalog:

```bash
trading-news validate-catalog --catalog .trading/entity-catalog.json
```

Enable it for new historical backfills and the live news stream:

```dotenv
TRADING_ENTITY_CATALOG_PATH=.trading/entity-catalog.json
```

## Reclassify an existing archive

The migration command writes a new JSON Lines archive and refuses to overwrite the source:

```bash
trading-news reclassify \
  --input .trading/history/news.jsonl \
  --output .trading/history/news-v2.jsonl \
  --catalog .trading/entity-catalog.json \
  --report .trading/history/news-v2.report.json
```

It preserves event IDs, `event_time`, and `knowledge_time`. The report includes old and new event-type counts, catch-all rates, secondary classifications, entity relation counts, numeric extraction coverage, source/output hashes, and catalog coverage.

An old archive cannot recover ticker copies that were already discarded by the earlier cross-symbol deduplication behavior. Re-run the historical news backfill to recover those records, then use the new archive for replay and dataset construction.

## Suggested validation sequence

1. Reclassify the existing archive and inspect the change in the `other` rate.
2. Manually label a stratified sample across every event type, including `other`.
3. Report precision, recall, and confusion by primary and secondary event type.
4. Re-run deterministic replay with the migrated archive.
5. Compare event-type concentration, turnover, drawdown, and returns on a later untouched window.

Do not select taxonomy rules or trading thresholds on the same period used as final evidence.
