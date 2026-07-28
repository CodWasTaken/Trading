# Replay diagnostics

Version 0.8.0 adds deterministic diagnostics on top of the shared historical replay engine. These reports rerun the full strategy → risk → paper broker path for every requested threshold; they do not infer results by filtering a previous run.

## Run a report

Create an optional sector map:

```json
{
  "AAPL": "Technology",
  "MSFT": "Technology",
  "NVDA": "Technology"
}
```

Then run:

```bash
trading-research replay-report \
  --bars .trading/history/bars.jsonl \
  --news .trading/history/news.jsonl \
  --strategy explainable \
  --thresholds 0.08,0.12,0.16,0.20,0.24 \
  --reference-threshold 0.16 \
  --sector-map .trading/sectors.json \
  --bar-minutes 60 \
  --spread-bps 10 \
  --slippage-bps 2 \
  --regime-lookback 20 \
  --regime-momentum-threshold 0.01 \
  --verify-determinism \
  --output .trading/replays/explainable-diagnostics-2025.json
```

For champion replay, thresholds are expected-return edges rather than explainable combined scores. A synthetic champion remains disabled unless `--allow-synthetic-champion` is explicitly supplied for pipeline diagnostics.

## Report sections

### Threshold sensitivity

Each threshold receives a fresh strategy instance, portfolio, risk engine, broker, and historical clock. The report compares:

- ending equity and net return
- maximum drawdown
- traded notional
- proposal and fill counts
- risk rejection reasons
- semantic replay trace hash
- deltas versus the selected reference threshold

Use this section to identify fragility, not to select a winning threshold on the same period. Any threshold chosen after inspecting this report needs an untouched evaluation window.

### Regimes

The report builds an equal-weight return series from the replay symbols and aligns it to replay equity timestamps. It labels retrospective diagnostic buckets using rolling market momentum and a median volatility split:

- bull / sideways / bear
- low volatility / high volatility

Each bucket reports strategy and market returns, strategy hit rate, average market momentum and volatility, and strategy maximum drawdown. These labels are diagnostics only; they are not supplied to the strategy and therefore cannot create look-ahead leakage in the replay.

### Sectors

A user-supplied JSON object maps symbols to sector names. The report aggregates symbol cash-flow P&L, fill count, and filled notional into those groups. Symbols absent from the map are placed in `Unmapped` rather than guessed.

### Event types

Every proposal records the latest news event type and source visible at proposal time. The report groups proposals, approvals/resizes, rejections, fills, and filled notional by that point-in-time event type. `none` indicates a proposal with no visible news context.

## Interpretation boundaries

- Historical bars do not contain executable bid/ask quotes. Replay uses the explicit synthetic spread and slippage assumptions only for execution and risk simulation.
- Historical provider receipt time may be unavailable, so historical news `knowledge_time` can equal publication time.
- Symbol P&L is cash flow plus the final marked value of any remaining quantity; it is attribution, not a substitute for the portfolio equity curve.
- Threshold sweeps, regime splits, and sector/event slices increase multiple-testing risk.
- Historical replay cannot replace 60–90 trading days of live paper evidence, reconciliation checks, incident review, and independent production approval.
