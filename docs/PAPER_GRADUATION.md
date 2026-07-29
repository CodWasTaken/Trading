# Paper-only graduation

Paper graduation is a fail-closed operational review of a historically promoted
champion. It never changes the execution mode and never authorizes real-money
trading.

## Required evidence

The operator exports a JSON evidence document with:

- the exact champion model version and `alpaca-paper` execution mode;
- at least 60 unique chronological trading sessions (90 preferred);
- daily ending equity;
- reconciled long/short trade attribution by symbol, sector, news use, event
  type, and regime;
- execution, borrow, margin-interest, dividend-replacement, and optional FX
  funding costs;
- daily broker reconciliation observations;
- operational and data-quality incidents; and
- a path and SHA-256 for the immutable source ledger.

The evaluator verifies the source ledger itself. It also requires a deterministic
shared-engine replay, replay diagnostics, and one or more healthy drift and
calibration reports for the exact champion version.

## Gates

The versioned defaults in `config/graduation/paper-only-v1.json` require:

- 60 paper trading days;
- zero unexplained reconciliation differences;
- zero unresolved data-quality incidents;
- zero borrow violations;
- positive combined, long, and short net PnL after execution and financing
  costs;
- drawdown below 15% and 95% tail loss no greater than 5%;
- no symbol above 20% of positive net PnL, sector above 35%, event type above
  50%, or regime above 50%; and
- exact PnL reconciliation between daily equity and attribution.

Missing attribution, monitoring, hashes, replay evidence, or borrow validation
fails closed. A failed review still creates an immutable registry event so the
decision cannot disappear.

## Run

```bash
trading-graduation \
  --registry .trading/models \
  --paper-evidence .trading/graduation/paper-evidence.json \
  --replay .trading/replay/report.json \
  --replay-diagnostics .trading/replay/diagnostics.json \
  --monitoring .trading/monitoring/latest.json \
  --gate-config config/graduation/paper-only-v1.json \
  --output .trading/graduation/report.json
```

The command exits with status 2 when any gate fails. Outputs are write-once, and
the same champion/evidence hash cannot be evaluated twice.

Passing writes `paper_only_graduated`, `execution_scope=paper_only`, and
`live_money_authorized=false`. Historical and paper results are not evidence of
future profitability.
