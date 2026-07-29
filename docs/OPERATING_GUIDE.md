# Research-to-paper operating guide

This workflow preserves point-in-time information, sealed evaluation, and the
paper-only safety boundary.

## 1. Freeze the universe

Select the versioned universe manifest effective for the experiment date.
Archive the inclusion evidence and source hashes. Never substitute today’s
largest companies into an earlier period. Every dataset, candidate generation,
holdout, replay, and paper run must carry the same manifest hash.

## 2. Collect immutable bars and news

Archive raw hourly bars and news before normalization. Preserve event time,
knowledge time, provider identifiers, revisions, and source hashes. News is
eligible only when `knowledge_time <= decision_time`. Do not fabricate human
labels: independently complete the existing reviewer packets before requiring
review agreement.

## 3. Build point-in-time features

Compute each feature using only facts known at the hourly decision time. The
five-hour target is the realized return over the next five hourly bars and is a
historical training label. It is never a live feature and is never available to
the live model at decision time. News is one feature family, not an order.

## 4. Seal calibration and holdout splits

Use chronological splits and purge every training label whose five-bar outcome
crosses a later boundary. Hash the inputs, outputs, cost manifest, universe
manifest, and statistical plan. Previously inspected periods cannot later be
declared untouched.

## 5. Run 100-candidate calibration generations

Generate or supply exactly 100 unique deterministic configurations. Evaluate
them only with nested, purged, chronological calibration folds. Keep every
configuration, failure, metric, and artifact. Use bounded parallelism and
resumption. Run no more than ten generations and stop after three without the
predeclared meaningful improvement.

## 6. Select three finalists

Rank by the predeclared composite score, not raw return. Freeze exactly the top
three eligible configurations and declare `candidate_family_size=3`. Calibration
results do not authorize promotion.

## 7. Score finalists once

Score only the three frozen finalists on the sealed holdout, once per model and
holdout hash. Apply the declared multiple-comparison correction. Never score all
100 candidates or tune thresholds on the holdout.

## 8. Promote or reject

Apply the versioned fail-closed calibration, holdout, concentration, and short
safety gates. Promote at most one historical champion. Select no champion when
all finalists fail. Calibration quality alone, including calibration curves,
cannot justify promotion.

## 9. Run deterministic replay

Replay the champion through the shared strategy, risk, signed portfolio, and
paper broker engine twice. Require identical semantic traces. Review long,
short, combined, news/non-news, symbol, sector, event-type, regime, borrow, and
financing attribution.

## 10. Deploy to Alpaca paper

Configure only `TRADING_EXECUTION_MODE=alpaca-paper` and the paper API endpoint.
Verify the frozen universe and cost manifests at startup. Reconcile actual
paper fills and fail closed on missing borrow status. This repository has no
live-money execution mode.

## 11. Monitor for 60–90 trading days

Collect at least 60 trading days; prefer 90. Reconcile every day with zero
unexplained differences. Track data-quality incidents, borrow violations, drift,
calibration, cost-adjusted combined/long/short performance, tail loss, drawdown,
and symbol/sector/event/regime dependence. Run `trading-graduation` against the
immutable ledger-backed evidence. Passing means continued paper evaluation only.

## 12. Roll back and retrain

Pause paper execution on unhealthy monitoring, reconciliation gaps, borrow
violations, or gate failures. Roll the champion alias back through registry
history without deleting artifacts. Retraining begins a new declared candidate
generation and uses fresh calibration/holdout evidence; it never reopens or
relabels an inspected sealed period.

Polish capital-gains tax remains a separate estimate, not a per-trade execution
cost and not tax advice. No synthetic fixture, calibration result, historical
period, or paper graduation establishes future profitability.
