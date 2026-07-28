from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .broker import InternalPaperBroker
from .config import Settings
from .domain import DecisionStatus, LiveEvent, NewsEvent, Quote
from .engine import TradingEngine
from .historical import HistoricalBar
from .portfolio import Portfolio
from .risk import RiskEngine
from .store import EventStore
from .strategy import Strategy


@dataclass
class ReplayClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current

    def set(self, value: datetime) -> None:
        self.current = value.astimezone(UTC)


@dataclass(frozen=True)
class ReplayResult:
    report: dict[str, object]
    trace: list[dict[str, object]]


def quote_from_bar(
    bar: HistoricalBar,
    *,
    bar_interval: timedelta,
    spread_bps: float,
) -> Quote:
    if spread_bps < 0:
        raise ValueError("spread_bps must be non-negative")
    feature_time = bar.timestamp.astimezone(UTC) + bar_interval
    half_spread = bar.close * spread_bps / 20_000
    bid = bar.close - half_spread
    ask = bar.close + half_spread
    if bid <= 0 or ask <= bid:
        raise ValueError("spread_bps produced an invalid synthetic quote")
    return Quote(
        symbol=bar.symbol,
        bid=bid,
        ask=ask,
        event_time=feature_time,
        knowledge_time=feature_time,
    )


async def run_historical_replay(
    bars: list[HistoricalBar],
    news: list[NewsEvent],
    settings: Settings,
    strategy_factory: Callable[[], Strategy],
    *,
    strategy_name: str,
    bar_interval: timedelta,
    spread_bps: float = 10.0,
    slippage_bps: float = 2.0,
    source_metadata: dict[str, object] | None = None,
) -> ReplayResult:
    if not bars:
        raise ValueError("Historical replay requires at least one bar")
    if bar_interval <= timedelta(0):
        raise ValueError("bar_interval must be positive")
    if settings.trading_starting_cash <= 0:
        raise ValueError("trading_starting_cash must be positive")
    if slippage_bps < 0:
        raise ValueError("slippage_bps must be non-negative")

    ordered_bars = sorted(
        bars,
        key=lambda item: (item.timestamp.astimezone(UTC), item.symbol),
    )
    symbols = sorted({bar.symbol for bar in ordered_bars})
    symbol_set = set(symbols)
    ordered_news = sorted(
        (item for item in news if item.symbol in symbol_set),
        key=lambda item: (
            item.knowledge_time.astimezone(UTC),
            item.event_time.astimezone(UTC),
            item.symbol,
            item.headline,
        ),
    )
    first_time = ordered_bars[0].timestamp.astimezone(UTC) + bar_interval
    clock = ReplayClock(first_time)
    maximum_events = max(2_000, (len(ordered_bars) + len(ordered_news)) * 8 + 100)
    store = EventStore(max_events=maximum_events)
    portfolio = Portfolio(settings.trading_starting_cash)
    risk = RiskEngine(settings, clock=clock)
    engine = TradingEngine(
        store,
        portfolio,
        risk,
        strategy_factory(),
        InternalPaperBroker(slippage_bps=slippage_bps),
    )

    news_cursor = 0
    maximum_drawdown = 0.0
    peak_equity = settings.trading_starting_cash
    equity_points = 0
    for bar in ordered_bars:
        quote = quote_from_bar(
            bar,
            bar_interval=bar_interval,
            spread_bps=spread_bps,
        )
        clock.set(quote.knowledge_time)
        while (
            news_cursor < len(ordered_news)
            and ordered_news[news_cursor].knowledge_time <= quote.knowledge_time
        ):
            await engine.process_news(ordered_news[news_cursor])
            news_cursor += 1
        await engine.process_quote(quote)
        snapshot = portfolio.snapshot()
        peak_equity = max(peak_equity, snapshot.equity)
        if peak_equity > 0:
            maximum_drawdown = max(
                maximum_drawdown,
                (peak_equity - snapshot.equity) / peak_equity,
            )
        equity_points += 1

    final_time = ordered_bars[-1].timestamp.astimezone(UTC) + bar_interval
    final_snapshot = portfolio.snapshot()
    final_portfolio = final_snapshot.model_dump(mode="json")
    final_portfolio["as_of"] = final_time.isoformat()

    chronological_events = list(reversed(store.live_events))
    trace = [_canonical_event(index, event) for index, event in enumerate(chronological_events)]
    trace_bytes = b"".join(
        (
            json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for event in trace
    )
    event_counts = Counter(event.type for event in chronological_events)
    rejection_reasons: Counter[str] = Counter()
    for decision in store.decisions:
        if decision.status == DecisionStatus.REJECTED:
            rejection_reasons.update(decision.reasons)

    fills_by_symbol: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"fills": 0, "notional": 0.0}
    )
    traded_notional = 0.0
    for fill in store.fills:
        notional = fill.quantity * fill.price
        traded_notional += notional
        fills_by_symbol[fill.symbol]["fills"] = int(
            fills_by_symbol[fill.symbol]["fills"]
        ) + 1
        fills_by_symbol[fill.symbol]["notional"] = float(
            fills_by_symbol[fill.symbol]["notional"]
        ) + notional

    starting_cash = settings.trading_starting_cash
    report: dict[str, object] = {
        "schema_version": 1,
        "replay_kind": "shared_engine_historical_replay",
        "strategy": strategy_name,
        "symbols": symbols,
        "start": first_time.isoformat(),
        "end": final_time.isoformat(),
        "sessions": len(
            {bar.timestamp.astimezone(UTC).date() for bar in ordered_bars}
        ),
        "bars": len(ordered_bars),
        "news_visible": news_cursor,
        "news_after_replay_end": len(ordered_news) - news_cursor,
        "configuration": {
            "starting_cash": starting_cash,
            "bar_interval_seconds": bar_interval.total_seconds(),
            "synthetic_spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "max_position_pct": settings.trading_max_position_pct,
            "max_gross_exposure_pct": settings.trading_max_gross_exposure_pct,
            "max_daily_loss_pct": settings.trading_max_daily_loss_pct,
            "max_drawdown_pct": settings.trading_max_drawdown_pct,
            "minimum_confidence": settings.trading_min_confidence,
            "max_spread_bps": settings.trading_max_spread_bps,
            "max_trades_per_day": settings.trading_max_trades_per_day,
        },
        "performance": {
            "ending_equity": final_snapshot.equity,
            "net_return": final_snapshot.equity / starting_cash - 1,
            "maximum_drawdown": maximum_drawdown,
            "traded_notional": traded_notional,
            "equity_points": equity_points,
        },
        "final_portfolio": final_portfolio,
        "events": {
            "counts": dict(sorted(event_counts.items())),
            "risk_rejection_reasons": dict(sorted(rejection_reasons.items())),
            "fills_by_symbol": {
                symbol: values for symbol, values in sorted(fills_by_symbol.items())
            },
            "trace_events": len(trace),
            "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        },
        "sources": source_metadata or {},
        "limitations": [
            "Historical OHLC bars have no executable bid/ask quotes; replay uses the configured synthetic spread solely for execution and risk simulation.",
            "Historical news knowledge_time may equal publication time because provider receipt time is unavailable.",
            "Replay evidence is not evidence of future profitability and does not replace live paper validation.",
        ],
    }
    return ReplayResult(report=report, trace=trace)


def _canonical_event(sequence: int, event: LiveEvent) -> dict[str, object]:
    volatile_keys = {
        "id",
        "proposal_id",
        "order_id",
        "created_at",
        "checked_at",
        "filled_at",
    }
    payload: dict[str, Any] = {
        key: value for key, value in event.payload.items() if key not in volatile_keys
    }
    return {
        "sequence": sequence,
        "type": event.type,
        "payload": payload,
    }
