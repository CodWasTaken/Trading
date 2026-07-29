import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from trading_app.config import Settings
from trading_app.domain import Fill, Quote, Side, SignalProposal
from trading_app.historical import HistoricalBar
from trading_app.portfolio import Portfolio
from trading_app.replay import ReplayClock, run_historical_replay
from trading_app.replay_cli import replay_history
from trading_app.risk import RiskEngine
from trading_app.strategy import ExplainableCatalystStrategy


def make_bars(start: datetime, days: int = 2) -> list[HistoricalBar]:
    bars: list[HistoricalBar] = []
    for day in range(days):
        session = start + timedelta(days=day)
        for index in range(8):
            for symbol, offset in (("AAPL", 0.0), ("MSFT", 30.0)):
                price = 100 + offset + day * 3 + index * 2
                bars.append(
                    HistoricalBar(
                        symbol=symbol,
                        timestamp=session + timedelta(hours=index),
                        open=price - 0.5,
                        high=price + 0.5,
                        low=price - 1.0,
                        close=price,
                        volume=10_000,
                    )
                )
    return bars


def replay_settings() -> Settings:
    return Settings(
        trading_starting_cash=100_000,
        trading_min_confidence=0.5,
        trading_max_spread_bps=35,
        trading_max_data_age_seconds=30,
        trading_max_trades_per_day=2,
        trading_max_position_pct=0.05,
        trading_max_gross_exposure_pct=0.60,
    )


def test_risk_engine_accepts_injected_historical_clock() -> None:
    now = datetime(2025, 1, 2, 15, tzinfo=UTC)
    clock = ReplayClock(now)
    engine = RiskEngine(replay_settings(), clock=clock)
    quote = Quote(
        symbol="AAPL",
        bid=99.95,
        ask=100.05,
        event_time=now,
        knowledge_time=now,
    )
    from trading_app.domain import PortfolioSnapshot, SignalProposal

    proposal = SignalProposal(
        symbol="AAPL",
        side=Side.BUY,
        confidence=0.8,
        expected_return=0.01,
        target_notional=1_000,
        reference_price=100,
        rationale=["historical test"],
        feature_snapshot={},
    )
    snapshot = PortfolioSnapshot(
        cash=100_000,
        equity=100_000,
        gross_exposure=0,
        daily_pnl=0,
        drawdown=0,
        positions=[],
        trades_today=0,
    )
    decision = engine.evaluate(proposal, snapshot, quote)
    assert "stale_market_data" not in decision.reasons


def test_portfolio_resets_daily_controls_on_new_session() -> None:
    portfolio = Portfolio(100_000)
    first = datetime(2025, 1, 2, 15, tzinfo=UTC)
    quote = Quote(
        symbol="AAPL",
        bid=99.95,
        ask=100.05,
        event_time=first,
        knowledge_time=first,
    )
    portfolio.mark(quote)
    portfolio.rollover_session(first)
    portfolio.apply_fill(
        Fill(
            order_id="11111111-1111-1111-1111-111111111111",
            symbol="AAPL",
            side=Side.BUY,
            quantity=10,
            price=100,
            slippage_bps=0,
        )
    )
    assert portfolio.trades_today == 1

    next_day = first + timedelta(days=1)
    portfolio.mark(
        Quote(
            symbol="AAPL",
            bid=100.95,
            ask=101.05,
            event_time=next_day,
            knowledge_time=next_day,
        )
    )
    assert portfolio.rollover_session(next_day) is True
    assert portfolio.trades_today == 0
    assert portfolio.day_start_equity == pytest.approx(100_010)


@pytest.mark.asyncio
async def test_shared_engine_replay_is_semantically_deterministic() -> None:
    bars = make_bars(datetime(2025, 1, 2, 14, 30, tzinfo=UTC))

    async def execute():
        return await run_historical_replay(
            bars,
            [],
            replay_settings(),
            ExplainableCatalystStrategy,
            strategy_name="explainable",
            bar_interval=timedelta(hours=1),
            spread_bps=10,
            slippage_bps=2,
        )

    first = await execute()
    second = await execute()
    first_events = first.report["events"]
    second_events = second.report["events"]
    assert isinstance(first_events, dict)
    assert isinstance(second_events, dict)
    assert first_events["trace_sha256"] == second_events["trace_sha256"]
    assert first.report["sessions"] == 2
    assert first_events["counts"]["proposal"] > 0
    trace_text = json.dumps(first.trace)
    assert "proposal_id" not in trace_text
    assert "order_id" not in trace_text


@pytest.mark.asyncio
async def test_replay_cli_writes_report_trace_and_verifies_twice(tmp_path: Path) -> None:
    bars_path = tmp_path / "bars.jsonl"
    news_path = tmp_path / "news.jsonl"
    report_path = tmp_path / "replay.json"
    trace_path = tmp_path / "replay.jsonl"
    bars = make_bars(datetime(2025, 1, 2, 14, 30, tzinfo=UTC), days=1)
    bars_path.write_text("".join(bar.model_dump_json() + "\n" for bar in bars))
    news_path.write_text("")

    report = await replay_history(
        str(bars_path),
        str(news_path),
        str(report_path),
        trace_output=str(trace_path),
        strategy_mode="explainable",
        registry_path=str(tmp_path / "models"),
        allow_synthetic_champion=False,
        verify_determinism=True,
        bar_minutes=60,
        spread_bps=10,
        slippage_bps=2,
        starting_cash=100_000,
    )

    assert report["determinism_verified"] is True
    assert report_path.exists()
    assert trace_path.exists()
    assert json.loads(report_path.read_text())["events"]["trace_sha256"]


class ScriptedLongShortStrategy:
    def __init__(self) -> None:
        self._steps = iter(
            (
                (Side.SELL, 1_000.0),
                (Side.BUY, 500.0),
                (Side.BUY, 1_000.0),
            )
        )

    def on_quote(
        self,
        quote: Quote,
        news: list[object],
        equity: float,
    ) -> SignalProposal | None:
        del news, equity
        try:
            side, target = next(self._steps)
        except StopIteration:
            return None
        return SignalProposal(
            symbol=quote.symbol,
            side=side,
            confidence=0.99,
            expected_return=0.01 if side == Side.BUY else -0.01,
            target_notional=target,
            reference_price=quote.mid,
            rationale=["deterministic long/short replay test"],
            feature_snapshot={},
        )


@pytest.mark.asyncio
async def test_replay_deterministically_opens_covers_and_reverses_short() -> None:
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    bars = [
        HistoricalBar(
            symbol="AAPL",
            timestamp=start + timedelta(hours=index),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price,
            volume=10_000,
        )
        for index, price in enumerate((100.0, 95.0, 90.0))
    ]
    settings = replay_settings().model_copy(
        update={
            "trading_easy_to_borrow_symbols": ["AAPL"],
            "trading_max_trades_per_day": 10,
        }
    )

    async def execute():
        return await run_historical_replay(
            bars,
            [],
            settings,
            ScriptedLongShortStrategy,
            strategy_name="scripted-long-short",
            bar_interval=timedelta(hours=1),
            spread_bps=10,
            slippage_bps=0,
        )

    first = await execute()
    second = await execute()
    assert first.report["events"]["trace_sha256"] == second.report["events"]["trace_sha256"]
    effects = [
        event["payload"]["effect"]
        for event in first.trace
        if event["type"] == "position_transition"
    ]
    assert effects == [
        "open_short",
        "partial_cover_short",
        "reverse_short_to_long",
    ]
    performance = first.report["performance"]
    assert performance["short_risk_attribution"]["realized_pnl"] > 0


@pytest.mark.asyncio
async def test_replay_deterministically_rejects_short_without_borrow() -> None:
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    bars = [
        HistoricalBar(
            symbol="AAPL",
            timestamp=start,
            open=100,
            high=101,
            low=99,
            close=100,
            volume=10_000,
        )
    ]
    result = await run_historical_replay(
        bars,
        [],
        replay_settings(),
        ScriptedLongShortStrategy,
        strategy_name="scripted-short-rejection",
        bar_interval=timedelta(hours=1),
    )
    assert result.report["events"]["risk_rejection_reasons"] == {
        "borrow_status_missing": 1
    }
    assert result.report["events"]["counts"].get("fill", 0) == 0
