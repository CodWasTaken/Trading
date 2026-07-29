from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from trading_app.borrow import (
    BorrowRecall,
    BorrowState,
    BorrowStatus,
    StaticBorrowProvider,
    forced_cover_for,
)
from trading_app.broker import InternalPaperBroker
from trading_app.config import Settings
from trading_app.domain import (
    DecisionStatus,
    Fill,
    PortfolioSnapshot,
    PositionEffect,
    Quote,
    Side,
    SignalProposal,
)
from trading_app.engine import TradingEngine
from trading_app.portfolio import Portfolio
from trading_app.risk import RiskEngine
from trading_app.store import EventStore


def fill(side: Side, quantity: float, price: float, symbol: str = "AAPL") -> Fill:
    return Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        slippage_bps=0,
    )


def proposal(side: Side, notional: float) -> SignalProposal:
    return SignalProposal(
        symbol="AAPL",
        side=side,
        confidence=0.9,
        expected_return=0.01 if side == Side.BUY else -0.01,
        target_notional=notional,
        reference_price=100,
        rationale=["signed accounting test"],
        feature_snapshot={},
    )


def test_short_entry_increase_partial_cover_and_full_cover() -> None:
    portfolio = Portfolio(100_000)

    opened = portfolio.apply_fill(fill(Side.SELL, 10, 100))
    increased = portfolio.apply_fill(fill(Side.SELL, 10, 110))
    partial = portfolio.apply_fill(fill(Side.BUY, 5, 90))

    assert opened.effect == PositionEffect.OPEN_SHORT
    assert increased.effect == PositionEffect.INCREASE_SHORT
    assert partial.effect == PositionEffect.PARTIAL_COVER_SHORT
    snapshot = portfolio.snapshot()
    position = snapshot.positions[0]
    assert position.quantity == -15
    assert position.average_price == pytest.approx(105)
    assert snapshot.cash == pytest.approx(101_650)
    assert snapshot.short_realized_pnl == pytest.approx(75)
    assert snapshot.short_unrealized_pnl == pytest.approx(225)
    assert snapshot.equity == pytest.approx(100_300)
    assert snapshot.gross_short_exposure == pytest.approx(1_350)
    assert snapshot.net_exposure == pytest.approx(-1_350)

    covered = portfolio.apply_fill(fill(Side.BUY, 15, 95))
    assert covered.effect == PositionEffect.COVER_SHORT
    final = portfolio.snapshot()
    assert not final.positions
    assert final.cash == pytest.approx(100_225)
    assert final.short_realized_pnl == pytest.approx(225)


def test_long_and_short_reversals_reset_average_entry_price() -> None:
    portfolio = Portfolio(100_000)
    portfolio.apply_fill(fill(Side.BUY, 10, 100))

    to_short = portfolio.apply_fill(fill(Side.SELL, 15, 110))
    short = portfolio.snapshot().positions[0]
    assert to_short.effect == PositionEffect.REVERSE_LONG_TO_SHORT
    assert to_short.realized_pnl == pytest.approx(100)
    assert short.quantity == -5
    assert short.average_price == pytest.approx(110)

    to_long = portfolio.apply_fill(fill(Side.BUY, 8, 90))
    long = portfolio.snapshot().positions[0]
    assert to_long.effect == PositionEffect.REVERSE_SHORT_TO_LONG
    assert to_long.realized_pnl == pytest.approx(100)
    assert long.quantity == 3
    assert long.average_price == pytest.approx(90)
    assert portfolio.snapshot().realized_pnl == pytest.approx(200)


def test_short_financing_and_dividend_replacement_reduce_equity() -> None:
    portfolio = Portfolio(
        100_000,
        borrow_rate_annual=0.365,
        dividend_replacement_rate_annual=0.365,
    )
    start = datetime(2025, 1, 2, 15, tzinfo=UTC)
    portfolio.apply_fill(fill(Side.SELL, 10, 100))
    portfolio.accrue_financing(start)
    charged = portfolio.accrue_financing(start + timedelta(days=1))
    dividend = portfolio.apply_dividend("AAPL", 0.50)

    assert charged == pytest.approx(2.0)
    assert dividend == pytest.approx(5.0)
    snapshot = portfolio.snapshot()
    assert snapshot.borrow_costs == pytest.approx(1.0)
    assert snapshot.dividend_replacement_costs == pytest.approx(6.0)
    assert snapshot.equity == pytest.approx(99_993)


def test_short_open_fails_closed_without_borrow_status() -> None:
    now = datetime(2025, 1, 2, 15, tzinfo=UTC)
    engine = RiskEngine(Settings(), clock=lambda: now)
    decision = engine.evaluate(
        proposal(Side.SELL, 1_000),
        PortfolioSnapshot(
            cash=100_000,
            equity=100_000,
            gross_exposure=0,
            daily_pnl=0,
            drawdown=0,
            positions=[],
            trades_today=0,
        ),
        Quote(
            symbol="AAPL",
            bid=99.9,
            ask=100.1,
            event_time=now,
            knowledge_time=now,
        ),
    )
    assert decision.status == DecisionStatus.REJECTED
    assert decision.reasons == ["borrow_status_missing"]


def test_easy_to_borrow_allows_short_but_quantity_and_status_reject() -> None:
    now = datetime(2025, 1, 2, 15, tzinfo=UTC)
    quote = Quote(
        symbol="AAPL",
        bid=100,
        ask=100.1,
        event_time=now,
        knowledge_time=now,
    )
    snapshot = Portfolio(100_000).snapshot()

    available = StaticBorrowProvider(
        {
            "AAPL": BorrowStatus(
                symbol="AAPL",
                state=BorrowState.EASY_TO_BORROW,
                available_quantity=20,
                checked_at=now,
            )
        }
    )
    approved = RiskEngine(
        Settings(),
        clock=lambda: now,
        borrow_provider=available,
    ).evaluate(proposal(Side.SELL, 1_000), snapshot, quote)
    assert approved.status == DecisionStatus.APPROVED

    insufficient = RiskEngine(
        Settings(),
        clock=lambda: now,
        borrow_provider=available,
    ).evaluate(proposal(Side.SELL, 3_000), snapshot, quote)
    assert insufficient.status == DecisionStatus.REJECTED
    assert insufficient.reasons == ["borrow_quantity_unavailable"]

    unavailable = StaticBorrowProvider(
        {
            "AAPL": BorrowStatus(
                symbol="AAPL",
                state=BorrowState.UNAVAILABLE,
                checked_at=now,
            )
        }
    )
    rejected = RiskEngine(
        Settings(),
        clock=lambda: now,
        borrow_provider=unavailable,
    ).evaluate(proposal(Side.SELL, 1_000), snapshot, quote)
    assert rejected.reasons == ["borrow_unavailable"]


def test_stale_borrow_status_rejects_short() -> None:
    now = datetime(2025, 1, 2, 15, tzinfo=UTC)
    provider = StaticBorrowProvider(
        {
            "AAPL": BorrowStatus(
                symbol="AAPL",
                state=BorrowState.EASY_TO_BORROW,
                checked_at=now - timedelta(minutes=6),
            )
        }
    )
    rejected = RiskEngine(
        Settings(trading_borrow_status_max_age_seconds=300),
        clock=lambda: now,
        borrow_provider=provider,
    ).evaluate(
        proposal(Side.SELL, 1_000),
        Portfolio(100_000).snapshot(),
        Quote(
            symbol="AAPL",
            bid=100,
            ask=100.1,
            event_time=now,
            knowledge_time=now,
        ),
    )
    assert rejected.reasons == ["borrow_status_not_current"]


def test_sell_reducing_long_does_not_require_borrow_but_reversal_does() -> None:
    now = datetime(2025, 1, 2, 15, tzinfo=UTC)
    portfolio = Portfolio(100_000)
    portfolio.apply_fill(fill(Side.BUY, 10, 100))
    quote = Quote(
        symbol="AAPL",
        bid=100,
        ask=100.1,
        event_time=now,
        knowledge_time=now,
    )

    reducing = RiskEngine(Settings(), clock=lambda: now).evaluate(
        proposal(Side.SELL, 500),
        portfolio.snapshot(),
        quote,
    )
    reversing = RiskEngine(Settings(), clock=lambda: now).evaluate(
        proposal(Side.SELL, 1_500),
        portfolio.snapshot(),
        quote,
    )
    assert reducing.status == DecisionStatus.APPROVED
    assert reversing.status == DecisionStatus.REJECTED
    assert reversing.reasons == ["borrow_status_missing"]


def test_borrow_recall_creates_bounded_forced_cover_instruction() -> None:
    now = datetime(2025, 1, 2, 15, tzinfo=UTC)
    instruction = forced_cover_for(
        BorrowRecall(
            symbol="AAPL",
            recalled_quantity=4,
            effective_at=now,
        ),
        -10,
    )
    assert instruction is not None
    assert instruction.quantity == 4
    assert forced_cover_for(
        BorrowRecall(symbol="AAPL", effective_at=now),
        10,
    ) is None


def test_short_limit_resizes_independently_of_long_limit() -> None:
    now = datetime(2025, 1, 2, 15, tzinfo=UTC)
    settings = Settings(
        trading_max_long_position_pct=0.05,
        trading_max_short_position_pct=0.01,
    )
    provider = StaticBorrowProvider(
        {
            "AAPL": BorrowStatus(
                symbol="AAPL",
                state=BorrowState.EASY_TO_BORROW,
                checked_at=now,
            )
        }
    )
    decision = RiskEngine(
        settings,
        clock=lambda: now,
        borrow_provider=provider,
    ).evaluate(
        proposal(Side.SELL, 3_000),
        Portfolio(100_000).snapshot(),
        Quote(
            symbol="AAPL",
            bid=100,
            ask=100.1,
            event_time=now,
            knowledge_time=now,
        ),
    )
    assert decision.status == DecisionStatus.RESIZED
    assert decision.approved_notional == pytest.approx(1_000)


class NoSignalStrategy:
    def on_quote(
        self,
        quote: Quote,
        news: list[object],
        equity: float,
    ) -> None:
        del quote, news, equity


@pytest.mark.asyncio
async def test_forced_cover_uses_paper_broker_and_is_audited() -> None:
    now = datetime(2025, 1, 2, 15, tzinfo=UTC)
    portfolio = Portfolio(100_000)
    portfolio.apply_fill(fill(Side.SELL, 10, 100))
    store = EventStore()
    engine = TradingEngine(
        store=store,
        portfolio=portfolio,
        risk=RiskEngine(Settings(), clock=lambda: now),
        strategy=NoSignalStrategy(),
        broker=InternalPaperBroker(slippage_bps=0),
    )
    instruction = forced_cover_for(
        BorrowRecall(
            symbol="AAPL",
            recalled_quantity=4,
            effective_at=now,
        ),
        portfolio.quantity("AAPL"),
    )
    assert instruction is not None

    await engine.process_forced_cover(
        instruction,
        Quote(
            symbol="AAPL",
            bid=89.9,
            ask=90,
            event_time=now,
            knowledge_time=now,
        ),
    )

    assert portfolio.quantity("AAPL") == pytest.approx(-6)
    assert portfolio.transitions[-1].effect == PositionEffect.PARTIAL_COVER_SHORT
    ledger = await store.snapshot()
    assert ledger["orders"][0]["position_effect"] == "partial_cover_short"
    assert ledger["system_events"][1]["type"] == "borrow_recall"
