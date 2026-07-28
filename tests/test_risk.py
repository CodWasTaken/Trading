from datetime import UTC, datetime, timedelta

from trading_app.config import Settings
from trading_app.domain import (
    DecisionStatus,
    PortfolioSnapshot,
    Quote,
    Side,
    SignalProposal,
)
from trading_app.risk import RiskEngine


def snapshot() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        cash=100_000,
        equity=100_000,
        gross_exposure=0,
        daily_pnl=0,
        drawdown=0,
        positions=[],
        trades_today=0,
    )


def proposal(confidence: float = 0.8, notional: float = 4_000) -> SignalProposal:
    return SignalProposal(
        symbol="AAPL",
        side=Side.BUY,
        confidence=confidence,
        expected_return=0.01,
        target_notional=notional,
        reference_price=200,
        rationale=["test"],
        feature_snapshot={},
    )


def test_approves_valid_proposal() -> None:
    engine = RiskEngine(Settings())
    decision = engine.evaluate(
        proposal(), snapshot(), Quote(symbol="AAPL", bid=199.9, ask=200.1)
    )
    assert decision.status == DecisionStatus.APPROVED
    assert decision.approved_notional == 4_000


def test_rejects_stale_data() -> None:
    engine = RiskEngine(Settings(trading_max_data_age_seconds=5))
    quote = Quote(
        symbol="AAPL",
        bid=199.9,
        ask=200.1,
        knowledge_time=datetime.now(UTC) - timedelta(seconds=6),
    )
    decision = engine.evaluate(proposal(), snapshot(), quote)
    assert decision.status == DecisionStatus.REJECTED
    assert "stale_market_data" in decision.reasons


def test_kill_switch_fails_closed() -> None:
    engine = RiskEngine(Settings())
    engine.set_kill_switch(True)
    decision = engine.evaluate(
        proposal(), snapshot(), Quote(symbol="AAPL", bid=199.9, ask=200.1)
    )
    assert decision.status == DecisionStatus.REJECTED
    assert "kill_switch_enabled" in decision.reasons
