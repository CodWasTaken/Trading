import pytest

from trading_app.config import Settings
from trading_app.domain import Quote, Side, SignalProposal
from trading_app.engine import TradingEngine
from trading_app.portfolio import Portfolio
from trading_app.risk import RiskEngine
from trading_app.store import EventStore


class AlwaysBuyStrategy:
    def on_quote(self, quote, news, equity):
        return SignalProposal(
            symbol=quote.symbol,
            side=Side.BUY,
            confidence=0.9,
            expected_return=0.01,
            target_notional=1_000,
            reference_price=quote.mid,
            rationale=["test"],
            feature_snapshot={},
        )


class FailingBroker:
    async def execute(self, order, quote):
        raise RuntimeError("broker unavailable")


@pytest.mark.asyncio
async def test_execution_failure_is_audited_without_mutating_portfolio() -> None:
    settings = Settings()
    store = EventStore()
    portfolio = Portfolio(100_000)
    engine = TradingEngine(
        store=store,
        portfolio=portfolio,
        risk=RiskEngine(settings),
        strategy=AlwaysBuyStrategy(),
        broker=FailingBroker(),
    )
    await engine.process_quote(Quote(symbol="AAPL", bid=99.9, ask=100.1))
    snapshot = await store.snapshot()

    assert not snapshot["fills"]
    assert snapshot["system_events"][0]["type"] == "execution_error"
    assert portfolio.snapshot().positions == []
