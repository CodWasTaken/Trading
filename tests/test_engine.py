import pytest

from trading_app.broker import InternalPaperBroker
from trading_app.config import Settings
from trading_app.domain import Quote
from trading_app.engine import TradingEngine
from trading_app.portfolio import Portfolio
from trading_app.risk import RiskEngine
from trading_app.store import EventStore
from trading_app.strategy import ExplainableCatalystStrategy


@pytest.mark.asyncio
async def test_end_to_end_decision_can_fill() -> None:
    settings = Settings(trading_min_confidence=0.5)
    store = EventStore()
    portfolio = Portfolio(100_000)
    engine = TradingEngine(
        store=store,
        portfolio=portfolio,
        risk=RiskEngine(settings),
        strategy=ExplainableCatalystStrategy(),
        broker=InternalPaperBroker(),
    )

    prices = [100, 100.8, 101.5, 102.4, 103.5]
    for price in prices:
        await engine.process_quote(
            Quote(symbol="AAPL", bid=price - 0.01, ask=price + 0.01)
        )

    ledger = await store.snapshot()
    assert ledger["proposals"]
    assert ledger["decisions"]
    assert ledger["fills"]
    assert portfolio.snapshot().positions
