import pytest

from trading_app.broker import InternalPaperBroker
from trading_app.domain import Order, Quote, Side, SignalProposal


@pytest.mark.asyncio
async def test_buy_fill_is_conservative() -> None:
    proposal = SignalProposal(
        symbol="AAPL",
        side=Side.BUY,
        confidence=0.8,
        expected_return=0.01,
        target_notional=1_000,
        reference_price=100,
        rationale=["test"],
        feature_snapshot={},
    )
    order = Order(
        proposal_id=proposal.id,
        symbol="AAPL",
        side=Side.BUY,
        quantity=10,
        requested_price=100.1,
    )
    fill = await InternalPaperBroker(slippage_bps=2).execute(
        order, Quote(symbol="AAPL", bid=99.9, ask=100.1)
    )
    assert fill.price > 100.1
    assert fill.slippage_bps == 2
