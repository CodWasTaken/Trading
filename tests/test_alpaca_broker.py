import httpx
import pytest

from trading_app.broker import AlpacaPaperBroker
from trading_app.config import Settings
from trading_app.domain import Order, Quote, Side, SignalProposal


@pytest.mark.asyncio
async def test_alpaca_broker_waits_for_actual_fill_and_reads_positions() -> None:
    calls = {"order": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v2/orders":
            return httpx.Response(
                200,
                json={
                    "id": "broker-order-1",
                    "status": "new",
                    "filled_qty": "0",
                    "filled_avg_price": None,
                },
            )
        if request.method == "GET" and request.url.path == "/v2/orders/broker-order-1":
            calls["order"] += 1
            return httpx.Response(
                200,
                json={
                    "id": "broker-order-1",
                    "status": "filled",
                    "filled_qty": "10",
                    "filled_avg_price": "100.25",
                },
            )
        if request.method == "GET" and request.url.path == "/v2/positions":
            return httpx.Response(
                200,
                json=[
                    {
                        "symbol": "AAPL",
                        "qty": "10",
                        "avg_entry_price": "100.25",
                        "market_value": "1010.00",
                    }
                ],
            )
        return httpx.Response(404)

    settings = Settings(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        trading_order_poll_interval_seconds=0,
        trading_order_fill_timeout_seconds=1,
    )
    broker = AlpacaPaperBroker(settings, transport=httpx.MockTransport(handler))
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
    fill = await broker.execute(order, Quote(symbol="AAPL", bid=99.9, ask=100.1))
    positions = await broker.get_positions()
    await broker.close()

    assert calls["order"] == 1
    assert fill.price == 100.25
    assert fill.quantity == 10
    assert positions[0].quantity == 10
