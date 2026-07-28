from __future__ import annotations

from typing import Protocol

import httpx

from .config import Settings
from .domain import Fill, Order, Quote, Side


class Broker(Protocol):
    async def execute(self, order: Order, quote: Quote) -> Fill: ...


class InternalPaperBroker:
    def __init__(self, slippage_bps: float = 2.0) -> None:
        self.slippage_bps = slippage_bps

    async def execute(self, order: Order, quote: Quote) -> Fill:
        multiplier = self.slippage_bps / 10_000
        if order.side == Side.BUY:
            price = quote.ask * (1 + multiplier)
        else:
            price = quote.bid * (1 - multiplier)
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=price,
            slippage_bps=self.slippage_bps,
        )


class AlpacaPaperBroker:
    def __init__(self, settings: Settings) -> None:
        key, secret = settings.require_alpaca_credentials()
        self.base_url = settings.alpaca_trading_base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            timeout=15,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        )

    async def submit(self, order: Order) -> dict[str, object]:
        payload = {
            "symbol": order.symbol,
            "qty": str(order.quantity),
            "side": order.side.value,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": str(order.id),
        }
        response = await self.client.post(f"{self.base_url}/v2/orders", json=payload)
        response.raise_for_status()
        result: dict[str, object] = response.json()
        return result

    async def execute(self, order: Order, quote: Quote) -> Fill:
        # Alpaca fills asynchronously. The vertical slice submits the order and records an
        # estimated fill; production must consume trade_updates and reconcile the actual fill.
        await self.submit(order)
        estimated = quote.ask if order.side == Side.BUY else quote.bid
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=estimated,
            slippage_bps=0,
        )

    async def close(self) -> None:
        await self.client.aclose()
