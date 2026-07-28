from __future__ import annotations

import asyncio
from time import monotonic
from typing import Protocol

import httpx
from pydantic import BaseModel

from .config import Settings
from .domain import Fill, Order, Quote, Side


class ExecutionError(RuntimeError):
    pass


class BrokerPosition(BaseModel):
    symbol: str
    quantity: float
    average_entry_price: float
    market_value: float


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
    TERMINAL_FAILURES = {"canceled", "expired", "rejected", "replaced", "suspended"}

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key, secret = settings.require_alpaca_credentials()
        self.base_url = settings.alpaca_trading_base_url.rstrip("/")
        self.fill_timeout_seconds = settings.trading_order_fill_timeout_seconds
        self.poll_interval_seconds = settings.trading_order_poll_interval_seconds
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=15,
            transport=transport,
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
        response = await self.client.post("/v2/orders", json=payload)
        response.raise_for_status()
        result: dict[str, object] = response.json()
        return result

    async def get_order(self, broker_order_id: str) -> dict[str, object]:
        response = await self.client.get(f"/v2/orders/{broker_order_id}")
        response.raise_for_status()
        result: dict[str, object] = response.json()
        return result

    async def cancel_order(self, broker_order_id: str) -> None:
        response = await self.client.delete(f"/v2/orders/{broker_order_id}")
        if response.status_code not in {204, 404, 422}:
            response.raise_for_status()

    async def execute(self, order: Order, quote: Quote) -> Fill:
        latest = await self.submit(order)
        broker_order_id = str(latest.get("id", ""))
        if not broker_order_id:
            raise ExecutionError("Broker response did not include an order id")
        deadline = monotonic() + self.fill_timeout_seconds
        while True:
            status = str(latest.get("status", "")).lower()
            filled_quantity = float(latest.get("filled_qty") or 0)
            average_price_raw = latest.get("filled_avg_price")
            if status == "filled" and filled_quantity > 0 and average_price_raw is not None:
                return self._fill_from_broker(
                    order, quote, filled_quantity, float(average_price_raw)
                )
            if status in self.TERMINAL_FAILURES:
                raise ExecutionError(f"Broker order ended with status {status}")
            if monotonic() >= deadline:
                await self.cancel_order(broker_order_id)
                if filled_quantity > 0 and average_price_raw is not None:
                    return self._fill_from_broker(
                        order, quote, filled_quantity, float(average_price_raw)
                    )
                raise ExecutionError("Timed out waiting for a broker fill; order was cancelled")
            await asyncio.sleep(self.poll_interval_seconds)
            latest = await self.get_order(broker_order_id)

    @staticmethod
    def _fill_from_broker(
        order: Order,
        quote: Quote,
        quantity: float,
        price: float,
    ) -> Fill:
        if order.side == Side.BUY:
            slippage_bps = (price / quote.ask - 1) * 10_000
        else:
            slippage_bps = (quote.bid / price - 1) * 10_000
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            slippage_bps=slippage_bps,
        )

    async def get_positions(self) -> list[BrokerPosition]:
        response = await self.client.get("/v2/positions")
        response.raise_for_status()
        return [
            BrokerPosition(
                symbol=item["symbol"],
                quantity=float(item["qty"]),
                average_entry_price=float(item["avg_entry_price"]),
                market_value=float(item["market_value"]),
            )
            for item in response.json()
        ]

    async def close(self) -> None:
        await self.client.aclose()
