from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from .broker import Broker
from .domain import DecisionStatus, NewsEvent, Order, Quote
from .portfolio import Portfolio
from .risk import RiskEngine
from .store import EventStore
from .strategy import Strategy


class TradingEngine:
    def __init__(
        self,
        store: EventStore,
        portfolio: Portfolio,
        risk: RiskEngine,
        strategy: Strategy,
        broker: Broker,
    ) -> None:
        self.store = store
        self.portfolio = portfolio
        self.risk = risk
        self.strategy = strategy
        self.broker = broker
        self.running = False
        self._tasks: set[asyncio.Task[None]] = set()

    async def process_quote(self, quote: Quote) -> None:
        await self.store.set_quote(quote)
        self.portfolio.mark(quote)
        self.portfolio.rollover_session(quote.event_time)
        news = await self.store.recent_news_for(quote.symbol)
        snapshot = self.portfolio.snapshot()
        proposal = self.strategy.on_quote(quote, news, snapshot.equity)
        if proposal is None:
            return

        await self.store.add_proposal(proposal)
        decision = self.risk.evaluate(proposal, snapshot, quote)
        await self.store.add_decision(decision)
        if decision.status == DecisionStatus.REJECTED:
            return

        quantity = decision.approved_notional / quote.ask
        if proposal.side.value == "sell":
            quantity = decision.approved_notional / quote.bid
        order = Order(
            proposal_id=proposal.id,
            symbol=proposal.symbol,
            side=proposal.side,
            quantity=quantity,
            requested_price=quote.ask if proposal.side.value == "buy" else quote.bid,
        )
        await self.store.add_order(order)
        try:
            fill = await self.broker.execute(order, quote)
        except Exception as error:
            await self.store.add_system_event(
                "execution_error",
                {
                    "order_id": str(order.id),
                    "proposal_id": str(proposal.id),
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            return
        self.portfolio.apply_fill(fill)
        await self.store.add_fill(fill)

    async def process_news(self, event: NewsEvent) -> None:
        await self.store.add_news(event)

    async def run_demo(self, feed: AsyncIterator[Quote | NewsEvent]) -> None:
        self.running = True
        try:
            async for item in feed:
                if not self.running:
                    break
                if isinstance(item, Quote):
                    await self.process_quote(item)
                else:
                    await self.process_news(item)
        finally:
            self.running = False

    async def run_streams(
        self,
        quotes: AsyncIterator[Quote],
        news: AsyncIterator[NewsEvent],
    ) -> None:
        self.running = True

        async def consume_quotes() -> None:
            async for quote in quotes:
                if not self.running:
                    return
                await self.process_quote(quote)

        async def consume_news() -> None:
            async for item in news:
                if not self.running:
                    return
                await self.process_news(item)

        self._tasks = {
            asyncio.create_task(consume_quotes()),
            asyncio.create_task(consume_news()),
        }
        try:
            await asyncio.gather(*self._tasks)
        finally:
            self.running = False
            self._tasks.clear()

    async def stop(self) -> None:
        self.running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
