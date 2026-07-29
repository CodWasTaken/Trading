from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from uuid import uuid4

from .borrow import ForcedCoverInstruction
from .broker import Broker
from .domain import DecisionStatus, NewsEvent, Order, Quote, Side
from .portfolio import Portfolio, classify_position_effect
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
        self.portfolio.accrue_financing(quote.event_time)
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
            position_effect=classify_position_effect(
                self.portfolio.quantity(proposal.symbol),
                proposal.side,
                quantity,
            ),
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
        transition = self.portfolio.apply_fill(fill)
        await self.store.add_fill(fill)
        await self.store.add_system_event(
            "position_transition",
            transition.model_dump(mode="json"),
        )

    async def process_forced_cover(
        self,
        instruction: ForcedCoverInstruction,
        quote: Quote,
    ) -> None:
        """Execute a paper-only lender recall instruction without creating a signal."""
        if quote.symbol != instruction.symbol:
            raise ValueError("forced-cover quote symbol does not match instruction")
        current = self.portfolio.quantity(instruction.symbol)
        if current >= 0:
            return
        quantity = min(abs(current), instruction.quantity)
        order = Order(
            proposal_id=uuid4(),
            symbol=instruction.symbol,
            side=Side.BUY,
            quantity=quantity,
            requested_price=quote.ask,
            position_effect=classify_position_effect(current, Side.BUY, quantity),
        )
        await self.store.add_system_event(
            "borrow_recall",
            instruction.model_dump(mode="json"),
        )
        await self.store.add_order(order)
        try:
            fill = await self.broker.execute(order, quote)
        except Exception as error:
            await self.store.add_system_event(
                "forced_cover_execution_error",
                {
                    "order_id": str(order.id),
                    "symbol": order.symbol,
                    "quantity": order.quantity,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            return
        transition = self.portfolio.apply_fill(fill)
        await self.store.add_fill(fill)
        await self.store.add_system_event(
            "position_transition",
            transition.model_dump(mode="json"),
        )

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
