from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from copy import deepcopy
from datetime import UTC, datetime

from .domain import Fill, LiveEvent, NewsEvent, Order, Quote, RiskDecision, SignalProposal
from .persistence import EventSink


class EventStore:
    """Thread-safe live ledger with an optional durable append-only sink."""

    def __init__(
        self,
        max_events: int = 2_000,
        sink: EventSink | None = None,
    ) -> None:
        self.quotes: dict[str, Quote] = {}
        self.news: deque[NewsEvent] = deque(maxlen=max_events)
        self.proposals: deque[SignalProposal] = deque(maxlen=max_events)
        self.decisions: deque[RiskDecision] = deque(maxlen=max_events)
        self.orders: deque[Order] = deque(maxlen=max_events)
        self.fills: deque[Fill] = deque(maxlen=max_events)
        self.system_events: deque[LiveEvent] = deque(maxlen=max_events)
        self.live_events: deque[LiveEvent] = deque(maxlen=max_events)
        self._subscribers: set[asyncio.Queue[LiveEvent]] = set()
        self._lock = asyncio.Lock()
        self._sink = sink

    async def publish(self, event: LiveEvent) -> None:
        if self._sink is not None:
            self._sink.append(event.type, event.payload, event.created_at, event.created_at)
        async with self._lock:
            self.live_events.appendleft(event)
            subscribers = tuple(self._subscribers)
        for queue in subscribers:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)

    async def set_quote(self, quote: Quote) -> None:
        async with self._lock:
            self.quotes[quote.symbol] = quote
        await self.publish(LiveEvent(type="quote", payload=quote.model_dump(mode="json")))

    async def add_news(self, item: NewsEvent) -> None:
        async with self._lock:
            self.news.appendleft(item)
        await self.publish(LiveEvent(type="news", payload=item.model_dump(mode="json")))

    async def add_proposal(self, item: SignalProposal) -> None:
        async with self._lock:
            self.proposals.appendleft(item)
        await self.publish(LiveEvent(type="proposal", payload=item.model_dump(mode="json")))

    async def add_decision(self, item: RiskDecision) -> None:
        async with self._lock:
            self.decisions.appendleft(item)
        await self.publish(
            LiveEvent(type="risk_decision", payload=item.model_dump(mode="json"))
        )

    async def add_order(self, item: Order) -> None:
        async with self._lock:
            self.orders.appendleft(item)
        await self.publish(LiveEvent(type="order", payload=item.model_dump(mode="json")))

    async def add_fill(self, item: Fill) -> None:
        async with self._lock:
            self.fills.appendleft(item)
        await self.publish(LiveEvent(type="fill", payload=item.model_dump(mode="json")))

    async def add_system_event(self, kind: str, payload: dict[str, object]) -> None:
        event = LiveEvent(type=kind, payload=payload)
        async with self._lock:
            self.system_events.appendleft(event)
        await self.publish(event)

    async def subscribe(self) -> AsyncIterator[LiveEvent]:
        queue: asyncio.Queue[LiveEvent] = asyncio.Queue(maxsize=200)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    async def health(self, maximum_age_seconds: int) -> dict[str, object]:
        now = datetime.now(UTC)
        async with self._lock:
            quote_ages = {
                symbol: max(0.0, (now - quote.knowledge_time).total_seconds())
                for symbol, quote in self.quotes.items()
            }
        stale = sorted(
            symbol for symbol, age in quote_ages.items() if age > maximum_age_seconds
        )
        return {
            "quote_age_seconds": quote_ages,
            "stale_symbols": stale,
            "healthy": bool(quote_ages) and not stale,
        }

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            return {
                "quotes": {k: v.model_dump(mode="json") for k, v in self.quotes.items()},
                "news": [x.model_dump(mode="json") for x in list(self.news)[:100]],
                "proposals": [x.model_dump(mode="json") for x in list(self.proposals)[:100]],
                "decisions": [x.model_dump(mode="json") for x in list(self.decisions)[:100]],
                "orders": [x.model_dump(mode="json") for x in list(self.orders)[:100]],
                "fills": [x.model_dump(mode="json") for x in list(self.fills)[:100]],
                "system_events": [
                    x.model_dump(mode="json") for x in list(self.system_events)[:100]
                ],
            }

    async def recent_news_for(self, symbol: str, limit: int = 10) -> list[NewsEvent]:
        async with self._lock:
            return deepcopy([item for item in self.news if item.symbol == symbol][:limit])
