from __future__ import annotations

import asyncio
import json
import math
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import websockets

from .config import Settings
from .domain import NewsEvent, Quote
from .entities import EntityCatalog
from .point_in_time_news import PointInTimeNewsIntelligence


class DemoFeed:
    def __init__(self, symbols: list[str], interval_seconds: float = 3.0, seed: int = 7) -> None:
        self.symbols = symbols
        self.interval_seconds = interval_seconds
        self.random = random.Random(seed)
        self._prices = {
            symbol: price
            for symbol, price in zip(
                symbols, [225.0, 510.0, 180.0, 235.0, 730.0], strict=False
            )
        }
        for symbol in symbols:
            self._prices.setdefault(symbol, self.random.uniform(50, 400))
        self._step = 0

    async def stream(self) -> AsyncIterator[Quote | NewsEvent]:
        headlines = [
            ("raises full-year guidance after stronger demand", 0.85, "guidance"),
            ("announces expanded share repurchase programme", 0.65, "buyback"),
            ("faces regulatory inquiry into product practices", -0.75, "regulatory"),
            ("reports supply disruption at key facility", -0.62, "supply_chain"),
            ("launches new enterprise AI product", 0.55, "product_launch"),
        ]
        while True:
            self._step += 1
            for index, symbol in enumerate(self.symbols):
                cyclical = math.sin((self._step + index * 3) / 7) * 0.0018
                shock = self.random.gauss(0, 0.0012)
                self._prices[symbol] *= 1 + cyclical + shock
                mid = self._prices[symbol]
                spread = max(0.01, mid * self.random.uniform(0.00005, 0.00025))
                now = datetime.now(UTC)
                yield Quote(
                    symbol=symbol,
                    bid=round(mid - spread / 2, 4),
                    ask=round(mid + spread / 2, 4),
                    event_time=now,
                    knowledge_time=now,
                )

            if self._step % 8 == 0:
                symbol = self.symbols[(self._step // 8) % len(self.symbols)]
                text, sentiment, event_type = headlines[(self._step // 8) % len(headlines)]
                now = datetime.now(UTC)
                yield NewsEvent(
                    symbol=symbol,
                    headline=f"{symbol} {text}",
                    source="demo-wire",
                    sentiment=sentiment,
                    novelty=0.85,
                    source_quality=0.90,
                    event_type=event_type,
                    event_time=now,
                    knowledge_time=now,
                )
            await asyncio.sleep(self.interval_seconds)


class AlpacaWebSocketFeed:
    """Official Alpaca stock and news WebSocket consumer with enrichment."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key, self.secret = settings.require_alpaca_credentials()
        self.news_intelligence = PointInTimeNewsIntelligence(
            entity_catalog=EntityCatalog.load(settings.trading_entity_catalog_path)
        )

    async def _authenticate(self, socket: Any) -> None:
        await socket.send(
            json.dumps({"action": "auth", "key": self.key, "secret": self.secret})
        )
        raw = await socket.recv()
        messages = json.loads(raw)
        if not any(item.get("T") == "success" for item in messages):
            raise RuntimeError(f"Alpaca stream authentication failed: {messages}")

    async def stream_quotes(self) -> AsyncIterator[Quote]:
        async with websockets.connect(self.settings.alpaca_data_stream_url) as socket:
            await self._authenticate(socket)
            await socket.send(
                json.dumps({"action": "subscribe", "quotes": self.settings.trading_symbols})
            )
            async for raw in socket:
                for item in json.loads(raw):
                    if item.get("T") != "q":
                        continue
                    now = datetime.now(UTC)
                    yield Quote(
                        symbol=item["S"],
                        bid=float(item["bp"]),
                        ask=float(item["ap"]),
                        event_time=datetime.fromisoformat(item["t"].replace("Z", "+00:00")),
                        knowledge_time=now,
                    )

    async def stream_news(self) -> AsyncIterator[NewsEvent]:
        async with websockets.connect(self.settings.alpaca_news_stream_url) as socket:
            await self._authenticate(socket)
            await socket.send(
                json.dumps({"action": "subscribe", "news": self.settings.trading_symbols})
            )
            async for raw in socket:
                for item in json.loads(raw):
                    if item.get("T") != "n":
                        continue
                    now = datetime.now(UTC)
                    event_time = datetime.fromisoformat(
                        item.get("created_at", now.isoformat()).replace("Z", "+00:00")
                    )
                    for symbol in item.get("symbols", []):
                        event = self.news_intelligence.enrich(
                            symbol=symbol,
                            headline=item.get("headline", "Untitled news item"),
                            source=item.get("source", "alpaca"),
                            summary=item.get("summary", ""),
                            event_time=event_time,
                            knowledge_time=now,
                        )
                        if event is not None:
                            yield event
