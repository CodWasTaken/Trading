from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel

from .config import Settings
from .news_intelligence import NewsIntelligence


class HistoricalBar(BaseModel):
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int | None = None
    vwap: float | None = None


class AlpacaHistoricalClient:
    """Paginated Alpaca historical bars and news client."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        key, secret = settings.require_alpaca_credentials()
        self.base_url = settings.alpaca_data_base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30,
            transport=transport,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        )
        self.news_intelligence = NewsIntelligence()

    async def iter_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        *,
        timeframe: str = "1Min",
        feed: str = "iex",
        adjustment: str = "all",
        page_limit: int = 10_000,
    ) -> AsyncIterator[HistoricalBar]:
        params: dict[str, str | int] = {
            "symbols": ",".join(symbol.upper() for symbol in symbols),
            "start": start.astimezone(UTC).isoformat(),
            "end": end.astimezone(UTC).isoformat(),
            "timeframe": timeframe,
            "feed": feed,
            "adjustment": adjustment,
            "sort": "asc",
            "limit": page_limit,
        }
        while True:
            response = await self.client.get("/v2/stocks/bars", params=params)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            bars = payload.get("bars", {})
            for symbol, records in bars.items():
                for item in records:
                    yield HistoricalBar(
                        symbol=symbol,
                        timestamp=datetime.fromisoformat(item["t"].replace("Z", "+00:00")),
                        open=float(item["o"]),
                        high=float(item["h"]),
                        low=float(item["l"]),
                        close=float(item["c"]),
                        volume=float(item["v"]),
                        trade_count=(int(item["n"]) if item.get("n") is not None else None),
                        vwap=(float(item["vw"]) if item.get("vw") is not None else None),
                    )
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = str(token)

    async def iter_news(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        *,
        include_content: bool = False,
        page_limit: int = 50,
    ) -> AsyncIterator[object]:
        params: dict[str, str | int | bool] = {
            "symbols": ",".join(symbol.upper() for symbol in symbols),
            "start": start.astimezone(UTC).isoformat(),
            "end": end.astimezone(UTC).isoformat(),
            "sort": "asc",
            "include_content": include_content,
            "limit": page_limit,
        }
        while True:
            response = await self.client.get("/v1beta1/news", params=params)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            for article in payload.get("news", []):
                event_time = datetime.fromisoformat(
                    article["created_at"].replace("Z", "+00:00")
                )
                summary = article.get("summary", "")
                for symbol in article.get("symbols", []):
                    event = self.news_intelligence.enrich(
                        symbol=symbol,
                        headline=article.get("headline", "Untitled news item"),
                        source=article.get("source", "alpaca"),
                        summary=summary,
                        event_time=event_time,
                        # Historical payloads do not contain provider receipt time. Use the
                        # source publication time and mark this limitation in dataset metadata.
                        knowledge_time=event_time,
                    )
                    if event is not None:
                        yield event
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = str(token)

    async def close(self) -> None:
        await self.client.aclose()
