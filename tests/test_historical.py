from datetime import UTC, datetime

import httpx
import pytest

from trading_app.config import Settings
from trading_app.historical import AlpacaHistoricalClient


@pytest.mark.asyncio
async def test_historical_client_paginates_bars_and_enriches_news() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/stocks/bars":
            page = request.url.params.get("page_token")
            if page is None:
                return httpx.Response(
                    200,
                    json={
                        "bars": {
                            "AAPL": [
                                {
                                    "t": "2026-01-02T14:30:00Z",
                                    "o": 100,
                                    "h": 101,
                                    "l": 99.5,
                                    "c": 100.5,
                                    "v": 1000,
                                    "n": 100,
                                    "vw": 100.2,
                                }
                            ]
                        },
                        "next_page_token": "page-2",
                    },
                )
            return httpx.Response(
                200,
                json={
                    "bars": {
                        "AAPL": [
                            {
                                "t": "2026-01-02T14:31:00Z",
                                "o": 100.5,
                                "h": 102,
                                "l": 100,
                                "c": 101.5,
                                "v": 1200,
                                "n": 110,
                                "vw": 101.1,
                            }
                        ]
                    },
                    "next_page_token": None,
                },
            )
        if request.url.path == "/v1beta1/news":
            return httpx.Response(
                200,
                json={
                    "news": [
                        {
                            "created_at": "2026-01-02T14:30:30Z",
                            "headline": "Apple raises guidance after strong revenue growth",
                            "summary": "The company increased its annual outlook.",
                            "source": "Reuters",
                            "symbols": ["AAPL"],
                        }
                    ],
                    "next_page_token": None,
                },
            )
        return httpx.Response(404)

    settings = Settings(
        alpaca_api_key="key",
        alpaca_api_secret="secret",
        alpaca_data_base_url="https://data.example.test",
    )
    client = AlpacaHistoricalClient(
        settings, transport=httpx.MockTransport(handler)
    )
    start = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    end = datetime(2026, 1, 2, 15, 30, tzinfo=UTC)
    bars = [bar async for bar in client.iter_bars(["AAPL"], start, end)]
    news = [item async for item in client.iter_news(["AAPL"], start, end)]
    await client.close()

    assert len(bars) == 2
    assert bars[-1].close == 101.5
    assert len(news) == 1
    assert news[0].symbol == "AAPL"
    assert news[0].sentiment > 0
