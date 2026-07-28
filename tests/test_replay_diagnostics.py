import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_app.domain import NewsEvent
from trading_app.historical import HistoricalBar
from trading_app.replay_diagnostics import build_replay_diagnostics


def make_bars(start: datetime, count: int = 48) -> list[HistoricalBar]:
    bars: list[HistoricalBar] = []
    for index in range(count):
        cycle = index % 16
        direction = 1 if cycle < 8 else -1
        step = cycle if cycle < 8 else cycle - 8
        for symbol, offset in (("AAPL", 0.0), ("MSFT", 25.0)):
            price = 100 + offset + direction * step * 0.45 + index * 0.02
            bars.append(
                HistoricalBar(
                    symbol=symbol,
                    timestamp=start + timedelta(hours=index),
                    open=price - 0.05,
                    high=price + 0.20,
                    low=price - 0.20,
                    close=price,
                    volume=25_000,
                )
            )
    return bars


def make_news(start: datetime) -> list[NewsEvent]:
    return [
        NewsEvent(
            symbol=symbol,
            headline=f"{symbol} reports quarterly results",
            source="Reuters",
            sentiment=0.20,
            novelty=1.0,
            source_quality=0.95,
            event_type="earnings",
            event_time=start + timedelta(hours=3),
            knowledge_time=start + timedelta(hours=3),
        )
        for symbol in ("AAPL", "MSFT")
    ]


@pytest.mark.asyncio
async def test_replay_report_runs_full_threshold_and_attribution_diagnostics(
    tmp_path: Path,
) -> None:
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    bars = make_bars(start)
    news = make_news(start)
    bars_path = tmp_path / "bars.jsonl"
    news_path = tmp_path / "news.jsonl"
    sector_path = tmp_path / "sectors.json"
    output_path = tmp_path / "diagnostics.json"
    bars_path.write_text("".join(item.model_dump_json() + "\n" for item in bars))
    news_path.write_text("".join(item.model_dump_json() + "\n" for item in news))
    sector_path.write_text(json.dumps({"AAPL": "Technology", "MSFT": "Technology"}))

    report = await build_replay_diagnostics(
        str(bars_path),
        str(news_path),
        str(output_path),
        strategy_mode="explainable",
        registry_path=str(tmp_path / "models"),
        allow_synthetic_champion=False,
        verify_determinism=True,
        bar_minutes=60,
        spread_bps=10,
        slippage_bps=2,
        starting_cash=100_000,
        thresholds=[0.08, 0.16, 1.0],
        reference_threshold=0.16,
        sector_map_path=str(sector_path),
        regime_lookback=5,
        regime_momentum_threshold=0.005,
    )

    assert output_path.exists()
    assert report["determinism_verified"] is True
    sensitivity = report["threshold_sensitivity"]
    assert isinstance(sensitivity, list)
    assert len(sensitivity) == 3
    low = next(item for item in sensitivity if item["threshold"] == 0.08)
    high = next(item for item in sensitivity if item["threshold"] == 1.0)
    assert low["proposals"] > high["proposals"]
    assert low["trace_sha256"] != high["trace_sha256"]

    regimes = report["regimes"]
    assert isinstance(regimes, dict)
    assert regimes["buckets"]

    sectors = report["sectors"]
    assert isinstance(sectors, dict)
    groups = sectors["groups"]
    assert "Technology" in groups
    assert sorted(groups["Technology"]["symbols"]) == ["AAPL", "MSFT"]

    event_types = report["event_types"]
    assert isinstance(event_types, dict)
    assert event_types["groups"]["earnings"]["proposals"] > 0
    assert json.loads(output_path.read_text())["reference_threshold"] == 0.16


@pytest.mark.asyncio
async def test_replay_report_marks_missing_sector_mappings(tmp_path: Path) -> None:
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    bars_path = tmp_path / "bars.jsonl"
    news_path = tmp_path / "news.jsonl"
    output_path = tmp_path / "diagnostics.json"
    bars = make_bars(start, count=24)
    bars_path.write_text("".join(item.model_dump_json() + "\n" for item in bars))
    news_path.write_text("")

    report = await build_replay_diagnostics(
        str(bars_path),
        str(news_path),
        str(output_path),
        strategy_mode="explainable",
        registry_path=str(tmp_path / "models"),
        allow_synthetic_champion=False,
        verify_determinism=False,
        bar_minutes=60,
        spread_bps=10,
        slippage_bps=2,
        starting_cash=100_000,
        thresholds=[0.08],
        reference_threshold=0.08,
        sector_map_path=None,
        regime_lookback=5,
        regime_momentum_threshold=0.005,
    )

    sectors = report["sectors"]
    assert isinstance(sectors, dict)
    assert "Unmapped" in sectors["groups"]
