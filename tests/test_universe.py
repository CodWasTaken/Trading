import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from trading_app.cli import build_dataset
from trading_app.domain import NewsEvent
from trading_app.historical import HistoricalBar
from trading_app.universe import (
    assert_universe_binding,
    assert_universe_symbols,
    load_universe_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config/universes/us-liquid-large-cap-v1.json"


def test_frozen_manifest_has_48_diversified_common_stocks_and_valid_hashes() -> None:
    manifest = load_universe_manifest(MANIFEST)
    assert len(manifest.symbols) == 48
    assert len(set(manifest.sectors.values())) >= 8
    assert all(item.security_type == "common_stock" for item in manifest.constituents)
    assert manifest.minimum_median_daily_dollar_volume_usd == 25_000_000
    assert manifest.minimum_price_usd == 5
    assert manifest.minimum_historical_coverage_years == 5
    assert manifest.shortability_requirement
    assert manifest.manifest_sha256


def test_manifest_content_and_source_tampering_fail_closed(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["minimum_price_usd"] = 6
    tampered_manifest = tmp_path / "tampered.json"
    tampered_manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_universe_manifest(tampered_manifest)

    universe_dir = tmp_path / "universe"
    source_dir = universe_dir / "sources"
    source_dir.mkdir(parents=True)
    copied_manifest = universe_dir / MANIFEST.name
    copied_source = source_dir / "us-liquid-large-cap-2025-01-02.csv"
    shutil.copyfile(MANIFEST, copied_manifest)
    shutil.copyfile(
        MANIFEST.parent / "sources/us-liquid-large-cap-2025-01-02.csv",
        copied_source,
    )
    copied_source.write_text(copied_source.read_text() + "\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_universe_manifest(copied_manifest)


def test_symbol_and_binding_mismatch_reject_with_details() -> None:
    manifest = load_universe_manifest(MANIFEST)
    with pytest.raises(ValueError, match="missing=.*MSFT"):
        assert_universe_symbols(["AAPL"], manifest, context="test run")
    binding = manifest.binding(MANIFEST)
    binding["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        assert_universe_binding(
            {"universe": binding},
            manifest,
            context="test dataset",
        )


def test_dataset_build_rejects_partial_universe(tmp_path: Path) -> None:
    start = datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    bars = [
        HistoricalBar(
            symbol="AAPL",
            timestamp=start + timedelta(hours=index),
            open=100,
            high=101,
            low=99,
            close=100 + index,
            volume=10_000,
        )
        for index in range(30)
    ]
    bars_path = tmp_path / "bars.jsonl"
    news_path = tmp_path / "news.jsonl"
    bars_path.write_text("".join(item.model_dump_json() + "\n" for item in bars))
    news_path.write_text(
        NewsEvent(
            symbol="AAPL",
            headline="Test",
            source="fixture",
            sentiment=0,
            novelty=1,
            source_quality=1,
            event_time=start,
            knowledge_time=start,
        ).model_dump_json()
        + "\n"
    )
    with pytest.raises(ValueError, match="historical bars universe mismatch"):
        build_dataset(
            str(bars_path),
            str(news_path),
            str(tmp_path / "dataset.jsonl"),
            lookback_bars=5,
            forecast_bars=5,
            bar_minutes=60,
            news_window_hours=24,
            universe_manifest_path=str(MANIFEST),
        )
