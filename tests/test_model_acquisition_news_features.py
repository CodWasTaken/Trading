from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_app.dataset import load_feature_dataset, write_feature_dataset
from trading_app.model_acquisition import acquire_huggingface_snapshot
from trading_app.news_ai_features import NEWS_AI_FEATURE_NAMES, join_news_ai_features
from trading_app.research import FeatureRow


class _FakeDownloader:
    def resolve_revision(self, repo_id: str, revision: str) -> str:
        assert repo_id == "example/model"
        assert revision == "v1"
        return "a" * 40

    def download(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        allow_patterns: tuple[str, ...],
        ignore_patterns: tuple[str, ...],
    ) -> Path:
        assert repo_id == "example/model"
        assert revision == "a" * 40
        assert allow_patterns == ("*.json", "*.bin")
        assert ignore_patterns == ("*.msgpack",)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        (destination / "weights.bin").write_bytes(b"weights")
        return destination


def test_model_acquisition_is_explicit_hash_bound_and_secret_free(tmp_path: Path) -> None:
    destination = tmp_path / "snapshot"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_id": "example/model",
                "revision": "v1",
                "destination": str(destination),
                "license": "Apache-2.0",
                "asset_kind": "foundation",
                "allow_patterns": ["*.json", "*.bin"],
                "ignore_patterns": ["*.msgpack"],
            }
        ),
        encoding="utf-8",
    )
    report = acquire_huggingface_snapshot(
        request,
        tmp_path / "report.json",
        downloader=_FakeDownloader(),
    )
    assert report["resolved_revision"] == "a" * 40
    assert len(str(report["weight_sha256"])) == 64
    assert report["controls"]["operator_triggered"] is True
    assert report["controls"]["startup_download"] is False
    serialized = json.dumps(report).lower()
    assert "token" not in serialized
    assert "secret" not in serialized


def _inference_record(
    *,
    symbol: str,
    knowledge_time: datetime,
    sentiment: float,
    confidence: float,
    parse_error: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "symbol": symbol,
        "knowledge_time": knowledge_time.isoformat(),
        "model_manifest_sha256": "b" * 64,
        "parse_error": parse_error,
        "normalized": None
        if parse_error is not None
        else {
            "sentiment": sentiment,
            "confidence": confidence,
            "primary_event_type": "earnings",
            "entities": [{"text": symbol, "entity_type": "organization"}],
        },
    }


def test_news_ai_join_uses_only_inference_known_by_feature_time(tmp_path: Path) -> None:
    start = datetime(2026, 1, 2, 14, tzinfo=UTC)
    rows = [
        FeatureRow(
            timestamp=start + timedelta(hours=index),
            symbol="AAPL",
            features=(0.01 * index,),
            target_return=-999.0 if index == 0 else 999.0,
            label_end_time=start + timedelta(hours=index + 5),
        )
        for index in range(2)
    ]
    dataset, _ = write_feature_dataset(
        tmp_path / "dataset.jsonl",
        rows,
        ("momentum",),
        {"dataset_role": "calibration", "sealed": False, "forecast_bars": 5},
    )
    inference = tmp_path / "news-ai.jsonl"
    records = [
        _inference_record(
            symbol="AAPL",
            knowledge_time=start,
            sentiment=0.8,
            confidence=0.5,
        ),
        _inference_record(
            symbol="AAPL",
            knowledge_time=start + timedelta(hours=2),
            sentiment=-1.0,
            confidence=1.0,
        ),
    ]
    inference.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    result = join_news_ai_features(
        dataset,
        inference,
        tmp_path / "joined.jsonl",
    )
    joined, feature_names, metadata = load_feature_dataset(result["output"])
    assert feature_names == ("momentum", *NEWS_AI_FEATURE_NAMES)
    sentiment_index = feature_names.index("news_ai_sentiment")
    assert [row.features[sentiment_index] for row in joined] == pytest.approx([0.8, 0.8])
    assert result["future_records_not_joined"] == 1
    assert metadata["news_ai_features"]["point_in_time_verified"] is True
