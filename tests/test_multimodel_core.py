from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_app.governed_models import GovernedModelRegistry
from trading_app.modeling import ElasticNetReturnModel, ModelCapabilities
from trading_app.research import FeatureRow
from trading_app.sequence_models import sequence_examples


def _rows(count: int = 80) -> list[FeatureRow]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        first = (index - count / 2) / count
        second = ((index % 7) - 3) / 10
        rows.append(
            FeatureRow(
                timestamp=start + timedelta(hours=index),
                symbol="AAPL",
                features=(first, second),
                target_return=0.02 * first - 0.01 * second,
                label_end_time=start + timedelta(hours=index + 5),
            )
        )
    return rows


def test_elastic_net_round_trip(tmp_path: Path) -> None:
    model = ElasticNetReturnModel(("momentum", "news_score"), alpha=1e-5)
    model.fit(_rows())
    before = model.predict((0.25, -0.1))
    artifact = tmp_path / "model.json"
    model.save(artifact)
    loaded = ElasticNetReturnModel.load(artifact)
    assert loaded.predict((0.25, -0.1)) == pytest.approx(before)


def test_registry_verifies_artifact_and_audits_paper_selection(tmp_path: Path) -> None:
    registry = GovernedModelRegistry(tmp_path / "models")
    model = ElasticNetReturnModel(("momentum", "news_score"), alpha=1e-5)
    model.fit(_rows())
    record = registry.register_any(
        model,
        {"net_return": 0.01},
        metadata={"universe": {"manifest_sha256": "a" * 64}},
    )
    registry.set_alias(
        "champion",
        record.version,
        reason="test promotion",
        action="promote",
    )
    with pytest.raises(ValueError, match="paused"):
        registry.select_active_paper(
            record.version,
            reason="operator selection",
            engine_running=True,
            kill_switch=False,
            runtime_feature_names=("momentum", "news_score", "volatility", "spread_bps"),
            universe_manifest_sha256="a" * 64,
        )
    selected = registry.select_active_paper(
        record.version,
        reason="operator selection",
        engine_running=False,
        kill_switch=False,
        runtime_feature_names=("momentum", "news_score", "volatility", "spread_bps"),
        universe_manifest_sha256="a" * 64,
    )
    assert selected.version == record.version
    assert registry.active_paper().version == record.version  # type: ignore[union-attr]
    event = registry.history(alias="active-paper")[-1]
    assert event["action"] == "select_active_paper_model"
    assert event["details"]["live_money_authorized"] is False  # type: ignore[index]


def test_registry_fails_closed_on_artifact_tampering(tmp_path: Path) -> None:
    registry = GovernedModelRegistry(tmp_path / "models")
    model = ElasticNetReturnModel(("momentum", "news_score"), alpha=1e-5)
    model.fit(_rows())
    record = registry.register_any(
        model,
        {"net_return": 0.01},
        metadata={"universe": {"manifest_sha256": "b" * 64}},
    )
    (registry.root / record.model_path).write_text("tampered", encoding="utf-8")
    assert registry.verify_artifact(record.version) is False
    with pytest.raises(ValueError, match="hash mismatch"):
        registry.load_any(record.version)


def test_sequence_examples_never_cross_symbol_or_use_future_rows() -> None:
    rows = _rows(12) + [
        FeatureRow(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index),
            symbol="MSFT",
            features=(0.1, 0.2),
            target_return=0.0,
            label_end_time=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index + 5),
        )
        for index in range(12)
    ]
    examples = sequence_examples(rows, 5)
    assert len(examples) == 16
    assert all(len(window) == 5 for window, _ in examples)


def test_non_return_model_is_not_selectable(tmp_path: Path) -> None:
    registry = GovernedModelRegistry(tmp_path / "models")
    model = ElasticNetReturnModel(("momentum", "news_score"), alpha=1e-5)
    model.fit(_rows())
    model.capabilities = ModelCapabilities(
        family="language",
        implementation="fingpt",
        task="news_intelligence",
        feature_names=("momentum", "news_score"),
        live_compatible=False,
    )
    record = registry.register_any(
        model,
        {"net_return": 0.0},
        metadata={"universe": {"manifest_sha256": "c" * 64}},
    )
    registry.set_alias("champion", record.version, reason="test", action="promote")
    blockers = registry.selection_failures(
        record.version,
        engine_running=False,
        kill_switch=False,
        runtime_feature_names=("momentum", "news_score", "volatility", "spread_bps"),
        universe_manifest_sha256="c" * 64,
    )
    assert "model_task_is_not_return_regression" in blockers
