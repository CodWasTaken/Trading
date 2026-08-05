from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_app.cost_model import CostModelConfig
from trading_app.dataset import write_feature_dataset
from trading_app.domain import NewsEvent
from trading_app.external_research import import_external_research
from trading_app.financial_news_ai import infer_news_archive
from trading_app.historical import HistoricalBar
from trading_app.mixed_candidates import (
    DEFAULT_QUOTAS,
    _validate_candidate_report,
    generate_mixed_plan,
)
from trading_app.modeling import sha256_file
from trading_app.pretrained_features import (
    ForecastSummary,
    generate_foundation_sidecar,
    join_foundation_sidecar,
    sha256_tree,
)
from trading_app.research import FeatureRow
from trading_app.rl_research import PortfolioResearchEnvironment, RLTrainingSpec


class _FakeForecaster:
    def __init__(self) -> None:
        self.histories: list[tuple[float, ...]] = []

    def forecast(self, history: tuple[float, ...], horizon: int) -> ForecastSummary:
        self.histories.append(history)
        terminal = history[-1] * 1.01
        point = tuple(terminal for _ in range(horizon))
        return ForecastSummary(
            point_forecast=point,
            quantiles={
                "q10": tuple(history[-1] * 0.99 for _ in range(horizon)),
                "q50": point,
                "q90": tuple(history[-1] * 1.02 for _ in range(horizon)),
            },
            expected_return=0.01,
            uncertainty=0.03,
        )


class _FakeNewsBackend:
    def infer(self, prompt: str) -> str:
        assert "future returns" not in prompt.lower()
        return json.dumps(
            {
                "sentiment": 0.7,
                "primary_event_type": "earnings",
                "secondary_event_types": ["guidance"],
                "entities": [{"text": "AAPL", "entity_type": "organization"}],
                "relations": [],
                "confidence": 0.9,
            }
        )


def _write_foundation_spec(tmp_path: Path) -> Path:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"immutable-test-weights")
    spec = tmp_path / "foundation.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "chronos",
                "model_id": "test/chronos",
                "revision": "test-revision",
                "local_path": str(weights),
                "weight_sha256": sha256_tree(weights),
                "license": "Apache-2.0",
                "context_length": 20,
                "prediction_length": 5,
                "quantile_levels": [0.1, 0.5, 0.9],
            }
        ),
        encoding="utf-8",
    )
    return spec


def test_foundation_sidecar_uses_past_closes_and_joins_by_feature_time(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars_path = tmp_path / "bars.jsonl"
    bars = [
        HistoricalBar(
            symbol="AAPL",
            timestamp=start + timedelta(hours=index),
            open=100 + index,
            high=101 + index,
            low=99 + index,
            close=100 + index,
            volume=1_000,
        )
        for index in range(22)
    ]
    bars_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in bars),
        encoding="utf-8",
    )
    sidecar = tmp_path / "sidecar.jsonl"
    forecaster = _FakeForecaster()
    result = generate_foundation_sidecar(
        bars_path,
        sidecar,
        _write_foundation_spec(tmp_path),
        forecaster=forecaster,
    )
    assert result["records"] == 3
    assert len(forecaster.histories) == 3
    assert all(len(history) == 20 for history in forecaster.histories)
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert all(record["controls"]["target_return_accessed"] is False for record in records)

    rows = [
        FeatureRow(
            timestamp=datetime.fromisoformat(record["feature_time"]),
            symbol="AAPL",
            features=(0.1, 0.2),
            target_return=999.0,
            label_end_time=datetime.fromisoformat(record["forecast_end_time"]),
        )
        for record in records
    ]
    dataset, _ = write_feature_dataset(
        tmp_path / "dataset.jsonl",
        rows,
        ("momentum", "news_score"),
        {"dataset_role": "calibration", "sealed": False, "forecast_bars": 5},
    )
    joined = join_foundation_sidecar(dataset, sidecar, tmp_path / "joined.jsonl")
    assert joined["rows"] == 3
    assert joined["feature_names"][-2:] == [
        "chronos_expected_return",
        "chronos_uncertainty",
    ]


def test_local_news_inference_is_immutable_and_has_no_order_authority(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "news-weights.bin"
    weights.write_bytes(b"news-test")
    spec = tmp_path / "news-spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "FinGPT/test",
                "revision": "revision",
                "local_model_path": str(weights),
                "weight_sha256": sha256_tree(weights),
                "license": "research",
            }
        ),
        encoding="utf-8",
    )
    event = NewsEvent(
        symbol="AAPL",
        headline="Apple reports earnings",
        source="test",
        sentiment=0,
        novelty=1,
        source_quality=1,
        event_time=datetime(2026, 1, 2, tzinfo=UTC),
        knowledge_time=datetime(2026, 1, 2, tzinfo=UTC),
    )
    source = tmp_path / "news.jsonl"
    source.write_text(event.model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "inference.jsonl"
    report = infer_news_archive(source, output, spec, backend=_FakeNewsBackend())
    assert report["parse_failures"] == 0
    record = json.loads(output.read_text())
    assert record["knowledge_time"] == event.knowledge_time.isoformat()
    assert record["normalized"]["primary_event_type"] == "earnings"
    assert record["controls"]["direct_order_authority"] is False


def _rl_rows(target_multiplier: float = 1.0) -> list[FeatureRow]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[FeatureRow] = []
    for index in range(4):
        for symbol_index, symbol in enumerate(("AAPL", "MSFT")):
            rows.append(
                FeatureRow(
                    timestamp=start + timedelta(hours=index),
                    symbol=symbol,
                    features=(0.1 * index, 0.2 * symbol_index),
                    target_return=target_multiplier * (0.01 if symbol == "AAPL" else -0.01),
                    label_end_time=start + timedelta(hours=index + 5),
                )
            )
    return rows


def test_rl_observation_excludes_future_target_and_reward_charges_costs() -> None:
    spec = RLTrainingSpec(algorithm="ppo", total_timesteps=100)
    first = PortfolioResearchEnvironment(_rl_rows(1.0), CostModelConfig(), spec)
    second = PortfolioResearchEnvironment(_rl_rows(100.0), CostModelConfig(), spec)
    assert first.reset() == second.reset()
    _, reward, terminated, info = first.step([1.0, -1.0])
    assert terminated is False
    assert info["execution_cost"] > 0
    assert info["gross_short_exposure"] <= spec.max_gross_short
    assert reward < info["period_return"]


def test_mixed_generation_is_exact_deterministic_and_quota_bound() -> None:
    first = generate_mixed_plan(generation=1, seed=42)
    second = generate_mixed_plan(generation=1, seed=42)
    assert first == second
    assert len(first.candidates) == 100
    assert Counter(item.category for item in first.candidates) == Counter(DEFAULT_QUOTAS)
    assert len({item.fingerprint for item in first.candidates}) == 100
    assert all(item.holdout_access == "forbidden" for item in first.candidates)


def test_mixed_candidate_report_rejects_holdout_access(tmp_path: Path) -> None:
    spec = generate_mixed_plan(generation=1, seed=7).candidates[0]
    model = tmp_path / "model.json"
    metrics = tmp_path / "metrics.json"
    model.write_text("{}", encoding="utf-8")
    metrics.write_text("{}", encoding="utf-8")
    report = {
        "candidate_id": spec.candidate_id,
        "status": "complete",
        "category": spec.category,
        "implementation": spec.implementation,
        "holdout_accessed": True,
        "model_path": str(model),
        "metrics_path": str(metrics),
        "composite_score": 1.0,
    }
    with pytest.raises(ValueError, match="accessed holdout"):
        _validate_candidate_report(report, spec, metrics)


def test_external_research_is_non_authoritative_and_hash_bound(tmp_path: Path) -> None:
    configuration = tmp_path / "qlib.yaml"
    configuration.write_text("market: test\n", encoding="utf-8")
    spec = tmp_path / "external.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "qlib",
                "tool_version": "test",
                "configuration_path": str(configuration),
                "configuration_sha256": sha256_file(configuration),
                "universe_manifest_sha256": "a" * 64,
                "dataset_sha256": "b" * 64,
                "model_identifier": "test-model",
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "qlib.json"
    source.write_text(
        json.dumps({"predictions": [{"score": 1}], "metrics": {"ic": 0.01}}),
        encoding="utf-8",
    )
    output = tmp_path / "evidence.json"
    result = import_external_research(source, output, spec)
    assert result["controls"]["authoritative_for_promotion"] is False
    assert result["controls"]["must_pass_native_holdout"] is True
