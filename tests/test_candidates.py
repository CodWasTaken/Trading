from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from trading_app.candidates import (
    CandidateConfig,
    _nested_folds,
    generate_candidate_configs,
    rank_candidates,
    run_candidate_generation,
    should_stop_generations,
    validate_candidate_configs,
)
from trading_app.dataset import write_feature_dataset
from trading_app.research import FeatureRow

FEATURES = ("momentum", "news_score", "volatility", "spread_bps")
COST_CONFIG = Path(__file__).parents[1] / "config" / "costs" / "conservative-us-paper-v1.json"


def _rows(timestamp_count: int = 96) -> list[FeatureRow]:
    start = datetime(2024, 1, 2, 14, tzinfo=UTC)
    rows = []
    for index in range(timestamp_count):
        for symbol_index, symbol in enumerate(("AAA", "BBB")):
            direction = 1 if (index + symbol_index) % 8 < 4 else -1
            momentum = direction * (0.002 + index / 1_000_000)
            news = direction * 0.3 if index % 7 == 0 else 0.0
            volatility = 0.01 + (index % 5) / 10_000
            spread = 5.0 + symbol_index
            rows.append(
                FeatureRow(
                    timestamp=start + timedelta(hours=index),
                    symbol=symbol,
                    features=(momentum, news, volatility, spread),
                    target_return=direction * 0.003 - 0.0002 * symbol_index,
                    label_end_time=start + timedelta(hours=index + 5),
                )
            )
    return rows


def _dataset(tmp_path: Path, *, role: str = "calibration", sealed: bool = False) -> Path:
    dataset, _ = write_feature_dataset(
        tmp_path / f"{role}.jsonl",
        _rows(),
        FEATURES,
        {
            "dataset_role": role,
            "sealed": sealed,
            "forecast_bars": 5,
        },
    )
    return dataset


def test_candidate_generation_is_exact_deterministic_and_unique() -> None:
    first = generate_candidate_configs(FEATURES, seed=42)
    second = generate_candidate_configs(FEATURES, seed=42)
    different = generate_candidate_configs(FEATURES, seed=43)

    assert len(first) == 100
    assert first == second
    assert first != different
    assert len({candidate.fingerprint for candidate in first}) == 100
    assert all(candidate.forecast_horizon == 5 for candidate in first)


def test_duplicate_and_non_100_candidate_sets_are_rejected() -> None:
    configs = generate_candidate_configs(FEATURES, seed=42)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_candidate_configs([*configs[:-1], configs[0]])
    with pytest.raises(ValueError, match="exactly 100"):
        validate_candidate_configs(configs[:-1])


def test_nested_folds_purge_the_five_bar_label_boundary() -> None:
    folds = _nested_folds(_rows(), 8)

    assert len(folds) == 8
    assert sum(fold.purged_outer for fold in folds) > 0
    for fold in folds:
        test_start = min(row.timestamp for row in fold.test)
        inner_start = min(row.timestamp for row in fold.inner_validation)
        assert all(
            row.label_end_time is None or row.label_end_time < test_start for row in fold.train
        )
        assert all(
            row.label_end_time is None or row.label_end_time < inner_start
            for row in fold.inner_train
        )


def test_ranking_uses_score_then_stable_candidate_id() -> None:
    reports = [
        {"candidate_id": "candidate-b", "status": "complete", "composite_score": 2.0},
        {"candidate_id": "candidate-a", "status": "complete", "composite_score": 2.0},
        {"candidate_id": "candidate-c", "status": "complete", "composite_score": 3.0},
        {"candidate_id": "candidate-failed", "status": "failed"},
    ]

    assert [item["candidate_id"] for item in rank_candidates(reports)] == [
        "candidate-c",
        "candidate-a",
        "candidate-b",
        "candidate-failed",
    ]


def test_generation_evaluates_100_on_calibration_and_freezes_only_three(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    output = tmp_path / "generation-1"
    report = run_candidate_generation(
        str(dataset),
        str(output),
        generation=1,
        seed=1729,
        cost_config_path=str(COST_CONFIG),
        universe_manifest_path=None,
        max_workers=2,
        outer_folds=8,
    )

    assert report["candidate_count"] == 100
    assert report["completed"] == 100
    assert report["failed"] == 0
    assert report["candidate_family_size"] == 3
    assert len(report["ranking"]) == 100
    assert len(report["finalists"]) == 3
    assert len(list((output / "candidates").glob("*/config.json"))) == 100
    assert len(list((output / "candidates").glob("*/metrics.json"))) == 100
    assert not list((output / "candidates").glob("*/failure.json"))

    finalists = json.loads((output / "finalists.json").read_text(encoding="utf-8"))
    assert finalists["candidate_family_size"] == 3
    assert finalists["controls"]["sealed_holdout_not_accessed"] is True
    assert all(item["frozen"] for item in finalists["finalists"])

    resumed = run_candidate_generation(
        str(dataset),
        str(output),
        generation=1,
        seed=1729,
        cost_config_path=str(COST_CONFIG),
        universe_manifest_path=None,
        max_workers=2,
        outer_folds=8,
    )
    assert resumed["ranking"] == report["ranking"]
    assert resumed["finalists"] == report["finalists"]


def test_generation_cannot_open_sealed_holdout(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, role="untouched_holdout", sealed=True)

    with pytest.raises(ValueError, match="holdout data is forbidden"):
        run_candidate_generation(
            str(dataset),
            str(tmp_path / "forbidden"),
            generation=1,
            seed=1729,
            cost_config_path=str(COST_CONFIG),
            universe_manifest_path=None,
        )
    assert not (tmp_path / "forbidden").exists()


def test_generation_lifecycle_stops_after_three_stale_or_ten_total() -> None:
    assert not should_stop_generations(
        [1.0, 1.1, 1.15],
        meaningful_improvement=0.01,
    )
    assert should_stop_generations(
        [1.0, 1.1, 1.2, 1.201, 1.202, 1.203],
        meaningful_improvement=0.01,
    )
    assert should_stop_generations(
        [float(value) for value in range(10)],
        meaningful_improvement=0.01,
    )


def test_custom_candidate_seed_is_part_of_resumption_manifest(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    output = tmp_path / "generation"
    configs = generate_candidate_configs(FEATURES, seed=5)
    changed = [config.model_copy(update={"seed": config.seed + 1}) for config in configs]
    run_candidate_generation(
        str(dataset),
        str(output),
        generation=1,
        seed=5,
        cost_config_path=str(COST_CONFIG),
        universe_manifest_path=None,
        configs=configs,
        max_workers=2,
    )
    with pytest.raises(ValueError, match="manifest does not match"):
        run_candidate_generation(
            str(dataset),
            str(output),
            generation=1,
            seed=5,
            cost_config_path=str(COST_CONFIG),
            universe_manifest_path=None,
            configs=changed,
            max_workers=2,
        )


def test_candidate_config_rejects_short_exposure_above_safety_ceiling() -> None:
    with pytest.raises(ValueError):
        CandidateConfig(
            ridge=0.001,
            positive_threshold=0.001,
            negative_threshold=0.001,
            feature_subset=FEATURES,
            cost_stress_multiplier=1,
            max_long_exposure=0.5,
            max_short_exposure=0.31,
            seed=1,
        )
