from datetime import UTC, datetime, timedelta

import pytest

from trading_app.dataset import load_feature_dataset, write_feature_dataset
from trading_app.holdout import evaluate_untouched_holdout, split_feature_dataset
from trading_app.model_registry import ModelRegistry
from trading_app.research import BacktestMetrics, FeatureRow, RidgeReturnModel


def _rows(count: int = 18) -> list[FeatureRow]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    returns = (0.008, 0.012, 0.005, 0.015, 0.009, 0.02)
    return [
        FeatureRow(
            timestamp=start + timedelta(days=index),
            symbol="AAPL",
            features=(float(index),),
            target_return=returns[index % len(returns)],
            label_end_time=start + timedelta(days=index + 2),
        )
        for index in range(count)
    ]


def _write_source(tmp_path):
    source = tmp_path / "features.jsonl"
    write_feature_dataset(
        source,
        _rows(),
        ("trend",),
        {
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_kind": "historical_point_in_time",
            "forecast_bars": 2,
        },
    )
    return source


def _metrics() -> BacktestMetrics:
    return BacktestMetrics(
        observations=600,
        net_return=0.12,
        annualized_return=0.12,
        sharpe=1.1,
        max_drawdown=0.05,
        hit_rate=0.56,
        turnover=20,
        average_trade_return=0.006,
        folds=6,
        excess_return_vs_benchmark=0.02,
        news_sharpe_delta=0.10,
    )


def test_split_purges_calibration_labels_crossing_holdout_boundary(tmp_path) -> None:
    source = _write_source(tmp_path)
    boundary = datetime(2025, 1, 11, tzinfo=UTC)

    report = split_feature_dataset(
        str(source),
        str(tmp_path / "calibration.jsonl"),
        str(tmp_path / "holdout.jsonl"),
        split_time=boundary,
    )
    calibration, calibration_names, calibration_metadata = load_feature_dataset(
        tmp_path / "calibration.jsonl"
    )
    holdout, holdout_names, holdout_metadata = load_feature_dataset(
        tmp_path / "holdout.jsonl"
    )

    assert calibration_names == holdout_names == ("trend",)
    assert all(row.timestamp < boundary for row in calibration)
    assert all(row.label_end_time is None or row.label_end_time < boundary for row in calibration)
    assert all(row.timestamp >= boundary for row in holdout)
    assert report["purged_boundary_rows"] == 2
    assert calibration_metadata["dataset_role"] == "calibration"
    assert holdout_metadata["dataset_role"] == "untouched_holdout"
    assert holdout_metadata["sealed"] is True
    assert calibration_metadata["split"]["split_id"] == holdout_metadata["split"]["split_id"]


def test_frozen_model_scores_holdout_once_and_can_pass_governed_promotion(tmp_path) -> None:
    source = _write_source(tmp_path)
    boundary = datetime(2025, 1, 11, tzinfo=UTC)
    split_feature_dataset(
        str(source),
        str(tmp_path / "calibration.jsonl"),
        str(tmp_path / "holdout.jsonl"),
        split_time=boundary,
    )
    calibration, feature_names, _ = load_feature_dataset(tmp_path / "calibration.jsonl")
    model = RidgeReturnModel(feature_names)
    model.fit(calibration)
    registry = ModelRegistry(tmp_path / "models")
    record = registry.register(
        model,
        _metrics(),
        metadata={
            "dataset": str(tmp_path / "calibration.jsonl"),
            "dataset_sha256": "calibration-hash",
            "validation": {
                "prediction_threshold": -1.0,
                "transaction_cost_bps": 0.0,
                "effective_periods_per_year": 52.0,
            },
        },
    )

    with pytest.raises(ValueError, match="untouched_holdout_evaluation_missing"):
        registry.promote(record.version)

    report = evaluate_untouched_holdout(
        str(tmp_path / "holdout.jsonl"),
        str(tmp_path / "models"),
        str(tmp_path / "holdout-report.json"),
        version=record.version,
    )
    assert report["metrics"]["net_return"] > 0
    assert report["controls"]["threshold_frozen_from_registration"] is True
    evaluations = registry.holdout_evaluations(version=record.version)
    assert len(evaluations) == 1
    assert evaluations[0]["split_id"] == report["holdout"]["split_id"]

    with pytest.raises(ValueError, match="already been scored"):
        evaluate_untouched_holdout(
            str(tmp_path / "holdout.jsonl"),
            str(tmp_path / "models"),
            str(tmp_path / "holdout-report-two.json"),
            version=record.version,
        )

    promoted = registry.promote(
        record.version,
        minimum_holdout_net_return=-1.0,
        minimum_holdout_sharpe=-1.0,
        maximum_holdout_drawdown=1.0,
        minimum_holdout_observations=1,
        minimum_holdout_excess_return=-1.0,
    )
    assert promoted.version == record.version
    champion = registry.champion()
    assert champion is not None
    assert champion.version == record.version
    promotion = registry.history(alias="champion")[-1]
    assert promotion["details"]["holdout_evaluation"]["id"] == evaluations[0]["id"]


def test_holdout_evaluation_rejects_calibration_dataset(tmp_path) -> None:
    source = _write_source(tmp_path)
    split_feature_dataset(
        str(source),
        str(tmp_path / "calibration.jsonl"),
        str(tmp_path / "holdout.jsonl"),
        split_time=datetime(2025, 1, 11, tzinfo=UTC),
    )
    calibration, feature_names, _ = load_feature_dataset(tmp_path / "calibration.jsonl")
    model = RidgeReturnModel(feature_names)
    model.fit(calibration)
    registry = ModelRegistry(tmp_path / "models")
    record = registry.register(
        model,
        _metrics(),
        metadata={
            "validation": {
                "prediction_threshold": -1.0,
                "transaction_cost_bps": 0.0,
                "effective_periods_per_year": 52.0,
            }
        },
    )

    with pytest.raises(ValueError, match="not a sealed untouched_holdout"):
        evaluate_untouched_holdout(
            str(tmp_path / "calibration.jsonl"),
            str(tmp_path / "models"),
            str(tmp_path / "invalid-report.json"),
            version=record.version,
        )
