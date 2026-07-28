import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from trading_app.dataset import load_feature_dataset, write_feature_dataset
from trading_app.model_registry import ModelRegistry
from trading_app.monitoring import evaluate_model_monitoring
from trading_app.research import BacktestMetrics, FeatureRow, RidgeReturnModel


def _feature_values(cycles: int, *, shift: float = 0.0) -> list[tuple[float, float]]:
    base = [(-1.0 + index * 0.1) for index in range(20)]
    return [
        (value + shift, (value + shift) ** 2)
        for _ in range(cycles)
        for value in base
    ]


def _rows(
    start: datetime,
    cycles: int,
    *,
    shift: float = 0.0,
    invert_target: bool = False,
) -> list[FeatureRow]:
    rows = []
    for index, features in enumerate(_feature_values(cycles, shift=shift)):
        target = 0.004 + 0.012 * features[0]
        if invert_target:
            target = -target
        rows.append(
            FeatureRow(
                timestamp=start + timedelta(hours=index),
                symbol="AAPL" if index % 2 == 0 else "MSFT",
                features=features,
                target_return=target,
                label_end_time=start + timedelta(hours=index + 1),
            )
        )
    return rows


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _setup(tmp_path):
    reference_path = tmp_path / "reference.jsonl"
    recent_path = tmp_path / "recent.jsonl"
    start = datetime(2025, 1, 1, tzinfo=UTC)
    write_feature_dataset(
        reference_path,
        _rows(start, 10),
        ("signal", "signal_squared"),
        {"dataset_role": "calibration"},
    )
    reference_rows, feature_names, _ = load_feature_dataset(reference_path)
    model = RidgeReturnModel(feature_names, ridge=1e-8)
    model.fit(reference_rows)
    registry = ModelRegistry(tmp_path / "models")
    record = registry.register(
        model,
        _metrics(),
        metadata={
            "dataset": str(reference_path),
            "dataset_sha256": _sha256(reference_path),
            "validation": {
                "prediction_threshold": 0.0,
                "transaction_cost_bps": 0.0,
                "effective_periods_per_year": 1638.0,
            },
        },
    )
    registry.set_alias(
        "champion",
        record.version,
        reason="monitoring fixture",
        action="test_champion",
    )
    return reference_path, recent_path, start, record


def test_monitoring_accepts_stable_later_distribution(tmp_path) -> None:
    reference_path, recent_path, start, record = _setup(tmp_path)
    write_feature_dataset(
        recent_path,
        _rows(start + timedelta(hours=300), 6),
        ("signal", "signal_squared"),
        {"dataset_role": "matured_monitoring"},
    )

    report = evaluate_model_monitoring(
        str(reference_path),
        str(recent_path),
        str(tmp_path / "models"),
        str(tmp_path / "monitor.json"),
        version=record.version,
        minimum_recent_rows=100,
        minimum_active_signals=20,
        maximum_feature_psi=0.25,
        maximum_feature_mean_shift=0.25,
        maximum_prediction_mean_shift=0.25,
        maximum_rmse_ratio=2.0,
        minimum_active_hit_rate=0.9,
        minimum_calibration_slope=0.9,
        maximum_calibration_slope=1.1,
    )

    assert report["healthy"] is True
    assert report["recommended_action"] == "continue_paper"
    assert report["alerts"] == []
    assert report["calibration"]["recent"]["calibration_slope"] == pytest.approx(
        1.0, abs=1e-4
    )
    assert report["controls"]["reference_hash_matches_registered_calibration"] is True


def test_monitoring_flags_shift_and_realized_degradation(tmp_path) -> None:
    reference_path, recent_path, start, record = _setup(tmp_path)
    write_feature_dataset(
        recent_path,
        _rows(
            start + timedelta(hours=300),
            6,
            shift=4.0,
            invert_target=True,
        ),
        ("signal", "signal_squared"),
        {"dataset_role": "matured_monitoring"},
    )

    report = evaluate_model_monitoring(
        str(reference_path),
        str(recent_path),
        str(tmp_path / "models"),
        str(tmp_path / "monitor-shifted.json"),
        version=record.version,
        minimum_recent_rows=100,
        minimum_active_signals=20,
        maximum_feature_psi=0.10,
        maximum_feature_mean_shift=0.50,
        maximum_prediction_mean_shift=0.50,
        maximum_rmse_ratio=1.25,
        minimum_active_hit_rate=0.50,
        minimum_calibration_slope=0.25,
        maximum_calibration_slope=1.75,
    )

    names = {alert["name"] for alert in report["alerts"]}
    assert report["healthy"] is False
    assert report["recommended_action"] == "pause_and_review"
    assert "prediction_mean_shift" in names
    assert "active_hit_rate" in names
    assert "calibration_slope_low" in names
    assert any(name.startswith("feature_psi:") for name in names)
    assert any(name.startswith("feature_mean_shift:") for name in names)


def test_monitoring_rejects_wrong_reference_hash(tmp_path) -> None:
    reference_path, recent_path, start, record = _setup(tmp_path)
    write_feature_dataset(
        recent_path,
        _rows(start + timedelta(hours=300), 6),
        ("signal", "signal_squared"),
        {"dataset_role": "matured_monitoring"},
    )
    registry = ModelRegistry(tmp_path / "models")
    index = registry.summary()
    models = dict(index["models"])
    payload = dict(models[record.version])
    metadata = dict(payload["metadata"])
    metadata["dataset_sha256"] = "wrong-hash"
    payload["metadata"] = metadata
    models[record.version] = payload
    index["models"] = models
    registry._write(index)

    with pytest.raises(ValueError, match="registered calibration dataset"):
        evaluate_model_monitoring(
            str(reference_path),
            str(recent_path),
            str(tmp_path / "models"),
            str(tmp_path / "monitor-invalid.json"),
            version=record.version,
        )
