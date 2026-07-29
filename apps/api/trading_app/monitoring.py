from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev

from .cost_model import CostModelConfig, legacy_cost_model
from .dataset import dataset_metadata_path, load_feature_dataset
from .model_registry import ModelRegistry
from .research import _ScoredRow, _simulate, metrics_dict

_Z_BOUNDS = (
    float("-inf"),
    -2.0,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    2.0,
    float("inf"),
)


def evaluate_model_monitoring(
    reference_path: str,
    recent_path: str,
    registry_path: str,
    output_path: str,
    *,
    version: str | None = None,
    alias: str = "champion",
    minimum_recent_rows: int = 100,
    minimum_active_signals: int = 20,
    maximum_feature_psi: float = 0.25,
    maximum_feature_mean_shift: float = 1.0,
    maximum_prediction_mean_shift: float = 1.0,
    maximum_rmse_ratio: float = 2.0,
    minimum_active_hit_rate: float = 0.50,
    minimum_calibration_slope: float = 0.25,
    maximum_calibration_slope: float = 1.75,
) -> dict[str, object]:
    reference_source = Path(reference_path)
    recent_source = Path(recent_path)
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"Monitoring output already exists: {destination}")
    if reference_source.resolve() == recent_source.resolve():
        raise ValueError("Reference and recent datasets must differ")
    if minimum_recent_rows < 1:
        raise ValueError("minimum_recent_rows must be positive")
    if minimum_active_signals < 0:
        raise ValueError("minimum_active_signals must not be negative")
    if maximum_feature_psi <= 0:
        raise ValueError("maximum_feature_psi must be positive")
    if maximum_feature_mean_shift <= 0 or maximum_prediction_mean_shift <= 0:
        raise ValueError("Mean-shift gates must be positive")
    if maximum_rmse_ratio <= 0:
        raise ValueError("maximum_rmse_ratio must be positive")
    if not 0 <= minimum_active_hit_rate <= 1:
        raise ValueError("minimum_active_hit_rate must be between 0 and 1")
    if minimum_calibration_slope >= maximum_calibration_slope:
        raise ValueError("Calibration slope bounds are invalid")

    reference_rows, reference_names, reference_metadata = load_feature_dataset(
        reference_source
    )
    recent_rows, recent_names, recent_metadata = load_feature_dataset(recent_source)
    if tuple(reference_names) != tuple(recent_names):
        raise ValueError(
            "Reference and recent feature schemas differ: "
            f"reference={reference_names}, recent={recent_names}"
        )
    reference_end = max(row.timestamp for row in reference_rows)
    recent_start = min(row.timestamp for row in recent_rows)
    if recent_start <= reference_end:
        raise ValueError(
            "Recent monitoring data must begin after the reference calibration dataset"
        )

    registry = ModelRegistry(registry_path)
    record = registry.get(version) if version else registry.alias(alias)
    if record is None:
        raise ValueError(f"No model is assigned to alias {alias!r}")
    model = registry.load(record.version)
    if tuple(model.feature_names) != tuple(reference_names):
        raise ValueError(
            "Model feature schema does not match monitoring datasets: "
            f"model={model.feature_names}, dataset={reference_names}"
        )

    reference_sha256 = _sha256(reference_source)
    registered_reference_sha256 = record.metadata.get("dataset_sha256")
    if registered_reference_sha256 and registered_reference_sha256 != reference_sha256:
        raise ValueError(
            "Reference dataset hash does not match the model's registered calibration dataset"
        )
    validation = record.metadata.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Registered model lacks frozen validation configuration")
    try:
        threshold = float(validation["prediction_threshold"])
        transaction_cost_bps = float(validation["transaction_cost_bps"])
        periods_per_year = float(validation["effective_periods_per_year"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Registered model validation configuration is incomplete"
        ) from error
    raw_cost_model = validation.get("cost_model")
    cost_model = (
        legacy_cost_model(transaction_cost_bps)
        if raw_cost_model is None
        else CostModelConfig.model_validate(raw_cost_model)
    )
    if validation.get("cost_model_sha256") not in {
        None,
        cost_model.manifest_sha256,
    }:
        raise ValueError("Registered cost model hash does not match its manifest")

    reference_predictions = [model.predict(row.features) for row in reference_rows]
    recent_predictions = [model.predict(row.features) for row in recent_rows]
    reference_targets = [row.target_return for row in reference_rows]
    recent_targets = [row.target_return for row in recent_rows]

    feature_diagnostics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(reference_names):
        reference_values = [row.features[index] for row in reference_rows]
        recent_values = [row.features[index] for row in recent_rows]
        reference_mean = fmean(reference_values)
        reference_std = max(pstdev(reference_values), 1e-12)
        recent_mean = fmean(recent_values)
        recent_std = pstdev(recent_values) if len(recent_values) > 1 else 0.0
        feature_diagnostics[name] = {
            "reference_mean": reference_mean,
            "reference_std": reference_std,
            "recent_mean": recent_mean,
            "recent_std": recent_std,
            "standardized_mean_shift": abs(recent_mean - reference_mean) / reference_std,
            "std_ratio": recent_std / reference_std,
            "population_stability_index": _population_stability_index(
                reference_values,
                recent_values,
                reference_mean,
                reference_std,
            ),
        }

    reference_prediction_mean = fmean(reference_predictions)
    reference_prediction_std = max(pstdev(reference_predictions), 1e-12)
    recent_prediction_mean = fmean(recent_predictions)
    recent_prediction_std = (
        pstdev(recent_predictions) if len(recent_predictions) > 1 else 0.0
    )
    prediction_diagnostics = {
        "reference_mean": reference_prediction_mean,
        "reference_std": reference_prediction_std,
        "recent_mean": recent_prediction_mean,
        "recent_std": recent_prediction_std,
        "standardized_mean_shift": (
            abs(recent_prediction_mean - reference_prediction_mean)
            / reference_prediction_std
        ),
        "reference_active_rate": sum(
            prediction > threshold for prediction in reference_predictions
        )
        / len(reference_predictions),
        "recent_active_rate": sum(
            prediction > threshold for prediction in recent_predictions
        )
        / len(recent_predictions),
    }

    reference_performance = _calibration_diagnostics(
        reference_predictions,
        reference_targets,
        threshold,
    )
    recent_performance = _calibration_diagnostics(
        recent_predictions,
        recent_targets,
        threshold,
    )
    reference_scored = [
        _ScoredRow(row=row, prediction=prediction)
        for row, prediction in zip(reference_rows, reference_predictions, strict=True)
    ]
    recent_scored = [
        _ScoredRow(row=row, prediction=prediction)
        for row, prediction in zip(recent_rows, recent_predictions, strict=True)
    ]
    reference_simulation = _simulate(
        reference_scored,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        folds=1,
        purged_rows=0,
        cost_model=cost_model,
    )
    recent_simulation = _simulate(
        recent_scored,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        folds=1,
        purged_rows=0,
        cost_model=cost_model,
    )

    alerts: list[dict[str, object]] = []
    if len(recent_rows) < minimum_recent_rows:
        alerts.append(
            _alert(
                "insufficient_recent_rows",
                len(recent_rows),
                minimum_recent_rows,
                "minimum",
            )
        )
    for name, diagnostics in feature_diagnostics.items():
        if diagnostics["population_stability_index"] > maximum_feature_psi:
            alerts.append(
                _alert(
                    f"feature_psi:{name}",
                    diagnostics["population_stability_index"],
                    maximum_feature_psi,
                    "maximum",
                )
            )
        if diagnostics["standardized_mean_shift"] > maximum_feature_mean_shift:
            alerts.append(
                _alert(
                    f"feature_mean_shift:{name}",
                    diagnostics["standardized_mean_shift"],
                    maximum_feature_mean_shift,
                    "maximum",
                )
            )
    if (
        prediction_diagnostics["standardized_mean_shift"]
        > maximum_prediction_mean_shift
    ):
        alerts.append(
            _alert(
                "prediction_mean_shift",
                prediction_diagnostics["standardized_mean_shift"],
                maximum_prediction_mean_shift,
                "maximum",
            )
        )
    if recent_performance["active_signals"] < minimum_active_signals:
        alerts.append(
            _alert(
                "insufficient_active_signals",
                recent_performance["active_signals"],
                minimum_active_signals,
                "minimum",
            )
        )
    elif recent_performance["active_hit_rate"] < minimum_active_hit_rate:
        alerts.append(
            _alert(
                "active_hit_rate",
                recent_performance["active_hit_rate"],
                minimum_active_hit_rate,
                "minimum",
            )
        )
    reference_rmse = max(reference_performance["rmse"], 1e-12)
    rmse_ratio = recent_performance["rmse"] / reference_rmse
    if rmse_ratio > maximum_rmse_ratio:
        alerts.append(
            _alert("rmse_ratio", rmse_ratio, maximum_rmse_ratio, "maximum")
        )
    recent_slope = recent_performance["calibration_slope"]
    if recent_slope < minimum_calibration_slope:
        alerts.append(
            _alert(
                "calibration_slope_low",
                recent_slope,
                minimum_calibration_slope,
                "minimum",
            )
        )
    if recent_slope > maximum_calibration_slope:
        alerts.append(
            _alert(
                "calibration_slope_high",
                recent_slope,
                maximum_calibration_slope,
                "maximum",
            )
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "model_drift_and_calibration_monitoring",
        "created_at": datetime.now(UTC).isoformat(),
        "healthy": not alerts,
        "recommended_action": "continue_paper" if not alerts else "pause_and_review",
        "model": {
            "version": record.version,
            "model_path": record.model_path,
            "model_sha256": _sha256(registry.root / record.model_path),
            "alias": None if version else alias,
            "registered_at": record.created_at,
        },
        "reference": _dataset_evidence(
            reference_source,
            reference_metadata,
            reference_rows,
        ),
        "recent": _dataset_evidence(recent_source, recent_metadata, recent_rows),
        "frozen_configuration": {
            "feature_names": list(reference_names),
            "prediction_threshold": threshold,
            "transaction_cost_bps": transaction_cost_bps,
            "cost_model": cost_model.model_dump(mode="json"),
            "cost_model_sha256": cost_model.manifest_sha256,
            "effective_periods_per_year": periods_per_year,
        },
        "gates": {
            "minimum_recent_rows": minimum_recent_rows,
            "minimum_active_signals": minimum_active_signals,
            "maximum_feature_psi": maximum_feature_psi,
            "maximum_feature_mean_shift": maximum_feature_mean_shift,
            "maximum_prediction_mean_shift": maximum_prediction_mean_shift,
            "maximum_rmse_ratio": maximum_rmse_ratio,
            "minimum_active_hit_rate": minimum_active_hit_rate,
            "minimum_calibration_slope": minimum_calibration_slope,
            "maximum_calibration_slope": maximum_calibration_slope,
        },
        "alerts": alerts,
        "features": feature_diagnostics,
        "predictions": prediction_diagnostics,
        "calibration": {
            "reference": reference_performance,
            "recent": recent_performance,
            "rmse_ratio": rmse_ratio,
        },
        "paper_simulation": {
            "reference": metrics_dict(reference_simulation.metrics),
            "recent": metrics_dict(recent_simulation.metrics),
            "recent_symbols": recent_simulation.symbols,
        },
        "controls": {
            "reference_hash_matches_registered_calibration": (
                not registered_reference_sha256
                or registered_reference_sha256 == reference_sha256
            ),
            "recent_window_strictly_after_reference": True,
            "model_weights_frozen": True,
            "threshold_and_costs_frozen": True,
            "realized_targets_required": True,
            "output_immutable": True,
        },
        "limitations": [
            "Drift thresholds are governance rules, not universal statistical constants.",
            "A healthy report does not prove future profitability.",
            "Recent outcomes become available only after each configured forecast horizon matures.",
            "Monitoring data must not be recycled into training without starting a new governed research cycle.",
        ],
        "output": str(destination),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _calibration_diagnostics(
    predictions: list[float],
    targets: list[float],
    threshold: float,
) -> dict[str, float | int]:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Predictions and targets must be non-empty and aligned")
    errors = [prediction - target for prediction, target in zip(predictions, targets, strict=True)]
    active = [
        (prediction, target)
        for prediction, target in zip(predictions, targets, strict=True)
        if prediction > threshold
    ]
    prediction_mean = fmean(predictions)
    target_mean = fmean(targets)
    prediction_variance = fmean(
        (prediction - prediction_mean) ** 2 for prediction in predictions
    )
    covariance = fmean(
        (prediction - prediction_mean) * (target - target_mean)
        for prediction, target in zip(predictions, targets, strict=True)
    )
    slope = covariance / prediction_variance if prediction_variance > 1e-18 else 0.0
    intercept = target_mean - slope * prediction_mean
    target_variance = fmean((target - target_mean) ** 2 for target in targets)
    correlation = (
        covariance / math.sqrt(prediction_variance * target_variance)
        if prediction_variance > 1e-18 and target_variance > 1e-18
        else 0.0
    )
    return {
        "observations": len(predictions),
        "mae": fmean(abs(error) for error in errors),
        "rmse": math.sqrt(fmean(error**2 for error in errors)),
        "prediction_target_correlation": correlation,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "active_signals": len(active),
        "active_hit_rate": (
            0.0 if not active else sum(target > 0 for _, target in active) / len(active)
        ),
        "active_average_realized_return": (
            0.0 if not active else fmean(target for _, target in active)
        ),
    }


def _population_stability_index(
    reference: list[float],
    recent: list[float],
    mean: float,
    scale: float,
) -> float:
    reference_counts = [0] * (len(_Z_BOUNDS) - 1)
    recent_counts = [0] * (len(_Z_BOUNDS) - 1)
    for value in reference:
        reference_counts[_bin_index((value - mean) / scale)] += 1
    for value in recent:
        recent_counts[_bin_index((value - mean) / scale)] += 1
    epsilon = 1e-6
    score = 0.0
    for reference_count, recent_count in zip(reference_counts, recent_counts, strict=True):
        reference_rate = max(reference_count / len(reference), epsilon)
        recent_rate = max(recent_count / len(recent), epsilon)
        score += (recent_rate - reference_rate) * math.log(recent_rate / reference_rate)
    return score


def _bin_index(value: float) -> int:
    for index, (lower, upper) in enumerate(
        zip(_Z_BOUNDS[:-1], _Z_BOUNDS[1:], strict=True)
    ):
        if lower <= value < upper:
            return index
    return len(_Z_BOUNDS) - 2


def _dataset_evidence(
    path: Path,
    metadata: dict[str, object],
    rows: list[object],
) -> dict[str, object]:
    timestamps = [getattr(row, "timestamp") for row in rows]
    metadata_path = dataset_metadata_path(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "rows": len(rows),
        "start": min(timestamps).astimezone(UTC).isoformat(),
        "end": max(timestamps).astimezone(UTC).isoformat(),
        "metadata": metadata,
    }


def _alert(
    name: str,
    observed: float | int,
    threshold: float | int,
    direction: str,
) -> dict[str, object]:
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "direction": direction,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
