from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dataset import dataset_metadata_path, load_feature_dataset, write_feature_dataset
from .model_registry import ModelRegistry
from .research import _ScoredRow, _simulate, metrics_dict


def split_feature_dataset(
    dataset_path: str,
    calibration_output: str,
    holdout_output: str,
    *,
    split_time: datetime,
    manifest_path: str | None = None,
) -> dict[str, object]:
    source = Path(dataset_path)
    calibration_destination = Path(calibration_output)
    holdout_destination = Path(holdout_output)
    destinations = {
        source.resolve(),
        calibration_destination.resolve(),
        holdout_destination.resolve(),
    }
    if len(destinations) != 3:
        raise ValueError("Source, calibration output, and holdout output must differ")
    for destination in (calibration_destination, holdout_destination):
        if destination.exists() or dataset_metadata_path(destination).exists():
            raise ValueError(f"Dataset split output already exists: {destination}")

    boundary = _as_utc(split_time)
    rows, feature_names, source_metadata = load_feature_dataset(source)
    source_sha256 = _sha256(source)
    source_metadata_path = dataset_metadata_path(source)
    source_metadata_sha256 = _sha256(source_metadata_path)
    split_id = hashlib.sha256(
        (
            f"{source_sha256}|{source_metadata_sha256}|{boundary.isoformat()}|"
            + ",".join(feature_names)
        ).encode()
    ).hexdigest()

    calibration_rows = []
    holdout_rows = []
    purged_boundary_rows = []
    for row in rows:
        if row.timestamp >= boundary:
            holdout_rows.append(row)
        elif row.label_end_time is not None and row.label_end_time >= boundary:
            purged_boundary_rows.append(row)
        else:
            calibration_rows.append(row)

    if not calibration_rows:
        raise ValueError("Split produced no calibration rows")
    if not holdout_rows:
        raise ValueError("Split produced no untouched holdout rows")
    if max(row.timestamp for row in calibration_rows) >= boundary:
        raise RuntimeError("Calibration split crossed the holdout boundary")
    if any(
        row.label_end_time is not None and row.label_end_time >= boundary
        for row in calibration_rows
    ):
        raise RuntimeError("Calibration labels overlap the untouched holdout boundary")

    created_at = datetime.now(UTC).isoformat()
    common_metadata: dict[str, object] = {
        "created_at": created_at,
        "dataset_kind": source_metadata.get("dataset_kind", "feature_dataset"),
        "parent_dataset": {
            "path": str(source),
            "sha256": source_sha256,
            "metadata_path": str(source_metadata_path),
            "metadata_sha256": source_metadata_sha256,
        },
        "split": {
            "split_id": split_id,
            "boundary": boundary.isoformat(),
            "method": "chronological_feature_time_with_label_end_boundary_purge",
            "purged_boundary_rows": len(purged_boundary_rows),
        },
        "point_in_time_controls": {
            "calibration_feature_time_before_boundary": True,
            "calibration_label_end_before_boundary": True,
            "holdout_feature_time_at_or_after_boundary": True,
            "future_returns_not_used_to_choose_boundary": True,
        },
        "source_metadata": source_metadata,
    }
    calibration_path, calibration_metadata_path = write_feature_dataset(
        calibration_destination,
        calibration_rows,
        feature_names,
        {
            **common_metadata,
            "dataset_role": "calibration",
            "sealed": False,
        },
    )
    holdout_path, holdout_metadata_path = write_feature_dataset(
        holdout_destination,
        holdout_rows,
        feature_names,
        {
            **common_metadata,
            "dataset_role": "untouched_holdout",
            "sealed": True,
            "sealed_at": created_at,
            "usage_policy": "score each frozen model version at most once",
        },
    )

    manifest_destination = (
        Path(manifest_path)
        if manifest_path
        else holdout_destination.with_suffix(".split-manifest.json")
    )
    if manifest_destination.exists():
        raise ValueError(f"Split manifest already exists: {manifest_destination}")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_kind": "calibration_holdout_split",
        "split_id": split_id,
        "created_at": created_at,
        "boundary": boundary.isoformat(),
        "source": {
            "path": str(source),
            "rows": len(rows),
            "sha256": source_sha256,
            "metadata_sha256": source_metadata_sha256,
        },
        "calibration": {
            "path": str(calibration_path),
            "rows": len(calibration_rows),
            "sha256": _sha256(calibration_path),
            "metadata_path": str(calibration_metadata_path),
            "metadata_sha256": _sha256(calibration_metadata_path),
        },
        "holdout": {
            "path": str(holdout_path),
            "rows": len(holdout_rows),
            "sha256": _sha256(holdout_path),
            "metadata_path": str(holdout_metadata_path),
            "metadata_sha256": _sha256(holdout_metadata_path),
            "sealed": True,
        },
        "purged_boundary_rows": len(purged_boundary_rows),
        "controls": common_metadata["point_in_time_controls"],
        "limitations": [
            "A holdout remains independent only while humans avoid inspecting outcomes and tuning to them.",
            "The split boundary must be selected before examining holdout performance.",
            "Previously inspected periods cannot be made untouched retroactively by this command.",
        ],
    }
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_destination)}


def evaluate_untouched_holdout(
    dataset_path: str,
    registry_path: str,
    output_path: str,
    *,
    version: str | None = None,
) -> dict[str, object]:
    source = Path(dataset_path)
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"Holdout evaluation output already exists: {destination}")

    rows, feature_names, metadata = load_feature_dataset(source)
    if metadata.get("dataset_role") != "untouched_holdout" or metadata.get("sealed") is not True:
        raise ValueError("Dataset is not a sealed untouched_holdout split")
    split_payload = metadata.get("split")
    if not isinstance(split_payload, dict) or not split_payload.get("split_id"):
        raise ValueError("Holdout metadata does not define a split_id")

    registry = ModelRegistry(registry_path)
    model_record = registry.get(version) if version else registry.challenger()
    if model_record is None:
        raise ValueError("No challenger is registered; specify --version explicitly")
    model = registry.load(model_record.version)
    if tuple(model.feature_names) != tuple(feature_names):
        raise ValueError(
            "Holdout feature names do not match the frozen model: "
            f"dataset={feature_names}, model={model.feature_names}"
        )

    validation = model_record.metadata.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Registered model lacks frozen validation configuration")
    try:
        threshold = float(validation["prediction_threshold"])
        transaction_cost_bps = float(validation["transaction_cost_bps"])
        periods_per_year = float(validation["effective_periods_per_year"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Registered model validation configuration is incomplete; "
            "threshold and cost assumptions must be frozen before holdout scoring"
        ) from error

    dataset_sha256 = _sha256(source)
    metadata_source = dataset_metadata_path(source)
    metadata_sha256 = _sha256(metadata_source)
    registry.assert_holdout_available(model_record.version, dataset_sha256)

    scored = [_ScoredRow(row=row, prediction=model.predict(row.features)) for row in rows]
    candidate = _simulate(
        scored,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        folds=1,
        purged_rows=0,
    )
    benchmark = _simulate(
        scored,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        folds=1,
        purged_rows=0,
        always_long=True,
    )
    metrics = replace(
        candidate.metrics,
        benchmark_net_return=benchmark.metrics.net_return,
        excess_return_vs_benchmark=(
            candidate.metrics.net_return - benchmark.metrics.net_return
        ),
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "untouched_holdout_evaluation",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "model": {
            "version": model_record.version,
            "model_path": model_record.model_path,
            "registered_at": model_record.created_at,
            "calibration_dataset_sha256": model_record.metadata.get("dataset_sha256"),
        },
        "holdout": {
            "path": str(source),
            "sha256": dataset_sha256,
            "metadata_path": str(metadata_source),
            "metadata_sha256": metadata_sha256,
            "split_id": split_payload["split_id"],
            "boundary": split_payload.get("boundary"),
            "rows": len(rows),
        },
        "frozen_configuration": {
            "feature_names": list(feature_names),
            "prediction_threshold": threshold,
            "transaction_cost_bps": transaction_cost_bps,
            "effective_periods_per_year": periods_per_year,
        },
        "metrics": metrics_dict(metrics),
        "diagnostics": {
            "equal_weight_long": metrics_dict(benchmark.metrics),
            "symbols": candidate.symbols,
        },
        "controls": {
            "model_weights_frozen": True,
            "feature_schema_matched": True,
            "threshold_frozen_from_registration": True,
            "transaction_costs_frozen_from_registration": True,
            "single_evaluation_per_model_and_holdout_hash": True,
            "holdout_not_used_for_model_fit": True,
        },
        "limitations": [
            "A one-time score does not prevent a human from informally overfitting future experiments to the result.",
            "Promotion still requires paper-trading evidence after historical gates pass.",
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_sha256 = hashlib.sha256(payload.encode()).hexdigest()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(payload, encoding="utf-8")
    try:
        evaluation = registry.record_holdout_evaluation(
            model_record.version,
            dataset_path=str(source),
            dataset_sha256=dataset_sha256,
            metadata_sha256=metadata_sha256,
            split_id=str(split_payload["split_id"]),
            report_path=str(destination),
            report_sha256=report_sha256,
            metrics=metrics_dict(metrics),
            diagnostics=report["diagnostics"],
            frozen_configuration=report["frozen_configuration"],
        )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {**report, "registry_evaluation_id": evaluation["id"], "output": str(destination)}


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
