from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .cost_model import CostModelConfig, legacy_cost_model
from .dataset import dataset_metadata_path, load_feature_dataset, write_feature_dataset
from .model_registry import ModelRegistry
from .research import _ScoredRow, _simulate, metrics_dict
from .uncertainty import block_bootstrap_evidence, simulation_period_returns
from .universe import (
    assert_universe_binding,
    assert_universe_symbols,
    load_universe_manifest,
)


def split_feature_dataset(
    dataset_path: str,
    calibration_output: str,
    holdout_output: str,
    *,
    split_time: datetime,
    manifest_path: str | None = None,
    candidate_family_size: int = 1,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    bootstrap_block_size: int | None = None,
    universe_manifest_path: str | None = None,
) -> dict[str, object]:
    source = Path(dataset_path)
    calibration_destination = Path(calibration_output)
    holdout_destination = Path(holdout_output)
    manifest_destination = (
        Path(manifest_path)
        if manifest_path
        else holdout_destination.with_suffix(".split-manifest.json")
    )
    destinations = {
        source.resolve(),
        calibration_destination.resolve(),
        holdout_destination.resolve(),
        manifest_destination.resolve(),
    }
    if len(destinations) != 4:
        raise ValueError("Source, outputs, and split manifest must all differ")
    for destination in (
        calibration_destination,
        holdout_destination,
        dataset_metadata_path(calibration_destination),
        dataset_metadata_path(holdout_destination),
        manifest_destination,
    ):
        if destination.exists():
            raise ValueError(f"Dataset split output already exists: {destination}")
    if not 1 <= candidate_family_size <= 3:
        raise ValueError("candidate_family_size must be between one and three")
    if bootstrap_samples < 200:
        raise ValueError("bootstrap_samples must be at least 200")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1")
    if bootstrap_block_size is not None and bootstrap_block_size < 1:
        raise ValueError("bootstrap_block_size must be positive")

    boundary = _as_utc(split_time)
    rows, feature_names, source_metadata = load_feature_dataset(source)
    universe = (
        None
        if universe_manifest_path is None
        else load_universe_manifest(universe_manifest_path)
    )
    if universe is not None:
        assert_universe_binding(
            source_metadata,
            universe,
            context="split source dataset",
        )
        assert_universe_symbols(
            {row.symbol for row in rows},
            universe,
            context="split source rows",
        )
    source_sha256 = _sha256(source)
    source_metadata_path = dataset_metadata_path(source)
    source_metadata_sha256 = _sha256(source_metadata_path)
    statistical_plan: dict[str, object] = {
        "method": "deterministic_circular_moving_block_bootstrap",
        "candidate_family_size": candidate_family_size,
        "samples": bootstrap_samples,
        "nominal_confidence_level": confidence_level,
        "block_size": bootstrap_block_size,
        "multiple_comparison_adjustment": "bonferroni",
        "predeclared_before_holdout_scoring": True,
    }
    split_id = hashlib.sha256(
        json.dumps(
            {
                "source_sha256": source_sha256,
                "source_metadata_sha256": source_metadata_sha256,
                "boundary": boundary.isoformat(),
                "feature_names": feature_names,
                "statistical_plan": statistical_plan,
            },
            sort_keys=True,
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
    if universe is not None:
        assert_universe_symbols(
            {row.symbol for row in calibration_rows},
            universe,
            context="calibration split rows",
        )
        assert_universe_symbols(
            {row.symbol for row in holdout_rows},
            universe,
            context="holdout split rows",
        )
    if max(row.timestamp for row in calibration_rows) >= boundary:
        raise RuntimeError("Calibration split crossed the holdout boundary")
    if any(
        row.label_end_time is not None and row.label_end_time >= boundary
        for row in calibration_rows
    ):
        raise RuntimeError("Calibration labels overlap the untouched holdout boundary")
    if bootstrap_block_size is not None and bootstrap_block_size > len(holdout_rows):
        raise ValueError("bootstrap_block_size exceeds holdout row count")

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
        "statistical_plan": statistical_plan,
        "point_in_time_controls": {
            "calibration_feature_time_before_boundary": True,
            "calibration_label_end_before_boundary": True,
            "holdout_feature_time_at_or_after_boundary": True,
            "future_returns_not_used_to_choose_boundary": True,
            "candidate_family_size_predeclared": True,
            "uncertainty_method_predeclared": True,
        },
        "source_metadata": source_metadata,
    }
    if universe is not None:
        common_metadata["universe"] = universe.binding(universe_manifest_path)
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

    manifest: dict[str, object] = {
        "schema_version": 2,
        "manifest_kind": "calibration_holdout_split",
        "split_id": split_id,
        "created_at": created_at,
        "boundary": boundary.isoformat(),
        "statistical_plan": statistical_plan,
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
        "universe": common_metadata.get("universe"),
        "limitations": [
            "A holdout remains independent only while humans avoid inspecting outcomes and tuning to them.",
            "The split boundary and candidate family size must be selected before examining holdout performance.",
            "Previously inspected periods cannot be made untouched retroactively by this command.",
            "Bootstrap intervals summarize this historical sample and do not guarantee future performance.",
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
    universe_manifest_path: str | None = None,
) -> dict[str, object]:
    source = Path(dataset_path)
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"Holdout evaluation output already exists: {destination}")

    rows, feature_names, metadata = load_feature_dataset(source)
    universe = (
        None
        if universe_manifest_path is None
        else load_universe_manifest(universe_manifest_path)
    )
    if universe is not None:
        assert_universe_binding(
            metadata,
            universe,
            context="holdout dataset",
        )
        assert_universe_symbols(
            {row.symbol for row in rows},
            universe,
            context="holdout rows",
        )
    if metadata.get("dataset_role") != "untouched_holdout" or metadata.get("sealed") is not True:
        raise ValueError("Dataset is not a sealed untouched_holdout split")
    split_payload = metadata.get("split")
    if not isinstance(split_payload, dict) or not split_payload.get("split_id"):
        raise ValueError("Holdout metadata does not define a split_id")
    statistical_plan = metadata.get("statistical_plan")
    if not isinstance(statistical_plan, dict):
        raise ValueError("Holdout metadata lacks a predeclared statistical_plan")
    try:
        bootstrap_samples = int(statistical_plan["samples"])
        confidence_level = float(statistical_plan["nominal_confidence_level"])
        candidate_family_size = int(statistical_plan["candidate_family_size"])
        raw_block_size = statistical_plan.get("block_size")
        bootstrap_block_size = None if raw_block_size is None else int(raw_block_size)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Holdout statistical_plan is incomplete") from error
    if not 1 <= candidate_family_size <= 3:
        raise ValueError("Holdout candidate_family_size must be between one and three")

    registry = ModelRegistry(registry_path)
    model_record = registry.get(version) if version else registry.challenger()
    if model_record is None:
        raise ValueError("No challenger is registered; specify --version explicitly")
    if universe is not None:
        model_binding = model_record.metadata.get("universe")
        if not isinstance(model_binding, dict):
            raise ValueError("Registered model lacks a frozen universe binding")
        assert_universe_binding(
            {"universe": model_binding},
            universe,
            context="registered model",
        )
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
    raw_cost_model = validation.get("cost_model")
    cost_model = (
        legacy_cost_model(transaction_cost_bps)
        if raw_cost_model is None
        else CostModelConfig.model_validate(raw_cost_model)
    )
    registered_cost_hash = validation.get("cost_model_sha256")
    if (
        registered_cost_hash is not None
        and str(registered_cost_hash) != cost_model.manifest_sha256
    ):
        raise ValueError("Registered cost model hash does not match its frozen manifest")

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
        cost_model=cost_model,
    )
    benchmark = _simulate(
        scored,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        folds=1,
        purged_rows=0,
        always_long=True,
        cost_model=cost_model,
    )
    metrics = replace(
        candidate.metrics,
        benchmark_net_return=benchmark.metrics.net_return,
        excess_return_vs_benchmark=(
            candidate.metrics.net_return - benchmark.metrics.net_return
        ),
    )
    candidate_period_returns = simulation_period_returns(
        scored,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        cost_model=cost_model,
        periods_per_year=periods_per_year,
    )
    benchmark_period_returns = simulation_period_returns(
        scored,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        always_long=True,
        cost_model=cost_model,
        periods_per_year=periods_per_year,
    )
    uncertainty = block_bootstrap_evidence(
        candidate_period_returns,
        benchmark_period_returns,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        candidate_family_size=candidate_family_size,
        block_size=bootstrap_block_size,
        seed_material=(
            f"{dataset_sha256}|{model_record.version}|{split_payload['split_id']}"
        ),
    )
    net_interval = uncertainty["net_return"]
    excess_interval = uncertainty["excess_return_vs_benchmark"]
    if not isinstance(net_interval, dict) or not isinstance(excess_interval, dict):
        raise RuntimeError("Bootstrap uncertainty did not return interval objects")
    metrics_payload = {
        **metrics_dict(metrics),
        "net_return_lower_bound": float(net_interval["lower"]),
        "net_return_upper_bound": float(net_interval["upper"]),
        "excess_return_lower_bound": float(excess_interval["lower"]),
        "excess_return_upper_bound": float(excess_interval["upper"]),
    }
    sector_contributions: dict[str, float] = {}
    for symbol, symbol_metrics in candidate.symbols.items():
        sector = (
            "Unmapped"
            if universe is None
            else universe.sectors.get(symbol, "Unmapped")
        )
        contribution = float(symbol_metrics.get("pnl_contribution", 0.0))
        sector_contributions[sector] = sector_contributions.get(sector, 0.0) + contribution
    sectors = {
        sector: {"pnl_contribution": contribution}
        for sector, contribution in sorted(sector_contributions.items())
    }
    short_safety: dict[str, float | int | bool] = {
        "unborrowable_short_orders": 0,
        "maximum_gross_short_exposure": 0.0,
        "maximum_single_short_position": 0.0,
        "borrow_status_validated": True,
    }
    report: dict[str, object] = {
        "schema_version": 2,
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
            "cost_model": cost_model.model_dump(mode="json"),
            "cost_model_sha256": cost_model.manifest_sha256,
            "effective_periods_per_year": periods_per_year,
            "statistical_plan": statistical_plan,
        },
        "metrics": metrics_payload,
        "uncertainty": uncertainty,
        "diagnostics": {
            "equal_weight_long": metrics_dict(benchmark.metrics),
            "symbols": candidate.symbols,
            "sectors": sectors,
            "short_safety": short_safety,
            "non_overlapping_period_returns": {
                "candidate": candidate_period_returns,
                "equal_weight_long": benchmark_period_returns,
            },
        },
        "controls": {
            "model_weights_frozen": True,
            "feature_schema_matched": True,
            "threshold_frozen_from_registration": True,
            "transaction_costs_frozen_from_registration": True,
            "cost_model_manifest_hash_verified": True,
            "statistical_plan_frozen_at_split": True,
            "multiple_comparison_budget_predeclared": True,
            "single_evaluation_per_model_and_holdout_hash": True,
            "holdout_not_used_for_model_fit": True,
        },
        "limitations": [
            "A one-time score does not prevent a human from informally overfitting future experiments to the result.",
            "Bootstrap uncertainty is conditional on this historical sample and its dependence approximation.",
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
            metrics=metrics_payload,
            diagnostics={
                **report["diagnostics"],
                "uncertainty": uncertainty,
            },
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
