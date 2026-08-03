from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .candidates import (
    CandidateConfig,
    _Scored,
    _nested_folds,
    _select_features,
    _simulate_candidate,
)
from .cost_model import load_cost_model
from .dataset import load_feature_dataset
from .governed_models import GovernedModelRegistry
from .modeling import ContextTradingModel, ModelBuildSpec, TradingModel, build_model
from .research import FeatureRow
from .universe import assert_universe_binding, load_universe_manifest


def train_governed_model(
    dataset_path: str,
    registry_path: str,
    *,
    family: str,
    feature_subset: tuple[str, ...] | None = None,
    hyperparameters: dict[str, object] | None = None,
    seed: int = 1729,
    outer_folds: int = 8,
    positive_threshold: float = 0.0005,
    negative_threshold: float = 0.0005,
    cost_config_path: str = "config/costs/conservative-us-paper-v1.json",
    universe_manifest_path: str = "config/universes/us-liquid-large-cap-v1.json",
    periods_per_year: float = 327.6,
) -> dict[str, object]:
    rows, feature_names, metadata = load_feature_dataset(dataset_path)
    if metadata.get("dataset_role") != "calibration" or metadata.get("sealed") is True:
        raise ValueError("Multi-model training accepts calibration datasets only")
    forecast_horizon = int(metadata.get("forecast_bars", 5))
    selected_names = feature_subset or feature_names
    selected_rows = _select_features(rows, feature_names, selected_names)
    universe = load_universe_manifest(universe_manifest_path)
    assert_universe_binding(metadata, universe, context="multi-model calibration")
    cost_model = load_cost_model(cost_config_path)
    folds = _nested_folds(selected_rows, outer_folds)
    scored: list[_Scored] = []
    purged_rows = 0
    parameters = hyperparameters or {}
    for fold in folds:
        model = build_model(
            ModelBuildSpec(
                family=family,
                feature_names=selected_names,
                hyperparameters=parameters,
                seed=seed + fold.index,
                forecast_horizon=forecast_horizon,
            )
        )
        model.fit(fold.train)
        scored.extend(_score_fold(model, fold.train, fold.test, fold.index))
        purged_rows += fold.purged_outer + fold.purged_inner
    if not scored:
        raise ValueError("Model produced no out-of-sample predictions")
    configuration = CandidateConfig(
        ridge=1e-3,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        feature_subset=selected_names,
        cost_stress_multiplier=1.0,
        max_long_exposure=0.60,
        max_short_exposure=0.30,
        forecast_horizon=forecast_horizon,
        seed=seed,
    )
    metrics = _simulate_candidate(
        scored,
        configuration,
        cost_model,
        universe,
        periods_per_year,
    )
    metrics["purged_rows"] = purged_rows
    metrics["model_family"] = family
    final_model = build_model(
        ModelBuildSpec(
            family=family,
            feature_names=selected_names,
            hyperparameters=parameters,
            seed=seed,
            forecast_horizon=forecast_horizon,
        )
    )
    final_model.fit(selected_rows)
    registry = GovernedModelRegistry(registry_path)
    record = registry.register_any(
        final_model,
        metrics,
        metadata={
            "dataset_path": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "dataset_metadata": metadata,
            "universe": universe.binding(universe_manifest_path),
            "validation": {
                "method": "nested_purged_chronological_walk_forward",
                "outer_folds": outer_folds,
                "prediction_threshold": positive_threshold,
                "negative_prediction_threshold": negative_threshold,
                "effective_periods_per_year": periods_per_year,
                "cost_model": cost_model.model_dump(mode="json"),
                "cost_model_sha256": cost_model.manifest_sha256,
            },
        },
        hyperparameters=parameters,
        seed=seed,
    )
    return {
        "model_version": record.version,
        "model_family": family,
        "metrics": metrics,
        "artifact": record.model_path,
        "execution_scope": "paper_only",
        "holdout_accessed": False,
    }


def _score_fold(
    model: TradingModel,
    training_rows: list[FeatureRow],
    test_rows: list[FeatureRow],
    fold_index: int,
) -> list[_Scored]:
    if not isinstance(model, ContextTradingModel):
        return [
            _Scored(row=row, prediction=model.predict(row.features), fold=fold_index)
            for row in test_rows
        ]
    test_keys = {(row.timestamp, row.symbol, row.label_end_time) for row in test_rows}
    context_by_symbol: dict[str, deque[tuple[float, ...]]] = defaultdict(
        lambda: deque(maxlen=model.capabilities.required_history)
    )
    scored: list[_Scored] = []
    for row in sorted(training_rows + test_rows, key=lambda item: (item.timestamp, item.symbol)):
        context = context_by_symbol[row.symbol]
        context.append(row.features)
        key = (row.timestamp, row.symbol, row.label_end_time)
        if key in test_keys and len(context) >= model.capabilities.required_history:
            scored.append(
                _Scored(
                    row=row,
                    prediction=model.predict_context(tuple(context)),
                    fold=fold_index,
                )
            )
    return scored


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
