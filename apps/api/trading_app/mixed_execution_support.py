from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .dataset import dataset_metadata_path, load_feature_dataset
from .mixed_candidates import MixedCandidateSpec, MixedGenerationPlan
from .modeling import sha256_file

if TYPE_CHECKING:
    from .mixed_execution_models import MixedExecutionResources


def verify_resources(
    plan: MixedGenerationPlan,
    resources: MixedExecutionResources,
    resource_directory: Path,
) -> dict[str, object]:
    required_keys = {
        dataset_key(spec)
        for spec in plan.candidates
        if spec.candidate_id not in resources.candidate_datasets
    }
    missing = sorted(required_keys - set(resources.datasets))
    if missing:
        raise ValueError(f"Mixed execution dataset resources are missing: {missing}")
    bindings: dict[str, object] = {}
    for key, raw_path in sorted(resources.datasets.items()):
        path = resolve(resource_directory, raw_path)
        bindings[key] = dataset_binding(path)
    for candidate_id, raw_path in sorted(resources.candidate_datasets.items()):
        if candidate_id not in {item.candidate_id for item in plan.candidates}:
            raise ValueError(f"Unknown candidate dataset override: {candidate_id}")
        path = resolve(resource_directory, raw_path)
        bindings[f"candidate:{candidate_id}"] = dataset_binding(path)
    cost = resolve(resource_directory, resources.cost_config_path)
    universe = resolve(resource_directory, resources.universe_manifest_path)
    if not cost.is_file() or not universe.is_file():
        raise ValueError("Cost or universe manifest is missing")
    bindings["cost_config"] = {"path": str(cost), "sha256": sha256_file(cost)}
    bindings["universe_manifest"] = {
        "path": str(universe),
        "sha256": sha256_file(universe),
    }
    return bindings


def dataset_binding(path: Path) -> dict[str, object]:
    _, _, metadata = load_feature_dataset(path)
    if metadata.get("dataset_role") != "calibration" or metadata.get("sealed") is True:
        raise ValueError(f"Mixed execution dataset is not calibration-only: {path}")
    metadata_path = dataset_metadata_path(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "metadata_sha256": sha256_file(metadata_path),
    }


def resolve_candidate_dataset(
    spec: MixedCandidateSpec,
    resources: MixedExecutionResources,
    resource_directory: Path,
) -> Path:
    raw = resources.candidate_datasets.get(spec.candidate_id)
    if raw is None:
        raw = resources.datasets[dataset_key(spec)]
    return resolve(resource_directory, raw)


def dataset_key(spec: MixedCandidateSpec) -> str:
    configured = spec.configuration.get("dataset_key")
    if isinstance(configured, str) and configured:
        return configured
    return "base"


def downstream_family(spec: MixedCandidateSpec) -> str:
    if spec.category in {"tabular_linear", "sequence"}:
        return spec.implementation
    configured = spec.configuration.get("downstream_family")
    return str(configured) if configured else "elastic_net"


def model_hyperparameters(
    spec: MixedCandidateSpec,
    family: str,
) -> dict[str, object]:
    configuration = spec.configuration
    if family == "ridge":
        return {"ridge": configuration_number(spec, "ridge", 1e-3)}
    if family == "elastic_net":
        return {
            "alpha": configuration_number(spec, "alpha", 1e-3),
            "l1_ratio": configuration_number(spec, "l1_ratio", 0.5),
            "iterations": 500,
        }
    if family == "lightgbm":
        return {
            "n_estimators": int(configuration.get("n_estimators", 150)),
            "learning_rate": float(configuration.get("learning_rate", 0.05)),
            "max_depth": int(configuration.get("max_depth", 4)),
        }
    if family == "xgboost":
        return {
            "n_estimators": int(configuration.get("n_estimators", 150)),
            "learning_rate": float(configuration.get("learning_rate", 0.05)),
            "max_depth": int(configuration.get("max_depth", 4)),
        }
    if family == "catboost":
        return {
            "iterations": int(configuration.get("iterations", 150)),
            "learning_rate": float(configuration.get("learning_rate", 0.05)),
            "depth": int(configuration.get("depth", 4)),
        }
    if family == "mlp":
        return {
            "hidden_layer_sizes": [32, 16],
            "learning_rate_init": 0.001,
        }
    if spec.category == "sequence":
        return {
            "context_length": int(configuration.get("context_length", 20)),
            "hidden_size": int(configuration.get("hidden_size", 32)),
            "learning_rate": float(configuration.get("learning_rate", 1e-3)),
            "epochs": int(configuration.get("epochs", 20)),
            "batch_size": int(configuration.get("batch_size", 64)),
        }
    return dict(configuration)


def feature_subset(
    spec: MixedCandidateSpec,
    feature_names: tuple[str, ...],
) -> tuple[str, ...]:
    if spec.category != "ensemble_ablation":
        return feature_names
    lowered = {name: name.lower() for name in feature_names}
    implementation = spec.implementation
    if implementation == "full_features":
        selected = feature_names
    elif implementation == "no_news":
        selected = tuple(name for name in feature_names if "news" not in lowered[name])
    elif implementation == "price_only":
        selected = tuple(
            name
            for name in feature_names
            if name in {"momentum", "volatility", "spread_bps"}
            or "price" in lowered[name]
        )
    elif implementation == "news_only":
        selected = tuple(name for name in feature_names if "news" in lowered[name])
    else:
        selected = tuple(
            name
            for name in feature_names
            if any(token in lowered[name] for token in ("timesfm", "chronos", "moirai"))
        )
    if not selected:
        raise ValueError(
            f"Ablation {implementation!r} selected no features from {feature_names}"
        )
    return selected


def configuration_number(
    spec: MixedCandidateSpec,
    key: str,
    default: float,
) -> float:
    value = spec.configuration.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Candidate configuration {key!r} must be numeric")
    return float(value)


def assert_report_matches_spec(
    report: dict[str, object],
    spec: MixedCandidateSpec,
) -> None:
    if report.get("candidate_id") != spec.candidate_id:
        raise ValueError("Candidate report ID mismatch")
    if report.get("status") != "complete":
        raise ValueError("Candidate report is not complete")
    if report.get("category") != spec.category:
        raise ValueError("Candidate report category mismatch")
    if report.get("implementation") != spec.implementation:
        raise ValueError("Candidate report implementation mismatch")
    if report.get("holdout_accessed") is not False:
        raise ValueError("Candidate report indicates holdout access")
    if report.get("configuration_fingerprint") != spec.fingerprint:
        raise ValueError("Candidate report configuration fingerprint mismatch")


def resolve(base: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base / path).resolve()
