from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .candidates import _composite_score
from .dataset import load_feature_dataset
from .mixed_candidates import MixedCandidateSpec
from .mixed_execution_support import (
    configuration_number,
    downstream_family,
    feature_subset,
    model_hyperparameters,
    resolve,
    resolve_candidate_dataset,
)
from .model_training import train_governed_model
from .modeling import sha256_file
from .rl_candidate_evaluation import evaluate_and_train_rl_candidate
from .rl_research import RLRewardConfig, RLTrainingSpec


class MixedExecutionResources(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    datasets: dict[str, str]
    candidate_datasets: dict[str, str] = Field(default_factory=dict)
    cost_config_path: str = "config/costs/conservative-us-paper-v1.json"
    universe_manifest_path: str = "config/universes/us-liquid-large-cap-v1.json"
    max_workers: int = Field(default=1, ge=1, le=8)
    outer_folds: int = Field(default=8, ge=8)
    periods_per_year: float = Field(default=327.6, gt=0)

    @model_validator(mode="after")
    def validate_resources(self) -> MixedExecutionResources:
        if self.schema_version != 1:
            raise ValueError("Unsupported mixed execution resource schema_version")
        if "base" not in self.datasets:
            raise ValueError("Mixed execution resources require a base dataset")
        return self


class MixedCandidateExecutor(Protocol):
    def execute(
        self,
        spec: MixedCandidateSpec,
        candidate_directory: Path,
        resources: MixedExecutionResources,
        resource_directory: Path,
    ) -> dict[str, object]: ...


class NativeMixedCandidateExecutor:
    def execute(
        self,
        spec: MixedCandidateSpec,
        candidate_directory: Path,
        resources: MixedExecutionResources,
        resource_directory: Path,
    ) -> dict[str, object]:
        dataset = resolve_candidate_dataset(
            spec,
            resources,
            resource_directory,
        )
        if spec.category == "reinforcement_learning":
            return self._execute_rl(
                spec,
                candidate_directory,
                resources,
                resource_directory,
                dataset,
            )
        return self._execute_return_model(
            spec,
            candidate_directory,
            resources,
            resource_directory,
            dataset,
        )

    def _execute_return_model(
        self,
        spec: MixedCandidateSpec,
        candidate_directory: Path,
        resources: MixedExecutionResources,
        resource_directory: Path,
        dataset: Path,
    ) -> dict[str, object]:
        _, feature_names, _ = load_feature_dataset(dataset)
        family = downstream_family(spec)
        selected_features = feature_subset(spec, feature_names)
        hyperparameters = model_hyperparameters(spec, family)
        registry = candidate_directory / "registry"
        result = train_governed_model(
            str(dataset),
            str(registry),
            family=family,
            feature_subset=selected_features,
            hyperparameters=hyperparameters,
            seed=spec.seed,
            outer_folds=resources.outer_folds,
            positive_threshold=configuration_number(
                spec,
                "positive_threshold",
                0.0005,
            ),
            negative_threshold=configuration_number(
                spec,
                "negative_threshold",
                0.0005,
            ),
            cost_config_path=str(
                resolve(resource_directory, resources.cost_config_path)
            ),
            universe_manifest_path=str(
                resolve(resource_directory, resources.universe_manifest_path)
            ),
            periods_per_year=resources.periods_per_year,
        )
        metrics = dict(result["metrics"])
        metrics.setdefault("news_ablation_improvement", 0.0)
        score, components = _composite_score(metrics)
        model_path = (registry / str(result["artifact"])).resolve()
        if not model_path.is_file():
            raise RuntimeError(f"Candidate model artifact is missing: {model_path}")
        return {
            "candidate_id": spec.candidate_id,
            "status": "complete",
            "category": spec.category,
            "implementation": spec.implementation,
            "configuration_fingerprint": spec.fingerprint,
            "configuration": spec.configuration,
            "dataset_path": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "metrics": metrics,
            "score_components": components,
            "composite_score": score,
            "holdout_accessed": False,
            "execution_scope": "paper_only",
            "live_money_authorized": False,
        }

    def _execute_rl(
        self,
        spec: MixedCandidateSpec,
        candidate_directory: Path,
        resources: MixedExecutionResources,
        resource_directory: Path,
        dataset: Path,
    ) -> dict[str, object]:
        reward = RLRewardConfig(
            drawdown_penalty=configuration_number(
                spec,
                "reward_drawdown_penalty",
                1.0,
            ),
            turnover_penalty=configuration_number(
                spec,
                "reward_turnover_penalty",
                0.01,
            ),
        )
        training = RLTrainingSpec(
            algorithm=spec.implementation,
            seed=spec.seed,
            total_timesteps=int(
                configuration_number(spec, "total_timesteps", 10_000)
            ),
            periods_per_year=resources.periods_per_year,
            reward=reward,
        )
        result = evaluate_and_train_rl_candidate(
            dataset,
            candidate_directory / "policy",
            training,
            resolve(resource_directory, resources.cost_config_path),
            resolve(resource_directory, resources.universe_manifest_path),
            outer_folds=resources.outer_folds,
        )
        metrics = dict(result["metrics"])
        score, components = _composite_score(metrics)
        model_path = Path(str(result["artifact"])).resolve()
        return {
            "candidate_id": spec.candidate_id,
            "status": "complete",
            "category": spec.category,
            "implementation": spec.implementation,
            "configuration_fingerprint": spec.fingerprint,
            "configuration": spec.configuration,
            "dataset_path": str(dataset),
            "dataset_sha256": sha256_file(dataset),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "metrics": metrics,
            "score_components": components,
            "composite_score": score,
            "holdout_accessed": False,
            "live_compatible": False,
            "execution_scope": "paper_only",
            "live_money_authorized": False,
        }
