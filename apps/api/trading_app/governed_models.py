from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .model_registry import ModelRecord, ModelRegistry
from .modeling import (
    ModelArtifactManifest,
    ModelProvenance,
    RidgeModelAdapter,
    TradingModel,
    artifact_extension,
    artifact_manifest,
    load_model,
    sha256_file,
)
from .research import BacktestMetrics, metrics_dict


class GovernedModelRegistry(ModelRegistry):
    """ModelRegistry extension for heterogeneous hash-verified paper models."""

    def register_any(
        self,
        model: TradingModel,
        metrics: BacktestMetrics | dict[str, object],
        *,
        metadata: dict[str, object] | None = None,
        hyperparameters: dict[str, object] | None = None,
        seed: int = 1729,
        provenance: ModelProvenance | None = None,
        set_challenger: bool = True,
    ) -> ModelRecord:
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]
        suffix = artifact_extension(model)
        model_path = self.root / f"model-{version}{suffix}"
        model.save(model_path)
        manifest = artifact_manifest(
            model,
            model_path,
            hyperparameters=hyperparameters or {},
            seed=seed,
            provenance=provenance,
        )
        record = ModelRecord(
            version=version,
            created_at=datetime.now(UTC).isoformat(),
            model_path=model_path.name,
            metrics=(metrics_dict(metrics) if isinstance(metrics, BacktestMetrics) else dict(metrics)),
            metadata={
                **(metadata or {}),
                "model_artifact": manifest.model_dump(mode="json"),
                "execution_scope": "paper_only",
                "live_money_authorized": False,
            },
        )
        index = self._read()
        models = dict(index["models"])
        models[version] = asdict(record)
        index["models"] = models
        if set_challenger:
            self._change_alias(
                index,
                "challenger",
                version,
                action="register_challenger",
                reason="heterogeneous_model_registered",
                details={
                    "family": model.capabilities.family,
                    "implementation": model.capabilities.implementation,
                    "task": model.capabilities.task,
                    "artifact_sha256": manifest.artifact_sha256,
                },
            )
        self._write(index)
        return record

    def manifest(self, version: str) -> ModelArtifactManifest | None:
        raw = self.get(version).metadata.get("model_artifact")
        return None if not isinstance(raw, dict) else ModelArtifactManifest.model_validate(raw)

    def verify_artifact(self, version: str) -> bool:
        record = self.get(version)
        path = self.root / record.model_path
        if not path.is_file():
            return False
        manifest = self.manifest(version)
        return True if manifest is None else sha256_file(path) == manifest.artifact_sha256

    def load_any(self, version: str) -> TradingModel:
        record = self.get(version)
        path = self.root / record.model_path
        if not path.is_file():
            raise FileNotFoundError(f"Registered model artifact is missing: {path}")
        manifest = self.manifest(version)
        if manifest is None:
            return RidgeModelAdapter(self.load(version))
        if sha256_file(path) != manifest.artifact_sha256:
            raise ValueError("Registered model artifact hash mismatch")
        model = load_model(path)
        if model.capabilities != manifest.capabilities:
            raise ValueError("Model artifact capabilities do not match registry metadata")
        return model

    def active_paper(self) -> ModelRecord | None:
        return self.alias("active-paper")

    def select_active_paper(
        self,
        version: str,
        *,
        reason: str,
        engine_running: bool,
        kill_switch: bool,
        runtime_feature_names: tuple[str, ...],
        universe_manifest_sha256: str,
    ) -> ModelRecord:
        failures = self.selection_failures(
            version,
            engine_running=engine_running,
            kill_switch=kill_switch,
            runtime_feature_names=runtime_feature_names,
            universe_manifest_sha256=universe_manifest_sha256,
        )
        if failures:
            raise ValueError("Model is not eligible for paper selection: " + ", ".join(failures))
        manifest = self.manifest(version)
        assert manifest is not None
        return self.set_alias(
            "active-paper",
            version,
            reason=reason,
            action="select_active_paper_model",
            details={
                "artifact_sha256": manifest.artifact_sha256,
                "compatibility_checks": {
                    "engine_paused_or_kill_switched": True,
                    "feature_schema": list(runtime_feature_names),
                    "universe_manifest_sha256": universe_manifest_sha256,
                    "historically_promoted": True,
                    "monitoring_not_unhealthy": True,
                },
                "execution_scope": "paper_only",
                "live_money_authorized": False,
            },
        )

    def selection_failures(
        self,
        version: str,
        *,
        engine_running: bool,
        kill_switch: bool,
        runtime_feature_names: tuple[str, ...],
        universe_manifest_sha256: str,
    ) -> list[str]:
        record = self.get(version)
        manifest = self.manifest(version)
        failures: list[str] = []
        if engine_running and not kill_switch:
            failures.append("engine_must_be_paused_or_kill_switched")
        if manifest is None:
            failures.append("legacy_model_missing_capability_manifest")
            return failures
        capabilities = manifest.capabilities
        if not self.verify_artifact(version):
            failures.append("artifact_hash_verification_failed")
        if capabilities.task != "return_regression":
            failures.append("model_task_is_not_return_regression")
        if not capabilities.live_compatible:
            failures.append("model_not_live_compatible")
        if capabilities.required_history > 20:
            failures.append("required_history_exceeds_runtime_limit")
        if not set(capabilities.feature_names).issubset(set(runtime_feature_names)):
            failures.append("runtime_feature_schema_incompatible")
        promoted = self._was_promoted(version)
        if not promoted:
            failures.append("historical_promotion_missing")
        status = record.metadata.get("monitoring_status")
        if status in {"pause_and_review", "unhealthy", "invalidated"}:
            failures.append("monitoring_unhealthy")
        registered_universe = _registered_universe_hash(record)
        if registered_universe != universe_manifest_sha256:
            failures.append("runtime_universe_manifest_mismatch")
        return failures

    def _was_promoted(self, version: str) -> bool:
        current = self.champion()
        if current is not None and current.version == version:
            return True
        return any(
            event.get("alias") == "champion"
            and event.get("version") == version
            and event.get("action") == "promote"
            for event in self.history(alias="champion")
        )

    def catalogue(
        self,
        *,
        engine_running: bool,
        kill_switch: bool,
        runtime_feature_names: tuple[str, ...],
        universe_manifest_sha256: str,
    ) -> dict[str, object]:
        index = self.summary()
        aliases = {str(name): str(version) for name, version in dict(index["aliases"]).items()}
        entries: list[dict[str, object]] = []
        for version in sorted(dict(index["models"]), reverse=True):
            record = self.get(version)
            manifest = self.manifest(version)
            failures = self.selection_failures(
                version,
                engine_running=engine_running,
                kill_switch=kill_switch,
                runtime_feature_names=runtime_feature_names,
                universe_manifest_sha256=universe_manifest_sha256,
            )
            holdout = self.latest_holdout_evaluation(version)
            graduations = self.paper_graduations(version=version)
            entries.append(
                {
                    "version": version,
                    "created_at": record.created_at,
                    "aliases": sorted(name for name, target in aliases.items() if target == version),
                    "capabilities": None if manifest is None else manifest.capabilities.model_dump(mode="json"),
                    "provenance": None if manifest is None else manifest.provenance.model_dump(mode="json"),
                    "artifact_sha256": None if manifest is None else manifest.artifact_sha256,
                    "artifact_verified": self.verify_artifact(version),
                    "metrics": record.metrics,
                    "holdout": holdout,
                    "paper_graduation": graduations[-1] if graduations else None,
                    "monitoring_status": record.metadata.get("monitoring_status", "not_evaluated"),
                    "eligible_for_active_paper": not failures,
                    "selection_blockers": failures,
                    "metadata": _safe_metadata(record.metadata),
                }
            )
        return {
            "execution_scope": "paper_only",
            "live_money_authorized": False,
            "aliases": aliases,
            "models": entries,
        }

    def compare_versions(self, versions: list[str]) -> dict[str, object]:
        if not 1 <= len(versions) <= 8:
            raise ValueError("Compare between one and eight model versions")
        return {
            "versions": [
                {
                    "version": version,
                    "metrics": self.get(version).metrics,
                    "holdout": self.latest_holdout_evaluation(version),
                    "manifest": (
                        None
                        if self.manifest(version) is None
                        else self.manifest(version).model_dump(mode="json")
                    ),
                }
                for version in versions
            ]
        }


def _registered_universe_hash(record: ModelRecord) -> str | None:
    direct = record.metadata.get("universe")
    if isinstance(direct, dict) and isinstance(direct.get("manifest_sha256"), str):
        return str(direct["manifest_sha256"])
    dataset = record.metadata.get("dataset_metadata")
    if isinstance(dataset, dict):
        universe = dataset.get("universe")
        if isinstance(universe, dict) and isinstance(universe.get("manifest_sha256"), str):
            return str(universe["manifest_sha256"])
    return None


def _safe_metadata(metadata: dict[str, object]) -> dict[str, object]:
    blocked = {"api_key", "secret", "token", "password", "credentials"}
    return {
        key: value
        for key, value in metadata.items()
        if not any(word in key.lower() for word in blocked)
    }
