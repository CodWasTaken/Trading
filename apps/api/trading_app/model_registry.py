from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .research import BacktestMetrics, RidgeReturnModel, metrics_dict


REGISTRY_SCHEMA_VERSION = 2
_ALIAS_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


@dataclass(frozen=True)
class ModelRecord:
    version: str
    created_at: str
    model_path: str
    metrics: dict[str, object]
    metadata: dict[str, object]


class ModelRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        if not self.index_path.exists():
            self._write(self._empty_index())

    def _empty_index(self) -> dict[str, object]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "models": {},
            "aliases": {},
            "alias_history": [],
        }

    def _read(self) -> dict[str, object]:
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Model registry index must be a JSON object")
        models = payload.get("models", {})
        aliases = payload.get("aliases", {})
        history = payload.get("alias_history", [])
        if not isinstance(models, dict):
            raise ValueError("Model registry models must be an object")
        if not isinstance(aliases, dict):
            raise ValueError("Model registry aliases must be an object")
        if not isinstance(history, list):
            raise ValueError("Model registry alias_history must be a list")
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "models": models,
            "aliases": aliases,
            "alias_history": history,
        }

    def _write(self, payload: dict[str, object]) -> None:
        normalized = {
            **payload,
            "schema_version": REGISTRY_SCHEMA_VERSION,
        }
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(normalized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.index_path)

    def register(
        self,
        model: RidgeReturnModel,
        metrics: BacktestMetrics,
        metadata: dict[str, object] | None = None,
        *,
        set_challenger: bool = True,
    ) -> ModelRecord:
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S") + "-" + uuid4().hex[:8]
        model_path = self.root / f"model-{version}.json"
        model.save(model_path)
        record = ModelRecord(
            version=version,
            created_at=datetime.now(UTC).isoformat(),
            model_path=model_path.name,
            metrics=metrics_dict(metrics),
            metadata=metadata or {},
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
                reason="new_model_registered",
                details={"model_created_at": record.created_at},
            )
        self._write(index)
        return record

    def get(self, version: str) -> ModelRecord:
        index = self._read()
        payload = dict(index["models"]).get(version)
        if payload is None:
            raise KeyError(f"Unknown model version: {version}")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid model record for version: {version}")
        return ModelRecord(**payload)

    def load(self, version: str) -> RidgeReturnModel:
        record = self.get(version)
        model_path = self.root / record.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Registered model artifact is missing: {model_path}")
        return RidgeReturnModel.load(model_path)

    def alias(self, name: str) -> ModelRecord | None:
        alias_name = self._validate_alias(name)
        index = self._read()
        version = dict(index["aliases"]).get(alias_name)
        return None if version is None else self.get(str(version))

    def set_alias(
        self,
        name: str,
        version: str,
        *,
        reason: str,
        action: str = "set_alias",
        details: dict[str, object] | None = None,
    ) -> ModelRecord:
        alias_name = self._validate_alias(name)
        record = self.get(version)
        index = self._read()
        self._change_alias(
            index,
            alias_name,
            version,
            action=action,
            reason=reason,
            details=details,
        )
        self._write(index)
        return record

    def promote(
        self,
        version: str,
        *,
        minimum_folds: int = 3,
        minimum_sharpe: float = 0.0,
        maximum_drawdown: float = 0.20,
        minimum_observations: int = 100,
        minimum_excess_return: float | None = None,
        minimum_news_sharpe_delta: float | None = None,
        reason: str = "promotion_gates_passed",
    ) -> ModelRecord:
        record = self.get(version)
        metrics = record.metrics
        failures: list[str] = []
        if float(metrics.get("net_return", 0)) <= 0:
            failures.append("non_positive_net_return")
        if float(metrics.get("sharpe", 0)) <= minimum_sharpe:
            failures.append("sharpe_below_gate")
        if float(metrics.get("max_drawdown", 1)) >= maximum_drawdown:
            failures.append("drawdown_above_gate")
        if int(metrics.get("folds", 0)) < minimum_folds:
            failures.append("insufficient_walk_forward_folds")
        if int(metrics.get("observations", 0)) < minimum_observations:
            failures.append("insufficient_out_of_sample_observations")
        if (
            minimum_excess_return is not None
            and float(metrics.get("excess_return_vs_benchmark", 0))
            <= minimum_excess_return
        ):
            failures.append("benchmark_excess_return_below_gate")
        if (
            minimum_news_sharpe_delta is not None
            and float(metrics.get("news_sharpe_delta", 0))
            <= minimum_news_sharpe_delta
        ):
            failures.append("news_ablation_delta_below_gate")
        if failures:
            raise ValueError("Model failed promotion gates: " + ", ".join(failures))

        gates: dict[str, object] = {
            "minimum_folds": minimum_folds,
            "minimum_sharpe": minimum_sharpe,
            "maximum_drawdown": maximum_drawdown,
            "minimum_observations": minimum_observations,
            "minimum_excess_return": minimum_excess_return,
            "minimum_news_sharpe_delta": minimum_news_sharpe_delta,
        }
        return self.set_alias(
            "champion",
            version,
            reason=reason,
            action="promote",
            details={
                "promotion_gates": gates,
                "registered_metrics": metrics,
            },
        )

    def rollback(
        self,
        version: str | None = None,
        *,
        reason: str = "operator_rollback",
    ) -> ModelRecord:
        index = self._read()
        aliases = dict(index["aliases"])
        current = aliases.get("champion")
        if current is None:
            raise ValueError("Cannot roll back because no champion alias is set")

        target = version
        if target is None:
            for event in reversed(list(index["alias_history"])):
                if not isinstance(event, dict):
                    continue
                if event.get("alias") != "champion":
                    continue
                if event.get("version") != current:
                    continue
                previous = event.get("previous_version")
                if previous:
                    target = str(previous)
                    break
        if target is None:
            raise ValueError("No previous champion is available for rollback")
        if target == current:
            raise ValueError("Rollback target is already the champion")
        record = self.get(target)
        self._change_alias(
            index,
            "champion",
            target,
            action="rollback",
            reason=reason,
            details={"rolled_back_from": current},
        )
        self._write(index)
        return record

    def champion(self) -> ModelRecord | None:
        return self.alias("champion")

    def challenger(self) -> ModelRecord | None:
        return self.alias("challenger")

    def load_champion(self) -> RidgeReturnModel | None:
        record = self.champion()
        return None if record is None else self.load(record.version)

    def load_challenger(self) -> RidgeReturnModel | None:
        record = self.challenger()
        return None if record is None else self.load(record.version)

    def history(
        self,
        *,
        alias: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        if limit is not None and limit < 1:
            raise ValueError("History limit must be positive")
        alias_name = self._validate_alias(alias) if alias is not None else None
        events = [
            dict(event)
            for event in self._read()["alias_history"]
            if isinstance(event, dict)
            and (alias_name is None or event.get("alias") == alias_name)
        ]
        if limit is not None:
            events = events[-limit:]
        return events

    def summary(self) -> dict[str, object]:
        return self._read()

    def _change_alias(
        self,
        index: dict[str, object],
        alias: str,
        version: str,
        *,
        action: str,
        reason: str,
        details: dict[str, object] | None,
    ) -> None:
        alias_name = self._validate_alias(alias)
        models = dict(index["models"])
        if version not in models:
            raise KeyError(f"Unknown model version: {version}")
        aliases = dict(index["aliases"])
        previous = aliases.get(alias_name)
        aliases[alias_name] = version
        index["aliases"] = aliases
        history = list(index["alias_history"])
        history.append(
            {
                "id": uuid4().hex,
                "changed_at": datetime.now(UTC).isoformat(),
                "alias": alias_name,
                "previous_version": previous,
                "version": version,
                "action": action,
                "reason": reason.strip() or "unspecified",
                "details": details or {},
            }
        )
        index["alias_history"] = history

    @staticmethod
    def _validate_alias(name: str | None) -> str:
        value = (name or "").strip().lower()
        if not _ALIAS_PATTERN.fullmatch(value):
            raise ValueError(
                "Alias must start with a lowercase letter and contain only "
                "lowercase letters, digits, underscores, or hyphens"
            )
        return value
