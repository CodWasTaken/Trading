from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .research import BacktestMetrics, RidgeReturnModel, metrics_dict


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
            self._write({"models": {}, "aliases": {}})

    def _read(self) -> dict[str, object]:
        return json.loads(self.index_path.read_text())

    def _write(self, payload: dict[str, object]) -> None:
        temporary = self.index_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(self.index_path)

    def register(
        self,
        model: RidgeReturnModel,
        metrics: BacktestMetrics,
        metadata: dict[str, object] | None = None,
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
        models = dict(index.get("models", {}))
        models[version] = asdict(record)
        index["models"] = models
        self._write(index)
        return record

    def get(self, version: str) -> ModelRecord:
        index = self._read()
        payload = dict(index.get("models", {})).get(version)
        if payload is None:
            raise KeyError(f"Unknown model version: {version}")
        return ModelRecord(**payload)

    def promote(
        self,
        version: str,
        *,
        minimum_folds: int = 3,
        minimum_sharpe: float = 0.0,
        maximum_drawdown: float = 0.20,
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
        if failures:
            raise ValueError("Model failed promotion gates: " + ", ".join(failures))
        index = self._read()
        aliases = dict(index.get("aliases", {}))
        aliases["champion"] = version
        index["aliases"] = aliases
        self._write(index)
        return record

    def champion(self) -> ModelRecord | None:
        index = self._read()
        version = dict(index.get("aliases", {})).get("champion")
        return None if version is None else self.get(str(version))

    def load_champion(self) -> RidgeReturnModel | None:
        record = self.champion()
        if record is None:
            return None
        return RidgeReturnModel.load(self.root / record.model_path)

    def summary(self) -> dict[str, object]:
        return self._read()
