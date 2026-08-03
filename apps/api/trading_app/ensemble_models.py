from __future__ import annotations

import json
import math
from pathlib import Path

from .modeling import ModelCapabilities, TradingModel, load_model, sha256_file
from .research import FeatureRow


class EnsembleReturnModel:
    """Governed ensemble. Constituent artifacts are hash-bound and holdout-blind."""

    def __init__(
        self,
        constituents: list[tuple[str, str, float]],
        feature_names: tuple[str, ...],
        *,
        method: str = "weighted_average",
        disagreement_threshold: float | None = None,
        forecast_horizon: int = 5,
    ) -> None:
        if method not in {"weighted_average", "rank_average", "constrained_stacking"}:
            raise ValueError("Unsupported ensemble method")
        if not constituents:
            raise ValueError("Ensemble requires constituents")
        total = sum(max(0.0, weight) for _, _, weight in constituents)
        if total <= 0:
            raise ValueError("Ensemble weights must contain a positive value")
        self.constituents = [
            (path, digest, max(0.0, weight) / total) for path, digest, weight in constituents
        ]
        self.method = method
        self.disagreement_threshold = disagreement_threshold
        self.capabilities = ModelCapabilities(
            family="ensemble",
            implementation=method,
            feature_names=feature_names,
            forecast_horizon=forecast_horizon,
            supports_uncertainty=True,
        )
        self._models: list[tuple[TradingModel, float]] | None = None

    def _load(self) -> list[tuple[TradingModel, float]]:
        if self._models is None:
            loaded = []
            for path, declared_hash, weight in self.constituents:
                if sha256_file(path) != declared_hash:
                    raise ValueError(f"Ensemble constituent hash mismatch: {path}")
                model = load_model(path)
                if tuple(model.capabilities.feature_names) != self.capabilities.feature_names:
                    raise ValueError("Ensemble constituent feature schema mismatch")
                loaded.append((model, weight))
            self._models = loaded
        return self._models

    def fit(self, rows: list[FeatureRow]) -> None:
        raise RuntimeError("Ensemble constituents must be frozen before construction")

    def predict(self, features: tuple[float, ...]) -> float:
        predictions = [(model.predict(features), weight) for model, weight in self._load()]
        values = [value for value, _ in predictions]
        if self.disagreement_threshold is not None and len(values) > 1:
            mean = sum(values) / len(values)
            spread = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            if spread > self.disagreement_threshold:
                return 0.0
        if self.method == "rank_average":
            ordered = sorted(range(len(values)), key=lambda index: values[index])
            ranks = [0.0] * len(values)
            for rank, index in enumerate(ordered):
                ranks[index] = 0.0 if len(values) == 1 else rank / (len(values) - 1) - 0.5
            direction = 1.0 if sum(rank * weight for rank, (_, weight) in zip(ranks, predictions, strict=True)) > 0 else -1.0
            magnitude = sum(abs(value) * weight for value, weight in predictions)
            return direction * magnitude
        return sum(value * weight for value, weight in predictions)

    def save(self, path: str | Path) -> None:
        payload = {
            "model_type": "ensemble_return",
            "capabilities": self.capabilities.model_dump(mode="json"),
            "method": self.method,
            "disagreement_threshold": self.disagreement_threshold,
            "constituents": [
                {"path": item[0], "sha256": item[1], "weight": item[2]}
                for item in self.constituents
            ],
            "calibration_only_weight_fitting": True,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EnsembleReturnModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        capabilities = ModelCapabilities.model_validate(payload["capabilities"])
        return cls(
            [
                (str(item["path"]), str(item["sha256"]), float(item["weight"]))
                for item in payload["constituents"]
            ],
            capabilities.feature_names,
            method=str(payload["method"]),
            disagreement_threshold=(
                None
                if payload.get("disagreement_threshold") is None
                else float(payload["disagreement_threshold"])
            ),
            forecast_horizon=capabilities.forecast_horizon,
        )
