from __future__ import annotations

import hashlib
import importlib
import json
import math
import pickle
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .research import FeatureRow, RidgeReturnModel

ModelTask = Literal[
    "return_regression",
    "direction_classification",
    "quantile_forecasting",
    "volatility_forecasting",
    "portfolio_policy",
    "news_intelligence",
]


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    family: str
    implementation: str
    task: ModelTask = "return_regression"
    feature_names: tuple[str, ...]
    forecast_horizon: int = Field(default=5, gt=0)
    required_history: int = Field(default=1, gt=0)
    supports_uncertainty: bool = False
    live_compatible: bool = True
    deterministic: bool = True
    optional_dependency: str | None = None


class ModelProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str = "trained_locally"
    upstream_repository: str | None = None
    upstream_revision: str | None = None
    license: str | None = None
    weight_sha256: str | None = None
    limitations: tuple[str, ...] = ()


class ModelArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    capabilities: ModelCapabilities
    hyperparameters: dict[str, object]
    seed: int
    artifact_sha256: str
    python_version: str
    platform: str
    dependencies: dict[str, str]
    provenance: ModelProvenance


@runtime_checkable
class TradingModel(Protocol):
    capabilities: ModelCapabilities

    def fit(self, rows: list[FeatureRow]) -> None: ...

    def predict(self, features: tuple[float, ...]) -> float: ...

    def save(self, path: str | Path) -> None: ...


@runtime_checkable
class ContextTradingModel(TradingModel, Protocol):
    def predict_context(self, context: tuple[tuple[float, ...], ...]) -> float: ...


@dataclass(frozen=True)
class ModelBuildSpec:
    family: str
    feature_names: tuple[str, ...]
    hyperparameters: dict[str, object]
    seed: int
    forecast_horizon: int = 5


class RidgeModelAdapter:
    def __init__(self, model: RidgeReturnModel, forecast_horizon: int = 5) -> None:
        self.model = model
        self.capabilities = ModelCapabilities(
            family="linear",
            implementation="ridge",
            feature_names=tuple(model.feature_names),
            forecast_horizon=forecast_horizon,
        )

    def fit(self, rows: list[FeatureRow]) -> None:
        self.model.fit(rows)

    def predict(self, features: tuple[float, ...]) -> float:
        return self.model.predict(features)

    def save(self, path: str | Path) -> None:
        payload = self.model.to_dict()
        payload["model_type"] = "ridge_return"
        payload["forecast_horizon"] = self.capabilities.forecast_horizon
        _write_json(Path(path), payload)

    @classmethod
    def load(cls, path: str | Path) -> RidgeModelAdapter:
        payload = _load_json(Path(path))
        return cls(
            RidgeReturnModel.from_dict(payload),
            int(payload.get("forecast_horizon", 5)),
        )


class ElasticNetReturnModel:
    """Deterministic dependency-free elastic-net return regressor.

    Coordinate descent is intentionally compact and CPU-friendly. It is a stronger
    sparse linear baseline, not a profitability claim.
    """

    def __init__(
        self,
        feature_names: tuple[str, ...],
        *,
        alpha: float = 1e-3,
        l1_ratio: float = 0.5,
        iterations: int = 500,
        tolerance: float = 1e-9,
        seed: int = 1729,
        forecast_horizon: int = 5,
    ) -> None:
        if not feature_names:
            raise ValueError("At least one feature is required")
        if alpha <= 0 or not 0 <= l1_ratio <= 1:
            raise ValueError("Invalid elastic-net regularization")
        self.capabilities = ModelCapabilities(
            family="linear",
            implementation="elastic_net",
            feature_names=feature_names,
            forecast_horizon=forecast_horizon,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.iterations = iterations
        self.tolerance = tolerance
        self.seed = seed
        self.means = [0.0] * len(feature_names)
        self.scales = [1.0] * len(feature_names)
        self.intercept = 0.0
        self.weights = [0.0] * len(feature_names)
        self.fitted = False

    def fit(self, rows: list[FeatureRow]) -> None:
        width = len(self.capabilities.feature_names)
        if len(rows) <= width + 1:
            raise ValueError("Not enough rows to fit elastic net")
        if any(len(row.features) != width for row in rows):
            raise ValueError("Feature width does not match model")
        columns = list(zip(*(row.features for row in rows), strict=True))
        self.means = [fmean(column) for column in columns]
        self.scales = [max(pstdev(column), 1e-12) for column in columns]
        x = [self._standardize(row.features) for row in rows]
        y = [row.target_return for row in rows]
        self.intercept = fmean(y)
        centered = [value - self.intercept for value in y]
        self.weights = [0.0] * width
        residual = centered[:]
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1.0 - self.l1_ratio)
        order = list(range(width))
        random.Random(self.seed).shuffle(order)
        n = float(len(rows))
        for _ in range(self.iterations):
            largest = 0.0
            for column_index in order:
                old = self.weights[column_index]
                column = [row[column_index] for row in x]
                for index, value in enumerate(column):
                    residual[index] += old * value
                rho = sum(value * error for value, error in zip(column, residual, strict=True)) / n
                denominator = sum(value * value for value in column) / n + l2
                updated = _soft_threshold(rho, l1) / max(denominator, 1e-12)
                self.weights[column_index] = updated
                for index, value in enumerate(column):
                    residual[index] -= updated * value
                largest = max(largest, abs(updated - old))
            if largest <= self.tolerance:
                break
        self.fitted = True

    def _standardize(self, features: tuple[float, ...]) -> list[float]:
        return [
            (value - mean) / scale
            for value, mean, scale in zip(features, self.means, self.scales, strict=True)
        ]

    def predict(self, features: tuple[float, ...]) -> float:
        if not self.fitted:
            raise RuntimeError("Model is not fitted")
        if len(features) != len(self.weights):
            raise ValueError("Feature width does not match model")
        return self.intercept + sum(
            weight * value
            for weight, value in zip(self.weights, self._standardize(features), strict=True)
        )

    def save(self, path: str | Path) -> None:
        _write_json(
            Path(path),
            {
                "model_type": "elastic_net_return",
                "capabilities": self.capabilities.model_dump(mode="json"),
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "iterations": self.iterations,
                "tolerance": self.tolerance,
                "seed": self.seed,
                "means": self.means,
                "scales": self.scales,
                "intercept": self.intercept,
                "weights": self.weights,
                "fitted": self.fitted,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> ElasticNetReturnModel:
        payload = _load_json(Path(path))
        capabilities = ModelCapabilities.model_validate(payload["capabilities"])
        model = cls(
            capabilities.feature_names,
            alpha=float(payload["alpha"]),
            l1_ratio=float(payload["l1_ratio"]),
            iterations=int(payload["iterations"]),
            tolerance=float(payload["tolerance"]),
            seed=int(payload["seed"]),
            forecast_horizon=capabilities.forecast_horizon,
        )
        model.means = [float(value) for value in payload["means"]]
        model.scales = [float(value) for value in payload["scales"]]
        model.intercept = float(payload["intercept"])
        model.weights = [float(value) for value in payload["weights"]]
        model.fitted = bool(payload["fitted"])
        return model


class OptionalTabularReturnModel:
    """Lazy adapter for LightGBM, XGBoost, CatBoost, and sklearn MLP.

    Pickle is accepted only after the registry verifies the immutable artifact hash.
    """

    IMPLEMENTATIONS: dict[str, tuple[str, str, str]] = {
        "lightgbm": ("lightgbm", "LGBMRegressor", "lightgbm"),
        "xgboost": ("xgboost", "XGBRegressor", "xgboost"),
        "catboost": ("catboost", "CatBoostRegressor", "catboost"),
        "mlp": ("sklearn.neural_network", "MLPRegressor", "scikit-learn"),
    }

    def __init__(
        self,
        implementation: str,
        feature_names: tuple[str, ...],
        *,
        hyperparameters: dict[str, object] | None = None,
        seed: int = 1729,
        forecast_horizon: int = 5,
    ) -> None:
        if implementation not in self.IMPLEMENTATIONS:
            raise ValueError(f"Unsupported optional model: {implementation}")
        module, _, dependency = self.IMPLEMENTATIONS[implementation]
        self.capabilities = ModelCapabilities(
            family="tree" if implementation != "mlp" else "neural_tabular",
            implementation=implementation,
            feature_names=feature_names,
            forecast_horizon=forecast_horizon,
            deterministic=implementation != "mlp",
            optional_dependency=dependency,
        )
        self.module_name = module
        self.hyperparameters = hyperparameters or {}
        self.seed = seed
        self.estimator: Any = None

    def fit(self, rows: list[FeatureRow]) -> None:
        module_name, class_name, _ = self.IMPLEMENTATIONS[self.capabilities.implementation]
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise RuntimeError(
                f"Optional dependency for {self.capabilities.implementation} is not installed"
            ) from error
        estimator_class = getattr(module, class_name)
        parameters = dict(self.hyperparameters)
        if self.capabilities.implementation == "lightgbm":
            parameters.setdefault("random_state", self.seed)
            parameters.setdefault("n_jobs", 1)
            parameters.setdefault("verbosity", -1)
        elif self.capabilities.implementation == "xgboost":
            parameters.setdefault("random_state", self.seed)
            parameters.setdefault("n_jobs", 1)
            parameters.setdefault("tree_method", "hist")
        elif self.capabilities.implementation == "catboost":
            parameters.setdefault("random_seed", self.seed)
            parameters.setdefault("thread_count", 1)
            parameters.setdefault("verbose", False)
        else:
            parameters.setdefault("random_state", self.seed)
            parameters.setdefault("max_iter", 500)
        self.estimator = estimator_class(**parameters)
        self.estimator.fit([list(row.features) for row in rows], [row.target_return for row in rows])

    def predict(self, features: tuple[float, ...]) -> float:
        if self.estimator is None:
            raise RuntimeError("Model is not fitted")
        prediction = self.estimator.predict([list(features)])
        return float(prediction[0])

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        payload = {
            "model_type": "optional_tabular_return",
            "capabilities": self.capabilities.model_dump(mode="json"),
            "hyperparameters": self.hyperparameters,
            "seed": self.seed,
            "estimator": self.estimator,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(pickle.dumps(payload, protocol=5))

    @classmethod
    def load(cls, path: str | Path) -> OptionalTabularReturnModel:
        payload = pickle.loads(Path(path).read_bytes())  # noqa: S301 - registry hash verified first
        capabilities = ModelCapabilities.model_validate(payload["capabilities"])
        model = cls(
            capabilities.implementation,
            capabilities.feature_names,
            hyperparameters=dict(payload["hyperparameters"]),
            seed=int(payload["seed"]),
            forecast_horizon=capabilities.forecast_horizon,
        )
        model.estimator = payload["estimator"]
        return model


def build_model(spec: ModelBuildSpec) -> TradingModel:
    family = spec.family.lower()
    parameters = dict(spec.hyperparameters)
    if family == "ridge":
        return RidgeModelAdapter(
            RidgeReturnModel(spec.feature_names, ridge=float(parameters.get("ridge", 1e-3))),
            spec.forecast_horizon,
        )
    if family == "elastic_net":
        return ElasticNetReturnModel(
            spec.feature_names,
            alpha=float(parameters.get("alpha", 1e-3)),
            l1_ratio=float(parameters.get("l1_ratio", 0.5)),
            iterations=int(parameters.get("iterations", 500)),
            seed=spec.seed,
            forecast_horizon=spec.forecast_horizon,
        )
    if family in OptionalTabularReturnModel.IMPLEMENTATIONS:
        return OptionalTabularReturnModel(
            family,
            spec.feature_names,
            hyperparameters=parameters,
            seed=spec.seed,
            forecast_horizon=spec.forecast_horizon,
        )
    if family in {"lstm", "gru", "tcn", "patchtst", "itransformer"}:
        from .sequence_models import TorchSequenceReturnModel

        return TorchSequenceReturnModel(
            family,
            spec.feature_names,
            hyperparameters=parameters,
            seed=spec.seed,
            forecast_horizon=spec.forecast_horizon,
        )
    raise ValueError(f"Unsupported model family: {family}")


def load_model(path: str | Path) -> TradingModel:
    source = Path(path)
    if source.suffix == ".pkl":
        return OptionalTabularReturnModel.load(source)
    payload = _load_json(source)
    model_type = payload.get("model_type")
    if model_type == "ridge_return":
        return RidgeModelAdapter.load(source)
    if model_type == "elastic_net_return":
        return ElasticNetReturnModel.load(source)
    if model_type == "torch_sequence_return":
        from .sequence_models import TorchSequenceReturnModel

        return TorchSequenceReturnModel.load(source)
    if model_type == "ensemble_return":
        from .ensemble_models import EnsembleReturnModel

        return EnsembleReturnModel.load(source)
    raise ValueError(f"Unsupported model artifact type: {model_type!r}")


def artifact_extension(model: TradingModel) -> str:
    return ".pkl" if isinstance(model, OptionalTabularReturnModel) else ".json"


def artifact_manifest(
    model: TradingModel,
    artifact_path: str | Path,
    *,
    hyperparameters: dict[str, object],
    seed: int,
    provenance: ModelProvenance | None = None,
) -> ModelArtifactManifest:
    dependencies: dict[str, str] = {}
    optional = model.capabilities.optional_dependency
    if optional:
        try:
            from importlib.metadata import version

            dependencies[optional] = version(optional)
        except Exception:
            dependencies[optional] = "unknown"
    return ModelArtifactManifest(
        capabilities=model.capabilities,
        hyperparameters=hyperparameters,
        seed=seed,
        artifact_sha256=sha256_file(artifact_path),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        dependencies=dependencies,
        provenance=provenance or ModelProvenance(),
    )


def backend_availability() -> dict[str, dict[str, object]]:
    backends = {
        "ridge": None,
        "elastic_net": None,
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "mlp": "sklearn",
        "lstm": "torch",
        "gru": "torch",
        "tcn": "torch",
        "patchtst": "torch",
        "itransformer": "torch",
    }
    result: dict[str, dict[str, object]] = {}
    for name, module in backends.items():
        available = module is None or importlib.util.find_spec(module) is not None
        result[name] = {"available": available, "module": module}
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _soft_threshold(value: float, threshold: float) -> float:
    return math.copysign(max(abs(value) - threshold, 0.0), value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model artifact must be a JSON object")
    return payload
