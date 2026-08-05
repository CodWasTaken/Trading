from __future__ import annotations

import hashlib
import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cost_model import CostModelConfig, load_cost_model
from .dataset import dataset_metadata_path, load_feature_dataset
from .modeling import sha256_file
from .research import FeatureRow

RLAlgorithm = Literal["ppo", "a2c", "sac", "td3"]


class RLRewardConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    drawdown_penalty: float = Field(default=1.0, ge=0)
    turnover_penalty: float = Field(default=0.01, ge=0)
    concentration_penalty: float = Field(default=0.01, ge=0)
    gross_exposure_penalty: float = Field(default=0.05, ge=0)
    borrow_dependency_penalty: float = Field(default=0.01, ge=0)
    constraint_violation_penalty: float = Field(default=0.10, ge=0)


class RLTrainingSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    algorithm: RLAlgorithm
    seed: int = Field(default=1729, ge=0)
    total_timesteps: int = Field(default=25_000, ge=100)
    periods_per_year: float = Field(default=327.6, gt=0)
    max_long_position: float = Field(default=0.05, gt=0, le=1)
    max_short_position: float = Field(default=0.03, gt=0, le=0.30)
    max_gross_long: float = Field(default=0.60, gt=0, le=1)
    max_gross_short: float = Field(default=0.30, gt=0, le=1)
    reward: RLRewardConfig = Field(default_factory=RLRewardConfig)
    hyperparameters: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_schema(self) -> RLTrainingSpec:
        if self.schema_version != 1:
            raise ValueError("Unsupported RL training spec schema_version")
        if self.max_long_position > self.max_gross_long:
            raise ValueError("max_long_position cannot exceed max_gross_long")
        if self.max_short_position > self.max_gross_short:
            raise ValueError("max_short_position cannot exceed max_gross_short")
        return self

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


class PortfolioResearchEnvironment:
    """Dependency-free portfolio environment used by optional Gymnasium/SB3 adapters.

    Observations contain only current point-in-time features and previous positions.
    The realized target return is consumed only after an action to calculate reward.
    """

    def __init__(
        self,
        rows: list[FeatureRow],
        cost_model: CostModelConfig,
        spec: RLTrainingSpec,
    ) -> None:
        if not rows:
            raise ValueError("RL environment requires rows")
        grouped: dict[object, list[FeatureRow]] = defaultdict(list)
        for row in sorted(rows, key=lambda item: (item.timestamp, item.symbol)):
            grouped[row.timestamp].append(row)
        self._groups = [
            sorted(values, key=lambda item: item.symbol)
            for _, values in sorted(grouped.items())
        ]
        self.symbols = tuple(row.symbol for row in self._groups[0])
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError("RL environment group contains duplicate symbols")
        if any(tuple(row.symbol for row in group) != self.symbols for group in self._groups):
            raise ValueError("Every RL timestamp must contain the same frozen symbols")
        width = len(self._groups[0][0].features)
        if width < 1 or any(len(row.features) != width for group in self._groups for row in group):
            raise ValueError("RL environment feature schemas are inconsistent")
        self.feature_width = width
        self.cost_model = cost_model
        self.spec = spec
        self._index = 0
        self._positions = [0.0] * len(self.symbols)
        self._equity = 1.0
        self._peak = 1.0

    @property
    def observation_size(self) -> int:
        return len(self.symbols) * (self.feature_width + 1)

    @property
    def action_size(self) -> int:
        return len(self.symbols)

    def reset(self) -> tuple[float, ...]:
        self._index = 0
        self._positions = [0.0] * len(self.symbols)
        self._equity = 1.0
        self._peak = 1.0
        return self._observation()

    def step(
        self,
        actions: tuple[float, ...] | list[float],
    ) -> tuple[tuple[float, ...], float, bool, dict[str, object]]:
        if len(actions) != len(self.symbols):
            raise ValueError("RL action width does not match frozen universe")
        if self._index >= len(self._groups):
            raise RuntimeError("RL episode has already terminated")
        group = self._groups[self._index]
        raw = [max(-1.0, min(1.0, float(value))) for value in actions]
        targets = [
            value * self.spec.max_long_position
            if value >= 0
            else value * self.spec.max_short_position
            for value in raw
        ]
        violations = sum(
            float(value) != raw_value
            for value, raw_value in zip(actions, raw, strict=True)
        )
        targets, scaled = self._scale_exposures(targets)
        violations += int(scaled)
        period_return = 0.0
        turnover = 0.0
        financing = 0.0
        execution = 0.0
        short_gross = 0.0
        for index, row in enumerate(group):
            previous = self._positions[index]
            current = targets[index]
            trade_cost = self.cost_model.execution.trade_cost_fraction(previous, current)
            holding_cost = self.cost_model.execution.holding_cost_fraction(
                current,
                self.spec.periods_per_year,
            )
            period_return += current * row.target_return - trade_cost - holding_cost
            turnover += abs(current - previous)
            execution += trade_cost
            financing += holding_cost
            short_gross += max(0.0, -current)
        gross = sum(abs(value) for value in targets)
        concentration = 0.0 if gross == 0 else max(abs(value) for value in targets) / gross
        previous_drawdown = 0.0 if self._peak <= 0 else (self._peak - self._equity) / self._peak
        self._equity *= max(1e-12, 1 + period_return)
        self._peak = max(self._peak, self._equity)
        drawdown = 0.0 if self._peak <= 0 else (self._peak - self._equity) / self._peak
        drawdown_change = max(0.0, drawdown - previous_drawdown)
        excess_gross = max(0.0, gross - (self.spec.max_gross_long + self.spec.max_gross_short))
        reward = period_return - (
            self.spec.reward.drawdown_penalty * drawdown_change
            + self.spec.reward.turnover_penalty * turnover
            + self.spec.reward.concentration_penalty * concentration
            + self.spec.reward.gross_exposure_penalty * excess_gross
            + self.spec.reward.borrow_dependency_penalty * short_gross
            + self.spec.reward.constraint_violation_penalty * violations
        )
        self._positions = targets
        self._index += 1
        terminated = self._index >= len(self._groups)
        observation = (
            tuple(0.0 for _ in range(self.observation_size))
            if terminated
            else self._observation()
        )
        return observation, reward, terminated, {
            "period_return": period_return,
            "equity": self._equity,
            "drawdown": drawdown,
            "turnover": turnover,
            "execution_cost": execution,
            "financing_cost": financing,
            "gross_exposure": gross,
            "gross_short_exposure": short_gross,
            "concentration": concentration,
            "constraint_violations": violations,
        }

    def _observation(self) -> tuple[float, ...]:
        group = self._groups[self._index]
        flattened: list[float] = []
        for row, position in zip(group, self._positions, strict=True):
            flattened.extend(float(value) for value in row.features)
            flattened.append(position)
        return tuple(flattened)

    def _scale_exposures(self, positions: list[float]) -> tuple[list[float], bool]:
        long_gross = sum(max(0.0, value) for value in positions)
        short_gross = sum(max(0.0, -value) for value in positions)
        long_scale = (
            1.0
            if long_gross <= self.spec.max_gross_long
            else self.spec.max_gross_long / long_gross
        )
        short_scale = (
            1.0
            if short_gross <= self.spec.max_gross_short
            else self.spec.max_gross_short / short_gross
        )
        scaled = [
            value * long_scale if value >= 0 else value * short_scale
            for value in positions
        ]
        return scaled, long_scale < 1.0 or short_scale < 1.0


def make_gym_environment(core: PortfolioResearchEnvironment) -> Any:
    try:
        gymnasium = importlib.import_module("gymnasium")
        numpy = importlib.import_module("numpy")
    except ImportError as error:
        raise RuntimeError("Gymnasium and NumPy are required for RL training") from error

    class GymPortfolioEnvironment(gymnasium.Env):  # type: ignore[misc,valid-type]
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.action_space = gymnasium.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(core.action_size,),
                dtype=numpy.float32,
            )
            self.observation_space = gymnasium.spaces.Box(
                low=-numpy.inf,
                high=numpy.inf,
                shape=(core.observation_size,),
                dtype=numpy.float32,
            )

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, object] | None = None,
        ) -> tuple[Any, dict[str, object]]:
            del options
            super().reset(seed=seed)
            return numpy.asarray(core.reset(), dtype=numpy.float32), {}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, object]]:
            observation, reward, terminated, info = core.step(
                [float(value) for value in action.tolist()]
            )
            return (
                numpy.asarray(observation, dtype=numpy.float32),
                float(reward),
                terminated,
                False,
                info,
            )

    return GymPortfolioEnvironment()


def train_rl_policy(
    dataset_path: str | Path,
    output_path: str | Path,
    spec_path: str | Path,
    cost_config_path: str | Path,
) -> dict[str, object]:
    source = Path(dataset_path)
    destination = Path(output_path)
    if destination.exists() or destination.with_suffix(".zip").exists():
        raise ValueError(f"RL policy output already exists: {destination}")
    rows, feature_names, metadata = load_feature_dataset(source)
    if metadata.get("dataset_role") != "calibration" or metadata.get("sealed") is True:
        raise ValueError("RL policies may train on calibration datasets only")
    spec = RLTrainingSpec.model_validate_json(Path(spec_path).read_text(encoding="utf-8"))
    cost_model = load_cost_model(cost_config_path)
    core = PortfolioResearchEnvironment(rows, cost_model, spec)
    environment = make_gym_environment(core)
    try:
        stable_baselines = importlib.import_module("stable_baselines3")
        checker = importlib.import_module("stable_baselines3.common.env_checker")
    except ImportError as error:
        raise RuntimeError("Stable-Baselines3 is not installed") from error
    checker.check_env(environment, warn=True, skip_render_check=True)
    algorithm_class = {
        "ppo": stable_baselines.PPO,
        "a2c": stable_baselines.A2C,
        "sac": stable_baselines.SAC,
        "td3": stable_baselines.TD3,
    }[spec.algorithm]
    parameters = dict(spec.hyperparameters)
    parameters.setdefault("seed", spec.seed)
    parameters.setdefault("verbose", 0)
    model = algorithm_class("MlpPolicy", environment, **parameters)
    model.learn(total_timesteps=spec.total_timesteps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))
    artifact = destination if destination.is_file() else destination.with_suffix(".zip")
    if not artifact.is_file():
        raise RuntimeError("Stable-Baselines3 did not create the expected policy artifact")
    metadata_path = dataset_metadata_path(source)
    report = {
        "schema_version": 1,
        "report_kind": "calibration_only_rl_policy",
        "algorithm": spec.algorithm,
        "task": "portfolio_policy",
        "live_compatible": False,
        "paper_only": True,
        "dataset": {
            "path": str(source),
            "sha256": sha256_file(source),
            "metadata_sha256": sha256_file(metadata_path),
            "role": "calibration",
        },
        "feature_names": list(feature_names),
        "symbols": list(core.symbols),
        "spec": spec.model_dump(mode="json"),
        "spec_sha256": spec.manifest_sha256,
        "cost_model": cost_model.model_dump(mode="json"),
        "cost_model_sha256": cost_model.manifest_sha256,
        "artifact": {"path": str(artifact), "sha256": sha256_file(artifact)},
        "controls": {
            "holdout_access": "forbidden",
            "target_return_only_used_after_action": True,
            "normalized_action_space": [-1, 1],
            "risk_and_cost_penalties_enabled": True,
            "direct_live_selection_allowed": False,
        },
    }
    report_path = destination.with_suffix(".manifest.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "manifest": str(report_path)}


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
