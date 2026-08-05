from __future__ import annotations

import importlib
from collections import defaultdict
from datetime import datetime
from typing import Any

from .cost_model import CostModelConfig
from .research import FeatureRow
from .rl_research import PortfolioResearchEnvironment, RLTrainingSpec, make_gym_environment


def train_sb3_model(
    rows: list[FeatureRow],
    cost_model: CostModelConfig,
    spec: RLTrainingSpec,
) -> Any:
    try:
        stable_baselines = importlib.import_module("stable_baselines3")
    except ImportError as error:
        raise RuntimeError("Stable-Baselines3 is not installed") from error
    environment = make_gym_environment(PortfolioResearchEnvironment(rows, cost_model, spec))
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
    return model


def evaluate_sb3_model(
    model: Any,
    rows: list[FeatureRow],
    cost_model: CostModelConfig,
    spec: RLTrainingSpec,
) -> tuple[list[float], dict[str, object]]:
    core = PortfolioResearchEnvironment(rows, cost_model, spec)
    environment = make_gym_environment(core)
    observation, _ = environment.reset(seed=spec.seed)
    returns: list[float] = []
    symbol_pnl: dict[str, float] = defaultdict(float)
    long_contribution = 0.0
    short_contribution = 0.0
    turnover = 0.0
    active_positions = 0
    short_active_positions = 0
    terminated = False
    while not terminated:
        action, _ = model.predict(observation, deterministic=True)
        action_values = [float(value) for value in action.tolist()]
        group = core._groups[core._index]  # type: ignore[attr-defined]
        previous = list(core._positions)  # type: ignore[attr-defined]
        raw = [max(-1.0, min(1.0, value)) for value in action_values]
        targets = [
            value * spec.max_long_position
            if value >= 0
            else value * spec.max_short_position
            for value in raw
        ]
        targets, _ = core._scale_exposures(targets)  # type: ignore[attr-defined]
        observation, _, terminated, truncated, info = environment.step(action)
        if truncated:
            raise RuntimeError("RL evaluation episode was unexpectedly truncated")
        returns.append(float(info["period_return"]))
        for index, row in enumerate(group):
            current = targets[index]
            trade_cost = cost_model.execution.trade_cost_fraction(
                previous[index],
                current,
            )
            holding_cost = cost_model.execution.holding_cost_fraction(
                current,
                spec.periods_per_year,
            )
            net = current * row.target_return - trade_cost - holding_cost
            symbol_pnl[row.symbol] += net
            if current > 0:
                long_contribution += net
            elif current < 0:
                short_contribution += net
                short_active_positions += 1
            active_positions += int(current != 0)
        turnover += float(info["turnover"])
    return returns, {
        "symbol_pnl": dict(symbol_pnl),
        "long_contribution": long_contribution,
        "short_contribution": short_contribution,
        "turnover": turnover,
        "active_positions": active_positions,
        "short_active_positions": short_active_positions,
    }


def equal_weight_benchmark_returns(
    rows: list[FeatureRow],
    cost_model: CostModelConfig,
    periods_per_year: float,
) -> list[float]:
    groups: dict[datetime, list[FeatureRow]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item.timestamp, item.symbol)):
        groups[row.timestamp].append(row)
    previous: dict[str, float] = defaultdict(float)
    returns: list[float] = []
    for timestamp in sorted(groups):
        group = groups[timestamp]
        allocation = 1.0 / len(group)
        period = 0.0
        for row in group:
            position = allocation
            trade_cost = cost_model.execution.trade_cost_fraction(
                previous[row.symbol],
                position,
            )
            holding_cost = cost_model.execution.holding_cost_fraction(
                position,
                periods_per_year,
            )
            period += position * row.target_return - trade_cost - holding_cost
            previous[row.symbol] = position
        returns.append(period)
    return returns
