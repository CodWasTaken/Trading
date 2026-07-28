from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from itertools import groupby
from pathlib import Path
from statistics import fmean, pstdev


@dataclass(frozen=True)
class FeatureRow:
    timestamp: datetime
    symbol: str
    features: tuple[float, ...]
    target_return: float
    label_end_time: datetime | None = None


@dataclass(frozen=True)
class BacktestMetrics:
    observations: int
    net_return: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    turnover: float
    average_trade_return: float
    folds: int = 1
    purged_rows: int = 0
    test_periods: int = 0


class RidgeReturnModel:
    """Small deterministic ridge model with no external numerical dependency."""

    def __init__(self, feature_names: tuple[str, ...], ridge: float = 1e-3) -> None:
        if not feature_names:
            raise ValueError("At least one feature is required")
        self.feature_names = feature_names
        self.ridge = ridge
        self.means = [0.0] * len(feature_names)
        self.scales = [1.0] * len(feature_names)
        self.weights = [0.0] * (len(feature_names) + 1)
        self.fitted = False

    def fit(self, rows: list[FeatureRow]) -> None:
        if len(rows) <= len(self.feature_names) + 1:
            raise ValueError("Not enough rows to fit the model")
        if any(len(row.features) != len(self.feature_names) for row in rows):
            raise ValueError("Feature width does not match feature names")

        columns = list(zip(*(row.features for row in rows), strict=True))
        self.means = [fmean(column) for column in columns]
        self.scales = [max(pstdev(column), 1e-12) for column in columns]
        design = [[1.0, *self._standardize(row.features)] for row in rows]
        targets = [row.target_return for row in rows]
        width = len(self.feature_names) + 1
        normal = [[0.0 for _ in range(width)] for _ in range(width)]
        rhs = [0.0 for _ in range(width)]
        for vector, target in zip(design, targets, strict=True):
            for i in range(width):
                rhs[i] += vector[i] * target
                for j in range(width):
                    normal[i][j] += vector[i] * vector[j]
        for index in range(1, width):
            normal[index][index] += self.ridge
        self.weights = _solve_linear_system(normal, rhs)
        self.fitted = True

    def _standardize(self, features: tuple[float, ...]) -> list[float]:
        return [
            (value - mean) / scale
            for value, mean, scale in zip(features, self.means, self.scales, strict=True)
        ]

    def predict(self, features: tuple[float, ...]) -> float:
        if not self.fitted:
            raise RuntimeError("Model is not fitted")
        if len(features) != len(self.feature_names):
            raise ValueError("Feature width does not match model")
        vector = [1.0, *self._standardize(features)]
        return sum(weight * value for weight, value in zip(self.weights, vector, strict=True))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "ridge_return",
            "feature_names": list(self.feature_names),
            "ridge": self.ridge,
            "means": self.means,
            "scales": self.scales,
            "weights": self.weights,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RidgeReturnModel:
        model = cls(
            tuple(str(value) for value in payload["feature_names"]),
            float(payload["ridge"]),
        )
        model.means = [float(value) for value in payload["means"]]
        model.scales = [float(value) for value in payload["scales"]]
        model.weights = [float(value) for value in payload["weights"]]
        model.fitted = bool(payload.get("fitted", True))
        return model

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> RidgeReturnModel:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    for pivot_index in range(size):
        best = max(
            range(pivot_index, size),
            key=lambda row: abs(augmented[row][pivot_index]),
        )
        if abs(augmented[best][pivot_index]) < 1e-14:
            raise ValueError("Singular training matrix")
        augmented[pivot_index], augmented[best] = augmented[best], augmented[pivot_index]
        pivot = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot for value in augmented[pivot_index]]
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            augmented[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row_index], augmented[pivot_index], strict=True
                )
            ]
    return [augmented[index][-1] for index in range(size)]


def backtest(
    model: RidgeReturnModel,
    rows: list[FeatureRow],
    threshold: float = 0.0005,
    transaction_cost_bps: float = 5.0,
    periods_per_year: int = 252,
) -> BacktestMetrics:
    if not rows:
        raise ValueError("Backtest requires rows")
    ordered = sorted(rows, key=lambda item: (item.timestamp, item.symbol))
    period_returns: list[float] = []
    equity_curve = [1.0]
    previous_positions: dict[str, float] = defaultdict(float)
    turnover = 0.0
    wins = 0
    active = 0
    active_net_return = 0.0

    for _, timestamp_rows_iter in groupby(ordered, key=lambda item: item.timestamp):
        timestamp_rows = list(timestamp_rows_iter)
        period_net = 0.0
        allocation = 1.0 / len(timestamp_rows)
        for row in timestamp_rows:
            prediction = model.predict(row.features)
            position = 1.0 if prediction > threshold else 0.0
            trade_turnover = abs(position - previous_positions[row.symbol])
            cost = trade_turnover * transaction_cost_bps / 10_000
            net = position * row.target_return - cost
            period_net += allocation * net
            turnover += trade_turnover
            if position:
                active += 1
                wins += int(net > 0)
                active_net_return += net
            previous_positions[row.symbol] = position
        period_returns.append(period_net)
        equity_curve.append(equity_curve[-1] * (1 + period_net))

    net_return = equity_curve[-1] - 1
    mean_return = fmean(period_returns)
    volatility = pstdev(period_returns) if len(period_returns) > 1 else 0.0
    sharpe = 0.0 if volatility == 0 else mean_return / volatility * math.sqrt(periods_per_year)
    annualized = (equity_curve[-1] ** (periods_per_year / len(period_returns))) - 1
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, (peak - value) / peak)
    return BacktestMetrics(
        observations=len(rows),
        net_return=net_return,
        annualized_return=annualized,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        hit_rate=0.0 if active == 0 else wins / active,
        turnover=turnover,
        average_trade_return=0.0 if active == 0 else active_net_return / active,
        test_periods=len(period_returns),
    )


def walk_forward(
    rows: list[FeatureRow],
    feature_names: tuple[str, ...],
    minimum_train_rows: int = 120,
    test_rows: int = 20,
    transaction_cost_bps: float = 5.0,
    periods_per_year: int = 252,
    threshold: float = 0.0005,
) -> BacktestMetrics:
    if minimum_train_rows <= len(feature_names) + 1:
        raise ValueError("minimum_train_rows is too small for the feature width")
    if test_rows < 1:
        raise ValueError("test_rows must be positive")

    ordered = sorted(rows, key=lambda item: (item.timestamp, item.symbol))
    fold_returns: list[float] = []
    fold_drawdowns: list[float] = []
    fold_sharpes: list[float] = []
    fold_hits: list[float] = []
    total_turnover = 0.0
    total_observations = 0
    total_test_periods = 0
    total_purged_rows = 0
    folds = 0
    cursor = minimum_train_rows

    while cursor + test_rows <= len(ordered):
        while cursor < len(ordered) and ordered[cursor - 1].timestamp == ordered[cursor].timestamp:
            cursor += 1
        if cursor + test_rows > len(ordered):
            break

        test_end = cursor + test_rows
        while test_end < len(ordered) and ordered[test_end - 1].timestamp == ordered[test_end].timestamp:
            test_end += 1
        test_block = ordered[cursor:test_end]
        test_start = test_block[0].timestamp
        training_candidates = ordered[:cursor]
        training_rows = [
            row
            for row in training_candidates
            if row.label_end_time is None or row.label_end_time < test_start
        ]
        total_purged_rows += len(training_candidates) - len(training_rows)
        if len(training_rows) <= len(feature_names) + 1:
            cursor = test_end
            continue

        model = RidgeReturnModel(feature_names)
        model.fit(training_rows)
        metrics = backtest(
            model,
            test_block,
            threshold=threshold,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=periods_per_year,
        )
        fold_returns.append(metrics.net_return)
        fold_drawdowns.append(metrics.max_drawdown)
        fold_sharpes.append(metrics.sharpe)
        fold_hits.append(metrics.hit_rate)
        total_turnover += metrics.turnover
        total_observations += metrics.observations
        total_test_periods += metrics.test_periods
        folds += 1
        cursor = test_end

    if not folds:
        raise ValueError("Not enough rows for a walk-forward evaluation")
    compounded = math.prod(1 + value for value in fold_returns) - 1
    annualized = (
        (1 + compounded) ** (periods_per_year / total_test_periods) - 1
        if total_test_periods > 0 and compounded > -1
        else -1.0
    )
    return BacktestMetrics(
        observations=total_observations,
        net_return=compounded,
        annualized_return=annualized,
        sharpe=fmean(fold_sharpes),
        max_drawdown=max(fold_drawdowns),
        hit_rate=fmean(fold_hits),
        turnover=total_turnover,
        average_trade_return=compounded / max(total_turnover, 1.0),
        folds=folds,
        purged_rows=total_purged_rows,
        test_periods=total_test_periods,
    )


def synthetic_feature_rows(count: int = 600, seed: int = 17) -> list[FeatureRow]:
    """Deterministic research fixture; never treat this as market evidence."""
    if count < 10:
        raise ValueError("Synthetic dataset must have at least ten rows")
    randomizer = random.Random(seed)
    timestamp = datetime(2020, 1, 1, tzinfo=UTC)
    rows: list[FeatureRow] = []
    momentum = 0.0
    for index in range(count):
        news_score = randomizer.uniform(-1, 1) if index % 11 == 0 else 0.0
        volatility = abs(randomizer.gauss(0.012, 0.004))
        spread_bps = max(1.0, randomizer.gauss(8.0, 2.0))
        momentum = 0.65 * momentum + randomizer.gauss(0, 0.012)
        target = (
            0.22 * momentum
            + 0.006 * news_score
            - 0.08 * volatility
            - 0.00001 * spread_bps
            + randomizer.gauss(0, 0.002)
        )
        rows.append(
            FeatureRow(
                timestamp=timestamp + timedelta(days=index),
                symbol="SYNTH",
                features=(momentum, news_score, volatility, spread_bps),
                target_return=target,
            )
        )
    return rows


def metrics_dict(metrics: BacktestMetrics) -> dict[str, object]:
    return asdict(metrics)
