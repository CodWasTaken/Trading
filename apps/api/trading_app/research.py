from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import groupby
from pathlib import Path
from statistics import fmean, pstdev

from .cost_model import CostModelConfig, legacy_cost_model


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
    scored_observations: int = 0
    benchmark_net_return: float = 0.0
    excess_return_vs_benchmark: float = 0.0
    news_ablation_sharpe: float = 0.0
    news_sharpe_delta: float = 0.0
    execution_cost_return: float = 0.0
    financing_cost_return: float = 0.0


@dataclass(frozen=True)
class WalkForwardReport:
    metrics: BacktestMetrics
    equal_weight_long: BacktestMetrics
    no_news_ablation: BacktestMetrics | None
    symbols: dict[str, dict[str, float | int]]
    sectors: dict[str, dict[str, float | int]]
    short_safety: dict[str, float | int | bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "evaluation_method": "stitched_non_overlapping_out_of_sample",
            "cash": {
                "net_return": 0.0,
                "annualized_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            },
            "equal_weight_long": metrics_dict(self.equal_weight_long),
            "no_news_ablation": (
                None if self.no_news_ablation is None else metrics_dict(self.no_news_ablation)
            ),
            "symbols": self.symbols,
            "sectors": self.sectors,
            "short_safety": self.short_safety,
        }


@dataclass(frozen=True)
class _ScoredRow:
    row: FeatureRow
    prediction: float


@dataclass(frozen=True)
class _ScoredWalkForward:
    rows: list[_ScoredRow]
    folds: int
    purged_rows: int


@dataclass(frozen=True)
class _Simulation:
    metrics: BacktestMetrics
    symbols: dict[str, dict[str, float | int]]


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


def _selected_groups(scored_rows: list[_ScoredRow]) -> list[list[_ScoredRow]]:
    ordered = sorted(scored_rows, key=lambda item: (item.row.timestamp, item.row.symbol))
    selected: list[list[_ScoredRow]] = []
    next_available: datetime | None = None
    for timestamp, timestamp_rows_iter in groupby(ordered, key=lambda item: item.row.timestamp):
        timestamp_rows = list(timestamp_rows_iter)
        if next_available is not None and timestamp < next_available:
            continue
        selected.append(timestamp_rows)
        label_ends = [item.row.label_end_time for item in timestamp_rows]
        known_ends = [value for value in label_ends if value is not None]
        next_available = max(known_ends) if known_ends else timestamp
    return selected


def _metrics_from_returns(
    period_returns: list[float],
    *,
    observations: int,
    scored_observations: int,
    active: int,
    wins: int,
    active_net_return: float,
    turnover: float,
    folds: int,
    purged_rows: int,
    periods_per_year: float,
) -> BacktestMetrics:
    if not period_returns:
        raise ValueError("Backtest produced no non-overlapping periods")
    equity_curve = [1.0]
    for value in period_returns:
        equity_curve.append(equity_curve[-1] * (1 + value))
    net_return = equity_curve[-1] - 1
    mean_return = fmean(period_returns)
    volatility = pstdev(period_returns) if len(period_returns) > 1 else 0.0
    sharpe = 0.0 if volatility == 0 else mean_return / volatility * math.sqrt(periods_per_year)
    annualized = (
        (1 + net_return) ** (periods_per_year / len(period_returns)) - 1
        if net_return > -1
        else -1.0
    )
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, (peak - value) / peak)
    return BacktestMetrics(
        observations=observations,
        net_return=net_return,
        annualized_return=annualized,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        hit_rate=0.0 if active == 0 else wins / active,
        turnover=turnover,
        average_trade_return=0.0 if active == 0 else active_net_return / active,
        folds=folds,
        purged_rows=purged_rows,
        test_periods=len(period_returns),
        scored_observations=scored_observations,
    )


def _simulate(
    scored_rows: list[_ScoredRow],
    *,
    threshold: float,
    transaction_cost_bps: float,
    periods_per_year: float,
    folds: int,
    purged_rows: int,
    always_long: bool = False,
    cost_model: CostModelConfig | None = None,
) -> _Simulation:
    selected_groups = _selected_groups(scored_rows)
    previous_positions: dict[str, float] = defaultdict(float)
    period_returns: list[float] = []
    turnover = 0.0
    wins = 0
    active = 0
    active_net_return = 0.0
    observations = 0
    symbol_returns: dict[str, list[float]] = defaultdict(list)
    symbol_active: dict[str, int] = defaultdict(int)
    symbol_wins: dict[str, int] = defaultdict(int)
    symbol_turnover: dict[str, float] = defaultdict(float)
    symbol_total_net: dict[str, float] = defaultdict(float)
    symbol_pnl_contribution: dict[str, float] = defaultdict(float)
    total_execution_cost = 0.0
    total_financing_cost = 0.0
    resolved_costs = cost_model or legacy_cost_model(transaction_cost_bps)

    for timestamp_rows in selected_groups:
        allocation = 1.0 / len(timestamp_rows)
        period_net = 0.0
        for item in timestamp_rows:
            row = item.row
            position = 1.0 if always_long or item.prediction > threshold else 0.0
            previous = previous_positions[row.symbol]
            trade_turnover = abs(position - previous)
            execution_cost = resolved_costs.execution.trade_cost_fraction(
                previous,
                position,
            )
            financing_cost = resolved_costs.execution.holding_cost_fraction(
                position,
                periods_per_year,
            )
            cost = execution_cost + financing_cost
            net = position * row.target_return - cost
            period_net += allocation * net
            total_execution_cost += allocation * execution_cost
            total_financing_cost += allocation * financing_cost
            turnover += trade_turnover
            symbol_turnover[row.symbol] += trade_turnover
            symbol_returns[row.symbol].append(net)
            symbol_total_net[row.symbol] += net
            symbol_pnl_contribution[row.symbol] += allocation * net
            observations += 1
            active_net_return += net
            if position:
                active += 1
                wins += int(net > 0)
                symbol_active[row.symbol] += 1
                symbol_wins[row.symbol] += int(net > 0)
            previous_positions[row.symbol] = position
        period_returns.append(period_net)

    metrics = replace(
        _metrics_from_returns(
            period_returns,
            observations=observations,
            scored_observations=len(scored_rows),
            active=active,
            wins=wins,
            active_net_return=active_net_return,
            turnover=turnover,
            folds=folds,
            purged_rows=purged_rows,
            periods_per_year=periods_per_year,
        ),
        execution_cost_return=total_execution_cost,
        financing_cost_return=total_financing_cost,
    )
    symbols: dict[str, dict[str, float | int]] = {}
    for symbol, returns in sorted(symbol_returns.items()):
        equity = math.prod(1 + value for value in returns)
        active_count = symbol_active[symbol]
        symbols[symbol] = {
            "observations": len(returns),
            "active_signals": active_count,
            "hit_rate": 0.0 if active_count == 0 else symbol_wins[symbol] / active_count,
            "net_return": equity - 1,
            "pnl_contribution": symbol_pnl_contribution[symbol],
            "average_active_return": (
                0.0 if active_count == 0 else symbol_total_net[symbol] / active_count
            ),
            "turnover": symbol_turnover[symbol],
        }
    return _Simulation(metrics=metrics, symbols=symbols)


def backtest(
    model: RidgeReturnModel,
    rows: list[FeatureRow],
    threshold: float = 0.0005,
    transaction_cost_bps: float = 5.0,
    periods_per_year: float = 252,
    cost_model: CostModelConfig | None = None,
) -> BacktestMetrics:
    if not rows:
        raise ValueError("Backtest requires rows")
    scored = [_ScoredRow(row=row, prediction=model.predict(row.features)) for row in rows]
    return _simulate(
        scored,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        folds=1,
        purged_rows=0,
        cost_model=cost_model,
    ).metrics


def _score_walk_forward(
    rows: list[FeatureRow],
    feature_names: tuple[str, ...],
    *,
    minimum_train_rows: int,
    test_rows: int,
    ridge: float,
) -> _ScoredWalkForward:
    if minimum_train_rows <= len(feature_names) + 1:
        raise ValueError("minimum_train_rows is too small for the feature width")
    if test_rows < 1:
        raise ValueError("test_rows must be positive")

    ordered = sorted(rows, key=lambda item: (item.timestamp, item.symbol))
    scored_rows: list[_ScoredRow] = []
    total_purged_rows = 0
    folds = 0
    cursor = minimum_train_rows

    while cursor + test_rows <= len(ordered):
        while cursor < len(ordered) and ordered[cursor - 1].timestamp == ordered[cursor].timestamp:
            cursor += 1
        if cursor + test_rows > len(ordered):
            break
        test_end = cursor + test_rows
        while (
            test_end < len(ordered)
            and ordered[test_end - 1].timestamp == ordered[test_end].timestamp
        ):
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

        model = RidgeReturnModel(feature_names, ridge=ridge)
        model.fit(training_rows)
        scored_rows.extend(
            _ScoredRow(row=row, prediction=model.predict(row.features)) for row in test_block
        )
        folds += 1
        cursor = test_end

    if not folds:
        raise ValueError("Not enough rows for a walk-forward evaluation")
    return _ScoredWalkForward(rows=scored_rows, folds=folds, purged_rows=total_purged_rows)


def walk_forward(
    rows: list[FeatureRow],
    feature_names: tuple[str, ...],
    minimum_train_rows: int = 120,
    test_rows: int = 20,
    transaction_cost_bps: float = 5.0,
    periods_per_year: float = 252,
    threshold: float = 0.0005,
    ridge: float = 1e-3,
    cost_model: CostModelConfig | None = None,
) -> BacktestMetrics:
    scored = _score_walk_forward(
        rows,
        feature_names,
        minimum_train_rows=minimum_train_rows,
        test_rows=test_rows,
        ridge=ridge,
    )
    return _simulate(
        scored.rows,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        folds=scored.folds,
        purged_rows=scored.purged_rows,
        cost_model=cost_model,
    ).metrics


def walk_forward_report(
    rows: list[FeatureRow],
    feature_names: tuple[str, ...],
    *,
    minimum_train_rows: int = 120,
    test_rows: int = 20,
    transaction_cost_bps: float = 5.0,
    periods_per_year: float = 252,
    threshold: float = 0.0005,
    ridge: float = 1e-3,
    cost_model: CostModelConfig | None = None,
    sector_by_symbol: dict[str, str] | None = None,
) -> WalkForwardReport:
    scored = _score_walk_forward(
        rows,
        feature_names,
        minimum_train_rows=minimum_train_rows,
        test_rows=test_rows,
        ridge=ridge,
    )
    candidate = _simulate(
        scored.rows,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        folds=scored.folds,
        purged_rows=scored.purged_rows,
        cost_model=cost_model,
    )
    benchmark = _simulate(
        scored.rows,
        threshold=threshold,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
        folds=scored.folds,
        purged_rows=scored.purged_rows,
        always_long=True,
        cost_model=cost_model,
    )

    no_news_metrics: BacktestMetrics | None = None
    if "news_score" in feature_names and len(feature_names) > 1:
        news_index = feature_names.index("news_score")
        ablation_names = tuple(
            name for index, name in enumerate(feature_names) if index != news_index
        )
        ablation_rows = [
            FeatureRow(
                timestamp=row.timestamp,
                symbol=row.symbol,
                features=tuple(
                    value for index, value in enumerate(row.features) if index != news_index
                ),
                target_return=row.target_return,
                label_end_time=row.label_end_time,
            )
            for row in rows
        ]
        ablation_scored = _score_walk_forward(
            ablation_rows,
            ablation_names,
            minimum_train_rows=minimum_train_rows,
            test_rows=test_rows,
            ridge=ridge,
        )
        no_news_metrics = _simulate(
            ablation_scored.rows,
            threshold=threshold,
            transaction_cost_bps=transaction_cost_bps,
            periods_per_year=periods_per_year,
            folds=ablation_scored.folds,
            purged_rows=ablation_scored.purged_rows,
            cost_model=cost_model,
        ).metrics

    metrics = replace(
        candidate.metrics,
        benchmark_net_return=benchmark.metrics.net_return,
        excess_return_vs_benchmark=(
            candidate.metrics.net_return - benchmark.metrics.net_return
        ),
        news_ablation_sharpe=(
            0.0 if no_news_metrics is None else no_news_metrics.sharpe
        ),
        news_sharpe_delta=(
            0.0 if no_news_metrics is None else candidate.metrics.sharpe - no_news_metrics.sharpe
        ),
    )
    sectors: dict[str, dict[str, float | int]] = {}
    sector_contributions: dict[str, float] = defaultdict(float)
    for symbol, symbol_metrics in candidate.symbols.items():
        sector = (
            "Unmapped"
            if sector_by_symbol is None
            else sector_by_symbol.get(symbol, "Unmapped")
        )
        sector_contributions[sector] += float(symbol_metrics["pnl_contribution"])
    for sector, contribution in sorted(sector_contributions.items()):
        sectors[sector] = {"pnl_contribution": contribution}
    return WalkForwardReport(
        metrics=metrics,
        equal_weight_long=benchmark.metrics,
        no_news_ablation=no_news_metrics,
        symbols=candidate.symbols,
        sectors=sectors,
        short_safety={
            "unborrowable_short_orders": 0,
            "maximum_gross_short_exposure": 0.0,
            "maximum_single_short_position": 0.0,
            "borrow_status_validated": True,
        },
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
