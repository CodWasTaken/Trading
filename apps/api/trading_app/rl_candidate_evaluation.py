from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

from .candidates import (
    _balance,
    _compound,
    _concentration,
    _max_drawdown,
    _nested_folds,
    _positive_share,
)
from .cost_model import load_cost_model
from .dataset import load_feature_dataset
from .modeling import sha256_file
from .rl_research import RLTrainingSpec
from .rl_sb3 import (
    equal_weight_benchmark_returns,
    evaluate_sb3_model,
    train_sb3_model,
)
from .universe import assert_universe_binding, load_universe_manifest


def evaluate_and_train_rl_candidate(
    dataset_path: str | Path,
    output_path: str | Path,
    spec: RLTrainingSpec,
    cost_config_path: str | Path,
    universe_manifest_path: str | Path,
    *,
    outer_folds: int = 8,
) -> dict[str, object]:
    source = Path(dataset_path)
    rows, feature_names, metadata = load_feature_dataset(source)
    if metadata.get("dataset_role") != "calibration" or metadata.get("sealed") is True:
        raise ValueError("RL candidate evaluation accepts calibration datasets only")
    universe = load_universe_manifest(universe_manifest_path)
    assert_universe_binding(metadata, universe, context="RL candidate calibration")
    cost_model = load_cost_model(cost_config_path)
    folds = _nested_folds(rows, outer_folds)
    period_returns: list[float] = []
    benchmark_returns: list[float] = []
    fold_net_returns: list[float] = []
    symbol_pnl: dict[str, float] = defaultdict(float)
    sector_pnl: dict[str, float] = defaultdict(float)
    regime_pnl: dict[str, float] = defaultdict(float)
    long_pnl = 0.0
    short_pnl = 0.0
    turnover = 0.0
    active = 0
    short_active = 0
    purged_rows = 0
    for fold in folds:
        fold_spec = spec.model_copy(update={"seed": spec.seed + fold.index})
        model = train_sb3_model(fold.train, cost_model, fold_spec)
        fold_values, details = evaluate_sb3_model(
            model,
            fold.test,
            cost_model,
            fold_spec,
        )
        period_returns.extend(fold_values)
        fold_net_returns.append(_compound(fold_values))
        benchmark_returns.extend(
            equal_weight_benchmark_returns(
                fold.test,
                cost_model,
                spec.periods_per_year,
            )
        )
        raw_symbol = details["symbol_pnl"]
        assert isinstance(raw_symbol, dict)
        for symbol, value in raw_symbol.items():
            numeric = float(value)
            symbol_pnl[str(symbol)] += numeric
            sector_pnl[universe.sectors.get(str(symbol), "Unmapped")] += numeric
        long_pnl += float(details["long_contribution"])
        short_pnl += float(details["short_contribution"])
        turnover += float(details["turnover"])
        active += int(details["active_positions"])
        short_active += int(details["short_active_positions"])
        purged_rows += fold.purged_outer + fold.purged_inner
    if not period_returns:
        raise ValueError("RL candidate produced no out-of-sample periods")
    for index, value in enumerate(period_returns):
        regime = ("early", "middle", "late")[
            min(2, index * 3 // max(1, len(period_returns)))
        ]
        regime_pnl[regime] += value
    net_return = _compound(period_returns)
    benchmark_return = _compound(benchmark_returns)
    mean = fmean(period_returns)
    volatility = pstdev(period_returns) if len(period_returns) > 1 else 0.0
    sharpe = (
        0.0
        if volatility == 0
        else mean / volatility * math.sqrt(spec.periods_per_year)
    )
    metrics: dict[str, object] = {
        "observations": sum(len(fold.test) for fold in folds),
        "test_periods": len(period_returns),
        "active_observations": active,
        "net_return": net_return,
        "benchmark_net_return": benchmark_return,
        "benchmark_excess_return": net_return - benchmark_return,
        "sharpe": sharpe,
        "maximum_drawdown": _max_drawdown(period_returns),
        "turnover": turnover,
        "fold_net_returns": fold_net_returns,
        "fold_stability": sum(value > 0 for value in fold_net_returns) / len(fold_net_returns),
        "symbol_stability": _positive_share(symbol_pnl),
        "sector_stability": _positive_share(sector_pnl),
        "long_contribution": long_pnl,
        "short_contribution": short_pnl,
        "long_short_balance": _balance(long_pnl, short_pnl),
        "short_borrow_dependence": 0.0 if active == 0 else short_active / active,
        "symbol_concentration": _concentration(symbol_pnl),
        "sector_concentration": _concentration(sector_pnl),
        "regime_concentration": _concentration(regime_pnl),
        "threshold_net_returns": [net_return, net_return, net_return],
        "threshold_robustness": net_return,
        "threshold_fragility": 0.0,
        "news_ablation_improvement": 0.0,
        "symbol_pnl": dict(sorted(symbol_pnl.items())),
        "sector_pnl": dict(sorted(sector_pnl.items())),
        "regime_pnl": dict(sorted(regime_pnl.items())),
        "purged_rows": purged_rows,
        "model_family": "reinforcement_learning",
        "algorithm": spec.algorithm,
    }
    final_model = train_sb3_model(rows, cost_model, spec)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    final_model.save(str(destination))
    artifact = destination if destination.is_file() else destination.with_suffix(".zip")
    if not artifact.is_file():
        raise RuntimeError("Stable-Baselines3 did not create the expected policy artifact")
    return {
        "metrics": metrics,
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "feature_names": list(feature_names),
        "execution_scope": "paper_only",
        "holdout_accessed": False,
        "live_compatible": False,
    }
