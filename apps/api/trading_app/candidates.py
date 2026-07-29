from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev

from pydantic import BaseModel, ConfigDict, Field

from .cost_model import CostModelConfig, load_cost_model
from .dataset import dataset_metadata_path, load_feature_dataset
from .research import FeatureRow, RidgeReturnModel
from .universe import (
    UniverseManifest,
    assert_universe_binding,
    assert_universe_symbols,
    load_universe_manifest,
)

EXACT_CANDIDATES_PER_GENERATION = 100
MAX_GENERATIONS = 10
DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "net_return": 4.0,
    "benchmark_excess_return": 3.0,
    "sharpe": 0.20,
    "drawdown": -2.0,
    "fold_stability": 0.50,
    "symbol_stability": 0.50,
    "sector_stability": 0.50,
    "long_short_balance": 0.50,
    "news_ablation_improvement": 2.0,
    "threshold_robustness": 0.50,
    "symbol_concentration": -1.0,
    "sector_concentration": -0.70,
    "regime_concentration": -0.50,
    "turnover": -0.001,
    "threshold_fragility": -0.50,
    "few_active_observations": -0.50,
    "one_sided_performance": -0.50,
    "short_borrow_dependence": -0.50,
}


class CandidateConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    ridge: float = Field(gt=0)
    positive_threshold: float = Field(gt=0)
    negative_threshold: float = Field(gt=0)
    feature_subset: tuple[str, ...]
    cost_stress_multiplier: float = Field(ge=1)
    max_long_exposure: float = Field(gt=0, le=1)
    max_short_exposure: float = Field(gt=0, le=0.30)
    forecast_horizon: int = Field(default=5, gt=0)
    seed: int = Field(ge=0)

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"seed"})
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    @property
    def candidate_id(self) -> str:
        return f"candidate-{self.fingerprint[:16]}"


@dataclass(frozen=True)
class _Scored:
    row: FeatureRow
    prediction: float
    fold: int


@dataclass(frozen=True)
class _Fold:
    index: int
    train: list[FeatureRow]
    inner_train: list[FeatureRow]
    inner_validation: list[FeatureRow]
    test: list[FeatureRow]
    purged_outer: int
    purged_inner: int


def generate_candidate_configs(
    feature_names: tuple[str, ...],
    *,
    seed: int,
    forecast_horizon: int = 5,
) -> list[CandidateConfig]:
    if not feature_names:
        raise ValueError("Candidate generation requires feature names")
    subsets = _feature_subsets(feature_names)
    dimensions = itertools.product(
        (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
        (0.00025, 0.0005, 0.00075, 0.001, 0.0015),
        (0.00025, 0.0005, 0.00075, 0.001, 0.0015),
        subsets,
        (1.0, 1.25, 1.5, 2.0),
        (0.30, 0.45, 0.60),
        (0.10, 0.20, 0.30),
    )
    combinations = list(dimensions)
    generator = random.Random(seed)
    generator.shuffle(combinations)
    selected = combinations[:EXACT_CANDIDATES_PER_GENERATION]
    configs = [
        CandidateConfig(
            ridge=ridge,
            positive_threshold=positive,
            negative_threshold=negative,
            feature_subset=subset,
            cost_stress_multiplier=cost_stress,
            max_long_exposure=long_limit,
            max_short_exposure=short_limit,
            forecast_horizon=forecast_horizon,
            seed=seed + index,
        )
        for index, (
            ridge,
            positive,
            negative,
            subset,
            cost_stress,
            long_limit,
            short_limit,
        ) in enumerate(selected)
    ]
    validate_candidate_configs(configs)
    return configs


def validate_candidate_configs(configs: list[CandidateConfig]) -> None:
    if len(configs) != EXACT_CANDIDATES_PER_GENERATION:
        raise ValueError(
            f"Each generation requires exactly {EXACT_CANDIDATES_PER_GENERATION} "
            f"candidate configurations; received {len(configs)}"
        )
    fingerprints = [item.fingerprint for item in configs]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("Duplicate candidate configurations are not allowed")


def run_candidate_generation(
    dataset_path: str,
    output_directory: str,
    *,
    generation: int,
    seed: int,
    cost_config_path: str,
    universe_manifest_path: str | None,
    max_workers: int = 4,
    configs: list[CandidateConfig] | None = None,
    outer_folds: int = 8,
    periods_per_year: float = 327.6,
) -> dict[str, object]:
    if not 1 <= generation <= MAX_GENERATIONS:
        raise ValueError(f"generation must be between 1 and {MAX_GENERATIONS}")
    if not 1 <= max_workers <= 16:
        raise ValueError("max_workers must be between 1 and 16")
    if outer_folds < 8:
        raise ValueError("Nested calibration requires at least eight outer folds")
    source = Path(dataset_path)
    rows, feature_names, metadata = load_feature_dataset(source)
    if metadata.get("dataset_role") != "calibration" or metadata.get("sealed") is True:
        raise ValueError(
            "Candidate generations may evaluate calibration datasets only; "
            "sealed or untouched holdout data is forbidden"
        )
    if _as_int(metadata.get("forecast_bars", 5), "forecast_bars") != 5:
        raise ValueError("The default governed generation requires a five-bar horizon")
    universe: UniverseManifest | None = (
        None if universe_manifest_path is None else load_universe_manifest(universe_manifest_path)
    )
    universe_binding: dict[str, object] | None = None
    if universe is not None:
        assert universe_manifest_path is not None
        assert_universe_binding(metadata, universe, context="candidate calibration")
        assert_universe_symbols(
            {row.symbol for row in rows},
            universe,
            context="candidate calibration rows",
        )
        universe_binding = universe.binding(universe_manifest_path)
    cost_model = load_cost_model(cost_config_path)
    candidate_configs = configs or generate_candidate_configs(
        feature_names,
        seed=seed,
        forecast_horizon=5,
    )
    validate_candidate_configs(candidate_configs)
    if any(set(item.feature_subset) - set(feature_names) for item in candidate_configs):
        raise ValueError("Candidate feature subset is not present in the calibration dataset")
    if any(item.forecast_horizon != 5 for item in candidate_configs):
        raise ValueError("Every candidate in this generation must use five bars")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    candidates_dir = destination / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    manifest_path = destination / "generation-manifest.json"
    source_metadata = dataset_metadata_path(source)
    manifest = {
        "schema_version": 1,
        "generation": generation,
        "seed": seed,
        "candidate_count": EXACT_CANDIDATES_PER_GENERATION,
        "candidate_fingerprints": sorted(item.fingerprint for item in candidate_configs),
        "candidate_configurations_sha256": hashlib.sha256(
            _canonical_json(
                [
                    item.model_dump(mode="json")
                    for item in sorted(
                        candidate_configs,
                        key=lambda candidate: candidate.candidate_id,
                    )
                ]
            )
        ).hexdigest(),
        "dataset": {
            "path": str(source),
            "sha256": _sha256(source),
            "metadata_sha256": _sha256(source_metadata),
            "role": "calibration",
        },
        "universe": universe_binding,
        "cost_model": {
            "manifest": cost_model.model_dump(mode="json"),
            "manifest_sha256": cost_model.manifest_sha256,
        },
        "validation": {
            "method": "nested_purged_chronological_walk_forward",
            "outer_folds": outer_folds,
            "inner_split": "last_20_percent_of_each_outer_training_window",
            "holdout_access": "forbidden",
        },
        "composite_score": {
            "predeclared": True,
            "weights": DEFAULT_SCORE_WEIGHTS,
        },
        "resources": {
            "max_workers": max_workers,
            "bounded_parallel_execution": True,
            "resumable": True,
        },
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _stable_manifest(existing) != _stable_manifest(manifest):
            raise ValueError("Existing generation manifest does not match requested run")
    else:
        _write_json(manifest_path, manifest)

    reports: dict[str, dict[str, object]] = {}
    pending: list[CandidateConfig] = []
    for config in candidate_configs:
        candidate_dir = candidates_dir / config.candidate_id
        candidate_dir.mkdir(exist_ok=True)
        _write_json(candidate_dir / "config.json", config.model_dump(mode="json"))
        report_path = candidate_dir / "metrics.json"
        failure_path = candidate_dir / "failure.json"
        if report_path.exists():
            reports[config.candidate_id] = json.loads(report_path.read_text(encoding="utf-8"))
        elif failure_path.exists():
            reports[config.candidate_id] = json.loads(failure_path.read_text(encoding="utf-8"))
        else:
            pending.append(config)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _evaluate_and_persist,
                config,
                rows,
                feature_names,
                cost_model,
                universe,
                candidates_dir / config.candidate_id,
                outer_folds,
                periods_per_year,
            ): config
            for config in pending
        }
        for future in as_completed(futures):
            config = futures[future]
            try:
                reports[config.candidate_id] = future.result()
            except Exception as error:
                failure: dict[str, object] = {
                    "candidate_id": config.candidate_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "configuration_fingerprint": config.fingerprint,
                }
                _write_json(
                    candidates_dir / config.candidate_id / "failure.json",
                    failure,
                )
                reports[config.candidate_id] = failure

    ranked = rank_candidates(list(reports.values()))
    successful = [item for item in ranked if item.get("status") == "complete"]
    if len(successful) < 3:
        raise RuntimeError("Fewer than three candidates completed successfully")
    finalists = freeze_top_finalists(successful, destination, limit=3)
    summary = {
        **manifest,
        "status": "complete",
        "completed": len(successful),
        "failed": EXACT_CANDIDATES_PER_GENERATION - len(successful),
        "ranking": [
            {
                "rank": index,
                "candidate_id": item["candidate_id"],
                "composite_score": item["composite_score"],
                "metrics_path": item["metrics_path"],
            }
            for index, item in enumerate(successful, start=1)
        ],
        "finalists": finalists,
        "candidate_family_size": 3,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_json(destination / "generation-report.json", summary)
    return summary


def rank_candidates(reports: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        reports,
        key=lambda item: (
            item.get("status") != "complete",
            -_as_float(item.get("composite_score", -math.inf), "composite_score"),
            str(item.get("candidate_id", "")),
        ),
    )


def freeze_top_finalists(
    ranked_successful: list[dict[str, object]],
    destination: Path,
    *,
    limit: int,
) -> list[dict[str, object]]:
    if limit != 3:
        raise ValueError("Governed holdout finalist count must be exactly three")
    finalists = []
    for rank, report in enumerate(ranked_successful[:limit], start=1):
        metrics_path = Path(str(report["metrics_path"]))
        model_path = Path(str(report["model_path"]))
        finalists.append(
            {
                "rank": rank,
                "candidate_id": report["candidate_id"],
                "configuration_fingerprint": report["configuration_fingerprint"],
                "composite_score": report["composite_score"],
                "metrics_path": str(metrics_path),
                "metrics_sha256": _sha256(metrics_path),
                "model_path": str(model_path),
                "model_sha256": _sha256(model_path),
                "frozen": True,
            }
        )
    _write_json(
        destination / "finalists.json",
        {
            "schema_version": 1,
            "candidate_family_size": 3,
            "holdout_evaluations_allowed": 3,
            "finalists": finalists,
            "controls": {
                "all_100_ranked_on_calibration_only": True,
                "sealed_holdout_not_accessed": True,
                "finalists_frozen_before_holdout": True,
            },
        },
    )
    return finalists


def should_stop_generations(
    best_scores: list[float],
    *,
    meaningful_improvement: float,
    patience: int = 3,
) -> bool:
    if patience != 3:
        raise ValueError("Governed lifecycle patience is three generations")
    if len(best_scores) >= MAX_GENERATIONS:
        return True
    if len(best_scores) <= patience:
        return False
    prior_best = max(best_scores[:-patience])
    recent_best = max(best_scores[-patience:])
    return recent_best < prior_best + meaningful_improvement


def _evaluate_and_persist(
    config: CandidateConfig,
    rows: list[FeatureRow],
    feature_names: tuple[str, ...],
    cost_model: CostModelConfig,
    universe: UniverseManifest | None,
    candidate_dir: Path,
    outer_folds: int,
    periods_per_year: float,
) -> dict[str, object]:
    folds = _nested_folds(rows, outer_folds)
    selected_rows = _select_features(rows, feature_names, config.feature_subset)
    selected_by_key = {
        (row.timestamp, row.symbol, row.label_end_time): row for row in selected_rows
    }
    selected_folds = [
        _Fold(
            index=fold.index,
            train=_map_rows(fold.train, selected_by_key),
            inner_train=_map_rows(fold.inner_train, selected_by_key),
            inner_validation=_map_rows(fold.inner_validation, selected_by_key),
            test=_map_rows(fold.test, selected_by_key),
            purged_outer=fold.purged_outer,
            purged_inner=fold.purged_inner,
        )
        for fold in folds
    ]
    scored, inner_scores, purged_rows = _score_nested(
        selected_folds,
        config.feature_subset,
        config.ridge,
    )
    metrics = _simulate_candidate(
        scored,
        config,
        cost_model,
        universe,
        periods_per_year,
    )
    no_news_return = _as_float(metrics["net_return"], "net_return")
    if "news_score" in config.feature_subset and len(config.feature_subset) > 1:
        ablation = tuple(name for name in config.feature_subset if name != "news_score")
        ablation_rows = _select_features(
            rows,
            feature_names,
            ablation,
        )
        ablation_by_key = {
            (row.timestamp, row.symbol, row.label_end_time): row for row in ablation_rows
        }
        ablation_folds = [
            _Fold(
                index=fold.index,
                train=_map_rows(fold.train, ablation_by_key),
                inner_train=_map_rows(fold.inner_train, ablation_by_key),
                inner_validation=_map_rows(fold.inner_validation, ablation_by_key),
                test=_map_rows(fold.test, ablation_by_key),
                purged_outer=fold.purged_outer,
                purged_inner=fold.purged_inner,
            )
            for fold in folds
        ]
        ablation_scored, _, _ = _score_nested(
            ablation_folds,
            ablation,
            config.ridge,
        )
        no_news_return = _as_float(
            _simulate_candidate(
                ablation_scored,
                config.model_copy(update={"feature_subset": ablation}),
                cost_model,
                universe,
                periods_per_year,
            )["net_return"],
            "net_return",
        )
    metrics["news_ablation_improvement"] = (
        _as_float(metrics["net_return"], "net_return") - no_news_return
    )
    metrics["inner_validation_score_mean"] = fmean(inner_scores)
    metrics["purged_rows"] = purged_rows
    score, components = _composite_score(metrics)

    full_model = RidgeReturnModel(config.feature_subset, ridge=config.ridge)
    full_model.fit(selected_rows)
    model_path = candidate_dir / "model.json"
    full_model.save(model_path)
    report: dict[str, object] = {
        "candidate_id": config.candidate_id,
        "status": "complete",
        "configuration_fingerprint": config.fingerprint,
        "configuration": config.model_dump(mode="json"),
        "evaluation": {
            "dataset_role": "calibration",
            "holdout_accessed": False,
            "method": "nested_purged_chronological_walk_forward",
            "outer_folds": len(folds),
            "inner_validation_folds": len(inner_scores),
        },
        "metrics": metrics,
        "score_components": components,
        "composite_score": score,
        "metrics_path": str(candidate_dir / "metrics.json"),
        "model_path": str(model_path),
    }
    _write_json(candidate_dir / "metrics.json", report)
    return report


def _nested_folds(rows: list[FeatureRow], outer_folds: int) -> list[_Fold]:
    ordered = sorted(rows, key=lambda row: (row.timestamp, row.symbol))
    timestamps = sorted({row.timestamp for row in ordered})
    minimum_train_groups = max(10, len(timestamps) // 4)
    if len(timestamps) < minimum_train_groups + outer_folds:
        raise ValueError("Not enough chronological groups for eight nested folds")
    remaining = len(timestamps) - minimum_train_groups
    folds: list[_Fold] = []
    for index in range(outer_folds):
        start_index = minimum_train_groups + remaining * index // outer_folds
        end_index = minimum_train_groups + remaining * (index + 1) // outer_folds
        test_times = set(timestamps[start_index:end_index])
        if not test_times:
            raise ValueError("Nested fold has no test timestamps")
        test_start = min(test_times)
        training_candidates = [row for row in ordered if row.timestamp < test_start]
        train = [
            row
            for row in training_candidates
            if row.label_end_time is None or row.label_end_time < test_start
        ]
        train_times = sorted({row.timestamp for row in train})
        inner_index = max(1, int(len(train_times) * 0.8))
        if inner_index >= len(train_times):
            raise ValueError("Nested fold has no inner validation period")
        inner_boundary = train_times[inner_index]
        inner_candidates = [row for row in train if row.timestamp < inner_boundary]
        inner_train = [
            row
            for row in inner_candidates
            if row.label_end_time is None or row.label_end_time < inner_boundary
        ]
        inner_validation = [row for row in train if row.timestamp >= inner_boundary]
        test = [row for row in ordered if row.timestamp in test_times]
        folds.append(
            _Fold(
                index=index,
                train=train,
                inner_train=inner_train,
                inner_validation=inner_validation,
                test=test,
                purged_outer=len(training_candidates) - len(train),
                purged_inner=len(inner_candidates) - len(inner_train),
            )
        )
    return folds


def _score_nested(
    folds: list[_Fold],
    feature_names: tuple[str, ...],
    ridge: float,
) -> tuple[list[_Scored], list[float], int]:
    scored: list[_Scored] = []
    inner_scores: list[float] = []
    purged = 0
    for fold in folds:
        if len(fold.inner_train) <= len(feature_names) + 1:
            raise ValueError("Inner training fold is too small")
        inner_model = RidgeReturnModel(feature_names, ridge=ridge)
        inner_model.fit(fold.inner_train)
        inner_errors = [
            abs(inner_model.predict(row.features) - row.target_return)
            for row in fold.inner_validation
        ]
        inner_scores.append(-fmean(inner_errors))
        if len(fold.train) <= len(feature_names) + 1:
            raise ValueError("Outer training fold is too small")
        outer_model = RidgeReturnModel(feature_names, ridge=ridge)
        outer_model.fit(fold.train)
        scored.extend(
            _Scored(
                row=row,
                prediction=outer_model.predict(row.features),
                fold=fold.index,
            )
            for row in fold.test
        )
        purged += fold.purged_outer + fold.purged_inner
    return scored, inner_scores, purged


def _simulate_candidate(
    scored: list[_Scored],
    config: CandidateConfig,
    cost_model: CostModelConfig,
    universe: UniverseManifest | None,
    periods_per_year: float,
) -> dict[str, object]:
    ordered = _non_overlapping_scored(scored)
    groups: dict[datetime, list[_Scored]] = defaultdict(list)
    for item in ordered:
        groups[item.row.timestamp].append(item)
    previous: dict[str, float] = defaultdict(float)
    benchmark_previous: dict[str, float] = defaultdict(float)
    returns: list[float] = []
    benchmark_returns: list[float] = []
    fold_returns: dict[int, list[float]] = defaultdict(list)
    symbol_pnl: dict[str, float] = defaultdict(float)
    sector_pnl: dict[str, float] = defaultdict(float)
    regime_pnl: dict[str, float] = defaultdict(float)
    long_pnl = 0.0
    short_pnl = 0.0
    turnover = 0.0
    active = 0
    short_active = 0
    timestamps = sorted(groups)
    for time_index, timestamp in enumerate(timestamps):
        items = groups[timestamp]
        allocation = 1.0 / len(items)
        period_return = 0.0
        benchmark_return = 0.0
        for item in items:
            if item.prediction > config.positive_threshold:
                position = config.max_long_exposure
            elif item.prediction < -config.negative_threshold:
                position = -config.max_short_exposure
            else:
                position = 0.0
            prior = previous[item.row.symbol]
            trade_cost = (
                cost_model.execution.trade_cost_fraction(prior, position)
                * config.cost_stress_multiplier
            )
            holding_cost = (
                cost_model.execution.holding_cost_fraction(
                    position,
                    periods_per_year,
                )
                * config.cost_stress_multiplier
            )
            net = position * item.row.target_return - trade_cost - holding_cost
            weighted = allocation * net
            period_return += weighted
            symbol_pnl[item.row.symbol] += weighted
            sector = (
                "Unmapped"
                if universe is None
                else universe.sectors.get(item.row.symbol, "Unmapped")
            )
            sector_pnl[sector] += weighted
            regime = ("early", "middle", "late")[min(2, time_index * 3 // max(1, len(timestamps)))]
            regime_pnl[regime] += weighted
            if position > 0:
                long_pnl += weighted
            elif position < 0:
                short_pnl += weighted
                short_active += 1
            active += int(position != 0)
            turnover += abs(position - prior)
            previous[item.row.symbol] = position

            benchmark_cost = cost_model.execution.trade_cost_fraction(
                benchmark_previous[item.row.symbol],
                1.0,
            )
            benchmark_return += allocation * (item.row.target_return - benchmark_cost)
            benchmark_previous[item.row.symbol] = 1.0
        returns.append(period_return)
        benchmark_returns.append(benchmark_return)
        fold_returns[items[0].fold].append(period_return)
    net_return = _compound(returns)
    benchmark = _compound(benchmark_returns)
    mean = fmean(returns)
    volatility = pstdev(returns) if len(returns) > 1 else 0.0
    sharpe = 0.0 if volatility == 0 else mean / volatility * math.sqrt(periods_per_year)
    max_drawdown = _max_drawdown(returns)
    fold_net = [_compound(values) for _, values in sorted(fold_returns.items())]
    threshold_returns = [
        _threshold_net(scored, config, cost_model, multiplier, periods_per_year)
        for multiplier in (0.9, 1.0, 1.1)
    ]
    symbol_concentration = _concentration(symbol_pnl)
    sector_concentration = _concentration(sector_pnl)
    regime_concentration = _concentration(regime_pnl)
    balance = _balance(long_pnl, short_pnl)
    return {
        "observations": len(scored),
        "test_periods": len(returns),
        "active_observations": active,
        "net_return": net_return,
        "benchmark_net_return": benchmark,
        "benchmark_excess_return": net_return - benchmark,
        "sharpe": sharpe,
        "maximum_drawdown": max_drawdown,
        "turnover": turnover,
        "fold_net_returns": fold_net,
        "fold_stability": sum(value > 0 for value in fold_net) / len(fold_net),
        "symbol_stability": _positive_share(symbol_pnl),
        "sector_stability": _positive_share(sector_pnl),
        "long_contribution": long_pnl,
        "short_contribution": short_pnl,
        "long_short_balance": balance,
        "short_borrow_dependence": 0.0 if active == 0 else short_active / active,
        "symbol_concentration": symbol_concentration,
        "sector_concentration": sector_concentration,
        "regime_concentration": regime_concentration,
        "threshold_net_returns": threshold_returns,
        "threshold_robustness": min(threshold_returns),
        "threshold_fragility": max(threshold_returns) - min(threshold_returns),
        "symbol_pnl": dict(sorted(symbol_pnl.items())),
        "sector_pnl": dict(sorted(sector_pnl.items())),
        "regime_pnl": dict(sorted(regime_pnl.items())),
    }


def _composite_score(
    metrics: dict[str, object],
) -> tuple[float, dict[str, float]]:
    active_ratio = _as_float(metrics["active_observations"], "active_observations") / max(
        1,
        _as_int(metrics["observations"], "observations"),
    )
    components = {
        "net_return": _as_float(metrics["net_return"], "net_return"),
        "benchmark_excess_return": _as_float(
            metrics["benchmark_excess_return"], "benchmark_excess_return"
        ),
        "sharpe": _as_float(metrics["sharpe"], "sharpe"),
        "drawdown": _as_float(metrics["maximum_drawdown"], "maximum_drawdown"),
        "fold_stability": _as_float(metrics["fold_stability"], "fold_stability"),
        "symbol_stability": _as_float(metrics["symbol_stability"], "symbol_stability"),
        "sector_stability": _as_float(metrics["sector_stability"], "sector_stability"),
        "long_short_balance": _as_float(metrics["long_short_balance"], "long_short_balance"),
        "news_ablation_improvement": _as_float(
            metrics["news_ablation_improvement"], "news_ablation_improvement"
        ),
        "threshold_robustness": _as_float(metrics["threshold_robustness"], "threshold_robustness"),
        "symbol_concentration": _as_float(metrics["symbol_concentration"], "symbol_concentration"),
        "sector_concentration": _as_float(metrics["sector_concentration"], "sector_concentration"),
        "regime_concentration": _as_float(metrics["regime_concentration"], "regime_concentration"),
        "turnover": _as_float(metrics["turnover"], "turnover"),
        "threshold_fragility": _as_float(metrics["threshold_fragility"], "threshold_fragility"),
        "few_active_observations": 1.0 - active_ratio,
        "one_sided_performance": 1.0
        - _as_float(metrics["long_short_balance"], "long_short_balance"),
        "short_borrow_dependence": _as_float(
            metrics["short_borrow_dependence"], "short_borrow_dependence"
        ),
    }
    score = sum(DEFAULT_SCORE_WEIGHTS[name] * value for name, value in components.items())
    return score, components


def _threshold_net(
    scored: list[_Scored],
    config: CandidateConfig,
    cost_model: CostModelConfig,
    multiplier: float,
    periods_per_year: float,
) -> float:
    stressed = config.model_copy(
        update={
            "positive_threshold": config.positive_threshold * multiplier,
            "negative_threshold": config.negative_threshold * multiplier,
        }
    )
    previous: dict[str, float] = defaultdict(float)
    returns: list[float] = []
    groups: dict[datetime, list[_Scored]] = defaultdict(list)
    for item in _non_overlapping_scored(scored):
        groups[item.row.timestamp].append(item)
    for timestamp in sorted(groups):
        items = groups[timestamp]
        allocation = 1.0 / len(items)
        period_return = 0.0
        for item in sorted(items, key=lambda value: value.row.symbol):
            position = (
                stressed.max_long_exposure
                if item.prediction > stressed.positive_threshold
                else -stressed.max_short_exposure
                if item.prediction < -stressed.negative_threshold
                else 0.0
            )
            cost = (
                cost_model.execution.trade_cost_fraction(
                    previous[item.row.symbol],
                    position,
                )
                * stressed.cost_stress_multiplier
            )
            holding_cost = (
                cost_model.execution.holding_cost_fraction(
                    position,
                    periods_per_year,
                )
                * stressed.cost_stress_multiplier
            )
            period_return += allocation * (position * item.row.target_return - cost - holding_cost)
            previous[item.row.symbol] = position
        returns.append(period_return)
    return _compound(returns)


def _non_overlapping_scored(scored: list[_Scored]) -> list[_Scored]:
    groups: dict[datetime, list[_Scored]] = defaultdict(list)
    for item in sorted(scored, key=lambda value: (value.row.timestamp, value.row.symbol)):
        groups[item.row.timestamp].append(item)
    selected: list[_Scored] = []
    next_available: datetime | None = None
    for timestamp in sorted(groups):
        if next_available is not None and timestamp < next_available:
            continue
        items = groups[timestamp]
        selected.extend(items)
        label_ends = [
            item.row.label_end_time for item in items if item.row.label_end_time is not None
        ]
        next_available = max(label_ends) if label_ends else timestamp
    return selected


def _feature_subsets(feature_names: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    candidates = [
        feature_names,
        tuple(name for name in feature_names if name != "news_score"),
        feature_names[: max(1, min(2, len(feature_names)))],
        tuple(dict.fromkeys((feature_names[0], feature_names[-1]))),
    ]
    unique = []
    for subset in candidates:
        if subset and subset not in unique:
            unique.append(subset)
    return tuple(unique)


def _select_features(
    rows: list[FeatureRow],
    feature_names: tuple[str, ...],
    subset: tuple[str, ...],
) -> list[FeatureRow]:
    indices = [feature_names.index(name) for name in subset]
    return [
        FeatureRow(
            timestamp=row.timestamp,
            symbol=row.symbol,
            features=tuple(row.features[index] for index in indices),
            target_return=row.target_return,
            label_end_time=row.label_end_time,
        )
        for row in rows
    ]


def _map_rows(
    rows: list[FeatureRow],
    mapping: dict[tuple[datetime, str, datetime | None], FeatureRow],
) -> list[FeatureRow]:
    return [mapping[(row.timestamp, row.symbol, row.label_end_time)] for row in rows]


def _positive_share(values: dict[str, float]) -> float:
    return 0.0 if not values else sum(value > 0 for value in values.values()) / len(values)


def _concentration(values: dict[str, float]) -> float:
    denominator = sum(abs(value) for value in values.values())
    return 0.0 if denominator == 0 else max(abs(value) for value in values.values()) / denominator


def _balance(long_pnl: float, short_pnl: float) -> float:
    if long_pnl <= 0 or short_pnl <= 0:
        return 0.0
    denominator = abs(long_pnl) + abs(short_pnl)
    return 1.0 - abs(long_pnl - short_pnl) / denominator


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _compound(values: list[float]) -> float:
    return math.prod(1 + value for value in values) - 1


def _max_drawdown(values: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    maximum = 0.0
    for value in values:
        equity *= 1 + value
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _stable_manifest(payload: dict[str, object]) -> bytes:
    ignored = {"completed_at", "status", "ranking", "finalists", "completed", "failed"}
    return _canonical_json({key: value for key, value in payload.items() if key not in ignored})


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
