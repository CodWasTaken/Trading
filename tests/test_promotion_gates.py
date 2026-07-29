from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from trading_app.model_registry import ModelRegistry
from trading_app.promotion import PromotionGateConfig, load_promotion_gates
from trading_app.research import BacktestMetrics, RidgeReturnModel, synthetic_feature_rows

FEATURE_NAMES = ("momentum", "news_score", "volatility", "spread_bps")
GATE_CONFIG = Path(__file__).parents[1] / "config" / "promotion" / "strict-paper-v1.json"


def _metrics() -> BacktestMetrics:
    return BacktestMetrics(
        observations=300,
        scored_observations=1000,
        net_return=0.10,
        annualized_return=0.10,
        sharpe=0.51,
        max_drawdown=0.149,
        hit_rate=0.55,
        turnover=10,
        average_trade_return=0.001,
        folds=8,
        excess_return_vs_benchmark=0.01,
    )


def _diagnostics() -> dict[str, object]:
    return {
        "symbols": {f"SYM{index}": {"pnl_contribution": 0.02} for index in range(5)},
        "sectors": {f"Sector {index}": {"pnl_contribution": 0.02} for index in range(5)},
        "short_safety": {
            "unborrowable_short_orders": 0,
            "maximum_gross_short_exposure": 0.30,
            "maximum_single_short_position": 0.03,
            "borrow_status_validated": True,
        },
    }


def _register(
    tmp_path: Path,
    *,
    metrics: BacktestMetrics | None = None,
    diagnostics: object | None = None,
) -> tuple[ModelRegistry, str]:
    model = RidgeReturnModel(FEATURE_NAMES)
    model.fit(synthetic_feature_rows(220))
    registry = ModelRegistry(tmp_path / "registry")
    record = registry.register(
        model,
        metrics or _metrics(),
        metadata={
            "dataset": "calibration.jsonl",
            "diagnostics": _diagnostics() if diagnostics is None else diagnostics,
        },
    )
    return registry, record.version


def _record_holdout(
    registry: ModelRegistry,
    version: str,
    *,
    metrics: dict[str, object] | None = None,
    diagnostics: object | None = None,
    candidate_family_size: int = 3,
) -> None:
    registry.record_holdout_evaluation(
        version,
        dataset_path="sealed-holdout.jsonl",
        dataset_sha256="holdout-sha",
        metadata_sha256="metadata-sha",
        split_id="split-1",
        report_path="holdout-report.json",
        report_sha256="report-sha",
        metrics=metrics
        or {
            "observations": 300,
            "net_return": 0.05,
            "sharpe": 0.2,
            "max_drawdown": 0.149,
            "excess_return_vs_benchmark": 0.01,
            "net_return_lower_bound": 0.001,
            "excess_return_lower_bound": 0.001,
        },
        diagnostics=_diagnostics() if diagnostics is None else diagnostics,
        frozen_configuration={
            "statistical_plan": {
                "candidate_family_size": candidate_family_size,
            }
        },
    )


@pytest.mark.parametrize(
    ("metrics", "failure"),
    [
        (replace(_metrics(), folds=7), "insufficient_walk_forward_folds"),
        (
            replace(_metrics(), scored_observations=999),
            "insufficient_scored_calibration_observations",
        ),
        (replace(_metrics(), net_return=0.0), "non_positive_net_return"),
        (
            replace(_metrics(), excess_return_vs_benchmark=0.0),
            "benchmark_excess_return_below_gate",
        ),
        (replace(_metrics(), sharpe=0.50), "sharpe_below_gate"),
        (replace(_metrics(), max_drawdown=0.15), "drawdown_above_gate"),
    ],
)
def test_calibration_thresholds_fail_closed(
    tmp_path: Path,
    metrics: BacktestMetrics,
    failure: str,
) -> None:
    registry, version = _register(tmp_path, metrics=metrics)
    _record_holdout(registry, version)

    with pytest.raises(ValueError, match=failure):
        registry.promote(version)


@pytest.mark.parametrize(
    ("update", "failure"),
    [
        ({"observations": 299}, "holdout_observations_below_gate"),
        ({"net_return": 0.0}, "holdout_net_return_below_gate"),
        (
            {"excess_return_vs_benchmark": 0.0},
            "holdout_excess_return_below_gate",
        ),
        (
            {"net_return_lower_bound": 0.0},
            "holdout_net_return_lower_bound_below_gate",
        ),
        (
            {"excess_return_lower_bound": 0.0},
            "holdout_excess_return_lower_bound_below_gate",
        ),
        ({"max_drawdown": 0.15}, "holdout_drawdown_above_gate"),
    ],
)
def test_holdout_thresholds_fail_closed(
    tmp_path: Path,
    update: dict[str, object],
    failure: str,
) -> None:
    registry, version = _register(tmp_path)
    holdout = {
        "observations": 300,
        "net_return": 0.05,
        "sharpe": 0.2,
        "max_drawdown": 0.149,
        "excess_return_vs_benchmark": 0.01,
        "net_return_lower_bound": 0.001,
        "excess_return_lower_bound": 0.001,
        **update,
    }
    _record_holdout(registry, version, metrics=holdout)

    with pytest.raises(ValueError, match=failure):
        registry.promote(version)


def test_finalist_budget_above_three_is_rejected(tmp_path: Path) -> None:
    registry, version = _register(tmp_path)
    _record_holdout(registry, version, candidate_family_size=4)

    with pytest.raises(ValueError, match="candidate_family_size_above_gate"):
        registry.promote(version)


def test_historical_holdout_requirement_cannot_be_disabled(tmp_path: Path) -> None:
    registry, version = _register(tmp_path)

    with pytest.raises(
        ValueError,
        match="historical_holdout_requirement_cannot_be_disabled",
    ):
        registry.promote(version, require_holdout_evaluation=False)


def test_concentration_and_short_safety_gates_reject_violations(
    tmp_path: Path,
) -> None:
    diagnostics = _diagnostics()
    diagnostics["symbols"] = {
        "CONCENTRATED": {"pnl_contribution": 0.21},
        **{f"SYM{index}": {"pnl_contribution": 0.1975} for index in range(4)},
    }
    diagnostics["sectors"] = {
        "Concentrated": {"pnl_contribution": 0.36},
        "Other 1": {"pnl_contribution": 0.32},
        "Other 2": {"pnl_contribution": 0.32},
    }
    diagnostics["short_safety"] = {
        "unborrowable_short_orders": 1,
        "maximum_gross_short_exposure": 0.301,
        "maximum_single_short_position": 0.031,
        "borrow_status_validated": False,
    }
    registry, version = _register(tmp_path, diagnostics=diagnostics)
    _record_holdout(registry, version)

    with pytest.raises(ValueError) as raised:
        registry.promote(version)

    message = str(raised.value)
    assert "calibration_symbol_concentration_above_gate" in message
    assert "calibration_sector_concentration_above_gate" in message
    assert "calibration_unborrowable_short_orders_above_gate" in message
    assert "calibration_gross_short_exposure_above_gate" in message
    assert "calibration_single_short_position_above_gate" in message
    assert "calibration_borrow_status_not_validated" in message


def test_missing_attribution_and_short_evidence_never_defaults_to_zero(
    tmp_path: Path,
) -> None:
    registry, version = _register(tmp_path, diagnostics={})
    _record_holdout(registry, version)

    with pytest.raises(ValueError) as raised:
        registry.promote(version)

    message = str(raised.value)
    assert "calibration_symbol_attribution_missing" in message
    assert "calibration_sector_attribution_missing" in message
    assert "calibration_short_safety_evidence_missing" in message


def test_success_records_exact_gate_manifest(
    tmp_path: Path,
) -> None:
    registry, version = _register(tmp_path)
    _record_holdout(registry, version)

    promoted = registry.promote(version)
    event = registry.history(alias="champion")[-1]
    gates = event["details"]["promotion_gates"]

    assert promoted.version == version
    assert gates["minimum_calibration_folds"] == 8
    assert gates["minimum_calibration_observations"] == 1000
    assert gates["minimum_holdout_observations"] == 300
    assert gates["maximum_symbol_pnl_contribution"] == 0.20
    assert gates["maximum_sector_pnl_contribution"] == 0.35
    assert gates["maximum_gross_short_exposure"] == 0.30
    assert gates["maximum_single_short_position"] == 0.03
    assert gates["manifest_sha256"] == PromotionGateConfig().manifest_sha256


def test_bundled_gate_manifest_matches_code_defaults() -> None:
    loaded = load_promotion_gates(GATE_CONFIG)

    assert loaded == PromotionGateConfig()
    assert loaded.manifest_sha256 == PromotionGateConfig().manifest_sha256
