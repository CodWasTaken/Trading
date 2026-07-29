import pytest
from trading_app.model_registry import ModelRegistry
from trading_app.research import BacktestMetrics, RidgeReturnModel, synthetic_feature_rows

FEATURE_NAMES = ("momentum", "news_score", "volatility", "spread_bps")


def _governance_diagnostics() -> dict[str, object]:
    symbols = {f"SYM{index}": {"pnl_contribution": 0.02} for index in range(5)}
    sectors = {f"Sector {index}": {"pnl_contribution": 0.02} for index in range(5)}
    return {
        "symbols": symbols,
        "sectors": sectors,
        "short_safety": {
            "unborrowable_short_orders": 0,
            "maximum_gross_short_exposure": 0.30,
            "maximum_single_short_position": 0.03,
            "borrow_status_validated": True,
        },
    }


def test_registry_registers_and_promotes_champion(tmp_path) -> None:
    model = RidgeReturnModel(FEATURE_NAMES)
    model.fit(synthetic_feature_rows(200))
    metrics = BacktestMetrics(
        observations=100,
        net_return=0.12,
        annualized_return=0.12,
        sharpe=1.1,
        max_drawdown=0.05,
        hit_rate=0.56,
        turnover=20,
        average_trade_return=0.006,
        folds=8,
        scored_observations=1000,
        excess_return_vs_benchmark=0.02,
    )
    registry = ModelRegistry(tmp_path)
    record = registry.register(
        model,
        metrics,
        metadata={
            "dataset": "synthetic_fixture",
            "diagnostics": _governance_diagnostics(),
        },
    )
    registry.promote(record.version, require_holdout_evaluation=False)
    champion = registry.champion()
    assert champion is not None
    assert champion.version == record.version
    loaded = registry.load_champion()
    assert loaded is not None
    sample = (0.01, 0.3, 0.01, 8.0)
    assert loaded.predict(sample) == model.predict(sample)


def test_registry_can_require_benchmark_and_news_value(tmp_path) -> None:
    model = RidgeReturnModel(FEATURE_NAMES)
    model.fit(synthetic_feature_rows(200))
    metrics = BacktestMetrics(
        observations=500,
        net_return=0.12,
        annualized_return=0.12,
        sharpe=1.1,
        max_drawdown=0.05,
        hit_rate=0.56,
        turnover=20,
        average_trade_return=0.006,
        folds=8,
        scored_observations=1000,
        excess_return_vs_benchmark=-0.01,
        news_sharpe_delta=-0.10,
    )
    registry = ModelRegistry(tmp_path)
    record = registry.register(
        model,
        metrics,
        metadata={
            "dataset": "synthetic_fixture",
            "diagnostics": _governance_diagnostics(),
        },
    )

    with pytest.raises(ValueError) as raised:
        registry.promote(
            record.version,
            minimum_excess_return=0.0,
            minimum_news_sharpe_delta=0.0,
            require_holdout_evaluation=False,
        )

    message = str(raised.value)
    assert "benchmark_excess_return_below_gate" in message
    assert "news_ablation_delta_below_gate" in message
