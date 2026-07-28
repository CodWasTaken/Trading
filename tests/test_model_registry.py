import pytest

from trading_app.model_registry import ModelRegistry
from trading_app.research import BacktestMetrics, RidgeReturnModel, synthetic_feature_rows


FEATURE_NAMES = ("momentum", "news_score", "volatility", "spread_bps")


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
        folds=4,
    )
    registry = ModelRegistry(tmp_path)
    record = registry.register(model, metrics)
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
        folds=5,
        excess_return_vs_benchmark=-0.01,
        news_sharpe_delta=-0.10,
    )
    registry = ModelRegistry(tmp_path)
    record = registry.register(model, metrics)

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
