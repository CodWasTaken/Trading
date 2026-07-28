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
    registry.promote(record.version)
    champion = registry.champion()
    assert champion is not None
    assert champion.version == record.version
    loaded = registry.load_champion()
    assert loaded is not None
    sample = (0.01, 0.3, 0.01, 8.0)
    assert loaded.predict(sample) == model.predict(sample)
