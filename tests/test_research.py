from trading_app.research import (
    RidgeReturnModel,
    backtest,
    synthetic_feature_rows,
    walk_forward,
)


FEATURE_NAMES = ("momentum", "news_score", "volatility", "spread_bps")


def test_ridge_model_learns_predictive_relationship() -> None:
    rows = synthetic_feature_rows(300)
    model = RidgeReturnModel(FEATURE_NAMES)
    model.fit(rows[:240])
    metrics = backtest(model, rows[240:], transaction_cost_bps=2)
    assert metrics.observations == 60
    assert metrics.net_return > -0.10
    assert model.predict(rows[-1].features) != 0


def test_walk_forward_has_multiple_folds() -> None:
    metrics = walk_forward(
        synthetic_feature_rows(360),
        FEATURE_NAMES,
        minimum_train_rows=120,
        test_rows=40,
        transaction_cost_bps=2,
    )
    assert metrics.folds == 6
    assert metrics.observations == 240
