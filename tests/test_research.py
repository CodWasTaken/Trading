from trading_app.cost_model import CostModelConfig, ExecutionCostConfig
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


def test_explicit_conservative_costs_reduce_pre_tax_strategy_return() -> None:
    rows = synthetic_feature_rows(300)
    model = RidgeReturnModel(FEATURE_NAMES)
    model.fit(rows[:240])
    free = backtest(model, rows[240:], transaction_cost_bps=0)
    stressed = backtest(
        model,
        rows[240:],
        cost_model=CostModelConfig(
            execution=ExecutionCostConfig(
                observed_spread_bps=20,
                slippage_bps=5,
                commission_bps=1,
                regulatory_sell_fee_bps=0.5,
                market_impact_stress_bps=5,
            )
        ),
    )
    assert stressed.net_return < free.net_return
    assert stressed.execution_cost_return > 0
