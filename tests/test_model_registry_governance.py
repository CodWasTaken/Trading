import json

from trading_app.model_registry import REGISTRY_SCHEMA_VERSION, ModelRegistry
from trading_app.research import BacktestMetrics, RidgeReturnModel, synthetic_feature_rows


FEATURE_NAMES = ("momentum", "news_score", "volatility", "spread_bps")


def _model(seed: int) -> RidgeReturnModel:
    model = RidgeReturnModel(FEATURE_NAMES)
    model.fit(synthetic_feature_rows(220, seed=seed))
    return model


def _metrics(net_return: float = 0.12, sharpe: float = 1.1) -> BacktestMetrics:
    return BacktestMetrics(
        observations=600,
        net_return=net_return,
        annualized_return=net_return,
        sharpe=sharpe,
        max_drawdown=0.05,
        hit_rate=0.56,
        turnover=20,
        average_trade_return=0.006,
        folds=6,
        excess_return_vs_benchmark=0.02,
        news_sharpe_delta=0.10,
    )


def test_registration_sets_challenger_and_records_history(tmp_path) -> None:
    registry = ModelRegistry(tmp_path)
    record = registry.register(_model(17), _metrics())

    challenger = registry.challenger()
    assert challenger is not None
    assert challenger.version == record.version
    history = registry.history(alias="challenger")
    assert len(history) == 1
    assert history[0]["action"] == "register_challenger"
    assert history[0]["previous_version"] is None
    assert history[0]["version"] == record.version


def test_promotion_history_and_rollback_restore_previous_champion(tmp_path) -> None:
    registry = ModelRegistry(tmp_path)
    first = registry.register(_model(17), _metrics())
    registry.promote(first.version, reason="first approved candidate")
    second = registry.register(_model(19), _metrics(net_return=0.14, sharpe=1.3))
    registry.promote(second.version, reason="second approved candidate")

    champion = registry.champion()
    assert champion is not None
    assert champion.version == second.version

    restored = registry.rollback(reason="replay regression detected")
    assert restored.version == first.version
    champion = registry.champion()
    assert champion is not None
    assert champion.version == first.version

    champion_history = registry.history(alias="champion")
    assert [event["action"] for event in champion_history] == [
        "promote",
        "promote",
        "rollback",
    ]
    assert champion_history[-1]["details"]["rolled_back_from"] == second.version
    assert champion_history[-1]["reason"] == "replay regression detected"


def test_legacy_registry_is_read_and_upgraded_on_next_write(tmp_path) -> None:
    index = tmp_path / "registry.json"
    index.write_text(json.dumps({"models": {}, "aliases": {}}), encoding="utf-8")

    registry = ModelRegistry(tmp_path)
    summary = registry.summary()
    assert summary["schema_version"] == REGISTRY_SCHEMA_VERSION
    assert summary["alias_history"] == []

    registry.register(_model(23), _metrics())
    persisted = json.loads(index.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == REGISTRY_SCHEMA_VERSION
    assert persisted["aliases"]["challenger"]
    assert persisted["alias_history"][0]["action"] == "register_challenger"


def test_operator_alias_cannot_reference_unknown_model(tmp_path) -> None:
    registry = ModelRegistry(tmp_path)

    try:
        registry.set_alias("candidate", "missing", reason="test")
    except KeyError as error:
        assert "Unknown model version" in str(error)
    else:
        raise AssertionError("Expected an unknown model alias target to be rejected")
