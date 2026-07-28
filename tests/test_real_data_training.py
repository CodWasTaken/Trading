from datetime import UTC, datetime, timedelta

from trading_app.cli import build_dataset, train_dataset
from trading_app.dataset import (
    HISTORICAL_FEATURE_NAMES,
    HistoricalPointInTimeDatasetBuilder,
    load_feature_dataset,
    write_feature_dataset,
)
from trading_app.domain import NewsEvent, Quote
from trading_app.historical import HistoricalBar
from trading_app.model_strategy import ChampionModelStrategy
from trading_app.research import FeatureRow, RidgeReturnModel, walk_forward


def make_bars(
    symbol: str, start: datetime, count: int, offset: float = 0.0
) -> list[HistoricalBar]:
    result = []
    for index in range(count):
        price = 100 + offset + index * 0.05 + ((index % 5) - 2) * 0.01
        result.append(
            HistoricalBar(
                symbol=symbol,
                timestamp=start + timedelta(hours=index),
                open=price - 0.02,
                high=price + 0.10,
                low=price - 0.10,
                close=price,
                volume=1000,
            )
        )
    return result


def test_historical_dataset_uses_bar_close_time_and_news_knowledge_time(
    tmp_path,
) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    bars = make_bars("AAPL", start, 40)
    news = [
        NewsEvent(
            symbol="AAPL",
            headline="Apple raises guidance",
            source="Reuters",
            sentiment=0.8,
            novelty=1.0,
            source_quality=0.95,
            event_time=start + timedelta(hours=5),
            knowledge_time=start + timedelta(hours=8, minutes=30),
        )
    ]
    rows = HistoricalPointInTimeDatasetBuilder(
        lookback_bars=5,
        forecast_bars=2,
        bar_interval=timedelta(hours=1),
    ).build(bars, news)

    before = next(row for row in rows if row.timestamp == start + timedelta(hours=8))
    after = next(row for row in rows if row.timestamp == start + timedelta(hours=9))
    assert before.features[1] == 0.0
    assert after.features[1] > 0.0
    assert after.label_end_time == start + timedelta(hours=11)

    dataset, _ = write_feature_dataset(
        tmp_path / "features.jsonl",
        rows,
        HISTORICAL_FEATURE_NAMES,
        {"dataset_kind": "historical_point_in_time"},
    )
    loaded, feature_names, metadata = load_feature_dataset(dataset)
    assert loaded == rows
    assert feature_names == HISTORICAL_FEATURE_NAMES
    assert metadata["rows"] == len(rows)


def test_walk_forward_purges_training_labels_overlapping_test() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [
        FeatureRow(
            timestamp=start + timedelta(hours=index),
            symbol="AAPL",
            features=(index / 1000, 0.0, 0.01),
            target_return=0.001 if index % 2 == 0 else -0.0002,
            label_end_time=start + timedelta(hours=index + 5),
        )
        for index in range(180)
    ]
    metrics = walk_forward(
        rows,
        HISTORICAL_FEATURE_NAMES,
        minimum_train_rows=60,
        test_rows=20,
        periods_per_year=1638,
    )
    assert metrics.folds == 6
    assert metrics.observations == 120
    assert metrics.purged_rows == 30


def test_champion_accepts_historical_feature_subset() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [
        FeatureRow(
            timestamp=start + timedelta(hours=index),
            symbol="AAPL",
            features=(index / 1000, 0.1, 0.01),
            target_return=index / 100000,
        )
        for index in range(50)
    ]
    model = RidgeReturnModel(HISTORICAL_FEATURE_NAMES)
    model.fit(rows)
    strategy = ChampionModelStrategy(model)
    for index in range(20):
        strategy.on_quote(
            Quote(
                symbol="AAPL",
                bid=100 + index * 0.01,
                ask=100.02 + index * 0.01,
                event_time=start + timedelta(minutes=index),
                knowledge_time=start + timedelta(minutes=index),
            ),
            [],
            100_000,
        )


def test_cli_builds_and_trains_real_dataset(tmp_path) -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    bars = make_bars("AAPL", start, 140) + make_bars("MSFT", start, 140, 20)
    news = [
        NewsEvent(
            symbol="AAPL",
            headline="Apple raises guidance",
            source="Reuters",
            sentiment=0.8,
            novelty=1.0,
            source_quality=0.95,
            event_time=start + timedelta(hours=30),
            knowledge_time=start + timedelta(hours=30),
        )
    ]
    bars_path = tmp_path / "bars.jsonl"
    news_path = tmp_path / "news.jsonl"
    bars_path.write_text("".join(bar.model_dump_json() + "\n" for bar in bars))
    news_path.write_text("".join(item.model_dump_json() + "\n" for item in news))

    dataset_path = tmp_path / "dataset.jsonl"
    built = build_dataset(
        str(bars_path),
        str(news_path),
        str(dataset_path),
        lookback_bars=19,
        forecast_bars=5,
        bar_minutes=60,
        news_window_hours=24,
    )
    assert built["rows"] == 232

    result = train_dataset(
        str(dataset_path),
        str(tmp_path / "models"),
        promote=False,
        minimum_train_rows=80,
        test_rows=40,
        transaction_cost_bps=5.0,
        periods_per_year=1638,
        threshold=0.0005,
        ridge=0.001,
        minimum_folds=3,
        minimum_sharpe=0.0,
        maximum_drawdown=0.20,
        minimum_observations=100,
    )
    assert result["promoted"] is False
    assert result["rows"] == 232
    assert result["metrics"]["purged_rows"] > 0
