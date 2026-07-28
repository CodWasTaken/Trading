from datetime import UTC, datetime, timedelta

from trading_app.dataset import PointInTimeDatasetBuilder
from trading_app.domain import NewsEvent, Quote


def test_dataset_uses_news_knowledge_time_without_leakage() -> None:
    start = datetime(2026, 1, 2, 14, 30, tzinfo=UTC)
    quotes = [
        Quote(
            symbol="AAPL",
            bid=100 + index - 0.01,
            ask=100 + index + 0.01,
            event_time=start + timedelta(minutes=index),
            knowledge_time=start + timedelta(minutes=index),
        )
        for index in range(10)
    ]
    news = [
        NewsEvent(
            symbol="AAPL",
            headline="Apple raises guidance",
            source="Reuters",
            sentiment=0.8,
            novelty=1.0,
            source_quality=0.95,
            event_type="guidance",
            event_time=start + timedelta(minutes=1),
            # The article happened early but the system did not learn it until minute 6.
            knowledge_time=start + timedelta(minutes=6),
        )
    ]
    rows = PointInTimeDatasetBuilder(
        lookback_quotes=3,
        forecast_quotes=1,
    ).build(quotes, news)

    before_receipt = next(row for row in rows if row.timestamp == start + timedelta(minutes=5))
    after_receipt = next(row for row in rows if row.timestamp == start + timedelta(minutes=6))
    assert before_receipt.features[1] == 0.0
    assert after_receipt.features[1] > 0.0
    assert after_receipt.target_return > 0
