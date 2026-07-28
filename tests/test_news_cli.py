import json
from datetime import UTC, datetime

from trading_app.domain import NewsEvent
from trading_app.news_cli import reclassify_news


def test_reclassify_preserves_identity_and_point_in_time_fields(tmp_path) -> None:
    source = tmp_path / "news.jsonl"
    destination = tmp_path / "news-v2.jsonl"
    report_path = tmp_path / "report.json"
    event = NewsEvent(
        symbol="AAPL",
        headline="Apple raises guidance after revenue grows 12%",
        source="Reuters",
        sentiment=0.0,
        novelty=1.0,
        source_quality=0.96,
        event_type="earnings",
        event_time=datetime(2025, 1, 2, 14, tzinfo=UTC),
        knowledge_time=datetime(2025, 1, 2, 14, 1, tzinfo=UTC),
    )
    source.write_text(event.model_dump_json() + "\n", encoding="utf-8")

    report = reclassify_news(
        str(source),
        str(destination),
        report_path=str(report_path),
    )
    migrated = NewsEvent.model_validate_json(destination.read_text(encoding="utf-8"))

    assert migrated.id == event.id
    assert migrated.event_time == event.event_time
    assert migrated.knowledge_time == event.knowledge_time
    assert migrated.event_type == "guidance"
    assert "earnings" in migrated.secondary_event_types
    assert migrated.event_attributes["percentage_1"] == 12.0
    assert report["records"] == 1
    assert report["changed_primary_event_type"] == 1
    assert report["point_in_time_controls"]["source_archive_overwritten"] is False
    assert json.loads(report_path.read_text(encoding="utf-8"))["new_other_rate"] == 0.0


def test_reclassify_refuses_to_overwrite_source(tmp_path) -> None:
    source = tmp_path / "news.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    try:
        reclassify_news(str(source), str(source))
    except ValueError as error:
        assert "must differ" in str(error)
    else:
        raise AssertionError("Expected immutable-source protection")
