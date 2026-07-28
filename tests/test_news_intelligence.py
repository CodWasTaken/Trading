from datetime import UTC, datetime, timedelta

from trading_app.news_intelligence import NewsIntelligence


def test_news_is_classified_scored_and_deduplicated() -> None:
    intelligence = NewsIntelligence()
    now = datetime.now(UTC)
    first = intelligence.enrich(
        symbol="AAPL",
        headline="Apple raises guidance after record revenue growth",
        source="Reuters",
        event_time=now,
        knowledge_time=now,
    )
    duplicate = intelligence.enrich(
        symbol="AAPL",
        headline="Apple raises guidance after record revenue growth",
        source="Reuters",
        event_time=now + timedelta(minutes=1),
        knowledge_time=now + timedelta(minutes=1),
    )
    similar = intelligence.enrich(
        symbol="AAPL",
        headline="Apple raises annual guidance after strong revenue growth",
        source="Reuters",
        event_time=now + timedelta(minutes=2),
        knowledge_time=now + timedelta(minutes=2),
    )

    assert first is not None
    assert first.event_type == "earnings" or first.event_type == "guidance"
    assert first.sentiment > 0
    assert first.source_quality > 0.9
    assert duplicate is None
    assert similar is not None
    assert similar.novelty < first.novelty


def test_negative_regulatory_catalyst() -> None:
    event = NewsIntelligence().enrich(
        symbol="NVDA",
        headline="Regulator opens antitrust investigation after new probe",
        source="Benzinga",
        event_time=datetime.now(UTC),
    )
    assert event is not None
    assert event.event_type == "regulatory"
    assert event.sentiment < 0
