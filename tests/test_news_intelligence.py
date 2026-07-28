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
    same_story_other_symbol = intelligence.enrich(
        symbol="MSFT",
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
    assert first.event_type == "guidance"
    assert "earnings" in first.secondary_event_types
    assert first.sentiment > 0
    assert first.source_quality > 0.9
    assert first.extraction_confidence > 0.5
    assert first.entities[0].relation.value == "issuer"
    assert duplicate is None
    assert same_story_other_symbol is not None
    assert same_story_other_symbol.symbol == "MSFT"
    assert similar is not None
    assert similar.novelty < first.novelty
    assert similar.article_version == 2


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
    assert event.event_attributes["impact_direction"] == "negative"


def test_contract_and_numeric_attributes_are_extracted() -> None:
    event = NewsIntelligence().enrich(
        symbol="MSFT",
        headline="Microsoft wins $2.5 billion contract, value rises 12%",
        source="Reuters",
        event_time=datetime.now(UTC),
    )
    assert event is not None
    assert event.event_type == "contract"
    assert event.event_attributes["money_1_value"] == 2.5
    assert event.event_attributes["money_1_unit"] == "billion"
    assert event.event_attributes["percentage_1"] == 12.0
