from datetime import UTC, datetime

from trading_app.entities import EntityCatalog, IssuerProfile, RelatedEntity
from trading_app.point_in_time_news import PointInTimeNewsIntelligence


def test_news_links_relationships_only_after_effective_time() -> None:
    catalog = EntityCatalog(
        [
            IssuerProfile(
                ticker="ACME",
                canonical_name="Acme Holdings",
                related_entities=[
                    RelatedEntity(
                        canonical_name="Precision Parts Ltd",
                        relation="supplier",
                        aliases=["Precision Parts"],
                        effective_from=datetime(2025, 7, 1, tzinfo=UTC),
                    )
                ],
            )
        ]
    )
    intelligence = PointInTimeNewsIntelligence(
        entity_catalog=catalog,
        drop_exact_duplicates=False,
    )
    headline = "Acme signs a supply agreement with Precision Parts"

    before = intelligence.enrich(
        symbol="ACME",
        headline=headline,
        source="Reuters",
        event_time=datetime(2025, 6, 30, tzinfo=UTC),
        knowledge_time=datetime(2025, 6, 30, tzinfo=UTC),
    )
    after = intelligence.enrich(
        symbol="ACME",
        headline=headline,
        source="Reuters",
        event_time=datetime(2025, 7, 1, tzinfo=UTC),
        knowledge_time=datetime(2025, 7, 1, tzinfo=UTC),
    )

    assert before is not None
    assert after is not None
    assert [entity.relation.value for entity in before.entities] == ["issuer"]
    assert [entity.relation.value for entity in after.entities] == ["issuer", "supplier"]
