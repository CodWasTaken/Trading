import json
from datetime import UTC, datetime

import pytest

from trading_app.entities import EntityCatalog, EntityRelation


def write_catalog(tmp_path):
    path = tmp_path / "entities.json"
    path.write_text(
        json.dumps(
            {
                "issuers": [
                    {
                        "ticker": "ACME",
                        "canonical_name": "Acme Holdings",
                        "aliases": ["Acme"],
                        "sector": "Industrials",
                        "related_entities": [
                            {
                                "canonical_name": "Precision Parts Ltd",
                                "relation": "supplier",
                                "aliases": ["Precision Parts"],
                                "ticker": "PPL",
                                "effective_from": "2025-07-01T00:00:00Z",
                            },
                            {
                                "canonical_name": "Roadrunner Systems",
                                "relation": "competitor",
                                "aliases": ["Roadrunner"],
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_catalog_links_only_explicit_complete_aliases(tmp_path) -> None:
    catalog = EntityCatalog.load(write_catalog(tmp_path))
    entities = catalog.link(
        "ACME",
        "Acme signs a supply agreement with Precision Parts; a pineapple vendor is unrelated.",
        as_of=datetime(2025, 8, 1, tzinfo=UTC),
    )

    assert entities[0].relation == EntityRelation.ISSUER
    assert entities[0].canonical_name == "Acme Holdings"
    assert entities[0].sector == "Industrials"
    assert [(entity.canonical_name, entity.relation) for entity in entities[1:]] == [
        ("Precision Parts Ltd", EntityRelation.SUPPLIER)
    ]


def test_catalog_filters_relationships_by_point_in_time(tmp_path) -> None:
    catalog = EntityCatalog.load(write_catalog(tmp_path))
    text = "Acme signs a supply agreement with Precision Parts."

    before = catalog.link(
        "ACME",
        text,
        as_of=datetime(2025, 6, 30, 23, 59, tzinfo=UTC),
    )
    active = catalog.link(
        "ACME",
        text,
        as_of=datetime(2025, 7, 1, tzinfo=UTC),
    )

    assert [entity.relation for entity in before] == [EntityRelation.ISSUER]
    assert [entity.relation for entity in active] == [
        EntityRelation.ISSUER,
        EntityRelation.SUPPLIER,
    ]
    assert catalog.summary()["dated_relationships"] == 1


def test_catalog_rejects_invalid_effective_window(tmp_path) -> None:
    path = tmp_path / "invalid-window.json"
    path.write_text(
        json.dumps(
            {
                "issuers": [
                    {
                        "ticker": "ACME",
                        "canonical_name": "Acme",
                        "related_entities": [
                            {
                                "canonical_name": "Vendor",
                                "relation": "supplier",
                                "effective_from": "2026-01-01T00:00:00Z",
                                "effective_to": "2025-01-01T00:00:00Z",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="effective_from must be earlier"):
        EntityCatalog.load(path)


def test_catalog_rejects_ambiguous_aliases(tmp_path) -> None:
    path = tmp_path / "ambiguous.json"
    path.write_text(
        json.dumps(
            {
                "issuers": [
                    {
                        "ticker": "ACME",
                        "canonical_name": "Acme",
                        "related_entities": [
                            {
                                "canonical_name": "First Vendor",
                                "relation": "supplier",
                                "aliases": ["Shared Name"],
                            },
                            {
                                "canonical_name": "Second Vendor",
                                "relation": "competitor",
                                "aliases": ["Shared Name"],
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Ambiguous alias"):
        EntityCatalog.load(path)
