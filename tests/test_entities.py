import json

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
    )

    assert entities[0].relation == EntityRelation.ISSUER
    assert entities[0].canonical_name == "Acme Holdings"
    assert entities[0].sector == "Industrials"
    assert [(entity.canonical_name, entity.relation) for entity in entities[1:]] == [
        ("Precision Parts Ltd", EntityRelation.SUPPLIER)
    ]


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
