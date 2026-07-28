import csv
import json
from datetime import UTC, datetime
from uuid import UUID

import pytest

from trading_app.domain import NewsEvent
from trading_app.entities import EntityRelation, LinkedEntity
from trading_app.news_evaluation import (
    LABEL_COLUMNS,
    build_label_set,
    evaluate_label_set,
)


def _event(
    identifier: int,
    *,
    symbol: str,
    event_type: str,
    headline: str,
    secondary: list[str] | None = None,
) -> NewsEvent:
    timestamp = datetime(2025, 1, identifier + 1, 14, tzinfo=UTC)
    return NewsEvent(
        id=UUID(int=identifier),
        symbol=symbol,
        headline=headline,
        summary="Stored point-in-time summary.",
        source="Benzinga",
        sentiment=0.0,
        novelty=1.0,
        source_quality=0.86,
        event_type=event_type,
        secondary_event_types=secondary or [],
        extraction_confidence=0.75,
        entities=[
            LinkedEntity(
                canonical_name=symbol,
                relation=EntityRelation.ISSUER,
                ticker=symbol,
                matched_text=symbol,
                confidence=1.0,
                link_method="provider_symbol",
            )
        ],
        event_time=timestamp,
        knowledge_time=timestamp,
    )


def _write_archive(path, events: list[NewsEvent]) -> None:
    path.write_text(
        "".join(event.model_dump_json() + "\n" for event in events),
        encoding="utf-8",
    )


def test_label_sampling_is_deterministic_and_covers_strata(tmp_path) -> None:
    source = tmp_path / "news.jsonl"
    events = [
        _event(
            index,
            symbol="AAPL" if index % 2 else "MSFT",
            event_type="earnings" if index <= 6 else "other",
            headline=f"Headline {index}",
        )
        for index in range(1, 13)
    ]
    _write_archive(source, events)

    first = tmp_path / "labels-one.csv"
    second = tmp_path / "labels-two.csv"
    report = build_label_set(
        str(source),
        str(first),
        sample_size=8,
        minimum_per_stratum=1,
        seed="fixed-seed",
    )
    build_label_set(
        str(source),
        str(second),
        sample_size=8,
        minimum_per_stratum=1,
        seed="fixed-seed",
    )

    assert first.read_bytes() == second.read_bytes()
    assert report["sampling"]["selected_records"] == 8
    assert set(report["sampling"]["strata"]) >= {
        "earnings|AAPL",
        "earnings|MSFT",
        "other|AAPL",
        "other|MSFT",
    }
    with first.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["review_status"] == "pending" for row in rows)
    assert all(row["record_sha256"] for row in rows)


def test_label_evaluation_reports_primary_secondary_and_entities(tmp_path) -> None:
    source = tmp_path / "news.jsonl"
    events = [
        _event(
            1,
            symbol="AAPL",
            event_type="earnings",
            headline="Apple reports quarterly results",
            secondary=["guidance"],
        ),
        _event(
            2,
            symbol="AAPL",
            event_type="other",
            headline="Apple reports profit",
        ),
        _event(
            3,
            symbol="MSFT",
            event_type="legal",
            headline="Microsoft general update",
        ),
    ]
    _write_archive(source, events)
    labels = tmp_path / "labels.csv"
    build_label_set(
        str(source),
        str(labels),
        sample_size=3,
        minimum_per_stratum=1,
        seed="all-records",
    )

    with labels.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["event_id"]: row for row in rows}
    gold = {
        str(UUID(int=1)): ("earnings", '["guidance"]', by_id[str(UUID(int=1))]["predicted_entities_json"]),
        str(UUID(int=2)): ("earnings", "[]", by_id[str(UUID(int=2))]["predicted_entities_json"]),
        str(UUID(int=3)): ("other", "[]", "[]"),
    }
    for row in rows:
        primary, secondary, entities = gold[row["event_id"]]
        row["gold_primary_event_type"] = primary
        row["gold_secondary_event_types_json"] = secondary
        row["gold_entities_json"] = entities
        row["review_status"] = "labeled"
        row["reviewer"] = "analyst-one"

    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "evaluation.json"
    report = evaluate_label_set(
        str(labels),
        str(output),
        require_complete=True,
    )

    primary = report["primary_event_type"]
    assert primary["accuracy"] == pytest.approx(1 / 3)
    assert primary["per_class"]["earnings"]["precision"] == 1.0
    assert primary["per_class"]["earnings"]["recall"] == 0.5
    assert primary["per_class"]["other"]["recall"] == 0.0
    assert report["secondary_event_types"]["exact_match_rate"] == 1.0
    assert report["entity_links"]["micro_precision"] == pytest.approx(2 / 3)
    assert report["entity_links"]["micro_recall"] == 1.0
    assert report["entity_relations"]["per_class"]["issuer"]["false_positive"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["labels"]["rows"] == 3


def test_label_evaluation_rejects_modified_prediction_columns(tmp_path) -> None:
    source = tmp_path / "news.jsonl"
    _write_archive(
        source,
        [
            _event(
                1,
                symbol="AAPL",
                event_type="earnings",
                headline="Apple reports earnings",
            )
        ],
    )
    labels = tmp_path / "labels.csv"
    build_label_set(str(source), str(labels), sample_size=1)

    with labels.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["headline"] = "Tampered headline"
    rows[0]["gold_primary_event_type"] = "earnings"
    rows[0]["review_status"] = "labeled"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Immutable source columns were modified"):
        evaluate_label_set(str(labels), str(tmp_path / "evaluation.json"))
