import csv
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from trading_app.domain import NewsEvent
from trading_app.entities import EntityRelation, LinkedEntity
from trading_app.news_evaluation import LABEL_COLUMNS, build_label_set
from trading_app.news_review import (
    adjudicate_label_sets,
    assign_reviewer_packets,
    review_agreement,
)


def _event(identifier: int, *, symbol: str, event_type: str) -> NewsEvent:
    timestamp = datetime(2025, 1, identifier + 1, 14, tzinfo=UTC)
    return NewsEvent(
        id=UUID(int=identifier),
        symbol=symbol,
        headline=f"Stored headline {identifier}",
        summary="Stored point-in-time summary.",
        source="Benzinga",
        sentiment=0.0,
        novelty=1.0,
        source_quality=0.86,
        event_type=event_type,
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


def _template(tmp_path: Path, count: int = 9) -> Path:
    archive = tmp_path / "news.jsonl"
    events = [
        _event(
            index,
            symbol=("AAPL", "MSFT", "NVDA")[index % 3],
            event_type="earnings" if index % 2 else "other",
        )
        for index in range(1, count + 1)
    ]
    archive.write_text(
        "".join(event.model_dump_json() + "\n" for event in events),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    build_label_set(
        str(archive),
        str(labels),
        sample_size=count,
        minimum_per_stratum=1,
        seed="all-events",
    )
    return labels


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _complete_packet(path: Path, labels: dict[str, str]) -> None:
    rows = _read_rows(path)
    for row in rows:
        row["gold_primary_event_type"] = labels[row["event_id"]]
        row["gold_secondary_event_types_json"] = "[]"
        row["gold_entities_json"] = row["predicted_entities_json"]
        row["review_status"] = "labeled"
    _write_rows(path, rows)


def test_reviewer_assignment_is_balanced_deterministic_and_exact(tmp_path) -> None:
    labels = _template(tmp_path)
    reviewers = ["analyst-a", "analyst-b", "analyst-c"]

    first_report = assign_reviewer_packets(
        str(labels),
        str(tmp_path / "packets-one"),
        reviewers,
        reviews_per_item=2,
        seed="fixed-assignment",
    )
    second_report = assign_reviewer_packets(
        str(labels),
        str(tmp_path / "packets-two"),
        reviewers,
        reviews_per_item=2,
        seed="fixed-assignment",
    )

    occurrences: Counter[str] = Counter()
    packet_sizes: list[int] = []
    for reviewer in reviewers:
        first = Path(first_report["packets"][reviewer]["path"])
        second = Path(second_report["packets"][reviewer]["path"])
        assert first.read_bytes() == second.read_bytes()
        rows = _read_rows(first)
        packet_sizes.append(len(rows))
        assert {row["reviewer"] for row in rows} == {reviewer}
        occurrences.update(row["event_id"] for row in rows)

    assert set(occurrences.values()) == {2}
    assert max(packet_sizes) - min(packet_sizes) <= 1
    assert first_report["assignment"]["every_item_assigned_exactly"] == 2


def test_agreement_and_adjudication_preserve_disputes(tmp_path) -> None:
    labels = _template(tmp_path, count=3)
    assignment = assign_reviewer_packets(
        str(labels),
        str(tmp_path / "packets"),
        ["analyst-a", "analyst-b"],
        reviews_per_item=2,
        seed="full-overlap",
    )
    left = Path(assignment["packets"]["analyst-a"]["path"])
    right = Path(assignment["packets"]["analyst-b"]["path"])
    event_ids = [row["event_id"] for row in _read_rows(left)]

    _complete_packet(
        left,
        {
            event_ids[0]: "earnings",
            event_ids[1]: "earnings",
            event_ids[2]: "other",
        },
    )
    _complete_packet(
        right,
        {
            event_ids[0]: "earnings",
            event_ids[1]: "other",
            event_ids[2]: "other",
        },
    )

    agreement = review_agreement(
        [str(left), str(right)],
        str(tmp_path / "agreement.json"),
        require_complete=True,
    )
    primary = agreement["pairwise"]["analyst-a|analyst-b"]["primary_event_type"]
    assert primary["observed_agreement"] == pytest.approx(2 / 3)
    assert primary["cohen_kappa"] == pytest.approx(0.4)
    assert agreement["coverage"]["unanimous_primary_events"] == 2
    assert agreement["coverage"]["disputed_primary_events"] == 1

    adjudication = adjudicate_label_sets(
        [str(left), str(right)],
        str(tmp_path / "consensus.csv"),
        minimum_reviewers=2,
    )
    consensus_rows = _read_rows(tmp_path / "consensus.csv")
    by_id = {row["event_id"]: row for row in consensus_rows}
    assert by_id[event_ids[0]]["review_status"] == "labeled"
    assert by_id[event_ids[0]]["gold_primary_event_type"] == "earnings"
    assert by_id[event_ids[1]]["review_status"] == "pending"
    assert by_id[event_ids[1]]["gold_primary_event_type"] == ""
    assert "Primary disagreement" in by_id[event_ids[1]]["notes"]
    assert by_id[event_ids[2]]["review_status"] == "labeled"
    assert by_id[event_ids[2]]["gold_primary_event_type"] == "other"
    assert adjudication["outcomes"]["primary_consensus"] == 2
    assert adjudication["outcomes"]["primary_disagreement"] == 1


def test_agreement_rejects_tampered_source_columns(tmp_path) -> None:
    labels = _template(tmp_path, count=2)
    assignment = assign_reviewer_packets(
        str(labels),
        str(tmp_path / "packets"),
        ["analyst-a", "analyst-b"],
        reviews_per_item=2,
    )
    left = Path(assignment["packets"]["analyst-a"]["path"])
    right = Path(assignment["packets"]["analyst-b"]["path"])
    for path in (left, right):
        rows = _read_rows(path)
        for row in rows:
            row["gold_primary_event_type"] = "earnings"
            row["review_status"] = "labeled"
        _write_rows(path, rows)

    tampered = _read_rows(right)
    tampered[0]["headline"] = "Changed after assignment"
    _write_rows(right, tampered)

    with pytest.raises(ValueError, match="Immutable source columns were modified"):
        review_agreement([str(left), str(right)], str(tmp_path / "agreement.json"))
