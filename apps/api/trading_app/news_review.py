from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from .news_evaluation import (
    EVENT_TYPES,
    LABEL_COLUMNS,
    _load_label_rows,
    _parse_entities,
    _parse_event_type_list,
    _sha256,
    _validate_event_type,
    _validate_source_hash,
)

_REVIEW_STATUSES = frozenset({"pending", "labeled", "skip"})
_SLUG = re.compile(r"[^a-z0-9]+")


def assign_reviewer_packets(
    labels_path: str,
    output_dir: str,
    reviewers: list[str],
    *,
    reviews_per_item: int = 2,
    seed: str = "news-review-assignment-v1",
    report_path: str | None = None,
) -> dict[str, object]:
    source = Path(labels_path)
    rows = _load_label_rows(source)
    reviewer_names = _normalize_reviewers(reviewers)
    if reviews_per_item < 1 or reviews_per_item > len(reviewer_names):
        raise ValueError(
            "reviews_per_item must be between 1 and the number of reviewers"
        )

    for line_number, row in enumerate(rows, start=2):
        _validate_source_hash(row, source, line_number)
        status = _status(row, source, line_number)
        if status != "pending" or any(
            row[field].strip()
            for field in (
                "gold_primary_event_type",
                "gold_secondary_event_types_json",
                "gold_entities_json",
            )
        ):
            raise ValueError(
                "Reviewer packets must be created from an untouched pending label template; "
                f"found reviewed data at {source}:{line_number}"
            )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    slugs = {reviewer: _reviewer_slug(reviewer) for reviewer in reviewer_names}
    if len(set(slugs.values())) != len(slugs):
        raise ValueError("Reviewer names produce duplicate filename slugs")

    packet_paths = {
        reviewer: destination / f"{source.stem}.{slugs[reviewer]}.csv"
        for reviewer in reviewer_names
    }
    report_destination = (
        Path(report_path) if report_path else destination / "review-assignments.json"
    )
    existing = [str(path) for path in [*packet_paths.values(), report_destination] if path.exists()]
    if existing:
        raise ValueError(
            "Reviewer assignment outputs already exist; choose a new directory or remove: "
            + ", ".join(existing)
        )

    assignments: dict[str, list[dict[str, str]]] = {
        reviewer: [] for reviewer in reviewer_names
    }
    counts: Counter[str] = Counter()
    ranked_rows = sorted(
        rows,
        key=lambda row: _rank(seed, row["event_id"], "row"),
    )
    event_assignments: dict[str, list[str]] = {}
    for row in ranked_rows:
        ordered_reviewers = sorted(
            reviewer_names,
            key=lambda reviewer: (
                counts[reviewer],
                _rank(seed, row["event_id"], reviewer),
                reviewer,
            ),
        )
        selected = ordered_reviewers[:reviews_per_item]
        event_assignments[row["event_id"]] = selected
        for reviewer in selected:
            assigned = dict(row)
            assigned["reviewer"] = reviewer
            assignments[reviewer].append(assigned)
            counts[reviewer] += 1

    for reviewer, path in packet_paths.items():
        _write_rows(path, assignments[reviewer])

    pair_overlap: Counter[str] = Counter()
    for selected in event_assignments.values():
        for left, right in combinations(sorted(selected), 2):
            pair_overlap[f"{left}|{right}"] += 1

    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "news_review_assignment",
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "rows": len(rows),
        },
        "assignment": {
            "seed": seed,
            "reviewers": reviewer_names,
            "reviews_per_item": reviews_per_item,
            "rows_per_reviewer": dict(sorted(counts.items())),
            "pair_overlap": dict(sorted(pair_overlap.items())),
            "every_item_assigned_exactly": reviews_per_item,
            "method": "deterministic_least-loaded_assignment_with_hash_tiebreak",
        },
        "packets": {
            reviewer: {
                "path": str(packet_paths[reviewer]),
                "rows": len(assignments[reviewer]),
                "sha256": _sha256(packet_paths[reviewer]),
            }
            for reviewer in reviewer_names
        },
        "point_in_time_controls": {
            "immutable_source_columns_verified": True,
            "event_time_preserved": True,
            "knowledge_time_preserved": True,
            "assignment_does_not_use_future_returns": True,
        },
        "report": str(report_destination),
    }
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def review_agreement(
    label_paths: list[str],
    output_path: str,
    *,
    require_complete: bool = False,
) -> dict[str, object]:
    reviews = _load_review_files(label_paths)
    if len(reviews) < 2:
        raise ValueError("At least two independent reviewer files are required")

    pairwise: dict[str, object] = {}
    for left, right in combinations(sorted(reviews), 2):
        left_rows = reviews[left]
        right_rows = reviews[right]
        common = sorted(set(left_rows) & set(right_rows))
        if not common:
            raise ValueError(f"Reviewers {left!r} and {right!r} have no overlapping rows")

        if require_complete:
            incomplete = [
                event_id
                for event_id in common
                if _status(left_rows[event_id], Path(label_paths[0]), 0) != "labeled"
                or _status(right_rows[event_id], Path(label_paths[0]), 0) != "labeled"
            ]
            if incomplete:
                raise ValueError(
                    f"Review pair {left!r}/{right!r} has {len(incomplete)} overlapping rows "
                    "that are not labeled"
                )

        primary_pairs: list[tuple[str, str]] = []
        secondary_pairs: list[tuple[set[str], set[str]]] = []
        entity_pairs: list[tuple[set[tuple[str, str]], set[tuple[str, str]]]] = []
        relation_pairs: list[tuple[set[str], set[str]]] = []
        overlap_status: Counter[str] = Counter()
        for event_id in common:
            left_row = left_rows[event_id]
            right_row = right_rows[event_id]
            left_status = _status(left_row, Path(label_paths[0]), 0)
            right_status = _status(right_row, Path(label_paths[0]), 0)
            overlap_status[f"{left_status}|{right_status}"] += 1
            if left_status != "labeled" or right_status != "labeled":
                continue

            left_primary = left_row["gold_primary_event_type"].strip()
            right_primary = right_row["gold_primary_event_type"].strip()
            _validate_event_type(left_primary, Path(label_paths[0]), 0, "gold_primary_event_type")
            _validate_event_type(right_primary, Path(label_paths[0]), 0, "gold_primary_event_type")
            primary_pairs.append((left_primary, right_primary))

            left_secondary = left_row["gold_secondary_event_types_json"].strip()
            right_secondary = right_row["gold_secondary_event_types_json"].strip()
            if left_secondary and right_secondary:
                secondary_pairs.append(
                    (
                        set(
                            _parse_event_type_list(
                                left_secondary,
                                Path(label_paths[0]),
                                0,
                                "gold_secondary_event_types_json",
                            )
                        ),
                        set(
                            _parse_event_type_list(
                                right_secondary,
                                Path(label_paths[0]),
                                0,
                                "gold_secondary_event_types_json",
                            )
                        ),
                    )
                )

            left_entities = left_row["gold_entities_json"].strip()
            right_entities = right_row["gold_entities_json"].strip()
            if left_entities and right_entities:
                left_links, left_relations = _parse_entities(
                    left_entities,
                    Path(label_paths[0]),
                    0,
                    "gold_entities_json",
                )
                right_links, right_relations = _parse_entities(
                    right_entities,
                    Path(label_paths[0]),
                    0,
                    "gold_entities_json",
                )
                entity_pairs.append((left_links, right_links))
                relation_pairs.append((left_relations, right_relations))

        pair_key = f"{left}|{right}"
        pairwise[pair_key] = {
            "reviewers": [left, right],
            "overlap_rows": len(common),
            "overlap_status": dict(sorted(overlap_status.items())),
            "primary_event_type": _primary_agreement(primary_pairs),
            "secondary_event_types": _set_agreement(secondary_pairs),
            "entity_links": _set_agreement(entity_pairs),
            "entity_relations": _set_agreement(relation_pairs),
        }

    event_labels: dict[str, dict[str, str]] = defaultdict(dict)
    event_presence: Counter[int] = Counter()
    for reviewer, rows in reviews.items():
        for event_id, row in rows.items():
            event_presence[event_id] = 0
            if _status(row, Path(label_paths[0]), 0) == "labeled":
                event_labels[event_id][reviewer] = row["gold_primary_event_type"].strip()
    for event_id in event_presence:
        event_presence[event_id] = sum(event_id in rows for rows in reviews.values())

    labeled_overlap = {
        event_id: labels for event_id, labels in event_labels.items() if len(labels) >= 2
    }
    unanimous = sum(len(set(labels.values())) == 1 for labels in labeled_overlap.values())
    disputed = len(labeled_overlap) - unanimous
    coverage = Counter(event_presence.values())

    destination = Path(output_path)
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "news_inter_annotator_agreement",
        "reviewers": sorted(reviews),
        "files": {
            reviewer: {
                "path": str(_review_file_path(label_paths, reviewer, reviews)),
                "rows": len(rows),
            }
            for reviewer, rows in sorted(reviews.items())
        },
        "coverage": {
            "unique_events": len(event_presence),
            "files_per_event": {str(key): value for key, value in sorted(coverage.items())},
            "events_with_at_least_two_labels": len(labeled_overlap),
            "unanimous_primary_events": unanimous,
            "disputed_primary_events": disputed,
            "unanimous_primary_rate": unanimous / len(labeled_overlap)
            if labeled_overlap
            else 0.0,
        },
        "pairwise": pairwise,
        "point_in_time_controls": {
            "immutable_source_columns_verified": True,
            "agreement_uses_only_independent_human_labels": True,
            "future_returns_not_used": True,
        },
        "limitations": [
            "Pairwise Cohen's kappa is reported only on rows labeled by both reviewers.",
            "Agreement is not classifier accuracy; adjudicated gold labels are still required.",
            "Optional secondary and entity agreement excludes rows where either reviewer left the field blank.",
        ],
        "output": str(destination),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def adjudicate_label_sets(
    label_paths: list[str],
    output_path: str,
    *,
    minimum_reviewers: int = 2,
    report_path: str | None = None,
) -> dict[str, object]:
    if minimum_reviewers < 2:
        raise ValueError("minimum_reviewers must be at least 2")
    reviews = _load_review_files(label_paths)
    if len(reviews) < minimum_reviewers:
        raise ValueError("Not enough independent reviewer files for adjudication")

    order: list[str] = []
    grouped: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    canonical: dict[str, dict[str, str]] = {}
    for reviewer, rows in reviews.items():
        for event_id, row in rows.items():
            if event_id not in canonical:
                canonical[event_id] = dict(row)
                order.append(event_id)
            elif canonical[event_id]["record_sha256"] != row["record_sha256"]:
                raise ValueError(
                    f"Reviewer files contain different immutable data for event {event_id}"
                )
            grouped[event_id].append((reviewer, row))

    counts: Counter[str] = Counter()
    output_rows: list[dict[str, str]] = []
    for event_id in order:
        result = dict(canonical[event_id])
        candidates = grouped[event_id]
        labeled = [
            (reviewer, row)
            for reviewer, row in candidates
            if _status(row, Path(label_paths[0]), 0) == "labeled"
        ]
        skipped = [
            reviewer
            for reviewer, row in candidates
            if _status(row, Path(label_paths[0]), 0) == "skip"
        ]

        result["gold_primary_event_type"] = ""
        result["gold_secondary_event_types_json"] = ""
        result["gold_entities_json"] = ""
        result["reviewer"] = ""
        result["notes"] = ""

        if len(skipped) >= minimum_reviewers and not labeled:
            result["review_status"] = "skip"
            result["reviewer"] = "consensus:" + ",".join(sorted(skipped))
            result["notes"] = "Auto-adjudicated: at least the required reviewers marked this row skip."
            counts["consensus_skip"] += 1
            output_rows.append(result)
            continue

        if len(labeled) < minimum_reviewers:
            result["review_status"] = "pending"
            result["notes"] = (
                f"Adjudication pending: {len(labeled)} labeled reviews; "
                f"minimum required is {minimum_reviewers}."
            )
            counts["insufficient_reviews"] += 1
            output_rows.append(result)
            continue

        primary_by_reviewer = {
            reviewer: row["gold_primary_event_type"].strip()
            for reviewer, row in labeled
        }
        for primary in primary_by_reviewer.values():
            _validate_event_type(primary, Path(label_paths[0]), 0, "gold_primary_event_type")
        primary_values = set(primary_by_reviewer.values())
        if len(primary_values) != 1:
            result["review_status"] = "pending"
            result["reviewer"] = "adjudication-required"
            result["notes"] = "Primary disagreement: " + "; ".join(
                f"{reviewer}={label}"
                for reviewer, label in sorted(primary_by_reviewer.items())
            )
            counts["primary_disagreement"] += 1
            output_rows.append(result)
            continue

        result["gold_primary_event_type"] = next(iter(primary_values))
        result["gold_secondary_event_types_json"], secondary_consensus = _optional_consensus(
            labeled,
            "gold_secondary_event_types_json",
            parse_kind="event_types",
        )
        result["gold_entities_json"], entity_consensus = _optional_consensus(
            labeled,
            "gold_entities_json",
            parse_kind="entities",
        )
        result["review_status"] = "labeled"
        result["reviewer"] = "consensus:" + ",".join(
            sorted(reviewer for reviewer, _ in labeled)
        )
        notes = [
            f"Auto-adjudicated unanimous primary label from {len(labeled)} reviewers."
        ]
        if not secondary_consensus:
            notes.append("Secondary labels were blank or disagreed and require separate review.")
            counts["secondary_not_consensus"] += 1
        if not entity_consensus:
            notes.append("Entity labels were blank or disagreed and require separate review.")
            counts["entities_not_consensus"] += 1
        result["notes"] = " ".join(notes)
        counts["primary_consensus"] += 1
        output_rows.append(result)

    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"Adjudicated output already exists: {destination}")
    _write_rows(destination, output_rows)

    report_destination = (
        Path(report_path)
        if report_path
        else destination.with_suffix(".adjudication.json")
    )
    if report_destination.exists():
        raise ValueError(f"Adjudication report already exists: {report_destination}")
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "news_label_adjudication",
        "inputs": {
            reviewer: {
                "path": str(_review_file_path(label_paths, reviewer, reviews)),
                "rows": len(rows),
            }
            for reviewer, rows in sorted(reviews.items())
        },
        "minimum_reviewers": minimum_reviewers,
        "records": len(output_rows),
        "outcomes": dict(sorted(counts.items())),
        "output": {
            "path": str(destination),
            "sha256": _sha256(destination),
        },
        "report": str(report_destination),
        "controls": {
            "primary_labels_auto_accepted_only_when_unanimous": True,
            "disagreements_remain_pending": True,
            "immutable_source_columns_preserved": True,
            "optional_fields_require_unanimous_nonblank_values": True,
        },
    }
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_review_files(
    label_paths: list[str],
) -> dict[str, dict[str, dict[str, str]]]:
    reviews: dict[str, dict[str, dict[str, str]]] = {}
    for raw_path in label_paths:
        path = Path(raw_path)
        rows = _load_label_rows(path)
        reviewer_values: set[str] = set()
        indexed: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(rows, start=2):
            _validate_source_hash(row, path, line_number)
            _status(row, path, line_number)
            reviewer = row["reviewer"].strip()
            if reviewer:
                reviewer_values.add(reviewer)
            event_id = row["event_id"].strip()
            if event_id in indexed:
                raise ValueError(f"Duplicate event_id {event_id} in {path}")
            indexed[event_id] = row
        if len(reviewer_values) != 1:
            raise ValueError(
                f"Each review file must contain exactly one non-empty reviewer name; "
                f"found {sorted(reviewer_values)} in {path}"
            )
        reviewer_name = next(iter(reviewer_values))
        if reviewer_name in reviews:
            raise ValueError(f"Duplicate reviewer name across files: {reviewer_name}")
        reviews[reviewer_name] = indexed
    return reviews


def _review_file_path(
    label_paths: list[str],
    reviewer: str,
    reviews: dict[str, dict[str, dict[str, str]]],
) -> Path:
    for raw_path in label_paths:
        path = Path(raw_path)
        rows = _load_label_rows(path)
        names = {row["reviewer"].strip() for row in rows if row["reviewer"].strip()}
        if names == {reviewer}:
            return path
    raise RuntimeError(f"Unable to resolve review file for {reviewer}")


def _primary_agreement(pairs: list[tuple[str, str]]) -> dict[str, object]:
    if not pairs:
        return {
            "rows": 0,
            "status": "not_evaluated",
            "reason": "No overlapping rows were labeled by both reviewers",
        }
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    matches = 0
    for left, right in pairs:
        confusion[left][right] += 1
        matches += left == right
    total = len(pairs)
    observed = matches / total
    expected = sum(
        (left_counts[label] / total) * (right_counts[label] / total)
        for label in EVENT_TYPES
    )
    denominator = 1.0 - expected
    kappa = (observed - expected) / denominator if denominator else 1.0
    return {
        "rows": total,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "cohen_kappa": kappa,
        "matches": matches,
        "confusion": {
            label: dict(sorted(values.items()))
            for label, values in sorted(confusion.items())
        },
    }


def _set_agreement(pairs: list[tuple[set[Any], set[Any]]]) -> dict[str, object]:
    if not pairs:
        return {
            "rows": 0,
            "status": "not_evaluated",
            "reason": "No overlapping rows supplied labels from both reviewers",
        }
    intersections = 0
    total_items = 0
    exact = 0
    jaccard_total = 0.0
    for left, right in pairs:
        intersections += len(left & right)
        total_items += len(left) + len(right)
        exact += left == right
        union = left | right
        jaccard_total += len(left & right) / len(union) if union else 1.0
    micro_f1 = 2 * intersections / total_items if total_items else 1.0
    return {
        "rows": len(pairs),
        "exact_match_rate": exact / len(pairs),
        "mean_jaccard": jaccard_total / len(pairs),
        "micro_f1": micro_f1,
        "shared_items": intersections,
        "total_items_across_reviewers": total_items,
    }


def _optional_consensus(
    labeled: list[tuple[str, dict[str, str]]],
    field: str,
    *,
    parse_kind: str,
) -> tuple[str, bool]:
    raw_values = [row[field].strip() for _, row in labeled]
    if not all(raw_values):
        return "", False
    if parse_kind == "event_types":
        parsed = [
            tuple(
                sorted(
                    _parse_event_type_list(
                        raw,
                        Path("<adjudication>"),
                        0,
                        field,
                    )
                )
            )
            for raw in raw_values
        ]
    else:
        parsed = [
            _parse_entities(raw, Path("<adjudication>"), 0, field)[0]
            for raw in raw_values
        ]
    if any(value != parsed[0] for value in parsed[1:]):
        return "", False
    if parse_kind == "event_types":
        return json.dumps(list(parsed[0]), separators=(",", ":")), True
    return _canonical_json_list(raw_values[0]), True


def _canonical_json_list(raw: str) -> str:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list during adjudication")
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _status(row: dict[str, str], path: Path, line_number: int) -> str:
    status = row["review_status"].strip().lower() or "pending"
    if status not in _REVIEW_STATUSES:
        location = f"{path}:{line_number}" if line_number else str(path)
        raise ValueError(f"Invalid review_status {status!r} at {location}")
    return status


def _normalize_reviewers(reviewers: list[str]) -> list[str]:
    normalized = [reviewer.strip() for reviewer in reviewers if reviewer.strip()]
    if len(normalized) < 2:
        raise ValueError("At least two reviewer names are required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Reviewer names must be unique")
    return normalized


def _reviewer_slug(reviewer: str) -> str:
    slug = _SLUG.sub("-", reviewer.lower()).strip("-")
    if not slug:
        raise ValueError(f"Reviewer name {reviewer!r} cannot be converted to a filename")
    return slug


def _rank(seed: str, event_id: str, reviewer: str) -> str:
    return hashlib.sha256(f"{seed}|{event_id}|{reviewer}".encode()).hexdigest()
