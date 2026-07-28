from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC
from pathlib import Path
from typing import Any

from .domain import NewsEvent
from .entities import EntityRelation, normalize_entity_text
from .news_intelligence import _EVENT_RULES


EVENT_TYPES = tuple(rule.label for rule in _EVENT_RULES) + ("other",)


LABEL_COLUMNS = (
    "record_sha256",
    "event_id",
    "symbol",
    "event_time",
    "knowledge_time",
    "source",
    "headline",
    "summary",
    "predicted_primary_event_type",
    "predicted_secondary_event_types_json",
    "predicted_extraction_confidence",
    "predicted_entities_json",
    "gold_primary_event_type",
    "gold_secondary_event_types_json",
    "gold_entities_json",
    "review_status",
    "reviewer",
    "notes",
)
_SOURCE_COLUMNS = (
    "event_id",
    "symbol",
    "event_time",
    "knowledge_time",
    "source",
    "headline",
    "summary",
    "predicted_primary_event_type",
    "predicted_secondary_event_types_json",
    "predicted_extraction_confidence",
    "predicted_entities_json",
)
_REVIEW_STATUSES = frozenset({"pending", "labeled", "skip"})
_ENTITY_RELATIONS = frozenset(relation.value for relation in EntityRelation)


def build_label_set(
    input_path: str,
    output_path: str,
    *,
    sample_size: int = 400,
    minimum_per_stratum: int = 3,
    seed: str = "news-evaluation-v1",
    metadata_path: str | None = None,
) -> dict[str, object]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if minimum_per_stratum < 0:
        raise ValueError("minimum_per_stratum must not be negative")

    source = Path(input_path)
    destination = Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Input archive and label-set output paths must differ")

    events = _load_news(source)
    selected = _stratified_sample(
        events,
        sample_size=min(sample_size, len(events)),
        minimum_per_stratum=minimum_per_stratum,
        seed=seed,
    )
    rows = [_label_row(event) for event in selected]

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    strata = Counter(
        f"{row['predicted_primary_event_type']}|{row['symbol']}" for row in rows
    )
    event_types = Counter(row["predicted_primary_event_type"] for row in rows)
    symbols = Counter(row["symbol"] for row in rows)
    metadata_destination = (
        Path(metadata_path)
        if metadata_path
        else destination.with_suffix(".metadata.json")
    )
    metadata_destination.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "news_label_set",
        "source": {
            "path": str(source),
            "records": len(events),
            "sha256": _sha256(source),
        },
        "output": {
            "path": str(destination),
            "records": len(rows),
            "template_sha256": _sha256(destination),
        },
        "metadata": str(metadata_destination),
        "sampling": {
            "method": "deterministic_minimum_per_predicted_event_type_and_symbol_then_hash_fill",
            "seed": seed,
            "requested_sample_size": sample_size,
            "selected_records": len(rows),
            "minimum_per_stratum": minimum_per_stratum,
            "strata": dict(sorted(strata.items())),
            "event_types": dict(sorted(event_types.items())),
            "symbols": dict(sorted(symbols.items())),
        },
        "allowed_primary_event_types": list(EVENT_TYPES),
        "allowed_entity_relations": sorted(_ENTITY_RELATIONS),
        "labeling": {
            "review_status_values": sorted(_REVIEW_STATUSES),
            "required_for_primary_evaluation": [
                "review_status=labeled",
                "gold_primary_event_type",
            ],
            "optional_secondary_field": "gold_secondary_event_types_json",
            "optional_entity_field": "gold_entities_json",
            "empty_secondary_example": "[]",
            "empty_entity_example": "[]",
        },
        "point_in_time_controls": {
            "event_time_included": True,
            "knowledge_time_included": True,
            "deterministic_sampling": True,
            "immutable_source_columns_hashed_per_row": True,
            "future_text_not_joined": True,
            "spreadsheet_formula_prefixes_escaped": True,
        },
    }
    metadata_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def evaluate_label_set(
    labels_path: str,
    output_path: str,
    *,
    require_complete: bool = False,
    metadata_path: str | None = None,
) -> dict[str, object]:
    source = Path(labels_path)
    destination = Path(output_path)
    rows = _load_label_rows(source)
    metadata_source = (
        Path(metadata_path)
        if metadata_path
        else source.with_suffix(".metadata.json")
    )
    metadata = (
        json.loads(metadata_source.read_text(encoding="utf-8"))
        if metadata_source.exists()
        else None
    )

    status_counts: Counter[str] = Counter()
    primary_pairs: list[tuple[str, str]] = []
    secondary_pairs: list[tuple[set[str], set[str]]] = []
    entity_pairs: list[tuple[set[tuple[str, str]], set[tuple[str, str]]]] = []
    relation_pairs: list[tuple[set[str], set[str]]] = []
    reviewers: Counter[str] = Counter()

    for line_number, row in enumerate(rows, start=2):
        _validate_source_hash(row, source, line_number)
        status = row["review_status"].strip().lower() or "pending"
        if status not in _REVIEW_STATUSES:
            raise ValueError(
                f"Invalid review_status {status!r} at {source}:{line_number}"
            )
        status_counts[status] += 1
        if status != "labeled":
            continue

        gold_primary = row["gold_primary_event_type"].strip()
        predicted_primary = row["predicted_primary_event_type"].strip()
        _validate_event_type(gold_primary, source, line_number, "gold_primary_event_type")
        _validate_event_type(
            predicted_primary,
            source,
            line_number,
            "predicted_primary_event_type",
        )
        primary_pairs.append((gold_primary, predicted_primary))
        reviewer = row["reviewer"].strip()
        if reviewer:
            reviewers[reviewer] += 1

        gold_secondary_raw = row["gold_secondary_event_types_json"].strip()
        if gold_secondary_raw:
            gold_secondary = set(
                _parse_event_type_list(
                    gold_secondary_raw,
                    source,
                    line_number,
                    "gold_secondary_event_types_json",
                )
            )
            predicted_secondary = set(
                _parse_event_type_list(
                    row["predicted_secondary_event_types_json"],
                    source,
                    line_number,
                    "predicted_secondary_event_types_json",
                )
            )
            if gold_primary in gold_secondary:
                raise ValueError(
                    f"Gold secondary labels must not repeat the primary label at "
                    f"{source}:{line_number}"
                )
            secondary_pairs.append((gold_secondary, predicted_secondary))

        gold_entities_raw = row["gold_entities_json"].strip()
        if gold_entities_raw:
            gold_links, gold_relations = _parse_entities(
                gold_entities_raw,
                source,
                line_number,
                "gold_entities_json",
            )
            predicted_links, predicted_relations = _parse_entities(
                row["predicted_entities_json"],
                source,
                line_number,
                "predicted_entities_json",
            )
            entity_pairs.append((gold_links, predicted_links))
            relation_pairs.append((gold_relations, predicted_relations))

    if not primary_pairs:
        raise ValueError("No rows with review_status=labeled were found")
    if require_complete and status_counts["pending"]:
        raise ValueError(
            f"Label set is incomplete: {status_counts['pending']} rows remain pending"
        )

    primary_report = _classification_report(primary_pairs, EVENT_TYPES)
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "news_label_evaluation",
        "labels": {
            "path": str(source),
            "sha256": _sha256(source),
            "rows": len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "reviewers": dict(sorted(reviewers.items())),
        },
        "metadata": {
            "path": str(metadata_source) if metadata_source.exists() else None,
            "source_archive": metadata.get("source") if metadata else None,
            "sampling": metadata.get("sampling") if metadata else None,
        },
        "primary_event_type": primary_report,
        "secondary_event_types": (
            _multilabel_report(secondary_pairs, EVENT_TYPES)
            if secondary_pairs
            else {
                "rows": 0,
                "status": "not_evaluated",
                "reason": "No labeled rows supplied gold_secondary_event_types_json",
            }
        ),
        "entity_links": (
            _set_report(entity_pairs, labels=None)
            if entity_pairs
            else {
                "rows": 0,
                "status": "not_evaluated",
                "reason": "No labeled rows supplied gold_entities_json",
            }
        ),
        "entity_relations": (
            _set_report(
                [
                    (
                        {(relation, relation) for relation in gold},
                        {(relation, relation) for relation in predicted},
                    )
                    for gold, predicted in relation_pairs
                ],
                labels=sorted(_ENTITY_RELATIONS),
            )
            if relation_pairs
            else {
                "rows": 0,
                "status": "not_evaluated",
                "reason": "No labeled rows supplied gold_entities_json",
            }
        ),
        "point_in_time_controls": {
            "immutable_source_columns_verified": True,
            "event_time_present_in_label_rows": True,
            "knowledge_time_present_in_label_rows": True,
            "evaluation_uses_only_human_labels_and_stored_predictions": True,
        },
        "limitations": [
            "Metrics describe only the labeled sample and inherit its sampling design.",
            "A sample selected from one archive is not an untouched validation archive.",
            "Entity exact-link matching uses relation plus ticker when available, otherwise normalized canonical name.",
            "Inter-annotator agreement is not estimated unless multiple independent label sets are created.",
        ],
    }
    report["output"] = str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_news(path: Path) -> list[NewsEvent]:
    events: list[NewsEvent] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = NewsEvent.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid NewsEvent at {path}:{line_number}") from error
            identifier = str(event.id)
            if identifier in seen_ids:
                raise ValueError(f"Duplicate NewsEvent id {identifier} at {path}:{line_number}")
            seen_ids.add(identifier)
            events.append(event)
    if not events:
        raise ValueError(f"No news records found in {path}")
    return events


def _stratified_sample(
    events: list[NewsEvent],
    *,
    sample_size: int,
    minimum_per_stratum: int,
    seed: str,
) -> list[NewsEvent]:
    ranked = sorted(events, key=lambda event: _sample_rank(event, seed))
    by_stratum: dict[tuple[str, str], list[NewsEvent]] = defaultdict(list)
    for event in ranked:
        by_stratum[(event.event_type, event.symbol)].append(event)

    selected: list[NewsEvent] = []
    selected_ids: set[str] = set()
    ordered_strata = sorted(by_stratum)
    for offset in range(minimum_per_stratum):
        for stratum in ordered_strata:
            if len(selected) >= sample_size:
                break
            if offset >= len(by_stratum[stratum]):
                continue
            event = by_stratum[stratum][offset]
            selected.append(event)
            selected_ids.add(str(event.id))
        if len(selected) >= sample_size:
            break

    if len(selected) < sample_size:
        for event in ranked:
            if str(event.id) in selected_ids:
                continue
            selected.append(event)
            selected_ids.add(str(event.id))
            if len(selected) >= sample_size:
                break

    return sorted(selected, key=lambda event: _sample_rank(event, seed))


def _sample_rank(event: NewsEvent, seed: str) -> str:
    payload = (
        f"{seed}|{event.id}|{event.symbol}|{event.event_type}|"
        f"{event.knowledge_time.astimezone(UTC).isoformat()}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _label_row(event: NewsEvent) -> dict[str, str]:
    row = {
        "event_id": str(event.id),
        "symbol": event.symbol,
        "event_time": event.event_time.astimezone(UTC).isoformat(),
        "knowledge_time": event.knowledge_time.astimezone(UTC).isoformat(),
        "source": _spreadsheet_safe(event.source),
        "headline": _spreadsheet_safe(event.headline),
        "summary": _spreadsheet_safe(event.summary),
        "predicted_primary_event_type": event.event_type,
        "predicted_secondary_event_types_json": json.dumps(
            event.secondary_event_types,
            separators=(",", ":"),
        ),
        "predicted_extraction_confidence": f"{event.extraction_confidence:.12g}",
        "predicted_entities_json": json.dumps(
            [entity.model_dump(mode="json") for entity in event.entities],
            separators=(",", ":"),
            sort_keys=True,
        ),
        "gold_primary_event_type": "",
        "gold_secondary_event_types_json": "",
        "gold_entities_json": "",
        "review_status": "pending",
        "reviewer": "",
        "notes": "",
    }
    row["record_sha256"] = _row_hash(row)
    return row


def _load_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LABEL_COLUMNS:
            raise ValueError(
                f"Unexpected label-set columns in {path}; regenerate with trading-news sample-labels"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No label rows found in {path}")
    return rows


def _row_hash(row: dict[str, str]) -> str:
    payload = {column: row[column] for column in _SOURCE_COLUMNS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_source_hash(
    row: dict[str, str],
    path: Path,
    line_number: int,
) -> None:
    expected = row["record_sha256"].strip()
    actual = _row_hash(row)
    if not expected or expected != actual:
        raise ValueError(
            f"Immutable source columns were modified at {path}:{line_number}; "
            "regenerate the label set or restore the original prediction fields"
        )


def _validate_event_type(
    value: str,
    path: Path,
    line_number: int,
    field: str,
) -> None:
    if value not in EVENT_TYPES:
        raise ValueError(
            f"Invalid {field} {value!r} at {path}:{line_number}; "
            f"allowed values are {', '.join(EVENT_TYPES)}"
        )


def _parse_event_type_list(
    raw: str,
    path: Path,
    line_number: int,
    field: str,
) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {field} at {path}:{line_number}") from error
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON list of strings at {path}:{line_number}")
    normalized = list(dict.fromkeys(item.strip() for item in value if item.strip()))
    for event_type in normalized:
        _validate_event_type(event_type, path, line_number, field)
    return normalized


def _parse_entities(
    raw: str,
    path: Path,
    line_number: int,
    field: str,
) -> tuple[set[tuple[str, str]], set[str]]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {field} at {path}:{line_number}") from error
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list at {path}:{line_number}")

    links: set[tuple[str, str]] = set()
    relations: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(
                f"Each {field} item must be an object at {path}:{line_number}"
            )
        relation = str(item.get("relation", "")).strip()
        if relation not in _ENTITY_RELATIONS:
            raise ValueError(
                f"Invalid entity relation {relation!r} in {field} at "
                f"{path}:{line_number}"
            )
        ticker = str(item.get("ticker") or "").upper().strip()
        canonical_name = str(item.get("canonical_name") or "").strip()
        identity = ticker or normalize_entity_text(canonical_name)
        if not identity:
            raise ValueError(
                f"Each {field} item requires ticker or canonical_name at "
                f"{path}:{line_number}"
            )
        links.add((relation, identity))
        relations.add(relation)
    return links, relations


def _classification_report(
    pairs: list[tuple[str, str]],
    labels: tuple[str, ...],
) -> dict[str, object]:
    truth = Counter(gold for gold, _ in pairs)
    predicted = Counter(prediction for _, prediction in pairs)
    correct = sum(gold == prediction for gold, prediction in pairs)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for gold, prediction in pairs:
        confusion[gold][prediction] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    supported_metrics: list[tuple[float, float, float, int]] = []
    weighted_f1_numerator = 0.0
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = predicted[label] - true_positive
        false_negative = truth[label] - true_positive
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _f1(precision, recall)
        support = truth[label]
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted[label],
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
        if support:
            supported_metrics.append((precision, recall, f1, support))
            weighted_f1_numerator += f1 * support

    total = len(pairs)
    return {
        "rows": total,
        "accuracy": correct / total,
        "macro_precision_supported": _mean(item[0] for item in supported_metrics),
        "macro_recall_supported": _mean(item[1] for item in supported_metrics),
        "macro_f1_supported": _mean(item[2] for item in supported_metrics),
        "weighted_f1": weighted_f1_numerator / total,
        "gold_other_rate": truth["other"] / total,
        "predicted_other_rate": predicted["other"] / total,
        "per_class": per_class,
        "confusion": {
            gold: dict(sorted(predictions.items()))
            for gold, predictions in sorted(confusion.items())
        },
    }


def _multilabel_report(
    pairs: list[tuple[set[str], set[str]]],
    labels: tuple[str, ...],
) -> dict[str, object]:
    transformed = [
        (
            {(label, label) for label in gold},
            {(label, label) for label in predicted},
        )
        for gold, predicted in pairs
    ]
    return _set_report(transformed, labels=list(labels))


def _set_report(
    pairs: list[
        tuple[
            set[tuple[str, str]],
            set[tuple[str, str]],
        ]
    ],
    *,
    labels: list[str] | None,
) -> dict[str, object]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact_matches = 0
    per_label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for gold, predicted in pairs:
        intersection = gold & predicted
        true_positive += len(intersection)
        false_positive += len(predicted - gold)
        false_negative += len(gold - predicted)
        exact_matches += gold == predicted

        if labels is not None:
            gold_labels = {item[0] for item in gold}
            predicted_labels = {item[0] for item in predicted}
            for label in labels:
                if label in gold_labels and label in predicted_labels:
                    per_label_counts[label]["tp"] += 1
                elif label in predicted_labels:
                    per_label_counts[label]["fp"] += 1
                elif label in gold_labels:
                    per_label_counts[label]["fn"] += 1

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    report: dict[str, object] = {
        "rows": len(pairs),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": _f1(precision, recall),
        "exact_match_rate": exact_matches / len(pairs),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
    if labels is not None:
        report["per_class"] = {
            label: _counts_to_metrics(per_label_counts[label])
            for label in labels
        }
    return report


def _counts_to_metrics(counts: Counter[str]) -> dict[str, float | int]:
    precision = _safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = _safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "true_positive": counts["tp"],
        "false_positive": counts["fp"],
        "false_negative": counts["fn"],
    }


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def _mean(values: Any) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _spreadsheet_safe(value: str) -> str:
    if value and value[0] in "=+-@\t\r":
        return "'" + value
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
