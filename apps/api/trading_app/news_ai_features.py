from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .dataset import load_feature_dataset, write_feature_dataset
from .modeling import sha256_file
from .research import FeatureRow

NEWS_AI_FEATURE_NAMES = (
    "news_ai_sentiment",
    "news_ai_confidence",
    "news_ai_event_intensity",
    "news_ai_entity_count",
)


def join_news_ai_features(
    dataset_path: str | Path,
    inference_path: str | Path,
    output_path: str | Path,
    *,
    window_hours: int = 24,
    max_items: int = 5,
) -> dict[str, object]:
    if window_hours < 1:
        raise ValueError("window_hours must be positive")
    if max_items < 1:
        raise ValueError("max_items must be positive")
    dataset_source = Path(dataset_path)
    inference_source = Path(inference_path)
    rows, feature_names, metadata = load_feature_dataset(dataset_source)
    records = _load_inference_records(inference_source)
    by_symbol: dict[str, list[dict[str, object]]] = defaultdict(list)
    model_hashes: set[str] = set()
    parse_failures = 0
    for record in records:
        model_hash = record.get("model_manifest_sha256")
        if not isinstance(model_hash, str) or not model_hash:
            raise ValueError("News inference record lacks model_manifest_sha256")
        model_hashes.add(model_hash)
        if record.get("parse_error") is not None or not isinstance(record.get("normalized"), dict):
            parse_failures += 1
            continue
        symbol = str(record.get("symbol", "")).upper()
        if not symbol:
            raise ValueError("News inference record lacks symbol")
        knowledge_time = _parse_time(record.get("knowledge_time"))
        record["_knowledge_time"] = knowledge_time
        by_symbol[symbol].append(record)
    if len(model_hashes) != 1:
        raise ValueError("News AI feature join requires exactly one model manifest hash")
    for values in by_symbol.values():
        values.sort(key=lambda item: _record_time(item))

    cursors: dict[str, int] = defaultdict(int)
    visible: dict[str, list[dict[str, object]]] = defaultdict(list)
    joined: list[FeatureRow] = []
    for row in sorted(rows, key=lambda item: (item.timestamp, item.symbol)):
        symbol = row.symbol.upper()
        feature_time = _as_utc(row.timestamp)
        symbol_records = by_symbol.get(symbol, [])
        cursor = cursors[symbol]
        while cursor < len(symbol_records) and _record_time(symbol_records[cursor]) <= feature_time:
            visible[symbol].append(symbol_records[cursor])
            cursor += 1
        cursors[symbol] = cursor
        cutoff = feature_time - timedelta(hours=window_hours)
        visible[symbol] = [
            item for item in visible[symbol] if _record_time(item) >= cutoff
        ]
        features = _aggregate(visible[symbol][-max_items:])
        joined.append(
            FeatureRow(
                timestamp=row.timestamp,
                symbol=row.symbol,
                features=(*row.features, *features),
                target_return=row.target_return,
                label_end_time=row.label_end_time,
            )
        )
    max_feature_time = max(_as_utc(row.timestamp) for row in rows)
    future_records = sum(
        _record_time(record) > max_feature_time
        for values in by_symbol.values()
        for record in values
    )
    output_metadata = {
        **metadata,
        "news_ai_features": {
            "inference_path": str(inference_source),
            "inference_sha256": sha256_file(inference_source),
            "model_manifest_sha256": next(iter(model_hashes)),
            "window_hours": window_hours,
            "max_items": max_items,
            "parse_failures_excluded": parse_failures,
            "future_records_not_joined": future_records,
            "point_in_time_verified": True,
        },
    }
    destination, metadata_path = write_feature_dataset(
        output_path,
        joined,
        (*feature_names, *NEWS_AI_FEATURE_NAMES),
        output_metadata,
    )
    return {
        "output": str(destination),
        "metadata": str(metadata_path),
        "rows": len(joined),
        "feature_names": [*feature_names, *NEWS_AI_FEATURE_NAMES],
        "parse_failures_excluded": parse_failures,
        "future_records_not_joined": future_records,
    }


def _aggregate(records: list[dict[str, object]]) -> tuple[float, float, float, float]:
    if not records:
        return (0.0, 0.0, 0.0, 0.0)
    weighted_sentiment = 0.0
    total_weight = 0.0
    confidence_sum = 0.0
    event_sum = 0.0
    entity_sum = 0.0
    for record in records:
        normalized = record["normalized"]
        assert isinstance(normalized, dict)
        sentiment = _number(normalized.get("sentiment"), "sentiment")
        confidence = _number(normalized.get("confidence"), "confidence")
        confidence = max(0.0, min(1.0, confidence))
        weighted_sentiment += sentiment * confidence
        total_weight += confidence
        confidence_sum += confidence
        event_sum += confidence * int(normalized.get("primary_event_type") != "other")
        entities = normalized.get("entities", [])
        entity_sum += float(len(entities)) if isinstance(entities, list) else 0.0
    count = float(len(records))
    return (
        0.0 if total_weight == 0 else weighted_sentiment / total_weight,
        confidence_sum / count,
        event_sum / count,
        entity_sum / count,
    )


def _load_inference_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(payload)
    if not records:
        raise ValueError(f"No news inference records found in {path}")
    return records


def _record_time(record: dict[str, object]) -> datetime:
    value = record.get("_knowledge_time")
    if not isinstance(value, datetime):
        raise ValueError("News inference record lacks parsed knowledge time")
    return value


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Expected ISO-8601 knowledge_time")
    return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)
