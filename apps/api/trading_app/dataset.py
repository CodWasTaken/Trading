from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import pstdev
from typing import Any

from .domain import NewsEvent, Quote
from .historical import HistoricalBar
from .research import FeatureRow


FEATURE_NAMES = ("momentum", "news_score", "volatility", "spread_bps")
HISTORICAL_FEATURE_NAMES = ("momentum", "news_score", "volatility")
DATASET_SCHEMA_VERSION = 1


class PointInTimeDatasetBuilder:
    """Build labels using only observations known at each feature timestamp."""

    def __init__(
        self,
        *,
        lookback_quotes: int = 20,
        forecast_quotes: int = 5,
        news_window: timedelta = timedelta(hours=24),
    ) -> None:
        if lookback_quotes < 3:
            raise ValueError("lookback_quotes must be at least three")
        if forecast_quotes < 1:
            raise ValueError("forecast_quotes must be positive")
        self.lookback_quotes = lookback_quotes
        self.forecast_quotes = forecast_quotes
        self.news_window = news_window

    def build(self, quotes: list[Quote], news: list[NewsEvent]) -> list[FeatureRow]:
        quotes_by_symbol: dict[str, list[Quote]] = defaultdict(list)
        news_by_symbol: dict[str, list[NewsEvent]] = defaultdict(list)
        for quote in quotes:
            quotes_by_symbol[quote.symbol].append(quote)
        for item in news:
            news_by_symbol[item.symbol].append(item)
        rows: list[FeatureRow] = []
        for symbol, symbol_quotes in quotes_by_symbol.items():
            ordered_quotes = sorted(symbol_quotes, key=lambda item: item.knowledge_time)
            ordered_news = sorted(
                news_by_symbol.get(symbol, []), key=lambda item: item.knowledge_time
            )
            news_cursor = 0
            visible_news: list[NewsEvent] = []
            last_index = len(ordered_quotes) - self.forecast_quotes
            for index in range(self.lookback_quotes, last_index):
                current = ordered_quotes[index]
                while (
                    news_cursor < len(ordered_news)
                    and ordered_news[news_cursor].knowledge_time <= current.knowledge_time
                ):
                    visible_news.append(ordered_news[news_cursor])
                    news_cursor += 1
                cutoff = current.knowledge_time - self.news_window
                visible_news = [
                    item for item in visible_news if item.knowledge_time >= cutoff
                ]
                window = ordered_quotes[index - self.lookback_quotes : index + 1]
                mids = [quote.mid for quote in window]
                returns = [
                    mids[position] / mids[position - 1] - 1
                    for position in range(1, len(mids))
                ]
                momentum = mids[-1] / mids[0] - 1
                volatility = pstdev(returns) if len(returns) > 1 else 0.0
                weighted_news = [
                    item.sentiment * item.novelty * item.source_quality
                    for item in visible_news[-5:]
                ]
                news_score = (
                    sum(weighted_news) / len(weighted_news) if weighted_news else 0.0
                )
                future = ordered_quotes[index + self.forecast_quotes]
                target_return = future.mid / current.mid - 1
                rows.append(
                    FeatureRow(
                        timestamp=current.knowledge_time,
                        symbol=symbol,
                        features=(
                            momentum,
                            news_score,
                            volatility,
                            current.spread_bps,
                        ),
                        target_return=target_return,
                        label_end_time=future.knowledge_time,
                    )
                )
        return sorted(rows, key=lambda item: (item.timestamp, item.symbol))


class HistoricalPointInTimeDatasetBuilder:
    """Build live-compatible features from OHLC bars without inventing a spread."""

    def __init__(
        self,
        *,
        lookback_bars: int = 19,
        forecast_bars: int = 5,
        bar_interval: timedelta = timedelta(hours=1),
        news_window: timedelta = timedelta(hours=24),
    ) -> None:
        if lookback_bars < 3:
            raise ValueError("lookback_bars must be at least three")
        if forecast_bars < 1:
            raise ValueError("forecast_bars must be positive")
        if bar_interval <= timedelta(0):
            raise ValueError("bar_interval must be positive")
        self.lookback_bars = lookback_bars
        self.forecast_bars = forecast_bars
        self.bar_interval = bar_interval
        self.news_window = news_window

    def build(self, bars: list[HistoricalBar], news: list[NewsEvent]) -> list[FeatureRow]:
        bars_by_symbol: dict[str, dict[datetime, HistoricalBar]] = defaultdict(dict)
        news_by_symbol: dict[str, list[NewsEvent]] = defaultdict(list)
        for bar in bars:
            symbol = bar.symbol.upper()
            if min(bar.open, bar.high, bar.low, bar.close) <= 0:
                raise ValueError(f"Historical bar contains a non-positive price: {symbol}")
            timestamp = _as_utc(bar.timestamp)
            bars_by_symbol[symbol][timestamp] = bar.model_copy(
                update={"symbol": symbol, "timestamp": timestamp}
            )
        for item in news:
            symbol = item.symbol.upper()
            news_by_symbol[symbol].append(
                item.model_copy(
                    update={
                        "symbol": symbol,
                        "event_time": _as_utc(item.event_time),
                        "knowledge_time": _as_utc(item.knowledge_time),
                    }
                )
            )

        rows: list[FeatureRow] = []
        for symbol, timestamp_map in bars_by_symbol.items():
            ordered_bars = sorted(timestamp_map.values(), key=lambda item: item.timestamp)
            ordered_news = sorted(
                news_by_symbol.get(symbol, []), key=lambda item: item.knowledge_time
            )
            news_cursor = 0
            visible_news: list[NewsEvent] = []
            last_index = len(ordered_bars) - self.forecast_bars
            for index in range(self.lookback_bars, last_index):
                current = ordered_bars[index]
                feature_time = _as_utc(current.timestamp) + self.bar_interval
                while (
                    news_cursor < len(ordered_news)
                    and ordered_news[news_cursor].knowledge_time <= feature_time
                ):
                    visible_news.append(ordered_news[news_cursor])
                    news_cursor += 1
                cutoff = feature_time - self.news_window
                visible_news = [
                    item for item in visible_news if item.knowledge_time >= cutoff
                ]
                window = ordered_bars[index - self.lookback_bars : index + 1]
                closes = [bar.close for bar in window]
                returns = [
                    closes[position] / closes[position - 1] - 1
                    for position in range(1, len(closes))
                ]
                weighted_news = [
                    item.sentiment * item.novelty * item.source_quality
                    for item in visible_news[-5:]
                ]
                future = ordered_bars[index + self.forecast_bars]
                rows.append(
                    FeatureRow(
                        timestamp=feature_time,
                        symbol=symbol,
                        features=(
                            closes[-1] / closes[0] - 1,
                            (
                                sum(weighted_news) / len(weighted_news)
                                if weighted_news
                                else 0.0
                            ),
                            pstdev(returns) if len(returns) > 1 else 0.0,
                        ),
                        target_return=future.close / current.close - 1,
                        label_end_time=_as_utc(future.timestamp) + self.bar_interval,
                    )
                )
        return sorted(rows, key=lambda item: (item.timestamp, item.symbol))


def write_feature_dataset(
    output: str | Path,
    rows: list[FeatureRow],
    feature_names: tuple[str, ...],
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    if not rows:
        raise ValueError("Cannot write an empty feature dataset")
    if any(len(row.features) != len(feature_names) for row in rows):
        raise ValueError("Feature row width does not match feature names")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "timestamp": _as_utc(row.timestamp).isoformat(),
                "label_end_time": (
                    _as_utc(row.label_end_time).isoformat()
                    if row.label_end_time is not None
                    else None
                ),
                "symbol": row.symbol,
                "features": {
                    name: value
                    for name, value in zip(feature_names, row.features, strict=True)
                },
                "target_return": row.target_return,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    metadata_path = dataset_metadata_path(destination)
    payload = {
        **metadata,
        "schema_version": DATASET_SCHEMA_VERSION,
        "feature_names": list(feature_names),
        "rows": len(rows),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return destination, metadata_path


def load_feature_dataset(
    dataset: str | Path,
) -> tuple[list[FeatureRow], tuple[str, ...], dict[str, object]]:
    source = Path(dataset)
    metadata_path = dataset_metadata_path(source)
    if not metadata_path.exists():
        raise ValueError(f"Dataset metadata file is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if int(metadata.get("schema_version", 0)) != DATASET_SCHEMA_VERSION:
        raise ValueError("Unsupported dataset schema version")
    feature_names = tuple(str(value) for value in metadata.get("feature_names", []))
    if not feature_names:
        raise ValueError("Dataset metadata does not define feature names")

    rows: list[FeatureRow] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload: dict[str, Any] = json.loads(line)
            features_payload = payload.get("features")
            if not isinstance(features_payload, dict):
                raise ValueError(f"Dataset line {line_number} has invalid features")
            try:
                features = tuple(float(features_payload[name]) for name in feature_names)
                target_return = float(payload["target_return"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Dataset line {line_number} is invalid") from error
            values = (*features, target_return)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Dataset line {line_number} contains non-finite values")
            label_end = payload.get("label_end_time")
            rows.append(
                FeatureRow(
                    timestamp=_parse_datetime(payload["timestamp"]),
                    symbol=str(payload["symbol"]).upper(),
                    features=features,
                    target_return=target_return,
                    label_end_time=(
                        _parse_datetime(label_end) if label_end is not None else None
                    ),
                )
            )
    if not rows:
        raise ValueError("Dataset contains no feature rows")
    expected_rows = metadata.get("rows")
    if expected_rows is not None and int(expected_rows) != len(rows):
        raise ValueError(
            f"Dataset row count mismatch: metadata={expected_rows}, actual={len(rows)}"
        )
    return sorted(rows, key=lambda item: (item.timestamp, item.symbol)), feature_names, metadata


def dataset_metadata_path(dataset: str | Path) -> Path:
    return Path(dataset).with_suffix(".metadata.json")


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Expected ISO-8601 datetime string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
