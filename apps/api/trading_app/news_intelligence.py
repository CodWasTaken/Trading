from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict, deque
from datetime import UTC, datetime

from .domain import NewsEvent

_TOKEN = re.compile(r"[a-z0-9]+")

_EVENT_KEYWORDS: tuple[tuple[str, frozenset[str]], ...] = (
    ("earnings", frozenset({"earnings", "revenue", "eps", "quarterly", "profit"})),
    ("guidance", frozenset({"guidance", "outlook", "forecast", "expects", "raises", "cuts"})),
    ("merger_acquisition", frozenset({"acquire", "acquisition", "merger", "takeover", "buyout"})),
    ("regulatory", frozenset({"regulator", "regulatory", "probe", "investigation", "antitrust", "sec"})),
    ("legal", frozenset({"lawsuit", "court", "settlement", "verdict", "appeal"})),
    ("management", frozenset({"ceo", "cfo", "chair", "resigns", "appointed", "management"})),
    ("product_launch", frozenset({"launch", "unveils", "introduces", "product", "platform"})),
    ("cybersecurity", frozenset({"cyber", "breach", "ransomware", "hack", "security"})),
    ("financing", frozenset({"offering", "debt", "financing", "convertible", "shares"})),
    ("buyback", frozenset({"buyback", "repurchase"})),
    ("dividend", frozenset({"dividend", "distribution"})),
    ("supply_chain", frozenset({"supply", "shortage", "factory", "facility", "shipment"})),
    ("analyst_rating", frozenset({"upgrade", "downgrade", "rating", "price", "target"})),
)

_POSITIVE = frozenset(
    {
        "beat",
        "beats",
        "record",
        "growth",
        "strong",
        "raises",
        "raised",
        "upgrade",
        "upgraded",
        "approval",
        "approved",
        "wins",
        "expands",
        "buyback",
        "repurchase",
        "dividend",
        "surge",
        "profit",
    }
)
_NEGATIVE = frozenset(
    {
        "miss",
        "misses",
        "weak",
        "cuts",
        "cut",
        "downgrade",
        "downgraded",
        "probe",
        "investigation",
        "lawsuit",
        "breach",
        "shortage",
        "recall",
        "loss",
        "decline",
        "falls",
        "resigns",
        "bankruptcy",
    }
)

_SOURCE_QUALITY = {
    "sec": 1.0,
    "reuters": 0.96,
    "associated press": 0.94,
    "business wire": 0.92,
    "globe newswire": 0.90,
    "benzinga": 0.86,
    "alpaca": 0.82,
}


def tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))


def fingerprint(headline: str, source: str) -> str:
    normalized = " ".join(sorted(tokenize(headline)))
    return hashlib.sha256(f"{source.lower()}|{normalized}".encode()).hexdigest()


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


class NewsIntelligence:
    """Deterministic first-pass enrichment before higher-capacity NLP models."""

    def __init__(self, history_per_symbol: int = 100) -> None:
        self._history: dict[str, deque[frozenset[str]]] = defaultdict(
            lambda: deque(maxlen=history_per_symbol)
        )
        self._fingerprints: set[str] = set()

    def enrich(
        self,
        *,
        symbol: str,
        headline: str,
        source: str,
        event_time: datetime,
        knowledge_time: datetime | None = None,
        summary: str = "",
    ) -> NewsEvent | None:
        symbol = symbol.upper()
        identifier = fingerprint(headline, source)
        if identifier in self._fingerprints:
            return None

        tokens = tokenize(f"{headline} {summary}")
        similarities = [jaccard(tokens, prior) for prior in self._history[symbol]]
        novelty = 1.0 - max(similarities, default=0.0)
        positive = len(tokens & _POSITIVE)
        negative = len(tokens & _NEGATIVE)
        sentiment = math.tanh((positive - negative) / 2.0)
        event_type = "other"
        for category, keywords in _EVENT_KEYWORDS:
            if tokens & keywords:
                event_type = category
                break

        source_key = source.lower().strip()
        source_quality = next(
            (
                quality
                for known_source, quality in _SOURCE_QUALITY.items()
                if known_source in source_key
            ),
            0.72,
        )
        self._fingerprints.add(identifier)
        self._history[symbol].append(tokens)
        return NewsEvent(
            symbol=symbol,
            headline=headline.strip(),
            source=source.strip() or "unknown",
            sentiment=sentiment,
            novelty=max(0.0, min(1.0, novelty)),
            source_quality=source_quality,
            event_type=event_type,
            event_time=event_time.astimezone(UTC),
            knowledge_time=(knowledge_time or datetime.now(UTC)).astimezone(UTC),
        )
