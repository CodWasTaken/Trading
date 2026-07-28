from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime

from .domain import NewsEvent
from .entities import EntityCatalog

_TOKEN = re.compile(r"[a-z0-9]+")
_PERCENT = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")
_MONEY = re.compile(
    r"(?P<currency>US\$|\$|€|£)\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>trillion|billion|million|tn|bn|mm|m|b|t)?\b",
    re.IGNORECASE,
)
_EPS = re.compile(
    r"(?:\beps\b|earnings per share)[^$€£\d]{0,20}(?:US\$|\$|€|£)?\s*"
    r"(?P<value>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EventRule:
    label: str
    phrases: tuple[str, ...] = ()
    keywords: frozenset[str] = frozenset()


_EVENT_RULES: tuple[EventRule, ...] = (
    EventRule(
        "guidance",
        (
            "raises guidance",
            "raised guidance",
            "cuts guidance",
            "cut guidance",
            "full year outlook",
            "annual outlook",
            "revenue forecast",
            "earnings forecast",
        ),
        frozenset({"guidance", "outlook", "forecast"}),
    ),
    EventRule(
        "earnings",
        ("quarterly results", "reports quarterly", "earnings per share", "earnings report"),
        frozenset({"earnings", "revenue", "eps", "profit", "quarterly"}),
    ),
    EventRule(
        "merger_acquisition",
        ("to acquire", "agrees to acquire", "merger agreement", "takeover bid"),
        frozenset({"acquire", "acquisition", "merger", "takeover", "buyout"}),
    ),
    EventRule(
        "analyst_rating",
        ("price target", "initiates coverage", "raises target", "cuts target"),
        frozenset({"upgrade", "upgraded", "downgrade", "downgraded", "overweight", "underweight"}),
    ),
    EventRule(
        "regulatory",
        ("regulatory approval", "antitrust investigation", "regulatory inquiry", "sec investigation"),
        frozenset({"regulator", "regulatory", "antitrust", "probe", "investigation", "sec"}),
    ),
    EventRule(
        "legal",
        ("class action", "legal settlement", "court ruling"),
        frozenset({"lawsuit", "court", "settlement", "verdict", "appeal", "litigation"}),
    ),
    EventRule(
        "management",
        ("chief executive officer", "chief financial officer", "steps down", "named ceo"),
        frozenset({"ceo", "cfo", "chair", "resigns", "appointed", "management"}),
    ),
    EventRule(
        "cybersecurity",
        ("data breach", "cyber attack", "security incident"),
        frozenset({"cyber", "breach", "ransomware", "hack", "cybersecurity"}),
    ),
    EventRule(
        "financing",
        ("public offering", "secondary offering", "debt offering", "convertible notes"),
        frozenset({"offering", "debt", "financing", "convertible", "bonds"}),
    ),
    EventRule(
        "buyback",
        ("share repurchase", "stock repurchase"),
        frozenset({"buyback", "repurchase"}),
    ),
    EventRule(
        "dividend",
        ("declares dividend", "raises dividend", "cuts dividend"),
        frozenset({"dividend", "distribution"}),
    ),
    EventRule(
        "supply_chain",
        ("supply chain", "production delay", "factory shutdown", "shipment delay"),
        frozenset({"shortage", "factory", "facility", "shipment", "supplier"}),
    ),
    EventRule(
        "partnership",
        ("strategic partnership", "strategic alliance", "joint venture", "partners with"),
        frozenset({"partnership", "alliance", "collaboration"}),
    ),
    EventRule(
        "contract",
        ("wins contract", "awarded contract", "multi year contract", "purchase agreement"),
        frozenset({"contract", "order", "deal"}),
    ),
    EventRule(
        "restructuring",
        ("restructuring plan", "cost reduction plan", "chapter 11"),
        frozenset({"restructuring", "bankruptcy", "reorganization"}),
    ),
    EventRule(
        "workforce",
        ("job cuts", "workforce reduction", "plans layoffs"),
        frozenset({"layoffs", "layoff", "headcount"}),
    ),
    EventRule(
        "product_launch",
        ("launches new", "unveils new", "introduces new", "product launch"),
        frozenset({"launch", "launches", "unveils", "introduces"}),
    ),
    EventRule(
        "operations",
        ("opens facility", "closes facility", "capacity expansion", "production resumes"),
        frozenset({"capacity", "production", "operations"}),
    ),
    EventRule(
        "macro_geopolitical",
        ("interest rate", "trade restriction", "export controls", "economic slowdown"),
        frozenset({"tariff", "inflation", "recession", "sanctions", "geopolitical"}),
    ),
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
        "awarded",
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
        "layoffs",
        "delay",
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


@dataclass(frozen=True)
class EventClassification:
    primary: str
    secondary: tuple[str, ...]
    confidence: float
    scores: dict[str, float]


@dataclass(frozen=True)
class _HistoryItem:
    tokens: frozenset[str]
    article_version: int


def tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))


def fingerprint(headline: str, source: str, symbol: str = "") -> str:
    normalized = " ".join(sorted(tokenize(headline)))
    return hashlib.sha256(
        f"{symbol.upper().strip()}|{source.lower().strip()}|{normalized}".encode()
    ).hexdigest()


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def classify_event(text: str) -> EventClassification:
    lowered = " ".join(_TOKEN.findall(text.lower()))
    tokens = frozenset(lowered.split())
    scores: Counter[str] = Counter()
    for rule in _EVENT_RULES:
        for phrase in rule.phrases:
            if phrase in lowered:
                scores[rule.label] += 3.0
        scores[rule.label] += len(tokens & rule.keywords)

    ranked = sorted(
        ((label, score) for label, score in scores.items() if score > 0),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return EventClassification("other", (), 0.0, {})
    primary, primary_score = ranked[0]
    secondary = tuple(
        label
        for label, score in ranked[1:]
        if score >= 1.0 and score >= primary_score * 0.25
    )
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = min(1.0, 0.45 + primary_score * 0.09 + max(0.0, primary_score - second_score) * 0.04)
    return EventClassification(
        primary=primary,
        secondary=secondary,
        confidence=confidence,
        scores={label: score for label, score in ranked},
    )


def extract_event_attributes(text: str, sentiment: float) -> dict[str, float | str | bool]:
    attributes: dict[str, float | str | bool] = {
        "impact_direction": "positive" if sentiment > 0.15 else "negative" if sentiment < -0.15 else "neutral"
    }
    for index, match in enumerate(_PERCENT.finditer(text), start=1):
        if index > 3:
            break
        attributes[f"percentage_{index}"] = float(match.group(1))
    for index, match in enumerate(_MONEY.finditer(text), start=1):
        if index > 3:
            break
        attributes[f"money_{index}_currency"] = match.group("currency").upper()
        attributes[f"money_{index}_value"] = float(match.group("value"))
        unit = match.group("unit")
        if unit:
            attributes[f"money_{index}_unit"] = unit.lower()
    eps_match = _EPS.search(text)
    if eps_match is not None:
        attributes["eps_value"] = float(eps_match.group("value"))
    return attributes


class NewsIntelligence:
    """Deterministic point-in-time enrichment before higher-capacity NLP models."""

    def __init__(
        self,
        history_per_symbol: int = 100,
        *,
        entity_catalog: EntityCatalog | None = None,
        drop_exact_duplicates: bool = True,
    ) -> None:
        self._history: dict[str, deque[_HistoryItem]] = defaultdict(
            lambda: deque(maxlen=history_per_symbol)
        )
        self._fingerprints: set[str] = set()
        self.entity_catalog = entity_catalog or EntityCatalog()
        self.drop_exact_duplicates = drop_exact_duplicates

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
        symbol = symbol.upper().strip()
        clean_headline = headline.strip()
        clean_summary = summary.strip()
        clean_source = source.strip() or "unknown"
        identifier = fingerprint(clean_headline, clean_source, symbol)
        if self.drop_exact_duplicates and identifier in self._fingerprints:
            return None

        combined_text = f"{clean_headline} {clean_summary}".strip()
        tokens = tokenize(combined_text)
        similarities = [jaccard(tokens, prior.tokens) for prior in self._history[symbol]]
        novelty = 1.0 - max(similarities, default=0.0)
        related_versions = [
            prior.article_version
            for prior in self._history[symbol]
            if jaccard(tokens, prior.tokens) >= 0.72
        ]
        article_version = max(related_versions, default=0) + 1

        positive = len(tokens & _POSITIVE)
        negative = len(tokens & _NEGATIVE)
        sentiment = math.tanh((positive - negative) / 2.0)
        classification = classify_event(combined_text)
        source_key = clean_source.lower()
        source_quality = next(
            (
                quality
                for known_source, quality in _SOURCE_QUALITY.items()
                if known_source in source_key
            ),
            0.72,
        )

        self._fingerprints.add(identifier)
        self._history[symbol].append(_HistoryItem(tokens=tokens, article_version=article_version))
        return NewsEvent(
            symbol=symbol,
            headline=clean_headline,
            summary=clean_summary,
            source=clean_source,
            sentiment=sentiment,
            novelty=max(0.0, min(1.0, novelty)),
            source_quality=source_quality,
            event_type=classification.primary,
            secondary_event_types=list(classification.secondary),
            extraction_confidence=classification.confidence,
            entities=self.entity_catalog.link(symbol, combined_text),
            event_attributes=extract_event_attributes(combined_text, sentiment),
            content_fingerprint=identifier,
            article_version=article_version,
            event_time=event_time.astimezone(UTC),
            knowledge_time=(knowledge_time or datetime.now(UTC)).astimezone(UTC),
        )
