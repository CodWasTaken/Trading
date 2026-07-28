from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


_NORMALIZE = re.compile(r"[^a-z0-9]+")


class EntityRelation(StrEnum):
    ISSUER = "issuer"
    SUBSIDIARY = "subsidiary"
    SUPPLIER = "supplier"
    CUSTOMER = "customer"
    COMPETITOR = "competitor"
    PARTNER = "partner"
    REGULATOR = "regulator"
    PERSON = "person"


class LinkedEntity(BaseModel):
    canonical_name: str
    relation: EntityRelation
    ticker: str | None = None
    sector: str | None = None
    matched_text: str
    confidence: float = Field(ge=0, le=1)
    link_method: str


class RelatedEntity(BaseModel):
    canonical_name: str
    relation: EntityRelation
    aliases: list[str] = Field(default_factory=list)
    ticker: str | None = None
    effective_from: datetime | None = None
    effective_to: datetime | None = None

    @field_validator("canonical_name")
    @classmethod
    def require_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("canonical_name must not be empty")
        return value

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str | None) -> str | None:
        return value.upper().strip() if value else None

    @field_validator("effective_from", "effective_to")
    @classmethod
    def normalize_effective_time(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_effective_window(self) -> RelatedEntity:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from >= self.effective_to
        ):
            raise ValueError("effective_from must be earlier than effective_to")
        return self

    def active_at(self, as_of: datetime | None) -> bool:
        if as_of is None:
            return True
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        as_of = as_of.astimezone(UTC)
        return (
            (self.effective_from is None or self.effective_from <= as_of)
            and (self.effective_to is None or as_of < self.effective_to)
        )


class IssuerProfile(BaseModel):
    ticker: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    sector: str | None = None
    related_entities: list[RelatedEntity] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def normalize_profile_ticker(cls, value: str) -> str:
        value = value.upper().strip()
        if not value:
            raise ValueError("ticker must not be empty")
        return value

    @field_validator("canonical_name")
    @classmethod
    def require_canonical_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("canonical_name must not be empty")
        return value

    @model_validator(mode="after")
    def reject_ambiguous_aliases(self) -> IssuerProfile:
        owners: dict[str, str] = {}
        entries = [(self.canonical_name, "issuer"), *[(alias, "issuer") for alias in self.aliases]]
        for entity in self.related_entities:
            entries.append((entity.canonical_name, entity.canonical_name))
            entries.extend((alias, entity.canonical_name) for alias in entity.aliases)
        for alias, owner in entries:
            normalized = normalize_entity_text(alias)
            if not normalized:
                raise ValueError("entity aliases must contain letters or numbers")
            existing = owners.get(normalized)
            if existing is not None and existing != owner:
                raise ValueError(
                    f"Ambiguous alias {alias!r} is assigned to both {existing!r} and {owner!r}"
                )
            owners[normalized] = owner
        return self


class EntityCatalogPayload(BaseModel):
    issuers: list[IssuerProfile]

    @model_validator(mode="after")
    def reject_duplicate_tickers(self) -> EntityCatalogPayload:
        tickers = [profile.ticker for profile in self.issuers]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Entity catalog contains duplicate issuer tickers")
        return self


class EntityCatalog:
    """Explicit, context-scoped issuer relationship catalog.

    Relationships are never inferred from co-mentions. A related entity is linked only when
    an alias in the user-maintained catalog appears as a complete normalized phrase and the
    relationship is active at the supplied point-in-time boundary.
    """

    def __init__(self, profiles: list[IssuerProfile] | None = None) -> None:
        self._profiles = {profile.ticker: profile for profile in profiles or []}

    @classmethod
    def load(cls, path: str | Path | None) -> EntityCatalog:
        if path is None:
            return cls()
        source = Path(path)
        payload = EntityCatalogPayload.model_validate_json(source.read_text(encoding="utf-8"))
        return cls(payload.issuers)

    def link(
        self,
        symbol: str,
        text: str,
        *,
        as_of: datetime | None = None,
    ) -> list[LinkedEntity]:
        symbol = symbol.upper().strip()
        profile = self._profiles.get(symbol)
        issuer_name = profile.canonical_name if profile else symbol
        issuer_sector = profile.sector if profile else None
        linked = [
            LinkedEntity(
                canonical_name=issuer_name,
                relation=EntityRelation.ISSUER,
                ticker=symbol,
                sector=issuer_sector,
                matched_text=symbol,
                confidence=1.0,
                link_method="provider_symbol",
            )
        ]
        if profile is None:
            return linked

        normalized_text = f" {normalize_entity_text(text)} "
        matches: list[tuple[int, LinkedEntity]] = []
        for entity in profile.related_entities:
            if not entity.active_at(as_of):
                continue
            candidates = [entity.canonical_name, *entity.aliases]
            match = _longest_phrase_match(normalized_text, candidates)
            if match is None:
                continue
            matches.append(
                (
                    len(normalize_entity_text(match)),
                    LinkedEntity(
                        canonical_name=entity.canonical_name,
                        relation=entity.relation,
                        ticker=entity.ticker,
                        sector=None,
                        matched_text=match,
                        confidence=0.95,
                        link_method="catalog_alias",
                    ),
                )
            )
        linked.extend(
            item
            for _, item in sorted(matches, key=lambda pair: (-pair[0], pair[1].canonical_name))
        )
        return linked

    def summary(self) -> dict[str, object]:
        relationships = [
            entity
            for profile in self._profiles.values()
            for entity in profile.related_entities
        ]
        return {
            "issuers": len(self._profiles),
            "relationships": len(relationships),
            "dated_relationships": sum(
                entity.effective_from is not None or entity.effective_to is not None
                for entity in relationships
            ),
            "tickers": sorted(self._profiles),
        }


def normalize_entity_text(value: str) -> str:
    return " ".join(part for part in _NORMALIZE.sub(" ", value.lower()).split() if part)


def _longest_phrase_match(normalized_text: str, candidates: list[str]) -> str | None:
    ordered = sorted(
        candidates,
        key=lambda value: (-len(normalize_entity_text(value)), value.lower()),
    )
    for candidate in ordered:
        normalized = normalize_entity_text(candidate)
        if normalized and f" {normalized} " in normalized_text:
            return candidate
    return None
