from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UniverseConstituent(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    sector: str
    exchange: str
    security_type: str = "common_stock"


class UniverseSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class UniverseManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    universe_id: str
    effective_date: date
    constituents: tuple[UniverseConstituent, ...]
    inclusion_rules: tuple[str, ...]
    minimum_median_daily_dollar_volume_usd: float = Field(gt=0)
    minimum_price_usd: float = Field(gt=0)
    minimum_historical_coverage_years: float = Field(gt=0)
    shortability_requirement: str
    sources: tuple[UniverseSource, ...]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_constituents(self) -> UniverseManifest:
        symbols = self.symbols
        if not 40 <= len(symbols) <= 50:
            raise ValueError("Frozen major-stock universe must contain 40 to 50 symbols")
        if len(symbols) != len(set(symbols)):
            raise ValueError("Universe contains duplicate symbols")
        if len({item.sector for item in self.constituents}) < 8:
            raise ValueError("Universe must span at least eight sectors")
        if any(item.security_type != "common_stock" for item in self.constituents):
            raise ValueError("Universe may contain only common stocks")
        expected = manifest_hash(self.model_dump(mode="json"))
        if self.manifest_sha256 != expected:
            raise ValueError(
                "Universe manifest hash mismatch: "
                f"declared={self.manifest_sha256}, computed={expected}"
            )
        return self

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(item.symbol for item in self.constituents)

    @property
    def sectors(self) -> dict[str, str]:
        return {item.symbol: item.sector for item in self.constituents}

    def binding(self, manifest_path: str | Path) -> dict[str, object]:
        return {
            "universe_id": self.universe_id,
            "manifest_path": str(manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "effective_date": self.effective_date.isoformat(),
            "symbols": list(self.symbols),
        }


def manifest_hash(payload: dict[str, object]) -> str:
    canonical = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_universe_manifest(path: str | Path) -> UniverseManifest:
    source = Path(path)
    if not source.is_absolute() and not source.is_file():
        repository_source = Path(__file__).resolve().parents[3] / source
        if repository_source.is_file():
            source = repository_source
    manifest = UniverseManifest.model_validate_json(source.read_text(encoding="utf-8"))
    for item in manifest.sources:
        evidence = (source.parent / item.path).resolve()
        if not evidence.is_file():
            raise ValueError(f"Universe source is missing: {evidence}")
        actual = _sha256(evidence)
        if actual != item.sha256:
            raise ValueError(
                f"Universe source hash mismatch for {item.name}: "
                f"declared={item.sha256}, actual={actual}"
            )
        if evidence.suffix.lower() == ".csv":
            _assert_constituent_source(evidence, manifest)
    return manifest


def assert_universe_symbols(
    actual_symbols: set[str] | list[str] | tuple[str, ...],
    manifest: UniverseManifest,
    *,
    context: str,
) -> None:
    actual = {symbol.upper() for symbol in actual_symbols}
    expected = set(manifest.symbols)
    if actual != expected:
        raise ValueError(
            f"{context} universe mismatch for {manifest.universe_id}: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def assert_universe_time_range(
    timestamps: list[datetime],
    manifest: UniverseManifest,
    *,
    context: str,
) -> None:
    if not timestamps:
        raise ValueError(f"{context} has no timestamps")
    earliest = min(item.date() for item in timestamps)
    if earliest < manifest.effective_date:
        raise ValueError(
            f"{context} starts before frozen universe effective date: "
            f"start={earliest}, effective={manifest.effective_date}"
        )


def assert_universe_binding(
    metadata: dict[str, object],
    manifest: UniverseManifest,
    *,
    context: str,
) -> None:
    binding = metadata.get("universe")
    if not isinstance(binding, dict):
        raise ValueError(f"{context} metadata does not contain a frozen universe binding")
    if binding.get("manifest_sha256") != manifest.manifest_sha256:
        raise ValueError(f"{context} universe manifest hash mismatch")
    if binding.get("universe_id") != manifest.universe_id:
        raise ValueError(f"{context} universe_id mismatch")
    symbols = binding.get("symbols")
    if not isinstance(symbols, list):
        raise ValueError(f"{context} universe binding does not contain symbols")
    assert_universe_symbols(symbols, manifest, context=context)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_constituent_source(
    path: Path,
    manifest: UniverseManifest,
) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    source_rows = {
        (
            str(row.get("symbol", "")),
            str(row.get("sector", "")),
            str(row.get("exchange", "")),
            str(row.get("security_type", "")),
        )
        for row in rows
    }
    manifest_rows = {
        (item.symbol, item.sector, item.exchange, item.security_type)
        for item in manifest.constituents
    }
    if source_rows != manifest_rows:
        raise ValueError("Universe constituent source does not match manifest contents")
