from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .config import Settings, get_settings
from .domain import NewsEvent
from .historical import HistoricalBar
from .model_registry import ModelRegistry
from .model_strategy import ChampionModelStrategy
from .replay import ReplayResult, run_historical_replay
from .strategy import ExplainableCatalystStrategy, Strategy
from .universe import (
    assert_universe_binding,
    assert_universe_symbols,
    assert_universe_time_range,
    load_universe_manifest,
)

ModelType = TypeVar("ModelType", bound=BaseModel)


async def replay_history(
    bars_path: str,
    news_path: str,
    output: str,
    *,
    trace_output: str | None,
    strategy_mode: str,
    registry_path: str,
    allow_synthetic_champion: bool,
    verify_determinism: bool,
    bar_minutes: int,
    spread_bps: float,
    slippage_bps: float,
    starting_cash: float | None,
    signal_threshold: float | None = None,
    universe_manifest_path: str | None = None,
) -> dict[str, object]:
    bars_source = Path(bars_path)
    news_source = Path(news_path)
    bars, news = load_replay_inputs(bars_source, news_source)
    universe = (
        None
        if universe_manifest_path is None
        else load_universe_manifest(universe_manifest_path)
    )
    if universe is not None:
        assert_universe_symbols(
            {bar.symbol for bar in bars},
            universe,
            context="replay bars",
        )
        assert_universe_time_range(
            [bar.timestamp for bar in bars],
            universe,
            context="replay bars",
        )
    settings = replay_settings(starting_cash)
    strategy_factory, strategy_name, resolved_threshold = build_strategy_factory(
        strategy_mode,
        registry_path,
        allow_synthetic_champion=allow_synthetic_champion,
        signal_threshold=signal_threshold,
    )
    if universe is not None and strategy_mode == "champion":
        champion = ModelRegistry(registry_path).champion()
        if champion is None:
            raise ValueError("Champion replay requested but registry has no champion")
        assert_universe_binding(
            {"universe": champion.metadata.get("universe")},
            universe,
            context="replay champion",
        )
    sources = replay_source_metadata(bars_source, news_source, bars, news)
    if universe is not None:
        sources["universe"] = universe.binding(universe_manifest_path)

    async def execute() -> ReplayResult:
        return await run_historical_replay(
            bars,
            news,
            settings,
            strategy_factory,
            strategy_name=strategy_name,
            bar_interval=minutes(bar_minutes),
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            signal_threshold=resolved_threshold,
            source_metadata=sources,
        )

    result = await execute()
    report = result.report
    report["determinism_verified"] = False
    if verify_determinism:
        verification = await execute()
        first_events = report.get("events", {})
        second_events = verification.report.get("events", {})
        if not isinstance(first_events, dict) or not isinstance(second_events, dict):
            raise RuntimeError("Replay report did not contain event diagnostics")
        first_hash = str(first_events["trace_sha256"])
        second_hash = str(second_events["trace_sha256"])
        if first_hash != second_hash:
            raise RuntimeError(
                "Replay determinism verification failed: semantic trace hashes differ"
            )
        report["determinism_verified"] = True

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report["output"] = str(destination)
    if trace_output is not None:
        trace_destination = Path(trace_output)
        trace_destination.parent.mkdir(parents=True, exist_ok=True)
        trace_destination.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in result.trace),
            encoding="utf-8",
        )
        report["trace_output"] = str(trace_destination)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_strategy_factory(
    strategy_mode: str,
    registry_path: str,
    *,
    allow_synthetic_champion: bool,
    signal_threshold: float | None,
) -> tuple[Callable[[], Strategy], str, float]:
    if strategy_mode == "explainable":
        threshold = 0.16 if signal_threshold is None else signal_threshold
        if threshold <= 0:
            raise ValueError("Explainable signal threshold must be positive")

        def explainable_factory() -> Strategy:
            return ExplainableCatalystStrategy(
                minimum_combined=threshold,
                repeat_combined=max(0.30, threshold),
            )

        return explainable_factory, f"explainable:{threshold:g}", threshold
    registry = ModelRegistry(registry_path)
    record = registry.champion()
    if record is None:
        raise ValueError("Champion replay requested but the registry has no champion")
    dataset = record.metadata.get("dataset")
    if dataset == "synthetic_fixture" and not allow_synthetic_champion:
        raise ValueError(
            "Synthetic champion replay is disabled; use a genuine-data champion or pass "
            "--allow-synthetic-champion for pipeline diagnostics only"
        )
    model = registry.load_champion()
    if model is None:
        raise ValueError("Champion model artifact could not be loaded")
    threshold = 0.0005 if signal_threshold is None else signal_threshold
    if threshold <= 0:
        raise ValueError("Champion signal threshold must be positive")

    def champion_factory() -> Strategy:
        return ChampionModelStrategy(model, minimum_edge=threshold)

    return champion_factory, f"champion:{record.version}:{threshold:g}", threshold


def load_replay_inputs(
    bars_source: Path,
    news_source: Path,
) -> tuple[list[HistoricalBar], list[NewsEvent]]:
    return (
        _load_json_lines(bars_source, HistoricalBar),
        _load_json_lines(news_source, NewsEvent, allow_empty=True),
    )


def replay_settings(starting_cash: float | None) -> Settings:
    settings = get_settings()
    if starting_cash is not None:
        settings = settings.model_copy(update={"trading_starting_cash": starting_cash})
    return settings


def replay_source_metadata(
    bars_source: Path,
    news_source: Path,
    bars: list[HistoricalBar],
    news: list[NewsEvent],
) -> dict[str, object]:
    return {
        "bars": {
            "path": str(bars_source),
            "records": len(bars),
            "sha256": _sha256(bars_source),
        },
        "news": {
            "path": str(news_source),
            "records": len(news),
            "sha256": _sha256(news_source),
        },
    }


def _load_json_lines(
    path: Path,
    model_type: type[ModelType],
    *,
    allow_empty: bool = False,
) -> list[ModelType]:
    records: list[ModelType] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model_type.model_validate_json(line))
            except ValueError as error:
                raise ValueError(
                    f"Invalid {model_type.__name__} at {path}:{line_number}"
                ) from error
    if not records and not allow_empty:
        raise ValueError(f"No records found in {path}")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def minutes(value: int) -> timedelta:
    if value <= 0:
        raise ValueError("bar_minutes must be positive")
    return timedelta(minutes=value)
