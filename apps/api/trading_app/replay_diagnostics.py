from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path
from statistics import fmean, median, pstdev

from .historical import HistoricalBar
from .model_registry import ModelRegistry
from .replay import ReplayResult, run_historical_replay
from .replay_cli import (
    build_strategy_factory,
    load_replay_inputs,
    minutes,
    replay_settings,
    replay_source_metadata,
)
from .universe import (
    assert_universe_binding,
    assert_universe_symbols,
    assert_universe_time_range,
    load_universe_manifest,
)


async def build_replay_diagnostics(
    bars_path: str,
    news_path: str,
    output: str,
    *,
    strategy_mode: str,
    registry_path: str,
    allow_synthetic_champion: bool,
    verify_determinism: bool,
    bar_minutes: int,
    spread_bps: float,
    slippage_bps: float,
    starting_cash: float | None,
    thresholds: list[float],
    reference_threshold: float | None,
    sector_map_path: str | None,
    regime_lookback: int,
    regime_momentum_threshold: float,
    universe_manifest_path: str | None = None,
) -> dict[str, object]:
    if not thresholds:
        raise ValueError("At least one threshold is required")
    if any(value <= 0 for value in thresholds):
        raise ValueError("All thresholds must be positive")
    if regime_lookback < 3:
        raise ValueError("regime_lookback must be at least three")
    if regime_momentum_threshold <= 0:
        raise ValueError("regime_momentum_threshold must be positive")

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
            context="replay diagnostics bars",
        )
        assert_universe_time_range(
            [bar.timestamp for bar in bars],
            universe,
            context="replay diagnostics bars",
        )
        if strategy_mode == "champion":
            champion = ModelRegistry(registry_path).champion()
            if champion is None:
                raise ValueError(
                    "Champion replay diagnostics requested but registry has no champion"
                )
            assert_universe_binding(
                {"universe": champion.metadata.get("universe")},
                universe,
                context="replay diagnostics champion",
            )
    settings = replay_settings(starting_cash)
    sources = replay_source_metadata(bars_source, news_source, bars, news)
    if universe is not None:
        sources["universe"] = universe.binding(universe_manifest_path)
    interval = minutes(bar_minutes)
    unique_thresholds = list(dict.fromkeys(float(value) for value in thresholds))
    default_reference = 0.16 if strategy_mode == "explainable" else 0.0005
    target_reference = default_reference if reference_threshold is None else reference_threshold
    selected_reference = min(unique_thresholds, key=lambda value: abs(value - target_reference))

    replay_results: dict[float, ReplayResult] = {}
    threshold_summaries: list[dict[str, object]] = []
    all_verified = verify_determinism
    for threshold in unique_thresholds:
        factory, strategy_name, resolved_threshold = build_strategy_factory(
            strategy_mode,
            registry_path,
            allow_synthetic_champion=allow_synthetic_champion,
            signal_threshold=threshold,
        )

        async def execute() -> ReplayResult:
            return await run_historical_replay(
                bars,
                news,
                settings,
                factory,
                strategy_name=strategy_name,
                bar_interval=interval,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                signal_threshold=resolved_threshold,
                source_metadata=sources,
            )

        result = await execute()
        replay_results[threshold] = result
        verified = False
        if verify_determinism:
            verification = await execute()
            verified = _trace_hash(result) == _trace_hash(verification)
            if not verified:
                raise RuntimeError(
                    f"Replay determinism verification failed for threshold {threshold:g}"
                )
        all_verified = all_verified and verified
        performance = _mapping(result.report.get("performance"))
        events = _mapping(result.report.get("events"))
        counts = _mapping(events.get("counts"))
        threshold_summaries.append(
            {
                "threshold": threshold,
                "strategy": result.report["strategy"],
                "determinism_verified": verified,
                "trace_sha256": _trace_hash(result),
                "net_return": performance.get("net_return", 0.0),
                "ending_equity": performance.get("ending_equity", 0.0),
                "maximum_drawdown": performance.get("maximum_drawdown", 0.0),
                "traded_notional": performance.get("traded_notional", 0.0),
                "proposals": counts.get("proposal", 0),
                "fills": counts.get("fill", 0),
                "risk_rejection_reasons": events.get("risk_rejection_reasons", {}),
            }
        )

    reference_result = replay_results[selected_reference]
    reference_summary = next(
        item for item in threshold_summaries if item["threshold"] == selected_reference
    )
    for summary in threshold_summaries:
        summary["delta_vs_reference"] = {
            "net_return": float(summary["net_return"])
            - float(reference_summary["net_return"]),
            "maximum_drawdown": float(summary["maximum_drawdown"])
            - float(reference_summary["maximum_drawdown"]),
            "fills": int(summary["fills"]) - int(reference_summary["fills"]),
        }

    sector_map = _load_sector_map(sector_map_path)
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "historical_replay_diagnostics",
        "strategy_mode": strategy_mode,
        "reference_threshold": selected_reference,
        "threshold_sensitivity": threshold_summaries,
        "regimes": _regime_diagnostics(
            bars,
            reference_result,
            bar_minutes=bar_minutes,
            lookback=regime_lookback,
            momentum_threshold=regime_momentum_threshold,
        ),
        "sectors": _sector_diagnostics(reference_result, sector_map),
        "event_types": _event_type_diagnostics(reference_result),
        "sources": sources,
        "configuration": {
            "bar_minutes": bar_minutes,
            "synthetic_spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "regime_lookback": regime_lookback,
            "regime_momentum_threshold": regime_momentum_threshold,
            "sector_map": sector_map_path,
        },
        "determinism_verified": all_verified,
        "limitations": [
            "Threshold sensitivity is exploratory and must not be used to select a champion on the same evaluation window.",
            "Regime labels are retrospective diagnostics, not point-in-time model features.",
            "Sector results depend on the supplied symbol-to-sector map; missing symbols are reported as Unmapped.",
            "Historical replay remains paper-only evidence and does not replace sustained live paper validation.",
        ],
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report["output"] = str(destination)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _regime_diagnostics(
    bars: list[HistoricalBar],
    replay: ReplayResult,
    *,
    bar_minutes: int,
    lookback: int,
    momentum_threshold: float,
) -> dict[str, object]:
    market_returns = _equal_weight_market_returns(bars, bar_minutes)
    equity_returns = _equity_returns(replay)
    joined: list[dict[str, object]] = []
    rolling_market: list[float] = []
    for timestamp, market_return in market_returns:
        rolling_market.append(market_return)
        if len(rolling_market) > lookback:
            rolling_market.pop(0)
        if len(rolling_market) < lookback or timestamp not in equity_returns:
            continue
        momentum = math.prod(1 + value for value in rolling_market) - 1
        volatility = pstdev(rolling_market) if len(rolling_market) > 1 else 0.0
        joined.append(
            {
                "timestamp": timestamp,
                "strategy_return": equity_returns[timestamp],
                "market_return": market_return,
                "momentum": momentum,
                "volatility": volatility,
            }
        )
    if not joined:
        return {"volatility_split": 0.0, "buckets": {}}
    volatility_split = median(float(item["volatility"]) for item in joined)
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in joined:
        momentum = float(item["momentum"])
        if momentum > momentum_threshold:
            direction = "bull"
        elif momentum < -momentum_threshold:
            direction = "bear"
        else:
            direction = "sideways"
        volatility = (
            "high_vol" if float(item["volatility"]) > volatility_split else "low_vol"
        )
        buckets[f"{direction}_{volatility}"].append(item)
    return {
        "volatility_split": volatility_split,
        "buckets": {
            name: _regime_bucket(values) for name, values in sorted(buckets.items())
        },
    }


def _equal_weight_market_returns(
    bars: list[HistoricalBar],
    bar_minutes: int,
) -> list[tuple[str, float]]:
    interval_seconds = bar_minutes * 60
    ordered = sorted(
        bars,
        key=lambda item: (item.timestamp.astimezone(UTC), item.symbol),
    )
    previous: dict[str, float] = {}
    results: list[tuple[str, float]] = []
    for timestamp, group_iter in groupby(
        ordered, key=lambda item: item.timestamp.astimezone(UTC)
    ):
        returns: list[float] = []
        for bar in group_iter:
            if bar.symbol in previous:
                returns.append(bar.close / previous[bar.symbol] - 1)
            previous[bar.symbol] = bar.close
        close_time = datetime.fromtimestamp(
            timestamp.timestamp() + interval_seconds,
            tz=UTC,
        ).isoformat()
        results.append((close_time, fmean(returns) if returns else 0.0))
    return results


def _equity_returns(replay: ReplayResult) -> dict[str, float]:
    configuration = _mapping(replay.report.get("configuration"))
    previous = float(configuration.get("starting_cash", 0.0))
    returns: dict[str, float] = {}
    for point in replay.equity_curve:
        equity = float(point["equity"])
        returns[str(point["timestamp"])] = 0.0 if previous <= 0 else equity / previous - 1
        previous = equity
    return returns


def _regime_bucket(items: list[dict[str, object]]) -> dict[str, object]:
    strategy_returns = [float(item["strategy_return"]) for item in items]
    market_returns = [float(item["market_return"]) for item in items]
    return {
        "periods": len(items),
        "strategy_net_return": math.prod(1 + value for value in strategy_returns) - 1,
        "market_net_return": math.prod(1 + value for value in market_returns) - 1,
        "average_strategy_return": fmean(strategy_returns),
        "strategy_hit_rate": sum(value > 0 for value in strategy_returns) / len(items),
        "average_market_momentum": fmean(float(item["momentum"]) for item in items),
        "average_market_volatility": fmean(
            float(item["volatility"]) for item in items
        ),
        "maximum_drawdown": _maximum_drawdown(strategy_returns),
    }


def _sector_diagnostics(
    replay: ReplayResult,
    sector_map: dict[str, str],
) -> dict[str, object]:
    performance = _mapping(replay.report.get("performance"))
    events = _mapping(replay.report.get("events"))
    pnl_by_symbol = _mapping(performance.get("pnl_by_symbol"))
    fills_by_symbol = _mapping(events.get("fills_by_symbol"))
    sectors: dict[str, dict[str, object]] = defaultdict(
        lambda: {"symbols": [], "pnl": 0.0, "fills": 0, "filled_notional": 0.0}
    )
    symbols = sorted(set(pnl_by_symbol) | set(fills_by_symbol))
    symbol_rows: dict[str, dict[str, object]] = {}
    for symbol in symbols:
        sector = sector_map.get(symbol.upper(), "Unmapped")
        fill_data = _mapping(fills_by_symbol.get(symbol))
        pnl = float(pnl_by_symbol.get(symbol, 0.0))
        fills = int(fill_data.get("fills", 0))
        notional = float(fill_data.get("notional", 0.0))
        symbol_rows[symbol] = {
            "sector": sector,
            "pnl": pnl,
            "fills": fills,
            "filled_notional": notional,
        }
        sectors[sector]["symbols"].append(symbol)
        sectors[sector]["pnl"] = float(sectors[sector]["pnl"]) + pnl
        sectors[sector]["fills"] = int(sectors[sector]["fills"]) + fills
        sectors[sector]["filled_notional"] = (
            float(sectors[sector]["filled_notional"]) + notional
        )
    return {
        "symbols": symbol_rows,
        "groups": {name: values for name, values in sorted(sectors.items())},
    }


def _event_type_diagnostics(replay: ReplayResult) -> dict[str, object]:
    events = _mapping(replay.report.get("events"))
    activity = _mapping(events.get("event_type_activity"))
    total_proposals = sum(
        int(_mapping(value).get("proposals", 0)) for value in activity.values()
    )
    return {
        "total_proposals": total_proposals,
        "groups": activity,
    }


def _load_sector_map(path: str | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Sector map must be a JSON object of symbol to sector")
    return {str(symbol).upper(): str(sector) for symbol, sector in payload.items()}


def _maximum_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    drawdown = 0.0
    for value in returns:
        equity *= 1 + value
        peak = max(peak, equity)
        if peak > 0:
            drawdown = max(drawdown, (peak - equity) / peak)
    return drawdown


def _trace_hash(result: ReplayResult) -> str:
    events = _mapping(result.report.get("events"))
    return str(events.get("trace_sha256", ""))


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}
