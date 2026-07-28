from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .audit import audit_ledger
from .config import get_settings
from .dataset import (
    HISTORICAL_FEATURE_NAMES,
    HistoricalPointInTimeDatasetBuilder,
    load_feature_dataset,
    write_feature_dataset,
)
from .domain import NewsEvent
from .historical import AlpacaHistoricalClient, HistoricalBar
from .model_registry import ModelRegistry
from .replay_cli import replay_history
from .research import (
    RidgeReturnModel,
    metrics_dict,
    synthetic_feature_rows,
    walk_forward,
    walk_forward_report,
)


FEATURE_NAMES = ("momentum", "news_score", "volatility", "spread_bps")
ModelType = TypeVar("ModelType", bound=BaseModel)


def demo_train(rows: int, registry_path: str, promote: bool) -> dict[str, object]:
    dataset = synthetic_feature_rows(rows)
    metrics = walk_forward(dataset, FEATURE_NAMES)
    model = RidgeReturnModel(FEATURE_NAMES)
    model.fit(dataset)
    registry = ModelRegistry(registry_path)
    record = registry.register(
        model,
        metrics,
        metadata={
            "dataset": "synthetic_fixture",
            "warning": "For pipeline verification only; not market evidence",
            "rows": rows,
        },
    )
    promoted = False
    if promote:
        registry.promote(record.version)
        promoted = True
    return {
        "version": record.version,
        "promoted": promoted,
        "metrics": metrics_dict(metrics),
        "registry": registry_path,
    }


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


async def backfill(
    kind: str,
    symbols: list[str],
    start: datetime,
    end: datetime,
    output: str,
    timeframe: str,
) -> dict[str, object]:
    settings = get_settings()
    client = AlpacaHistoricalClient(settings)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with destination.open("w", encoding="utf-8") as handle:
            if kind == "bars":
                iterator = client.iter_bars(symbols, start, end, timeframe=timeframe)
            else:
                iterator = client.iter_news(symbols, start, end)
            async for item in iterator:
                handle.write(item.model_dump_json() + "\n")
                count += 1
    finally:
        await client.close()
    return {
        "kind": kind,
        "symbols": symbols,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "records": count,
        "output": str(destination),
    }


def build_dataset(
    bars_path: str,
    news_path: str,
    output: str,
    *,
    lookback_bars: int,
    forecast_bars: int,
    bar_minutes: int,
    news_window_hours: float,
) -> dict[str, object]:
    bars_source = Path(bars_path)
    news_source = Path(news_path)
    bars = _load_json_lines(bars_source, HistoricalBar)
    news = _load_json_lines(news_source, NewsEvent)
    builder = HistoricalPointInTimeDatasetBuilder(
        lookback_bars=lookback_bars,
        forecast_bars=forecast_bars,
        bar_interval=timedelta(minutes=bar_minutes),
        news_window=timedelta(hours=news_window_hours),
    )
    rows = builder.build(bars, news)
    if not rows:
        raise ValueError("Historical inputs did not produce any feature rows")
    symbols = sorted({row.symbol for row in rows})
    metadata: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_kind": "historical_point_in_time",
        "symbols": symbols,
        "start": rows[0].timestamp.isoformat(),
        "end": rows[-1].timestamp.isoformat(),
        "lookback_bars": lookback_bars,
        "forecast_bars": forecast_bars,
        "bar_interval_minutes": bar_minutes,
        "news_window_hours": news_window_hours,
        "sources": {
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
        },
        "point_in_time_controls": {
            "bar_feature_time": "bar_start_plus_interval",
            "purged_label_overlap": True,
            "news_visibility": "knowledge_time_lte_feature_time",
        },
        "limitations": [
            "Alpaca historical news does not expose provider receipt time; knowledge_time equals publication time.",
            "Historical OHLC bars do not contain bid/ask spread, so the model excludes spread_bps; live risk checks still enforce spread limits.",
        ],
    }
    destination, metadata_path = write_feature_dataset(
        output,
        rows,
        HISTORICAL_FEATURE_NAMES,
        metadata,
    )
    return {
        "output": str(destination),
        "metadata": str(metadata_path),
        "rows": len(rows),
        "symbols": symbols,
        "feature_names": list(HISTORICAL_FEATURE_NAMES),
        "start": rows[0].timestamp.isoformat(),
        "end": rows[-1].timestamp.isoformat(),
    }


def train_dataset(
    dataset_path: str,
    registry_path: str,
    *,
    promote: bool,
    minimum_train_rows: int,
    test_rows: int,
    transaction_cost_bps: float,
    periods_per_year: int,
    threshold: float,
    ridge: float,
    minimum_folds: int,
    minimum_sharpe: float,
    maximum_drawdown: float,
    minimum_observations: int,
    minimum_excess_return: float = 0.0,
    minimum_news_sharpe_delta: float = 0.0,
) -> dict[str, object]:
    rows, feature_names, dataset_metadata = load_feature_dataset(dataset_path)
    forecast_bars = max(1, int(dataset_metadata.get("forecast_bars", 1)))
    effective_periods_per_year = periods_per_year / forecast_bars
    report = walk_forward_report(
        rows,
        feature_names,
        minimum_train_rows=minimum_train_rows,
        test_rows=test_rows,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=effective_periods_per_year,
        threshold=threshold,
        ridge=ridge,
    )
    metrics = report.metrics
    diagnostics = report.to_dict()
    model = RidgeReturnModel(feature_names, ridge=ridge)
    model.fit(rows)
    registry = ModelRegistry(registry_path)
    record = registry.register(
        model,
        metrics,
        metadata={
            "dataset": str(Path(dataset_path)),
            "dataset_sha256": _sha256(Path(dataset_path)),
            "dataset_metadata": dataset_metadata,
            "rows": len(rows),
            "feature_names": list(feature_names),
            "validation": {
                "method": "stitched_non_overlapping_expanding_walk_forward_with_label_purge",
                "minimum_train_rows": minimum_train_rows,
                "test_rows": test_rows,
                "transaction_cost_bps": transaction_cost_bps,
                "source_periods_per_year": periods_per_year,
                "forecast_bars": forecast_bars,
                "effective_periods_per_year": effective_periods_per_year,
                "prediction_threshold": threshold,
            },
            "diagnostics": diagnostics,
            "warning": "Historical out-of-sample evidence only; paper trading validation is still required.",
        },
    )
    promoted = False
    promotion_error: str | None = None
    if promote:
        try:
            registry.promote(
                record.version,
                minimum_folds=minimum_folds,
                minimum_sharpe=minimum_sharpe,
                maximum_drawdown=maximum_drawdown,
                minimum_observations=minimum_observations,
                minimum_excess_return=minimum_excess_return,
                minimum_news_sharpe_delta=minimum_news_sharpe_delta,
            )
            promoted = True
        except ValueError as error:
            promotion_error = str(error)
    result: dict[str, object] = {
        "version": record.version,
        "registry": registry_path,
        "dataset": dataset_path,
        "rows": len(rows),
        "feature_names": list(feature_names),
        "metrics": metrics_dict(metrics),
        "diagnostics": diagnostics,
        "validation_periods": {
            "source_periods_per_year": periods_per_year,
            "forecast_bars": forecast_bars,
            "effective_periods_per_year": effective_periods_per_year,
        },
        "promote_requested": promote,
        "promoted": promoted,
        "promotion_gates": {
            "minimum_folds": minimum_folds,
            "minimum_sharpe": minimum_sharpe,
            "maximum_drawdown": maximum_drawdown,
            "minimum_observations": minimum_observations,
            "minimum_excess_return": minimum_excess_return,
            "minimum_news_sharpe_delta": minimum_news_sharpe_delta,
        },
    }
    if promotion_error is not None:
        result["promotion_error"] = promotion_error
    return result


def _load_json_lines(path: Path, model_type: type[ModelType]) -> list[ModelType]:
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
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trading data, research, registry, replay, and audit tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser(
        "demo-train", help="Verify the training pipeline on synthetic data"
    )
    train.add_argument("--rows", type=int, default=600)
    train.add_argument("--registry", default=".trading/models")
    train.add_argument("--promote", action="store_true")

    registry = subparsers.add_parser("registry", help="Print the model registry")
    registry.add_argument("--registry", default=".trading/models")

    history = subparsers.add_parser(
        "backfill", help="Download historical Alpaca bars or news to JSON Lines"
    )
    history.add_argument("kind", choices=("bars", "news"))
    history.add_argument("--symbols", required=True, help="Comma-separated tickers")
    history.add_argument("--start", required=True, help="ISO-8601 timestamp")
    history.add_argument("--end", required=True, help="ISO-8601 timestamp")
    history.add_argument("--output", required=True)
    history.add_argument("--timeframe", default="1Min")

    dataset = subparsers.add_parser(
        "build-dataset",
        help="Build a point-in-time feature dataset from historical bars and news",
    )
    dataset.add_argument("--bars", required=True, help="Historical bars JSON Lines")
    dataset.add_argument("--news", required=True, help="Historical news JSON Lines")
    dataset.add_argument("--output", required=True, help="Feature dataset JSON Lines")
    dataset.add_argument("--lookback-bars", type=int, default=19)
    dataset.add_argument("--forecast-bars", type=int, default=5)
    dataset.add_argument("--bar-minutes", type=int, default=60)
    dataset.add_argument("--news-window-hours", type=float, default=24.0)

    real_train = subparsers.add_parser(
        "train",
        help="Train and walk-forward validate a registered model from a feature dataset",
    )
    real_train.add_argument("--dataset", required=True)
    real_train.add_argument("--registry", default=".trading/models")
    real_train.add_argument("--promote", action="store_true")
    real_train.add_argument("--minimum-train-rows", type=int, default=500)
    real_train.add_argument("--test-rows", type=int, default=100)
    real_train.add_argument("--transaction-cost-bps", type=float, default=5.0)
    real_train.add_argument("--periods-per-year", type=int, default=1638)
    real_train.add_argument("--threshold", type=float, default=0.0005)
    real_train.add_argument("--ridge", type=float, default=0.001)
    real_train.add_argument("--minimum-folds", type=int, default=5)
    real_train.add_argument("--minimum-sharpe", type=float, default=0.25)
    real_train.add_argument("--maximum-drawdown", type=float, default=0.15)
    real_train.add_argument("--minimum-observations", type=int, default=500)
    real_train.add_argument("--minimum-excess-return", type=float, default=0.0)
    real_train.add_argument("--minimum-news-sharpe-delta", type=float, default=0.0)

    replay = subparsers.add_parser(
        "replay",
        help="Replay historical bars/news through the shared strategy-risk-broker engine",
    )
    replay.add_argument("--bars", required=True, help="Historical bars JSON Lines")
    replay.add_argument("--news", required=True, help="Historical news JSON Lines")
    replay.add_argument("--output", required=True, help="Replay report JSON")
    replay.add_argument("--trace-output", help="Optional canonical event trace JSON Lines")
    replay.add_argument(
        "--strategy", choices=("explainable", "champion"), default="explainable"
    )
    replay.add_argument("--registry", default=".trading/models")
    replay.add_argument("--allow-synthetic-champion", action="store_true")
    replay.add_argument("--verify-determinism", action="store_true")
    replay.add_argument("--bar-minutes", type=int, default=60)
    replay.add_argument("--spread-bps", type=float, default=10.0)
    replay.add_argument("--slippage-bps", type=float, default=2.0)
    replay.add_argument("--starting-cash", type=float)

    audit = subparsers.add_parser(
        "audit-ledger", help="Replay event relationships and report integrity failures"
    )
    audit.add_argument("--database", default=".trading/events.db")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.command == "demo-train":
        result = demo_train(arguments.rows, arguments.registry, arguments.promote)
    elif arguments.command == "backfill":
        result = asyncio.run(
            backfill(
                arguments.kind,
                [
                    part.strip().upper()
                    for part in arguments.symbols.split(",")
                    if part.strip()
                ],
                parse_datetime(arguments.start),
                parse_datetime(arguments.end),
                arguments.output,
                arguments.timeframe,
            )
        )
    elif arguments.command == "build-dataset":
        result = build_dataset(
            arguments.bars,
            arguments.news,
            arguments.output,
            lookback_bars=arguments.lookback_bars,
            forecast_bars=arguments.forecast_bars,
            bar_minutes=arguments.bar_minutes,
            news_window_hours=arguments.news_window_hours,
        )
    elif arguments.command == "train":
        result = train_dataset(
            arguments.dataset,
            arguments.registry,
            promote=arguments.promote,
            minimum_train_rows=arguments.minimum_train_rows,
            test_rows=arguments.test_rows,
            transaction_cost_bps=arguments.transaction_cost_bps,
            periods_per_year=arguments.periods_per_year,
            threshold=arguments.threshold,
            ridge=arguments.ridge,
            minimum_folds=arguments.minimum_folds,
            minimum_sharpe=arguments.minimum_sharpe,
            maximum_drawdown=arguments.maximum_drawdown,
            minimum_observations=arguments.minimum_observations,
            minimum_excess_return=arguments.minimum_excess_return,
            minimum_news_sharpe_delta=arguments.minimum_news_sharpe_delta,
        )
    elif arguments.command == "replay":
        result = asyncio.run(
            replay_history(
                arguments.bars,
                arguments.news,
                arguments.output,
                trace_output=arguments.trace_output,
                strategy_mode=arguments.strategy,
                registry_path=arguments.registry,
                allow_synthetic_champion=arguments.allow_synthetic_champion,
                verify_determinism=arguments.verify_determinism,
                bar_minutes=arguments.bar_minutes,
                spread_bps=arguments.spread_bps,
                slippage_bps=arguments.slippage_bps,
                starting_cash=arguments.starting_cash,
            )
        )
    elif arguments.command == "audit-ledger":
        result = audit_ledger(arguments.database).to_dict()
    else:
        result = ModelRegistry(arguments.registry).summary()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
