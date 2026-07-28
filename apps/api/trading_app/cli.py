from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from .config import get_settings
from .historical import AlpacaHistoricalClient
from .model_registry import ModelRegistry
from .research import RidgeReturnModel, metrics_dict, synthetic_feature_rows, walk_forward


FEATURE_NAMES = ("momentum", "news_score", "volatility", "spread_bps")


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
                iterator = client.iter_bars(
                    symbols, start, end, timeframe=timeframe
                )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trading data, research, and model registry tools"
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
                [part.strip().upper() for part in arguments.symbols.split(",") if part.strip()],
                parse_datetime(arguments.start),
                parse_datetime(arguments.end),
                arguments.output,
                arguments.timeframe,
            )
        )
    else:
        result = ModelRegistry(arguments.registry).summary()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
