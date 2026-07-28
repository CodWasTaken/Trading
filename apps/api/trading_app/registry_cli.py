from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .model_registry import ModelRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and operate the versioned paper-model registry"
    )
    parser.add_argument("--registry", default=".trading/models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Print models, aliases, and alias history")

    inspect = subparsers.add_parser("inspect", help="Print one registered model record")
    inspect.add_argument("version")

    history = subparsers.add_parser("history", help="Print append-only alias history")
    history.add_argument("--alias")
    history.add_argument("--limit", type=int)

    alias = subparsers.add_parser(
        "set-alias",
        help="Move a non-champion alias to a registered model",
    )
    alias.add_argument("alias")
    alias.add_argument("version")
    alias.add_argument("--reason", required=True)

    promote = subparsers.add_parser(
        "promote",
        help="Promote a registered challenger through quantitative gates",
    )
    promote.add_argument("version")
    promote.add_argument("--reason", default="operator_promotion")
    promote.add_argument("--minimum-folds", type=int, default=5)
    promote.add_argument("--minimum-sharpe", type=float, default=0.25)
    promote.add_argument("--maximum-drawdown", type=float, default=0.15)
    promote.add_argument("--minimum-observations", type=int, default=500)
    promote.add_argument("--minimum-excess-return", type=float, default=0.0)
    promote.add_argument("--minimum-news-sharpe-delta", type=float, default=0.0)

    rollback = subparsers.add_parser(
        "rollback",
        help="Restore the previous champion or an explicitly named registered model",
    )
    rollback.add_argument("--version")
    rollback.add_argument("--reason", default="operator_rollback")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    registry = ModelRegistry(arguments.registry)

    if arguments.command == "summary":
        result: object = registry.summary()
    elif arguments.command == "inspect":
        result = asdict(registry.get(arguments.version))
    elif arguments.command == "history":
        result = {
            "registry": arguments.registry,
            "alias": arguments.alias,
            "events": registry.history(alias=arguments.alias, limit=arguments.limit),
        }
    elif arguments.command == "set-alias":
        if arguments.alias.strip().lower() == "champion":
            raise SystemExit(
                "The champion alias can move only through promote or rollback"
            )
        record = registry.set_alias(
            arguments.alias,
            arguments.version,
            reason=arguments.reason,
            action="operator_set_alias",
        )
        result = {
            "alias": arguments.alias.strip().lower(),
            "version": record.version,
            "reason": arguments.reason,
            "aliases": registry.summary()["aliases"],
        }
    elif arguments.command == "promote":
        record = registry.promote(
            arguments.version,
            minimum_folds=arguments.minimum_folds,
            minimum_sharpe=arguments.minimum_sharpe,
            maximum_drawdown=arguments.maximum_drawdown,
            minimum_observations=arguments.minimum_observations,
            minimum_excess_return=arguments.minimum_excess_return,
            minimum_news_sharpe_delta=arguments.minimum_news_sharpe_delta,
            reason=arguments.reason,
        )
        result = {
            "promoted": True,
            "version": record.version,
            "reason": arguments.reason,
            "aliases": registry.summary()["aliases"],
        }
    else:
        record = registry.rollback(
            arguments.version,
            reason=arguments.reason,
        )
        result = {
            "rolled_back": True,
            "version": record.version,
            "reason": arguments.reason,
            "aliases": registry.summary()["aliases"],
        }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
