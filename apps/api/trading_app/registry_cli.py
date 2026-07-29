from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .model_registry import ModelRegistry
from .promotion import load_promotion_gates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and operate the versioned paper-model registry"
    )
    parser.add_argument("--registry", default=".trading/models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "summary",
        help="Print models, aliases, alias history, and holdout evaluations",
    )

    inspect = subparsers.add_parser("inspect", help="Print one registered model record")
    inspect.add_argument("version")

    history = subparsers.add_parser("history", help="Print append-only alias history")
    history.add_argument("--alias")
    history.add_argument("--limit", type=int)

    holdouts = subparsers.add_parser(
        "holdouts",
        help="Print immutable untouched-holdout evaluation records",
    )
    holdouts.add_argument("--version")

    alias = subparsers.add_parser(
        "set-alias",
        help="Move a non-champion alias to a registered model",
    )
    alias.add_argument("alias")
    alias.add_argument("version")
    alias.add_argument("--reason", required=True)

    promote = subparsers.add_parser(
        "promote",
        help="Promote a registered challenger through calibration and holdout gates",
    )
    promote.add_argument("version")
    promote.add_argument("--reason", default="operator_promotion")
    promote.add_argument(
        "--gate-config",
        default="config/promotion/strict-paper-v1.json",
    )
    promote.add_argument("--minimum-folds", type=int)
    promote.add_argument("--minimum-sharpe", type=float)
    promote.add_argument("--maximum-drawdown", type=float)
    promote.add_argument("--minimum-observations", type=int)
    promote.add_argument("--minimum-net-return", type=float)
    promote.add_argument("--minimum-excess-return", type=float)
    promote.add_argument("--minimum-news-sharpe-delta", type=float)
    promote.add_argument("--maximum-holdout-finalists", type=int)
    promote.add_argument("--minimum-holdout-net-return", type=float)
    promote.add_argument("--minimum-holdout-sharpe", type=float)
    promote.add_argument("--maximum-holdout-drawdown", type=float)
    promote.add_argument("--minimum-holdout-observations", type=int)
    promote.add_argument("--minimum-holdout-excess-return", type=float)
    promote.add_argument(
        "--minimum-holdout-net-return-lower-bound",
        type=float,
        help="Require the adjusted bootstrap lower bound to exceed this value",
    )
    promote.add_argument(
        "--minimum-holdout-excess-return-lower-bound",
        type=float,
        help="Require the adjusted benchmark-excess lower bound to exceed this value",
    )
    promote.add_argument("--maximum-symbol-pnl-contribution", type=float)
    promote.add_argument("--maximum-sector-pnl-contribution", type=float)
    promote.add_argument("--maximum-unborrowable-short-orders", type=int)
    promote.add_argument("--maximum-gross-short-exposure", type=float)
    promote.add_argument("--maximum-single-short-position", type=float)

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
    elif arguments.command == "holdouts":
        result = {
            "registry": arguments.registry,
            "version": arguments.version,
            "evaluations": registry.holdout_evaluations(version=arguments.version),
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
            gate_config=load_promotion_gates(arguments.gate_config),
            minimum_folds=arguments.minimum_folds,
            minimum_sharpe=arguments.minimum_sharpe,
            maximum_drawdown=arguments.maximum_drawdown,
            minimum_observations=arguments.minimum_observations,
            minimum_net_return=arguments.minimum_net_return,
            minimum_excess_return=arguments.minimum_excess_return,
            minimum_news_sharpe_delta=arguments.minimum_news_sharpe_delta,
            require_holdout_evaluation=True,
            maximum_holdout_finalists=arguments.maximum_holdout_finalists,
            minimum_holdout_net_return=arguments.minimum_holdout_net_return,
            minimum_holdout_sharpe=arguments.minimum_holdout_sharpe,
            maximum_holdout_drawdown=arguments.maximum_holdout_drawdown,
            minimum_holdout_observations=arguments.minimum_holdout_observations,
            minimum_holdout_excess_return=arguments.minimum_holdout_excess_return,
            minimum_holdout_net_return_lower_bound=(
                arguments.minimum_holdout_net_return_lower_bound
            ),
            minimum_holdout_excess_return_lower_bound=(
                arguments.minimum_holdout_excess_return_lower_bound
            ),
            maximum_symbol_pnl_contribution=(
                arguments.maximum_symbol_pnl_contribution
            ),
            maximum_sector_pnl_contribution=(
                arguments.maximum_sector_pnl_contribution
            ),
            maximum_unborrowable_short_orders=(
                arguments.maximum_unborrowable_short_orders
            ),
            maximum_gross_short_exposure=arguments.maximum_gross_short_exposure,
            maximum_single_short_position=arguments.maximum_single_short_position,
            reason=arguments.reason,
        )
        result = {
            "promoted": True,
            "version": record.version,
            "reason": arguments.reason,
            "holdout_evaluation": registry.latest_holdout_evaluation(record.version),
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
