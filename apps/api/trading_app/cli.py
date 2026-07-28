from __future__ import annotations

import argparse
import json

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trading research and model registry tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser(
        "demo-train", help="Verify the training pipeline on synthetic data"
    )
    train.add_argument("--rows", type=int, default=600)
    train.add_argument("--registry", default=".trading/models")
    train.add_argument("--promote", action="store_true")
    subparsers.add_parser("registry", help="Print the model registry")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.command == "demo-train":
        result = demo_train(arguments.rows, arguments.registry, arguments.promote)
    else:
        result = ModelRegistry(".trading/models").summary()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
