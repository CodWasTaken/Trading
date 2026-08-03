from __future__ import annotations

import argparse
import json

from .model_training import train_governed_model
from .modeling import backend_availability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and inspect governed model backends")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("backends", help="Show optional backend availability")
    train = subparsers.add_parser("train", help="Train one model on calibration data")
    train.add_argument("--dataset", required=True)
    train.add_argument("--registry", default=".trading/models")
    train.add_argument(
        "--family",
        required=True,
        choices=(
            "ridge",
            "elastic_net",
            "lightgbm",
            "xgboost",
            "catboost",
            "mlp",
            "lstm",
            "gru",
            "tcn",
            "patchtst",
            "itransformer",
        ),
    )
    train.add_argument("--features")
    train.add_argument("--hyperparameters", default="{}")
    train.add_argument("--seed", type=int, default=1729)
    train.add_argument("--outer-folds", type=int, default=8)
    train.add_argument("--positive-threshold", type=float, default=0.0005)
    train.add_argument("--negative-threshold", type=float, default=0.0005)
    train.add_argument(
        "--cost-config",
        default="config/costs/conservative-us-paper-v1.json",
    )
    train.add_argument(
        "--universe-manifest",
        default="config/universes/us-liquid-large-cap-v1.json",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "backends":
        print(json.dumps(backend_availability(), indent=2, sort_keys=True))
        return
    features = (
        None
        if not arguments.features
        else tuple(item.strip() for item in arguments.features.split(",") if item.strip())
    )
    parameters = json.loads(arguments.hyperparameters)
    if not isinstance(parameters, dict):
        raise ValueError("--hyperparameters must be a JSON object")
    report = train_governed_model(
        arguments.dataset,
        arguments.registry,
        family=arguments.family,
        feature_subset=features,
        hyperparameters=parameters,
        seed=arguments.seed,
        outer_folds=arguments.outer_folds,
        positive_threshold=arguments.positive_threshold,
        negative_threshold=arguments.negative_threshold,
        cost_config_path=arguments.cost_config,
        universe_manifest_path=arguments.universe_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
