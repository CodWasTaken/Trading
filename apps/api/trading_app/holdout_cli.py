from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from .holdout import evaluate_untouched_holdout, split_feature_dataset


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and score sealed calibration/holdout research datasets"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser(
        "split",
        help="Chronologically split a feature dataset and purge labels crossing the boundary",
    )
    split.add_argument("--dataset", required=True)
    split.add_argument("--calibration-output", required=True)
    split.add_argument("--holdout-output", required=True)
    split.add_argument("--split-time", required=True, help="ISO-8601 feature-time boundary")
    split.add_argument("--manifest")
    split.add_argument(
        "--candidate-family-size",
        type=int,
        default=1,
        help="Predeclared number of candidate models that may be tested on this holdout",
    )
    split.add_argument("--bootstrap-samples", type=int, default=2000)
    split.add_argument("--confidence-level", type=float, default=0.95)
    split.add_argument(
        "--bootstrap-block-size",
        type=int,
        help="Optional fixed moving-block length; defaults from holdout period count",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Score a frozen challenger once on a sealed untouched holdout",
    )
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--registry", default=".trading/models")
    evaluate.add_argument("--version", help="Defaults to the challenger alias")
    evaluate.add_argument("--output", required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "split":
        result = split_feature_dataset(
            arguments.dataset,
            arguments.calibration_output,
            arguments.holdout_output,
            split_time=_parse_datetime(arguments.split_time),
            manifest_path=arguments.manifest,
            candidate_family_size=arguments.candidate_family_size,
            bootstrap_samples=arguments.bootstrap_samples,
            confidence_level=arguments.confidence_level,
            bootstrap_block_size=arguments.bootstrap_block_size,
        )
    else:
        result = evaluate_untouched_holdout(
            arguments.dataset,
            arguments.registry,
            arguments.output,
            version=arguments.version,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
