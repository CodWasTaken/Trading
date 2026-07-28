from __future__ import annotations

import argparse
import json

from .monitoring import evaluate_model_monitoring


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure post-training feature drift and realized model calibration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Compare a later labelled window with the frozen calibration reference",
    )
    evaluate.add_argument("--reference", required=True)
    evaluate.add_argument("--recent", required=True)
    evaluate.add_argument("--registry", default=".trading/models")
    selector = evaluate.add_mutually_exclusive_group()
    selector.add_argument("--version")
    selector.add_argument("--alias", default="champion")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--minimum-recent-rows", type=int, default=100)
    evaluate.add_argument("--minimum-active-signals", type=int, default=20)
    evaluate.add_argument("--maximum-feature-psi", type=float, default=0.25)
    evaluate.add_argument("--maximum-feature-mean-shift", type=float, default=1.0)
    evaluate.add_argument("--maximum-prediction-mean-shift", type=float, default=1.0)
    evaluate.add_argument("--maximum-rmse-ratio", type=float, default=2.0)
    evaluate.add_argument("--minimum-active-hit-rate", type=float, default=0.50)
    evaluate.add_argument("--minimum-calibration-slope", type=float, default=0.25)
    evaluate.add_argument("--maximum-calibration-slope", type=float, default=1.75)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    report = evaluate_model_monitoring(
        arguments.reference,
        arguments.recent,
        arguments.registry,
        arguments.output,
        version=arguments.version,
        alias=arguments.alias,
        minimum_recent_rows=arguments.minimum_recent_rows,
        minimum_active_signals=arguments.minimum_active_signals,
        maximum_feature_psi=arguments.maximum_feature_psi,
        maximum_feature_mean_shift=arguments.maximum_feature_mean_shift,
        maximum_prediction_mean_shift=arguments.maximum_prediction_mean_shift,
        maximum_rmse_ratio=arguments.maximum_rmse_ratio,
        minimum_active_hit_rate=arguments.minimum_active_hit_rate,
        minimum_calibration_slope=arguments.minimum_calibration_slope,
        maximum_calibration_slope=arguments.maximum_calibration_slope,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["healthy"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
