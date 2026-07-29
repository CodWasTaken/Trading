from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from .candidates import CandidateConfig, run_candidate_generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly 100 governed calibration candidates per generation"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--cost-config",
        default="config/costs/conservative-us-paper-v1.json",
    )
    parser.add_argument(
        "--universe-manifest",
        default="config/universes/us-liquid-large-cap-v1.json",
    )
    parser.add_argument("--candidate-configs")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--outer-folds", type=int, default=8)
    parser.add_argument("--periods-per-year", type=float, default=327.6)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    configs = None
    if arguments.candidate_configs:
        payload = json.loads(Path(arguments.candidate_configs).read_text(encoding="utf-8"))
        configs = TypeAdapter(list[CandidateConfig]).validate_python(payload)
    report = run_candidate_generation(
        arguments.dataset,
        arguments.output,
        generation=arguments.generation,
        seed=arguments.seed,
        cost_config_path=arguments.cost_config,
        universe_manifest_path=arguments.universe_manifest,
        max_workers=arguments.max_workers,
        configs=configs,
        outer_folds=arguments.outer_folds,
        periods_per_year=arguments.periods_per_year,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
