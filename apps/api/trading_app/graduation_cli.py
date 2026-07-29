from __future__ import annotations

import argparse
import json

from .graduation import evaluate_paper_graduation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate immutable evidence for paper-only graduation"
    )
    parser.add_argument("--registry", default=".trading/models")
    parser.add_argument("--version")
    parser.add_argument("--paper-evidence", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--replay-diagnostics", required=True)
    parser.add_argument("--monitoring", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--gate-config",
        default="config/graduation/paper-only-v1.json",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    report = evaluate_paper_graduation(
        registry_path=arguments.registry,
        paper_evidence_path=arguments.paper_evidence,
        replay_report_path=arguments.replay,
        replay_diagnostics_path=arguments.replay_diagnostics,
        monitoring_report_paths=arguments.monitoring,
        output_path=arguments.output,
        gate_config_path=arguments.gate_config,
        version=arguments.version,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
