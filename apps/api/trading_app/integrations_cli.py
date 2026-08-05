from __future__ import annotations

import argparse
import json

from .external_research import import_external_research
from .financial_news_ai import (
    infer_news_archive,
    load_news_model_spec,
    verify_news_weights,
)
from .mixed_candidates import finalize_mixed_generation
from .mixed_execution import execute_mixed_generation
from .mixed_plan import write_executable_mixed_plan
from .model_acquisition import acquire_huggingface_snapshot
from .news_ai_features import join_news_ai_features
from .pretrained_features import (
    generate_foundation_sidecar,
    join_foundation_sidecar,
    load_pretrained_spec,
    verify_pretrained_weights,
)
from .rl_research import train_rl_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Governed pretrained, news-AI, RL, and external research integrations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser(
        "acquire-model",
        help="Explicitly acquire and hash a pinned Hugging Face model snapshot",
    )
    acquire.add_argument("--request", required=True)
    acquire.add_argument("--output", required=True)

    verify_foundation = subparsers.add_parser(
        "verify-foundation",
        help="Verify an operator-installed foundation-model weight snapshot",
    )
    verify_foundation.add_argument("--spec", required=True)

    foundation = subparsers.add_parser(
        "foundation-features",
        help="Generate past-only point-in-time foundation-model feature sidecars",
    )
    foundation.add_argument("--bars", required=True)
    foundation.add_argument("--output", required=True)
    foundation.add_argument("--spec", required=True)
    foundation.add_argument("--stride", type=int, default=1)

    join = subparsers.add_parser(
        "join-foundation",
        help="Join a verified foundation feature sidecar into a feature dataset",
    )
    join.add_argument("--dataset", required=True)
    join.add_argument("--sidecar", required=True)
    join.add_argument("--output", required=True)
    join.add_argument("--allow-missing", action="store_true")

    join_news = subparsers.add_parser(
        "join-news-ai",
        help="Join immutable news-AI annotations into a point-in-time dataset",
    )
    join_news.add_argument("--dataset", required=True)
    join_news.add_argument("--inference", required=True)
    join_news.add_argument("--output", required=True)
    join_news.add_argument("--window-hours", type=int, default=24)
    join_news.add_argument("--max-items", type=int, default=5)

    verify_news = subparsers.add_parser(
        "verify-news-model",
        help="Verify an operator-installed FinGPT-compatible model snapshot",
    )
    verify_news.add_argument("--spec", required=True)

    news = subparsers.add_parser(
        "news-infer",
        help="Run immutable local financial-news inference without order authority",
    )
    news.add_argument("--input", required=True)
    news.add_argument("--output", required=True)
    news.add_argument("--spec", required=True)

    rl = subparsers.add_parser(
        "rl-train",
        help="Train a calibration-only SB3 policy challenger",
    )
    rl.add_argument("--dataset", required=True)
    rl.add_argument("--output", required=True)
    rl.add_argument("--spec", required=True)
    rl.add_argument("--cost-config", required=True)

    external = subparsers.add_parser(
        "import-external",
        help="Normalize Qlib, FinRL, or LEAN research evidence",
    )
    external.add_argument("--input", required=True)
    external.add_argument("--output", required=True)
    external.add_argument("--spec", required=True)

    plan = subparsers.add_parser(
        "mixed-plan",
        help="Create an exact deterministic 100-candidate mixed-family plan",
    )
    plan.add_argument("--output", required=True)
    plan.add_argument("--generation", required=True, type=int)
    plan.add_argument("--seed", type=int, default=1729)

    execute = subparsers.add_parser(
        "mixed-execute",
        help="Run and resume all 100 mixed-family calibration candidates",
    )
    execute.add_argument("--plan", required=True)
    execute.add_argument("--resources", required=True)
    execute.add_argument("--output-dir", required=True)
    execute.add_argument("--retry-failed", action="store_true")

    finalize = subparsers.add_parser(
        "mixed-finalize",
        help="Require 100 successful calibration reports and freeze three finalists",
    )
    finalize.add_argument("--plan", required=True)
    finalize.add_argument("--reports", required=True)
    finalize.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "acquire-model":
        result = acquire_huggingface_snapshot(
            arguments.request,
            arguments.output,
        )
    elif arguments.command == "verify-foundation":
        result = verify_pretrained_weights(load_pretrained_spec(arguments.spec))
    elif arguments.command == "foundation-features":
        result = generate_foundation_sidecar(
            arguments.bars,
            arguments.output,
            arguments.spec,
            stride=arguments.stride,
        )
    elif arguments.command == "join-foundation":
        result = join_foundation_sidecar(
            arguments.dataset,
            arguments.sidecar,
            arguments.output,
            require_complete=not arguments.allow_missing,
        )
    elif arguments.command == "join-news-ai":
        result = join_news_ai_features(
            arguments.dataset,
            arguments.inference,
            arguments.output,
            window_hours=arguments.window_hours,
            max_items=arguments.max_items,
        )
    elif arguments.command == "verify-news-model":
        result = verify_news_weights(load_news_model_spec(arguments.spec))
    elif arguments.command == "news-infer":
        result = infer_news_archive(
            arguments.input,
            arguments.output,
            arguments.spec,
        )
    elif arguments.command == "rl-train":
        result = train_rl_policy(
            arguments.dataset,
            arguments.output,
            arguments.spec,
            arguments.cost_config,
        )
    elif arguments.command == "import-external":
        result = import_external_research(
            arguments.input,
            arguments.output,
            arguments.spec,
        )
    elif arguments.command == "mixed-plan":
        result = write_executable_mixed_plan(
            arguments.output,
            generation=arguments.generation,
            seed=arguments.seed,
        )
    elif arguments.command == "mixed-execute":
        result = execute_mixed_generation(
            arguments.plan,
            arguments.resources,
            arguments.output_dir,
            retry_failed=arguments.retry_failed,
        )
    else:
        result = finalize_mixed_generation(
            arguments.plan,
            arguments.reports,
            arguments.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
