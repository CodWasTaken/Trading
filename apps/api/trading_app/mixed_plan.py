from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .mixed_candidates import (
    DEFAULT_QUOTAS,
    CandidateCategory,
    MixedCandidateSpec,
    MixedGenerationPlan,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def generate_executable_mixed_plan(
    *,
    generation: int,
    seed: int,
) -> MixedGenerationPlan:
    generator = random.Random(seed)
    candidates: list[MixedCandidateSpec] = []
    for category, count in DEFAULT_QUOTAS.items():
        for index in range(count):
            candidate_seed = seed * 10_000 + len(candidates)
            implementation, configuration = _executable_configuration(
                category,
                index,
                candidate_seed,
                generator,
            )
            fingerprint = hashlib.sha256(
                _canonical_json(
                    {
                        "category": category,
                        "implementation": implementation,
                        "seed": candidate_seed,
                        "configuration": configuration,
                    }
                )
            ).hexdigest()
            candidates.append(
                MixedCandidateSpec(
                    candidate_id=f"mixed-{category}-{fingerprint[:12]}",
                    category=category,
                    implementation=implementation,
                    seed=candidate_seed,
                    configuration=configuration,
                )
            )
    generator.shuffle(candidates)
    return MixedGenerationPlan(
        generation=generation,
        seed=seed,
        candidates=tuple(candidates),
        quotas=dict(DEFAULT_QUOTAS),
    )


def write_executable_mixed_plan(
    output_path: str | Path,
    *,
    generation: int,
    seed: int,
) -> dict[str, object]:
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"Mixed candidate plan already exists: {destination}")
    plan = generate_executable_mixed_plan(generation=generation, seed=seed)
    payload = {
        **plan.model_dump(mode="json"),
        "manifest_sha256": plan.manifest_sha256,
        "controls": {
            "sealed_holdout_access": "forbidden",
            "all_candidate_types_have_native_execution_paths": True,
            "finalists_frozen_after_100_successful_reports": True,
        },
    }
    _write_json(destination, payload)
    return {"output": str(destination), **payload}


def _executable_configuration(
    category: CandidateCategory,
    index: int,
    seed: int,
    generator: random.Random,
) -> tuple[str, dict[str, object]]:
    if category == "tabular_linear":
        implementations = (
            "ridge",
            "elastic_net",
            "lightgbm",
            "xgboost",
            "catboost",
            "mlp",
        )
        implementation = implementations[index % len(implementations)]
        return implementation, {
            "ridge": 10 ** generator.uniform(-5, -1),
            "alpha": 10 ** generator.uniform(-5, -2),
            "l1_ratio": generator.choice((0.1, 0.5, 0.9)),
            "positive_threshold": generator.choice((0.00025, 0.0005, 0.001)),
            "negative_threshold": generator.choice((0.00025, 0.0005, 0.001)),
            "seed": seed,
        }
    if category == "sequence":
        implementations = ("lstm", "gru", "tcn", "patchtst", "itransformer")
        return implementations[index % len(implementations)], {
            "context_length": generator.choice((20, 40, 80)),
            "hidden_size": generator.choice((16, 32, 64)),
            "learning_rate": generator.choice((1e-4, 3e-4, 1e-3)),
            "epochs": generator.choice((10, 20, 30)),
            "seed": seed,
        }
    if category == "foundation":
        implementations = ("timesfm-2.5", "chronos-2", "moirai-2.0")
        implementation = implementations[index % len(implementations)]
        return implementation, {
            "dataset_key": implementation,
            "downstream_family": generator.choice(("ridge", "elastic_net")),
            "alpha": 10 ** generator.uniform(-5, -2),
            "l1_ratio": generator.choice((0.1, 0.5, 0.9)),
            "seed": seed,
        }
    if category == "reinforcement_learning":
        implementations = ("ppo", "a2c", "sac", "td3")
        return implementations[index % len(implementations)], {
            "dataset_key": "base",
            "reward_drawdown_penalty": generator.choice((0.5, 1.0, 2.0)),
            "reward_turnover_penalty": generator.choice((0.005, 0.01, 0.02)),
            "total_timesteps": generator.choice((10_000, 25_000, 50_000)),
            "seed": seed,
        }
    if category == "news_variant":
        implementations = (
            "current_news",
            "fingpt_sentiment",
            "fingpt_multitask",
            "no_news",
            "consensus_only",
        )
        implementation = implementations[index % len(implementations)]
        return implementation, {
            "dataset_key": implementation,
            "downstream_family": generator.choice(("ridge", "elastic_net")),
            "alpha": 10 ** generator.uniform(-5, -2),
            "l1_ratio": generator.choice((0.1, 0.5, 0.9)),
            "seed": seed,
        }
    implementations = (
        "full_features",
        "no_news",
        "price_only",
        "news_only",
        "foundation_only",
    )
    implementation = implementations[index % len(implementations)]
    return implementation, {
        "dataset_key": "base",
        "downstream_family": generator.choice(("ridge", "elastic_net")),
        "alpha": 10 ** generator.uniform(-5, -2),
        "l1_ratio": generator.choice((0.1, 0.5, 0.9)),
        "seed": seed,
    }


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
