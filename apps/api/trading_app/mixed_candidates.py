from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .modeling import sha256_file

CandidateCategory = Literal[
    "tabular_linear",
    "sequence",
    "foundation",
    "reinforcement_learning",
    "news_variant",
    "ensemble_ablation",
]

EXACT_MIXED_CANDIDATES = 100
DEFAULT_QUOTAS: dict[CandidateCategory, int] = {
    "tabular_linear": 25,
    "sequence": 20,
    "foundation": 15,
    "reinforcement_learning": 20,
    "news_variant": 10,
    "ensemble_ablation": 10,
}


class MixedCandidateSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    category: CandidateCategory
    implementation: str
    seed: int = Field(ge=0)
    configuration: dict[str, object]
    holdout_access: Literal["forbidden"] = "forbidden"

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json", exclude={"candidate_id"})
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


class MixedGenerationPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generation: int = Field(ge=1, le=10)
    seed: int = Field(ge=0)
    candidates: tuple[MixedCandidateSpec, ...]
    quotas: dict[CandidateCategory, int]
    calibration_only: bool = True
    candidate_family_size: int = 3

    @model_validator(mode="after")
    def validate_plan(self) -> MixedGenerationPlan:
        if self.schema_version != 1:
            raise ValueError("Unsupported mixed generation schema_version")
        if len(self.candidates) != EXACT_MIXED_CANDIDATES:
            raise ValueError("Mixed generation requires exactly 100 candidates")
        if self.candidate_family_size != 3:
            raise ValueError("Governed holdout candidate family size must be three")
        identifiers = [item.candidate_id for item in self.candidates]
        fingerprints = [item.fingerprint for item in self.candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Mixed generation candidate IDs must be unique")
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("Mixed generation configurations must be unique")
        counts = Counter(item.category for item in self.candidates)
        if dict(counts) != dict(self.quotas):
            raise ValueError(
                f"Mixed generation quotas do not match candidates: counts={dict(counts)}"
            )
        if sum(self.quotas.values()) != EXACT_MIXED_CANDIDATES:
            raise ValueError("Mixed generation quotas must sum to 100")
        if not self.calibration_only:
            raise ValueError("Mixed generation must remain calibration-only")
        return self

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


def generate_mixed_plan(
    *,
    generation: int,
    seed: int,
    quotas: dict[CandidateCategory, int] | None = None,
) -> MixedGenerationPlan:
    resolved = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    if sum(resolved.values()) != EXACT_MIXED_CANDIDATES:
        raise ValueError("Mixed candidate quotas must sum to exactly 100")
    generator = random.Random(seed)
    candidates: list[MixedCandidateSpec] = []
    for category in DEFAULT_QUOTAS:
        count = resolved.get(category, 0)
        for index in range(count):
            candidate_seed = seed * 10_000 + len(candidates)
            implementation, configuration = _candidate_configuration(
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
        quotas=resolved,
    )


def write_mixed_plan(
    output_path: str | Path,
    *,
    generation: int,
    seed: int,
) -> dict[str, object]:
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"Mixed candidate plan already exists: {destination}")
    plan = generate_mixed_plan(generation=generation, seed=seed)
    payload = {
        **plan.model_dump(mode="json"),
        "manifest_sha256": plan.manifest_sha256,
        "controls": {
            "sealed_holdout_access": "forbidden",
            "finalists_frozen_after_100_successful_reports": True,
            "external_models_receive_no_automatic_trust": True,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"output": str(destination), **payload}


def finalize_mixed_generation(
    plan_path: str | Path,
    reports_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, object]:
    plan_source = Path(plan_path)
    report_root = Path(reports_directory)
    destination = Path(output_directory)
    if destination.exists():
        raise ValueError(f"Mixed generation output already exists: {destination}")
    raw_plan = json.loads(plan_source.read_text(encoding="utf-8"))
    if not isinstance(raw_plan, dict):
        raise ValueError("Mixed candidate plan must be a JSON object")
    declared_hash = raw_plan.pop("manifest_sha256", None)
    raw_plan.pop("controls", None)
    plan = MixedGenerationPlan.model_validate(raw_plan)
    if declared_hash != plan.manifest_sha256:
        raise ValueError("Mixed candidate plan hash mismatch")
    reports: list[dict[str, object]] = []
    expected = {item.candidate_id: item for item in plan.candidates}
    for candidate_id, spec in expected.items():
        report_path = report_root / candidate_id / "metrics.json"
        if not report_path.is_file():
            raise ValueError(f"Candidate report is missing: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError(f"Candidate report must be an object: {report_path}")
        _validate_candidate_report(report, spec, report_path)
        reports.append(report)
    if len(reports) != EXACT_MIXED_CANDIDATES:
        raise ValueError("Exactly 100 successful candidate reports are required")
    ranked = sorted(
        reports,
        key=lambda item: (
            -_number(item.get("composite_score"), "composite_score"),
            str(item.get("candidate_id")),
        ),
    )
    finalists: list[dict[str, object]] = []
    for rank, report in enumerate(ranked[:3], start=1):
        model_path = Path(str(report["model_path"]))
        metrics_path = Path(str(report["metrics_path"]))
        finalists.append(
            {
                "rank": rank,
                "candidate_id": report["candidate_id"],
                "category": report["category"],
                "implementation": report["implementation"],
                "composite_score": report["composite_score"],
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "frozen": True,
            }
        )
    destination.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "report_kind": "mixed_model_generation",
        "status": "complete",
        "generation": plan.generation,
        "candidate_count": EXACT_MIXED_CANDIDATES,
        "completed": EXACT_MIXED_CANDIDATES,
        "failed": 0,
        "quotas": plan.quotas,
        "candidate_family_size": 3,
        "plan": {"path": str(plan_source), "sha256": sha256_file(plan_source)},
        "ranking": [
            {
                "rank": index,
                "candidate_id": item["candidate_id"],
                "category": item["category"],
                "implementation": item["implementation"],
                "composite_score": item["composite_score"],
            }
            for index, item in enumerate(ranked, start=1)
        ],
        "finalists": finalists,
        "controls": {
            "all_candidates_calibration_only": True,
            "holdout_accessed": False,
            "finalists_frozen_before_holdout": True,
            "holdout_evaluations_allowed": 3,
        },
    }
    report_path = destination / "generation-report.json"
    finalists_path = destination / "finalists.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    finalists_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_family_size": 3,
                "finalists": finalists,
                "controls": report["controls"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {**report, "output": str(report_path), "finalists_path": str(finalists_path)}


def _validate_candidate_report(
    report: dict[str, object],
    spec: MixedCandidateSpec,
    report_path: Path,
) -> None:
    if report.get("candidate_id") != spec.candidate_id:
        raise ValueError(f"Candidate ID mismatch in {report_path}")
    if report.get("status") != "complete":
        raise ValueError(f"Candidate did not complete successfully: {spec.candidate_id}")
    if report.get("category") != spec.category:
        raise ValueError(f"Candidate category mismatch: {spec.candidate_id}")
    if report.get("implementation") != spec.implementation:
        raise ValueError(f"Candidate implementation mismatch: {spec.candidate_id}")
    if report.get("holdout_accessed") is not False:
        raise ValueError(f"Candidate accessed holdout data: {spec.candidate_id}")
    model_path = Path(str(report.get("model_path", "")))
    metrics_path = Path(str(report.get("metrics_path", "")))
    if not model_path.is_file() or not metrics_path.is_file():
        raise ValueError(f"Candidate artifacts are missing: {spec.candidate_id}")
    _number(report.get("composite_score"), "composite_score")


def _candidate_configuration(
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
            "positive_threshold": generator.choice((0.00025, 0.0005, 0.001, 0.0015)),
            "negative_threshold": generator.choice((0.00025, 0.0005, 0.001, 0.0015)),
            "seed": seed,
        }
    if category == "sequence":
        implementations = ("lstm", "gru", "tcn", "patchtst", "itransformer")
        return implementations[index % len(implementations)], {
            "context_length": generator.choice((20, 40, 80, 160)),
            "hidden_size": generator.choice((16, 32, 64)),
            "learning_rate": generator.choice((1e-4, 3e-4, 1e-3)),
            "seed": seed,
        }
    if category == "foundation":
        implementations = ("timesfm-2.5", "chronos-2", "moirai-2.0")
        return implementations[index % len(implementations)], {
            "mode": generator.choice(("zero_shot_feature", "frozen_forecast_feature")),
            "context_length": generator.choice((128, 256, 512, 1024)),
            "prediction_length": 5,
            "seed": seed,
        }
    if category == "reinforcement_learning":
        implementations = ("ppo", "a2c", "sac", "td3")
        return implementations[index % len(implementations)], {
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
        return implementations[index % len(implementations)], {
            "confidence_floor": generator.choice((0.50, 0.65, 0.80)),
            "news_window_hours": generator.choice((6, 12, 24, 48)),
            "seed": seed,
        }
    implementations = (
        "weighted_average",
        "rank_average",
        "constrained_stacking",
        "disagreement_abstention",
        "feature_ablation",
    )
    return implementations[index % len(implementations)], {
        "constituent_count": generator.choice((2, 3, 5)),
        "disagreement_threshold": generator.choice((0.0005, 0.001, 0.002)),
        "seed": seed,
    }


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
