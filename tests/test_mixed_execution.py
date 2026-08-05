from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_app.dataset import write_feature_dataset
from trading_app.mixed_candidates import generate_mixed_plan, write_mixed_plan
from trading_app.mixed_execution import execute_mixed_generation
from trading_app.mixed_plan import generate_executable_mixed_plan
from trading_app.research import FeatureRow


class _FakeExecutor:
    def __init__(self, *, fail_candidate: str | None = None) -> None:
        self.fail_candidate = fail_candidate
        self.calls: list[str] = []

    def execute(  # type: ignore[no-untyped-def]
        self, spec, candidate_directory, resources, resource_directory
    ):
        del resources, resource_directory
        self.calls.append(spec.candidate_id)
        if spec.candidate_id == self.fail_candidate:
            raise RuntimeError("intentional candidate failure")
        model_path = candidate_directory / "model.json"
        model_path.write_text(json.dumps({"candidate": spec.candidate_id}), encoding="utf-8")
        return {
            "candidate_id": spec.candidate_id,
            "status": "complete",
            "category": spec.category,
            "implementation": spec.implementation,
            "configuration_fingerprint": spec.fingerprint,
            "model_path": str(model_path.resolve()),
            "composite_score": float(spec.seed),
            "holdout_accessed": False,
        }


def _dataset(tmp_path: Path, *, sealed: bool = False) -> Path:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        FeatureRow(
            timestamp=start + timedelta(hours=index),
            symbol="AAPL",
            features=(0.01 * index, 0.1),
            target_return=0.001,
            label_end_time=start + timedelta(hours=index + 5),
        )
        for index in range(20)
    ]
    dataset, _ = write_feature_dataset(
        tmp_path / ("holdout.jsonl" if sealed else "calibration.jsonl"),
        rows,
        ("momentum", "news_score"),
        {
            "dataset_role": "untouched_holdout" if sealed else "calibration",
            "sealed": sealed,
            "forecast_bars": 5,
        },
    )
    return dataset


def _resources(tmp_path: Path, dataset: Path) -> Path:
    plan = generate_mixed_plan(generation=1, seed=42)
    keys = {
        str(candidate.configuration.get("dataset_key", "base"))
        for candidate in plan.candidates
    }
    keys.add("base")
    cost = tmp_path / "cost.json"
    universe = tmp_path / "universe.json"
    cost.write_text("{}", encoding="utf-8")
    universe.write_text("{}", encoding="utf-8")
    resources = tmp_path / "resources.json"
    resources.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {key: str(dataset) for key in sorted(keys)},
                "cost_config_path": str(cost),
                "universe_manifest_path": str(universe),
                "max_workers": 4,
                "outer_folds": 8,
            }
        ),
        encoding="utf-8",
    )
    return resources


def test_mixed_executor_runs_exactly_100_and_freezes_three(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    write_mixed_plan(plan_path, generation=1, seed=42)
    executor = _FakeExecutor()
    report = execute_mixed_generation(
        plan_path,
        _resources(tmp_path, _dataset(tmp_path)),
        tmp_path / "run",
        executor=executor,
    )
    assert report["status"] == "complete"
    assert report["completed"] == 100
    assert report["failed"] == 0
    assert len(executor.calls) == 100
    finalized = report["finalized"]
    assert finalized["candidate_family_size"] == 3
    assert len(finalized["finalists"]) == 3
    assert len(list((tmp_path / "run" / "candidates").glob("*/metrics.json"))) == 100


def test_mixed_executor_is_resumable_and_retries_only_failures(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    write_mixed_plan(plan_path, generation=1, seed=42)
    plan = generate_mixed_plan(generation=1, seed=42)
    failed_id = plan.candidates[0].candidate_id
    resources = _resources(tmp_path, _dataset(tmp_path))
    first = _FakeExecutor(fail_candidate=failed_id)
    report = execute_mixed_generation(
        plan_path,
        resources,
        tmp_path / "run",
        executor=first,
    )
    assert report["status"] == "incomplete"
    assert report["completed"] == 99
    second = _FakeExecutor()
    resumed = execute_mixed_generation(
        plan_path,
        resources,
        tmp_path / "run",
        retry_failed=True,
        executor=second,
    )
    assert resumed["status"] == "complete"
    assert second.calls == [failed_id]


def test_mixed_executor_rejects_sealed_dataset_resources(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    write_mixed_plan(plan_path, generation=1, seed=42)
    with pytest.raises(ValueError, match="not calibration-only"):
        execute_mixed_generation(
            plan_path,
            _resources(tmp_path, _dataset(tmp_path, sealed=True)),
            tmp_path / "run",
            executor=_FakeExecutor(),
        )


def test_mixed_plan_uses_executable_ablation_and_downstream_configs() -> None:
    plan = generate_executable_mixed_plan(generation=1, seed=7)
    ablations = [item for item in plan.candidates if item.category == "ensemble_ablation"]
    assert {item.implementation for item in ablations} <= {
        "full_features",
        "no_news",
        "price_only",
        "news_only",
        "foundation_only",
    }
    derived = [
        item
        for item in plan.candidates
        if item.category in {"foundation", "news_variant", "ensemble_ablation"}
    ]
    assert all("downstream_family" in item.configuration for item in derived)
