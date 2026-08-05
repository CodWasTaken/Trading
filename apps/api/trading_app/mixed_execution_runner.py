from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .mixed_candidates import (
    EXACT_MIXED_CANDIDATES,
    MixedCandidateSpec,
    MixedGenerationPlan,
    finalize_mixed_generation,
)
from .mixed_execution_models import (
    MixedCandidateExecutor,
    MixedExecutionResources,
    NativeMixedCandidateExecutor,
)
from .mixed_execution_support import (
    assert_report_matches_spec,
    verify_resources,
)
from .modeling import sha256_file


def execute_mixed_generation(
    plan_path: str | Path,
    resources_path: str | Path,
    output_directory: str | Path,
    *,
    retry_failed: bool = False,
    executor: MixedCandidateExecutor | None = None,
) -> dict[str, object]:
    plan_source = Path(plan_path)
    resource_source = Path(resources_path)
    destination = Path(output_directory)
    plan = _load_plan(plan_source)
    resources = MixedExecutionResources.model_validate_json(
        resource_source.read_text(encoding="utf-8")
    )
    resource_directory = resource_source.parent
    resource_bindings = verify_resources(
        plan,
        resources,
        resource_directory,
    )
    destination.mkdir(parents=True, exist_ok=True)
    candidates_directory = destination / "candidates"
    candidates_directory.mkdir(exist_ok=True)
    manifest = {
        "schema_version": 1,
        "report_kind": "mixed_generation_execution_manifest",
        "generation": plan.generation,
        "plan": {"path": str(plan_source), "sha256": sha256_file(plan_source)},
        "resources": {
            "path": str(resource_source),
            "sha256": sha256_file(resource_source),
            "bindings": resource_bindings,
        },
        "candidate_count": EXACT_MIXED_CANDIDATES,
        "max_workers": resources.max_workers,
        "outer_folds": resources.outer_folds,
        "calibration_only": True,
        "holdout_access": "forbidden",
    }
    manifest_path = destination / "execution-manifest.json"
    _write_or_verify_manifest(manifest_path, manifest)
    implementation = executor or NativeMixedCandidateExecutor()
    reports: dict[str, dict[str, object]] = {}
    pending: list[MixedCandidateSpec] = []
    for spec in plan.candidates:
        candidate_directory = candidates_directory / spec.candidate_id
        candidate_directory.mkdir(exist_ok=True)
        config_path = candidate_directory / "config.json"
        if not config_path.exists():
            _write_json(config_path, spec.model_dump(mode="json"))
        metrics_path = candidate_directory / "metrics.json"
        failure_path = candidate_directory / "failure.json"
        if metrics_path.exists():
            report = _load_object(metrics_path)
            assert_report_matches_spec(report, spec)
            reports[spec.candidate_id] = report
        elif failure_path.exists() and not retry_failed:
            reports[spec.candidate_id] = _load_object(failure_path)
        else:
            if failure_path.exists():
                failure_path.unlink()
            pending.append(spec)

    with ThreadPoolExecutor(max_workers=resources.max_workers) as pool:
        futures = {
            pool.submit(
                implementation.execute,
                spec,
                candidates_directory / spec.candidate_id,
                resources,
                resource_directory,
            ): spec
            for spec in pending
        }
        for future in as_completed(futures):
            spec = futures[future]
            candidate_directory = candidates_directory / spec.candidate_id
            metrics_path = candidate_directory / "metrics.json"
            try:
                report = future.result()
                report["metrics_path"] = str(metrics_path.resolve())
                assert_report_matches_spec(report, spec)
                _write_json(metrics_path, report)
                reports[spec.candidate_id] = report
            except Exception as error:
                failure: dict[str, object] = {
                    "candidate_id": spec.candidate_id,
                    "status": "failed",
                    "category": spec.category,
                    "implementation": spec.implementation,
                    "configuration_fingerprint": spec.fingerprint,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "holdout_accessed": False,
                }
                _write_json(candidate_directory / "failure.json", failure)
                reports[spec.candidate_id] = failure

    successful = [
        report for report in reports.values() if report.get("status") == "complete"
    ]
    failed = [
        report for report in reports.values() if report.get("status") != "complete"
    ]
    summary: dict[str, object] = {
        **manifest,
        "status": "complete" if len(successful) == EXACT_MIXED_CANDIDATES else "incomplete",
        "completed": len(successful),
        "failed": len(failed),
        "failures": sorted(
            (
                {
                    "candidate_id": report.get("candidate_id"),
                    "error_type": report.get("error_type"),
                    "message": report.get("message"),
                }
                for report in failed
            ),
            key=lambda item: str(item["candidate_id"]),
        ),
        "resumable": True,
    }
    _write_json(destination / "execution-report.json", summary)
    if len(successful) == EXACT_MIXED_CANDIDATES:
        finalized = destination / "finalized"
        if finalized.exists():
            final_report = _load_object(finalized / "generation-report.json")
        else:
            final_report = finalize_mixed_generation(
                plan_source,
                candidates_directory,
                finalized,
            )
        summary["finalized"] = final_report
        _write_json(destination / "execution-report.json", summary)
    return summary


def _load_plan(path: Path) -> MixedGenerationPlan:
    payload = _load_object(path)
    declared_hash = payload.pop("manifest_sha256", None)
    payload.pop("controls", None)
    plan = MixedGenerationPlan.model_validate(payload)
    if declared_hash != plan.manifest_sha256:
        raise ValueError("Mixed candidate plan hash mismatch")
    return plan


def _write_or_verify_manifest(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        existing = _load_object(path)
        if existing != payload:
            raise ValueError("Existing mixed execution manifest does not match request")
        return
    _write_json(path, payload)


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
