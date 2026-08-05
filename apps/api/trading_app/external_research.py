from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .modeling import sha256_file

ExternalBackend = Literal["qlib", "finrl", "lean"]


class ExternalResearchImportSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    backend: ExternalBackend
    tool_version: str
    configuration_path: str
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    universe_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_identifier: str
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_schema(self) -> ExternalResearchImportSpec:
        if self.schema_version != 1:
            raise ValueError("Unsupported external import spec schema_version")
        return self

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


def import_external_research(
    input_path: str | Path,
    output_path: str | Path,
    spec_path: str | Path,
) -> dict[str, object]:
    source = Path(input_path)
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"External research output already exists: {destination}")
    spec = ExternalResearchImportSpec.model_validate_json(
        Path(spec_path).read_text(encoding="utf-8")
    )
    configuration = Path(spec.configuration_path)
    if not configuration.is_file():
        raise ValueError(f"External configuration is missing: {configuration}")
    actual_configuration_sha256 = sha256_file(configuration)
    if actual_configuration_sha256 != spec.configuration_sha256:
        raise ValueError("External configuration hash mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("External research input must be a JSON object")
    normalized = _normalize_payload(spec.backend, payload)
    report = {
        "schema_version": 1,
        "report_kind": "external_research_evidence",
        "backend": spec.backend,
        "tool_version": spec.tool_version,
        "model_identifier": spec.model_identifier,
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "configuration": {
            "path": str(configuration),
            "sha256": actual_configuration_sha256,
        },
        "universe_manifest_sha256": spec.universe_manifest_sha256,
        "dataset_sha256": spec.dataset_sha256,
        "spec_sha256": spec.manifest_sha256,
        "normalized": normalized,
        "controls": {
            "authoritative_for_promotion": False,
            "must_pass_native_replay": True,
            "must_pass_native_cost_model": True,
            "must_pass_native_holdout": True,
            "live_money_authorized": False,
        },
        "limitations": [
            *spec.limitations,
            "Imported evidence cannot move a registry alias or authorize execution.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "output": str(destination)}


def _normalize_payload(
    backend: ExternalBackend,
    payload: dict[str, object],
) -> dict[str, object]:
    if backend == "qlib":
        predictions = payload.get("predictions")
        metrics = payload.get("metrics")
        if not isinstance(predictions, list) or not isinstance(metrics, dict):
            raise ValueError("Qlib import requires predictions list and metrics object")
        return {
            "prediction_rows": len(predictions),
            "metrics": metrics,
            "predictions_sha256": hashlib.sha256(_canonical_json(predictions)).hexdigest(),
        }
    if backend == "finrl":
        policy = payload.get("policy")
        episodes = payload.get("episodes")
        if not isinstance(policy, dict) or not isinstance(episodes, list):
            raise ValueError("FinRL import requires policy object and episodes list")
        return {
            "policy": policy,
            "episode_count": len(episodes),
            "episodes_sha256": hashlib.sha256(_canonical_json(episodes)).hexdigest(),
        }
    orders = payload.get("orders")
    statistics = payload.get("statistics")
    if not isinstance(orders, list) or not isinstance(statistics, dict):
        raise ValueError("LEAN import requires orders list and statistics object")
    return {
        "order_count": len(orders),
        "statistics": statistics,
        "orders_sha256": hashlib.sha256(_canonical_json(orders)).hexdigest(),
    }


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
