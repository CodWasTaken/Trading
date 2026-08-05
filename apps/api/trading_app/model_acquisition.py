from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from .pretrained_features import sha256_tree

ModelAssetKind = Literal["foundation", "financial_news", "other"]


class HuggingFaceAcquisitionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    repo_id: str
    revision: str
    destination: str
    license: str
    asset_kind: ModelAssetKind
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_request(self) -> HuggingFaceAcquisitionRequest:
        if self.schema_version != 1:
            raise ValueError("Unsupported model acquisition schema_version")
        if not self.repo_id.strip() or not self.revision.strip():
            raise ValueError("repo_id and revision are required")
        if not self.license.strip():
            raise ValueError("An operator-reviewed licence declaration is required")
        return self

    @property
    def request_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


class SnapshotDownloader(Protocol):
    def resolve_revision(self, repo_id: str, revision: str) -> str: ...

    def download(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        allow_patterns: tuple[str, ...],
        ignore_patterns: tuple[str, ...],
    ) -> Path: ...


class HuggingFaceHubDownloader:
    """Explicit operator-triggered Hugging Face download; never used at startup."""

    def __init__(self) -> None:
        try:
            self._hub = importlib.import_module("huggingface_hub")
        except ImportError as error:
            raise RuntimeError("huggingface-hub is not installed") from error

    def resolve_revision(self, repo_id: str, revision: str) -> str:
        info = self._hub.HfApi().model_info(repo_id=repo_id, revision=revision)
        resolved = getattr(info, "sha", None)
        if not isinstance(resolved, str) or not resolved:
            raise RuntimeError("Hugging Face did not return a resolved revision SHA")
        return resolved

    def download(
        self,
        repo_id: str,
        revision: str,
        destination: Path,
        allow_patterns: tuple[str, ...],
        ignore_patterns: tuple[str, ...],
    ) -> Path:
        result = self._hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=list(allow_patterns) or None,
            ignore_patterns=list(ignore_patterns) or None,
        )
        return Path(result)


def acquire_huggingface_snapshot(
    request_path: str | Path,
    output_path: str | Path,
    *,
    downloader: SnapshotDownloader | None = None,
) -> dict[str, object]:
    request_source = Path(request_path)
    destination_report = Path(output_path)
    if destination_report.exists():
        raise ValueError(f"Acquisition report already exists: {destination_report}")
    request = HuggingFaceAcquisitionRequest.model_validate_json(
        request_source.read_text(encoding="utf-8")
    )
    destination = Path(request.destination)
    destination_has_content = destination.exists() and any(
        destination.iterdir() if destination.is_dir() else (destination,)
    )
    if destination_has_content:
        raise ValueError(f"Model destination must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    backend = downloader or HuggingFaceHubDownloader()
    resolved_revision = backend.resolve_revision(request.repo_id, request.revision)
    downloaded = backend.download(
        request.repo_id,
        resolved_revision,
        destination,
        request.allow_patterns,
        request.ignore_patterns,
    )
    if not destination.exists() or not any(destination.rglob("*")):
        raise RuntimeError("Model acquisition created no local files")
    if downloaded.resolve() != destination.resolve():
        if not downloaded.resolve().is_relative_to(destination.resolve()):
            raise RuntimeError("Model acquisition returned a path outside the declared destination")
    weight_sha256 = sha256_tree(destination)
    report = {
        "schema_version": 1,
        "report_kind": "operator_model_asset_acquisition",
        "provider": "huggingface",
        "repo_id": request.repo_id,
        "requested_revision": request.revision,
        "resolved_revision": resolved_revision,
        "destination": str(destination),
        "weight_sha256": weight_sha256,
        "license": request.license,
        "asset_kind": request.asset_kind,
        "request": {
            "path": str(request_source),
            "sha256": request.request_sha256,
        },
        "patterns": {
            "allow": list(request.allow_patterns),
            "ignore": list(request.ignore_patterns),
        },
        "controls": {
            "operator_triggered": True,
            "startup_download": False,
            "ci_download": False,
            "credentials_recorded": False,
            "live_money_authorized": False,
        },
        "limitations": [
            *request.limitations,
            "The declared licence must be reviewed by the operator for the pinned revision.",
            "Successful acquisition is not model validation or promotion evidence.",
        ],
    }
    destination_report.parent.mkdir(parents=True, exist_ok=True)
    destination_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {**report, "output": str(destination_report)}


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
