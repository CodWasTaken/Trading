from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .domain import NewsEvent
from .modeling import sha256_file
from .pretrained_features import sha256_tree

_ALLOWED_EVENT_TYPES = {
    "guidance",
    "earnings",
    "merger_acquisition",
    "analyst_rating",
    "regulatory",
    "legal",
    "management",
    "cybersecurity",
    "financing",
    "buyback",
    "dividend",
    "supply_chain",
    "partnership",
    "contract",
    "restructuring",
    "workforce",
    "product_launch",
    "operations",
    "macro_geopolitical",
    "other",
}


class FinancialEntity(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    entity_type: str


class FinancialRelation(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation: str
    source: str
    target: str


class StructuredNewsInference(BaseModel):
    model_config = ConfigDict(frozen=True)

    sentiment: float = Field(ge=-1, le=1)
    primary_event_type: str
    secondary_event_types: tuple[str, ...] = ()
    entities: tuple[FinancialEntity, ...] = ()
    relations: tuple[FinancialRelation, ...] = ()
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_event_types(self) -> StructuredNewsInference:
        if self.primary_event_type not in _ALLOWED_EVENT_TYPES:
            raise ValueError("Unsupported primary_event_type")
        if any(value not in _ALLOWED_EVENT_TYPES for value in self.secondary_event_types):
            raise ValueError("Unsupported secondary_event_type")
        return self


class NewsModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    model_id: str
    revision: str
    local_model_path: str
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str
    base_model_path: str | None = None
    task_version: str = "fingpt-structured-news-v1"
    max_new_tokens: int = Field(default=384, ge=32, le=2048)
    device: str = "cpu"
    trust_remote_code: bool = False
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_schema(self) -> NewsModelSpec:
        if self.schema_version != 1:
            raise ValueError("Unsupported news model spec schema_version")
        return self

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


class NewsInferenceBackend(Protocol):
    def infer(self, prompt: str) -> str: ...


class TransformersFinGPTBackend:
    """Local-only Transformers/PEFT backend for FinGPT-compatible checkpoints."""

    def __init__(self, spec: NewsModelSpec) -> None:
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError("Transformers is not installed") from error
        tokenizer_path = spec.base_model_path or spec.local_model_path
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=spec.trust_remote_code,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=spec.trust_remote_code,
            device_map=spec.device,
        )
        if spec.base_model_path is not None:
            try:
                peft = importlib.import_module("peft")
            except ImportError as error:
                raise RuntimeError("PEFT is required for a FinGPT adapter checkpoint") from error
            model = peft.PeftModel.from_pretrained(
                model,
                spec.local_model_path,
                local_files_only=True,
            )
        self._model = model
        self._max_new_tokens = spec.max_new_tokens

    def infer(self, prompt: str) -> str:
        encoded = self._tokenizer(prompt, return_tensors="pt")
        device = getattr(self._model, "device", None)
        if device is not None:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = self._model.generate(
            **encoded,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        prompt_length = int(encoded["input_ids"].shape[-1])
        return str(
            self._tokenizer.decode(
                generated[0][prompt_length:],
                skip_special_tokens=True,
            )
        )


def load_news_model_spec(path: str | Path) -> NewsModelSpec:
    spec = NewsModelSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))
    verify_news_weights(spec)
    return spec


def verify_news_weights(spec: NewsModelSpec) -> dict[str, object]:
    source = Path(spec.local_model_path)
    if not source.exists():
        raise ValueError(f"News model path does not exist: {source}")
    actual = sha256_tree(source)
    if actual != spec.weight_sha256:
        raise ValueError(
            "News model weight hash mismatch: "
            f"declared={spec.weight_sha256}, actual={actual}"
        )
    if spec.base_model_path is not None and not Path(spec.base_model_path).exists():
        raise ValueError("Configured FinGPT base_model_path does not exist")
    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "weight_sha256": actual,
        "license": spec.license,
        "manifest_sha256": spec.manifest_sha256,
    }


def build_news_prompt(event: NewsEvent, task_version: str) -> str:
    return (
        f"Task version: {task_version}\n"
        "Analyze the financial news using only the supplied text. Return one JSON object "
        "with keys sentiment, primary_event_type, secondary_event_types, entities, "
        "relations, confidence. sentiment must be between -1 and 1. "
        f"primary_event_type and secondary_event_types must use: {sorted(_ALLOWED_EVENT_TYPES)}. "
        "entities must contain text and entity_type. relations must contain relation, source, "
        "and target. Do not include prose outside the JSON.\n"
        f"Ticker: {event.symbol}\n"
        f"Headline: {event.headline}\n"
        f"Summary: {event.summary}\n"
        f"Source: {event.source}\n"
        f"Published/known at: {event.knowledge_time.isoformat()}"
    )


def parse_structured_news_output(raw_output: str) -> StructuredNewsInference:
    payload = _extract_json_object(raw_output)
    return StructuredNewsInference.model_validate(payload)


def infer_news_archive(
    input_path: str | Path,
    output_path: str | Path,
    spec_path: str | Path,
    *,
    backend: NewsInferenceBackend | None = None,
) -> dict[str, object]:
    source = Path(input_path)
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"News inference output already exists: {destination}")
    spec = load_news_model_spec(spec_path)
    model = backend or TransformersFinGPTBackend(spec)
    events = _load_news(source)
    records: list[dict[str, object]] = []
    failures = 0
    for event in events:
        prompt = build_news_prompt(event, spec.task_version)
        raw_output = model.infer(prompt)
        parse_error: str | None = None
        normalized: dict[str, object] | None
        try:
            inference = parse_structured_news_output(raw_output)
            normalized = inference.model_dump(mode="json")
        except ValueError as error:
            normalized = None
            parse_error = str(error)
            failures += 1
        records.append(
            {
                "schema_version": 1,
                "report_kind": "immutable_financial_news_inference",
                "news_event_id": str(event.id),
                "symbol": event.symbol,
                "article_hash": _article_hash(event),
                "event_time": event.event_time.isoformat(),
                "knowledge_time": event.knowledge_time.isoformat(),
                "inference_time": _utc_now_iso(),
                "model_id": spec.model_id,
                "model_revision": spec.revision,
                "model_manifest_sha256": spec.manifest_sha256,
                "weight_sha256": spec.weight_sha256,
                "task_version": spec.task_version,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_output": raw_output,
                "normalized": normalized,
                "parse_error": parse_error,
                "controls": {
                    "knowledge_time_preserved": True,
                    "future_returns_not_available": True,
                    "direct_order_authority": False,
                },
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "schema_version": 1,
        "report_kind": "financial_news_inference_archive",
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "model": spec.model_dump(mode="json"),
        "model_manifest_sha256": spec.manifest_sha256,
        "records": len(records),
        "parse_failures": failures,
        "limitations": [
            *spec.limitations,
            "Model output is a research feature candidate and never an order instruction.",
            "Improvement must be measured against independently reviewed human labels.",
        ],
    }
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "output": str(destination),
        "metadata": str(metadata_path),
        "records": len(records),
        "parse_failures": failures,
    }


def _extract_json_object(raw_output: str) -> dict[str, Any]:
    stripped = raw_output.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise ValueError("Model output did not contain a JSON object") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as error:
            raise ValueError("Model output contained invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON must be an object")
    return payload


def _load_news(path: Path) -> list[NewsEvent]:
    events: list[NewsEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(NewsEvent.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"Invalid NewsEvent at {path}:{line_number}") from error
    if not events:
        raise ValueError(f"No NewsEvent records found in {path}")
    return events


def _article_hash(event: NewsEvent) -> str:
    payload = {
        "symbol": event.symbol,
        "headline": event.headline,
        "summary": event.summary,
        "source": event.source,
        "event_time": event.event_time.isoformat(),
        "knowledge_time": event.knowledge_time.isoformat(),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
