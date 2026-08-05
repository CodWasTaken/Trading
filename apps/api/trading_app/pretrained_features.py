from __future__ import annotations

import hashlib
import importlib
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .dataset import load_feature_dataset, write_feature_dataset
from .historical import HistoricalBar
from .modeling import sha256_file
from .research import FeatureRow

FoundationProvider = Literal["timesfm", "chronos", "moirai"]


class PretrainedModelSpec(BaseModel):
    """Immutable local-weight declaration for a time-series foundation model."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    provider: FoundationProvider
    model_id: str
    revision: str
    local_path: str
    weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str
    context_length: int = Field(default=256, ge=20)
    prediction_length: int = Field(default=5, ge=1)
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    device: str = "cpu"
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_spec(self) -> PretrainedModelSpec:
        if self.schema_version != 1:
            raise ValueError("Unsupported pretrained model spec schema_version")
        if sorted(self.quantile_levels) != list(self.quantile_levels):
            raise ValueError("quantile_levels must be sorted")
        if any(not 0 < value < 1 for value in self.quantile_levels):
            raise ValueError("quantile_levels must be between zero and one")
        return self

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


class ForecastSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    point_forecast: tuple[float, ...]
    quantiles: dict[str, tuple[float, ...]]
    expected_return: float
    uncertainty: float = Field(ge=0)


class FoundationForecaster(Protocol):
    def forecast(self, history: tuple[float, ...], horizon: int) -> ForecastSummary: ...


class TimesFMForecaster:
    def __init__(self, spec: PretrainedModelSpec) -> None:
        self.spec = spec
        try:
            timesfm = importlib.import_module("timesfm")
            numpy = importlib.import_module("numpy")
        except ImportError as error:
            raise RuntimeError("TimesFM dependencies are not installed") from error
        model_class = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        config_class = getattr(timesfm, "ForecastConfig", None)
        if model_class is None or config_class is None:
            raise RuntimeError("Installed TimesFM package does not expose the 2.5 API")
        self._numpy = numpy
        self._model = model_class.from_pretrained(spec.local_path)
        self._model.compile(
            config_class(
                max_context=spec.context_length,
                max_horizon=spec.prediction_length,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )

    def forecast(self, history: tuple[float, ...], horizon: int) -> ForecastSummary:
        point, quantile = self._model.forecast(
            horizon=horizon,
            inputs=[self._numpy.asarray(history, dtype=float)],
        )
        point_values = tuple(float(value) for value in point[0].tolist())
        quantile_values: dict[str, tuple[float, ...]] = {}
        raw = quantile[0]
        # TimesFM 2.5 emits mean followed by deciles. Use matching requested levels.
        for level in self.spec.quantile_levels:
            index = max(1, min(9, round(level * 10)))
            quantile_values[_quantile_key(level)] = tuple(
                float(value) for value in raw[:, index].tolist()
            )
        return _summarize(history, point_values, quantile_values)


class ChronosForecaster:
    def __init__(self, spec: PretrainedModelSpec) -> None:
        self.spec = spec
        try:
            chronos = importlib.import_module("chronos")
            torch = importlib.import_module("torch")
        except ImportError as error:
            raise RuntimeError("Chronos dependencies are not installed") from error
        pipeline_class = (
            getattr(chronos, "Chronos2Pipeline", None)
            or getattr(chronos, "BaseChronosPipeline", None)
            or getattr(chronos, "ChronosPipeline", None)
        )
        if pipeline_class is None:
            raise RuntimeError("Installed Chronos package does not expose a pipeline API")
        self._torch = torch
        self._pipeline = pipeline_class.from_pretrained(
            spec.local_path,
            local_files_only=True,
            device_map=spec.device,
        )

    def forecast(self, history: tuple[float, ...], horizon: int) -> ForecastSummary:
        context = [self._torch.tensor(history, dtype=self._torch.float32)]
        levels = list(self.spec.quantile_levels)
        if hasattr(self._pipeline, "predict_quantiles"):
            quantiles, mean = self._pipeline.predict_quantiles(
                context,
                prediction_length=horizon,
                quantile_levels=levels,
            )
            mean_values = tuple(float(value) for value in mean[0].tolist())
            raw_quantiles = quantiles[0]
            mapped = {
                _quantile_key(level): tuple(
                    float(value) for value in raw_quantiles[:, index].tolist()
                )
                for index, level in enumerate(levels)
            }
            return _summarize(history, mean_values, mapped)
        samples = self._pipeline.predict(context, prediction_length=horizon)
        sample_values = samples[0]
        mean_values = tuple(float(value) for value in sample_values.mean(dim=0).tolist())
        mapped = {
            _quantile_key(level): tuple(
                float(value)
                for value in self._torch.quantile(sample_values, level, dim=0).tolist()
            )
            for level in levels
        }
        return _summarize(history, mean_values, mapped)


class MoiraiForecaster:
    def __init__(self, spec: PretrainedModelSpec) -> None:
        self.spec = spec
        try:
            self._pandas = importlib.import_module("pandas")
            pandas_dataset = importlib.import_module("gluonts.dataset.pandas")
            if "2.0" in spec.model_id or "moirai2" in spec.model_id.lower():
                moirai = importlib.import_module("uni2ts.model.moirai2")
                module = moirai.Moirai2Module.from_pretrained(spec.local_path)
                self._model = moirai.Moirai2Forecast(
                    module=module,
                    prediction_length=spec.prediction_length,
                    context_length=spec.context_length,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )
            else:
                moirai = importlib.import_module("uni2ts.model.moirai")
                module = moirai.MoiraiModule.from_pretrained(spec.local_path)
                self._model = moirai.MoiraiForecast(
                    module=module,
                    prediction_length=spec.prediction_length,
                    context_length=spec.context_length,
                    patch_size="auto",
                    num_samples=100,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )
            self._dataset_class = pandas_dataset.PandasDataset
        except ImportError as error:
            raise RuntimeError("Moirai/Uni2TS dependencies are not installed") from error
        self._predictor = self._model.create_predictor(batch_size=1)

    def forecast(self, history: tuple[float, ...], horizon: int) -> ForecastSummary:
        if horizon != self.spec.prediction_length:
            raise ValueError("Moirai predictor horizon must match its frozen specification")
        index = self._pandas.date_range("2000-01-01", periods=len(history), freq="h")
        series = self._pandas.Series(history, index=index)
        dataset = self._dataset_class({"target": series})
        forecast = next(iter(self._predictor.predict(dataset)))
        samples = forecast.samples
        point_values = tuple(float(value) for value in samples.mean(axis=0).tolist())
        mapped = {
            _quantile_key(level): tuple(
                float(value) for value in self._numpy_quantile(samples, level).tolist()
            )
            for level in self.spec.quantile_levels
        }
        return _summarize(history, point_values, mapped)

    @staticmethod
    def _numpy_quantile(samples: Any, level: float) -> Any:
        numpy = importlib.import_module("numpy")
        return numpy.quantile(samples, level, axis=0)


def load_pretrained_spec(path: str | Path) -> PretrainedModelSpec:
    spec = PretrainedModelSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))
    verify_pretrained_weights(spec)
    return spec


def verify_pretrained_weights(spec: PretrainedModelSpec) -> dict[str, object]:
    source = Path(spec.local_path)
    if not source.exists():
        raise ValueError(f"Pretrained model path does not exist: {source}")
    actual = sha256_tree(source)
    if actual != spec.weight_sha256:
        raise ValueError(
            "Pretrained weight hash mismatch: "
            f"declared={spec.weight_sha256}, actual={actual}"
        )
    return {
        "provider": spec.provider,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "local_path": str(source),
        "weight_sha256": actual,
        "license": spec.license,
        "manifest_sha256": spec.manifest_sha256,
    }


def build_forecaster(spec: PretrainedModelSpec) -> FoundationForecaster:
    if spec.provider == "timesfm":
        return TimesFMForecaster(spec)
    if spec.provider == "chronos":
        return ChronosForecaster(spec)
    if spec.provider == "moirai":
        return MoiraiForecaster(spec)
    raise ValueError(f"Unsupported foundation provider: {spec.provider}")


def generate_foundation_sidecar(
    bars_path: str | Path,
    output_path: str | Path,
    spec_path: str | Path,
    *,
    stride: int = 1,
    bar_interval: timedelta = timedelta(hours=1),
    forecaster: FoundationForecaster | None = None,
) -> dict[str, object]:
    if stride < 1:
        raise ValueError("stride must be positive")
    source = Path(bars_path)
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"Foundation sidecar output already exists: {destination}")
    spec = load_pretrained_spec(spec_path)
    backend = forecaster or build_forecaster(spec)
    bars = _load_json_lines(source, HistoricalBar)
    grouped: dict[str, list[HistoricalBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.symbol.upper()].append(bar)
    records: list[dict[str, object]] = []
    for symbol, symbol_bars in sorted(grouped.items()):
        ordered = sorted(symbol_bars, key=lambda item: item.timestamp)
        for index in range(spec.context_length - 1, len(ordered), stride):
            history_bars = ordered[index - spec.context_length + 1 : index + 1]
            history = tuple(float(item.close) for item in history_bars)
            if any(value <= 0 for value in history):
                raise ValueError(f"Non-positive close in foundation context for {symbol}")
            summary = backend.forecast(history, spec.prediction_length)
            history_end = _as_utc(ordered[index].timestamp) + bar_interval
            records.append(
                {
                    "schema_version": 1,
                    "symbol": symbol,
                    "feature_time": history_end.isoformat(),
                    "history_end_time": history_end.isoformat(),
                    "forecast_end_time": (
                        history_end + bar_interval * spec.prediction_length
                    ).isoformat(),
                    "provider": spec.provider,
                    "model_id": spec.model_id,
                    "model_revision": spec.revision,
                    "model_manifest_sha256": spec.manifest_sha256,
                    "weight_sha256": spec.weight_sha256,
                    "context_length": spec.context_length,
                    "prediction_length": spec.prediction_length,
                    "point_forecast": list(summary.point_forecast),
                    "quantiles": {
                        key: list(value) for key, value in summary.quantiles.items()
                    },
                    "expected_return": summary.expected_return,
                    "uncertainty": summary.uncertainty,
                    "controls": {
                        "past_closes_only": True,
                        "target_return_accessed": False,
                        "future_bars_accessed": False,
                    },
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "schema_version": 1,
        "report_kind": "foundation_feature_sidecar",
        "source_bars": {"path": str(source), "sha256": sha256_file(source)},
        "spec": spec.model_dump(mode="json"),
        "spec_sha256": spec.manifest_sha256,
        "records": len(records),
        "symbols": sorted(grouped),
        "point_in_time_controls": {
            "uses_only_past_closes": True,
            "feature_time_after_bar_close": True,
            "target_return_never_passed_to_forecaster": True,
        },
    }
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {
        "output": str(destination),
        "metadata": str(metadata_path),
        "records": len(records),
        "symbols": sorted(grouped),
        "provider": spec.provider,
    }


def join_foundation_sidecar(
    dataset_path: str | Path,
    sidecar_path: str | Path,
    output_path: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    dataset_source = Path(dataset_path)
    sidecar_source = Path(sidecar_path)
    rows, feature_names, metadata = load_feature_dataset(dataset_source)
    sidecar = _load_objects(sidecar_source)
    by_key: dict[tuple[datetime, str], dict[str, object]] = {}
    providers: set[str] = set()
    for record in sidecar:
        feature_time = _parse_time(record.get("feature_time"))
        history_end = _parse_time(record.get("history_end_time"))
        if history_end > feature_time:
            raise ValueError("Foundation sidecar history ends after feature_time")
        symbol = str(record.get("symbol", "")).upper()
        key = (feature_time, symbol)
        if key in by_key:
            raise ValueError(f"Duplicate foundation sidecar key: {key}")
        by_key[key] = record
        providers.add(str(record.get("provider", "foundation")))
    if len(providers) != 1:
        raise ValueError("A joined foundation sidecar must contain exactly one provider")
    provider = next(iter(providers)).replace("-", "_")
    extension_names = (
        f"{provider}_expected_return",
        f"{provider}_uncertainty",
    )
    joined: list[FeatureRow] = []
    missing: list[tuple[str, str]] = []
    for row in rows:
        record = by_key.get((_as_utc(row.timestamp), row.symbol.upper()))
        if record is None:
            missing.append((row.timestamp.isoformat(), row.symbol))
            continue
        joined.append(
            FeatureRow(
                timestamp=row.timestamp,
                symbol=row.symbol,
                features=(
                    *row.features,
                    _number(record.get("expected_return"), "expected_return"),
                    _number(record.get("uncertainty"), "uncertainty"),
                ),
                target_return=row.target_return,
                label_end_time=row.label_end_time,
            )
        )
    if missing and require_complete:
        preview = missing[:5]
        raise ValueError(
            f"Foundation sidecar is missing {len(missing)} dataset rows; examples={preview}"
        )
    if not joined:
        raise ValueError("Foundation sidecar did not match any dataset rows")
    output_metadata = {
        **metadata,
        "foundation_features": {
            "provider": provider,
            "sidecar_path": str(sidecar_source),
            "sidecar_sha256": sha256_file(sidecar_source),
            "joined_rows": len(joined),
            "missing_rows": len(missing),
            "point_in_time_verified": True,
        },
    }
    destination, metadata_path = write_feature_dataset(
        output_path,
        joined,
        (*feature_names, *extension_names),
        output_metadata,
    )
    return {
        "output": str(destination),
        "metadata": str(metadata_path),
        "rows": len(joined),
        "missing": len(missing),
        "feature_names": [*feature_names, *extension_names],
    }


def sha256_tree(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in source.rglob("*") if candidate.is_file()):
        relative = item.relative_to(source).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _summarize(
    history: tuple[float, ...],
    point_forecast: tuple[float, ...],
    quantiles: dict[str, tuple[float, ...]],
) -> ForecastSummary:
    if not point_forecast:
        raise ValueError("Foundation model returned an empty point forecast")
    final_price = point_forecast[-1]
    expected_return = final_price / history[-1] - 1
    terminal_quantiles = [values[-1] for values in quantiles.values() if values]
    uncertainty = (
        max(terminal_quantiles) - min(terminal_quantiles)
        if len(terminal_quantiles) >= 2
        else 0.0
    ) / history[-1]
    return ForecastSummary(
        point_forecast=point_forecast,
        quantiles=quantiles,
        expected_return=expected_return,
        uncertainty=max(0.0, uncertainty),
    )


def _load_json_lines(path: Path, model_type: type[BaseModel]) -> list[Any]:
    records: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model_type.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"Invalid record at {path}:{line_number}") from error
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _load_objects(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(payload)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Expected ISO-8601 timestamp")
    return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _quantile_key(level: float) -> str:
    return f"q{int(round(level * 100)):02d}"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
