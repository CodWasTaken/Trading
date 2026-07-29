from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .model_registry import ModelRegistry


class PaperDay(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_date: date
    ending_equity: float = Field(gt=0)


class PaperAttributionRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_date: date
    symbol: str
    sector: str
    direction: Literal["long", "short"]
    news_triggered: bool
    event_type: str
    regime: str
    trade_count: int = Field(ge=0)
    gross_pnl: float
    execution_costs: float = Field(ge=0)
    borrow_costs: float = Field(default=0.0, ge=0)
    margin_interest_costs: float = Field(default=0.0, ge=0)
    dividend_replacement_costs: float = Field(default=0.0, ge=0)
    borrow_validated: bool | None = None

    @property
    def financing_costs(self) -> float:
        return self.borrow_costs + self.margin_interest_costs + self.dividend_replacement_costs

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.execution_costs - self.financing_costs


class ReconciliationObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    observed_at: datetime
    mode: Literal["alpaca-paper"]
    status: Literal["matched", "mismatch", "error"]
    unexplained_differences: int = Field(ge=0)


class OperationalIncident(BaseModel):
    model_config = ConfigDict(frozen=True)

    incident_id: str
    category: str
    status: Literal["open", "resolved"]
    occurred_at: datetime
    resolved_at: datetime | None = None

    @model_validator(mode="after")
    def validate_resolution(self) -> OperationalIncident:
        if self.status == "resolved" and self.resolved_at is None:
            raise ValueError("Resolved incidents require resolved_at")
        if self.status == "open" and self.resolved_at is not None:
            raise ValueError("Open incidents cannot have resolved_at")
        return self


class PaperEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    evidence_kind: Literal["operator_exported_paper_evidence"]
    model_version: str
    execution_mode: Literal["alpaca-paper"]
    starting_equity: float = Field(gt=0)
    funding_fx_costs: float = Field(default=0.0, ge=0)
    days: tuple[PaperDay, ...]
    attribution: tuple[PaperAttributionRow, ...]
    reconciliation: tuple[ReconciliationObservation, ...]
    incidents: tuple[OperationalIncident, ...] = ()
    source_ledger_path: str
    source_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_evidence(self) -> PaperEvidence:
        if self.schema_version != 1:
            raise ValueError("Unsupported paper evidence schema_version")
        session_dates = [item.session_date for item in self.days]
        if not session_dates or len(session_dates) != len(set(session_dates)):
            raise ValueError("Paper evidence requires unique trading days")
        if session_dates != sorted(session_dates):
            raise ValueError("Paper evidence days must be chronological")
        known_dates = set(session_dates)
        if any(item.session_date not in known_dates for item in self.attribution):
            raise ValueError("Attribution row is outside the paper evidence window")
        if not self.attribution:
            raise ValueError("Paper evidence requires PnL attribution")
        return self


class GraduationGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    name: str = "paper-only-graduation-v1"
    minimum_trading_days: int = Field(default=60, ge=60)
    preferred_trading_days: int = Field(default=90, ge=60)
    minimum_combined_net_pnl: float = 0.0
    minimum_long_net_pnl: float = 0.0
    minimum_short_net_pnl: float = 0.0
    maximum_drawdown: float = Field(default=0.15, gt=0, le=1)
    maximum_tail_loss: float = Field(default=0.05, gt=0, le=1)
    maximum_symbol_pnl_contribution: float = Field(default=0.20, gt=0, le=1)
    maximum_sector_pnl_contribution: float = Field(default=0.35, gt=0, le=1)
    maximum_event_type_pnl_contribution: float = Field(default=0.50, gt=0, le=1)
    maximum_regime_pnl_contribution: float = Field(default=0.50, gt=0, le=1)
    maximum_unexplained_reconciliation_differences: int = Field(default=0, ge=0)
    maximum_unresolved_data_quality_incidents: int = Field(default=0, ge=0)
    maximum_borrow_violations: int = Field(default=0, ge=0)
    pnl_reconciliation_tolerance: float = Field(default=0.01, ge=0)

    @model_validator(mode="after")
    def validate_schema(self) -> GraduationGateConfig:
        if self.schema_version != 1:
            raise ValueError("Unsupported graduation gate schema_version")
        if self.preferred_trading_days < self.minimum_trading_days:
            raise ValueError("preferred_trading_days cannot be below the minimum")
        return self

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.model_dump(mode="json"))).hexdigest()


def load_graduation_gates(path: str | Path | None) -> GraduationGateConfig:
    if path is None:
        return GraduationGateConfig()
    return GraduationGateConfig.model_validate_json(_resolve_path(path).read_text(encoding="utf-8"))


def evaluate_paper_graduation(
    *,
    registry_path: str,
    paper_evidence_path: str,
    replay_report_path: str,
    replay_diagnostics_path: str,
    monitoring_report_paths: list[str],
    output_path: str,
    gate_config_path: str | None = None,
    version: str | None = None,
) -> dict[str, object]:
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"Graduation output already exists: {destination}")
    if not monitoring_report_paths:
        raise ValueError("At least one monitoring report is required")

    registry = ModelRegistry(registry_path)
    record = registry.get(version) if version else registry.champion()
    if record is None:
        raise ValueError("Paper graduation requires a historical champion")
    paper_path = Path(paper_evidence_path)
    replay_path = Path(replay_report_path)
    diagnostics_path = Path(replay_diagnostics_path)
    monitoring_paths = [Path(item) for item in monitoring_report_paths]
    evidence = PaperEvidence.model_validate_json(paper_path.read_text(encoding="utf-8"))
    paper_evidence_sha256 = _sha256(paper_path)
    if any(
        event.get("paper_evidence_sha256") == paper_evidence_sha256
        for event in registry.paper_graduations(version=record.version)
    ):
        raise ValueError("This model and paper evidence hash already have a graduation record")
    gates = load_graduation_gates(gate_config_path)
    replay = _load_object(replay_path)
    diagnostics = _load_object(diagnostics_path)
    monitoring = [_load_object(path) for path in monitoring_paths]

    failures: list[str] = []
    if evidence.model_version != record.version:
        failures.append("paper_evidence_model_version_mismatch")
    ledger_path = Path(evidence.source_ledger_path)
    if not ledger_path.is_absolute():
        ledger_path = paper_path.parent / ledger_path
    if not ledger_path.is_file():
        failures.append("source_ledger_missing")
        actual_ledger_sha256 = None
    else:
        actual_ledger_sha256 = _sha256(ledger_path)
        if actual_ledger_sha256 != evidence.source_ledger_sha256:
            failures.append("source_ledger_hash_mismatch")
    _check_replay(failures, replay, diagnostics, record.version)
    _check_monitoring(failures, monitoring, record.version)

    dates = [item.session_date for item in evidence.days]
    if len(dates) < gates.minimum_trading_days:
        failures.append("insufficient_paper_trading_days")
    preferred_duration_met = len(dates) >= gates.preferred_trading_days
    reconciliation_dates = {
        item.observed_at.date()
        for item in evidence.reconciliation
        if item.status == "matched" and item.unexplained_differences == 0
    }
    if not set(dates).issubset(reconciliation_dates):
        failures.append("daily_broker_reconciliation_missing")
    unexplained = sum(
        item.unexplained_differences
        for item in evidence.reconciliation
        if item.status != "matched" or item.unexplained_differences
    )
    if unexplained > gates.maximum_unexplained_reconciliation_differences:
        failures.append("unexplained_broker_reconciliation_differences")

    unresolved_data_quality = sum(
        item.category == "data_quality" and item.status == "open" for item in evidence.incidents
    )
    if unresolved_data_quality > gates.maximum_unresolved_data_quality_incidents:
        failures.append("unresolved_data_quality_incidents")
    borrow_violations = sum(
        item.category == "borrow_violation" for item in evidence.incidents
    ) + sum(
        item.direction == "short" and item.borrow_validated is not True
        for item in evidence.attribution
    )
    if borrow_violations > gates.maximum_borrow_violations:
        failures.append("borrow_violations")

    report_sections = _attribution_report(evidence)
    combined = _nested_net(report_sections, "combined")
    long_net = _nested_net(report_sections, "directions", "long")
    short_net = _nested_net(report_sections, "directions", "short")
    if combined <= gates.minimum_combined_net_pnl:
        failures.append("combined_performance_below_gate")
    if long_net <= gates.minimum_long_net_pnl:
        failures.append("long_performance_below_gate")
    if short_net <= gates.minimum_short_net_pnl:
        failures.append("short_performance_below_gate")

    daily = _daily_performance(evidence)
    if _numeric_metric(daily, "maximum_drawdown") >= gates.maximum_drawdown:
        failures.append("paper_drawdown_above_gate")
    if _numeric_metric(daily, "tail_loss_95") > gates.maximum_tail_loss:
        failures.append("paper_tail_loss_above_gate")
    if abs(_numeric_metric(daily, "attribution_difference")) > gates.pnl_reconciliation_tolerance:
        failures.append("paper_pnl_attribution_does_not_reconcile")

    concentrations = {
        "symbol": _concentration(report_sections["symbols"]),
        "sector": _concentration(report_sections["sectors"]),
        "event_type": _concentration(report_sections["event_types"]),
        "regime": _concentration(report_sections["regimes"]),
    }
    for name, limit in (
        ("symbol", gates.maximum_symbol_pnl_contribution),
        ("sector", gates.maximum_sector_pnl_contribution),
        ("event_type", gates.maximum_event_type_pnl_contribution),
        ("regime", gates.maximum_regime_pnl_contribution),
    ):
        value = concentrations[name]
        if value is None:
            failures.append(f"{name}_pnl_attribution_missing")
        elif value > limit:
            failures.append(f"{name}_pnl_concentration_above_gate")

    passed = not failures
    source_hashes = {
        "paper_evidence": paper_evidence_sha256,
        "source_ledger": {
            "path": str(ledger_path),
            "declared_sha256": evidence.source_ledger_sha256,
            "actual_sha256": actual_ledger_sha256,
        },
        "replay": _sha256(replay_path),
        "replay_diagnostics": _sha256(diagnostics_path),
        "monitoring": [{"path": str(path), "sha256": _sha256(path)} for path in monitoring_paths],
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "report_kind": "paper_only_graduation",
        "created_at": datetime.now(UTC).isoformat(),
        "model_version": record.version,
        "passed": passed,
        "status": "paper_only_graduated" if passed else "paper_only_not_graduated",
        "execution_scope": "paper_only",
        "live_money_authorized": False,
        "preferred_90_day_duration_met": preferred_duration_met,
        "failures": failures,
        "gates": {
            **gates.model_dump(mode="json"),
            "manifest_sha256": gates.manifest_sha256,
        },
        "paper_window": {
            "start": dates[0].isoformat(),
            "end": dates[-1].isoformat(),
            "trading_days": len(dates),
            "execution_mode": evidence.execution_mode,
            "source_ledger_sha256": evidence.source_ledger_sha256,
            "source_ledger_hash_verified": (actual_ledger_sha256 == evidence.source_ledger_sha256),
        },
        "performance": {
            **daily,
            **report_sections,
            "concentration": concentrations,
        },
        "operations": {
            "reconciliation_observations": len(evidence.reconciliation),
            "unexplained_reconciliation_differences": unexplained,
            "unresolved_data_quality_incidents": unresolved_data_quality,
            "borrow_violations": borrow_violations,
            "monitoring_reports": len(monitoring),
            "monitoring_healthy": all(item.get("healthy") is True for item in monitoring),
            "deterministic_full_engine_replay": (
                replay.get("replay_kind") == "shared_engine_historical_replay"
                and replay.get("determinism_verified") is True
                and diagnostics.get("determinism_verified") is True
            ),
        },
        "source_hashes": source_hashes,
        "controls": {
            "historical_champion_unchanged": True,
            "paper_evidence_operator_exported": True,
            "source_ledger_hash_verified": (actual_ledger_sha256 == evidence.source_ledger_sha256),
            "all_cost_families_reported": True,
            "broker_reconciliation_required_each_day": True,
            "no_live_execution_path": True,
        },
        "limitations": [
            "Graduation authorizes continued paper evaluation only.",
            "Operator-exported attribution must be reconciled to the immutable ledger.",
            "Historical replay and paper results do not establish future profitability.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    destination.write_text(payload, encoding="utf-8")
    report_sha256 = hashlib.sha256(payload.encode()).hexdigest()
    registry.record_paper_graduation(
        record.version,
        report_path=str(destination),
        report_sha256=report_sha256,
        paper_evidence_sha256=paper_evidence_sha256,
        passed=passed,
        status=str(report["status"]),
        gates=report["gates"],
    )
    return report


def _check_replay(
    failures: list[str],
    replay: dict[str, object],
    diagnostics: dict[str, object],
    version: str,
) -> None:
    if replay.get("replay_kind") != "shared_engine_historical_replay":
        failures.append("full_engine_replay_missing")
    if replay.get("determinism_verified") is not True:
        failures.append("replay_determinism_not_verified")
    if not str(replay.get("strategy", "")).startswith(f"champion:{version}:"):
        failures.append("replay_model_version_mismatch")
    events = replay.get("events")
    if not isinstance(events, dict) or not events.get("trace_sha256"):
        failures.append("replay_trace_hash_missing")
    if diagnostics.get("report_kind") != "historical_replay_diagnostics":
        failures.append("replay_diagnostics_missing")
    if diagnostics.get("strategy_mode") != "champion":
        failures.append("replay_diagnostics_not_champion")
    if diagnostics.get("determinism_verified") is not True:
        failures.append("diagnostic_replay_determinism_not_verified")


def _check_monitoring(
    failures: list[str],
    reports: list[dict[str, object]],
    version: str,
) -> None:
    for report in reports:
        if report.get("report_kind") != "model_drift_and_calibration_monitoring":
            failures.append("monitoring_report_kind_invalid")
            continue
        model = report.get("model")
        if not isinstance(model, dict) or model.get("version") != version:
            failures.append("monitoring_model_version_mismatch")
        if report.get("healthy") is not True:
            failures.append("monitoring_unhealthy")
        if report.get("recommended_action") != "continue_paper":
            failures.append("monitoring_does_not_recommend_continue_paper")


def _attribution_report(evidence: PaperEvidence) -> dict[str, object]:
    groups: dict[str, dict[str, dict[str, float | int]]] = {
        name: defaultdict(_empty_group)
        for name in (
            "directions",
            "news_split",
            "symbols",
            "sectors",
            "event_types",
            "regimes",
        )
    }
    totals = _empty_group()
    for row in evidence.attribution:
        keys = {
            "directions": row.direction,
            "news_split": "news_triggered" if row.news_triggered else "non_news",
            "symbols": row.symbol,
            "sectors": row.sector,
            "event_types": row.event_type,
            "regimes": row.regime,
        }
        _add_row(totals, row)
        for group_name, key in keys.items():
            _add_row(groups[group_name][key], row)
    return {
        "combined": totals,
        **{
            name: {key: value for key, value in sorted(values.items())}
            for name, values in groups.items()
        },
        "financing_costs": {
            "borrow": sum(item.borrow_costs for item in evidence.attribution),
            "margin_interest": sum(item.margin_interest_costs for item in evidence.attribution),
            "dividend_replacement": sum(
                item.dividend_replacement_costs for item in evidence.attribution
            ),
            "funding_fx": evidence.funding_fx_costs,
        },
    }


def _empty_group() -> dict[str, float | int]:
    return {
        "rows": 0,
        "trades": 0,
        "gross_pnl": 0.0,
        "execution_costs": 0.0,
        "financing_costs": 0.0,
        "net_pnl": 0.0,
    }


def _add_row(target: dict[str, float | int], row: PaperAttributionRow) -> None:
    target["rows"] = int(target["rows"]) + 1
    target["trades"] = int(target["trades"]) + row.trade_count
    target["gross_pnl"] = float(target["gross_pnl"]) + row.gross_pnl
    target["execution_costs"] = float(target["execution_costs"]) + row.execution_costs
    target["financing_costs"] = float(target["financing_costs"]) + row.financing_costs
    target["net_pnl"] = float(target["net_pnl"]) + row.net_pnl


def _daily_performance(evidence: PaperEvidence) -> dict[str, object]:
    attribution_by_day: dict[date, float] = defaultdict(float)
    for row in evidence.attribution:
        attribution_by_day[row.session_date] += row.net_pnl
    previous = evidence.starting_equity
    peak = previous
    maximum_drawdown = 0.0
    returns: list[float] = []
    differences: list[float] = []
    for day in evidence.days:
        daily_pnl = day.ending_equity - previous
        differences.append(daily_pnl - attribution_by_day[day.session_date])
        returns.append(daily_pnl / previous)
        peak = max(peak, day.ending_equity)
        maximum_drawdown = max(
            maximum_drawdown,
            (peak - day.ending_equity) / peak,
        )
        previous = day.ending_equity
    worst_count = max(1, math.ceil(len(returns) * 0.05))
    tail_mean = sum(sorted(returns)[:worst_count]) / worst_count
    return {
        "starting_equity": evidence.starting_equity,
        "ending_equity": evidence.days[-1].ending_equity,
        "net_return_after_execution_and_financing_costs": (
            evidence.days[-1].ending_equity / evidence.starting_equity - 1
        ),
        "maximum_drawdown": maximum_drawdown,
        "tail_loss_95": max(0.0, -tail_mean),
        "attribution_difference": sum(differences),
        "daily_returns": returns,
    }


def _concentration(raw: object) -> float | None:
    if not isinstance(raw, dict) or not raw:
        return None
    values = []
    for payload in raw.values():
        if not isinstance(payload, dict):
            return None
        value = payload.get("net_pnl")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    total = sum(values)
    if total <= 0:
        return None
    return max(max(value, 0.0) for value in values) / total


def _nested_net(payload: dict[str, object], *keys: str) -> float:
    current: object = payload
    for key in keys:
        if not isinstance(current, dict):
            return float("-inf")
        current = current.get(key)
    if not isinstance(current, dict):
        return float("-inf")
    value = current.get("net_pnl")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("-inf")
    return float(value)


def _numeric_metric(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Internal graduation metric is not numeric: {key}")
    return float(value)


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve_path(path: str | Path) -> Path:
    source = Path(path)
    if not source.is_absolute() and not source.is_file():
        repository_source = Path(__file__).resolve().parents[3] / source
        if repository_source.is_file():
            return repository_source
    return source


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
