from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from trading_app.graduation import (
    GraduationGateConfig,
    PaperEvidence,
    evaluate_paper_graduation,
    load_graduation_gates,
)
from trading_app.model_registry import ModelRegistry
from trading_app.research import BacktestMetrics, RidgeReturnModel, synthetic_feature_rows

GATE_CONFIG = Path(__file__).parents[1] / "config" / "graduation" / "paper-only-v1.json"


def _champion(tmp_path: Path) -> tuple[ModelRegistry, str]:
    model = RidgeReturnModel(("momentum", "news_score", "volatility", "spread_bps"))
    model.fit(synthetic_feature_rows(220))
    registry = ModelRegistry(tmp_path / "registry")
    record = registry.register(
        model,
        BacktestMetrics(
            observations=300,
            scored_observations=1000,
            net_return=0.10,
            annualized_return=0.10,
            sharpe=0.75,
            max_drawdown=0.10,
            hit_rate=0.55,
            turnover=20,
            average_trade_return=0.001,
            folds=8,
            excess_return_vs_benchmark=0.02,
        ),
        metadata={"dataset": "synthetic_fixture"},
    )
    registry.promote(record.version)
    return registry, record.version


def _write_evidence(
    path: Path,
    version: str,
    *,
    days: int = 60,
    reconciliation_status: str = "matched",
    incidents: list[dict[str, object]] | None = None,
) -> None:
    ledger = path.parent / "events.db"
    ledger.write_bytes(b"immutable-test-ledger")
    ledger_sha256 = hashlib.sha256(ledger.read_bytes()).hexdigest()
    start = date(2026, 1, 2)
    sessions = [start + timedelta(days=index) for index in range(days)]
    attribution = []
    for index, session in enumerate(sessions):
        symbol = f"SYM{index % 5}"
        sector = f"Sector {index % 5}"
        regime = "bull_low_vol" if index % 2 == 0 else "bear_high_vol"
        attribution.extend(
            [
                {
                    "session_date": session.isoformat(),
                    "symbol": symbol,
                    "sector": sector,
                    "direction": "long",
                    "news_triggered": True,
                    "event_type": "earnings",
                    "regime": regime,
                    "trade_count": 1,
                    "gross_pnl": 52.0,
                    "execution_costs": 2.0,
                    "borrow_validated": None,
                },
                {
                    "session_date": session.isoformat(),
                    "symbol": symbol,
                    "sector": sector,
                    "direction": "short",
                    "news_triggered": False,
                    "event_type": "none",
                    "regime": regime,
                    "trade_count": 1,
                    "gross_pnl": 52.0,
                    "execution_costs": 1.0,
                    "borrow_costs": 1.0,
                    "borrow_validated": True,
                },
            ]
        )
    payload = {
        "schema_version": 1,
        "evidence_kind": "operator_exported_paper_evidence",
        "model_version": version,
        "execution_mode": "alpaca-paper",
        "starting_equity": 100_000.0,
        "funding_fx_costs": 25.0,
        "days": [
            {
                "session_date": session.isoformat(),
                "ending_equity": 100_000.0 + (index + 1) * 100.0,
            }
            for index, session in enumerate(sessions)
        ],
        "attribution": attribution,
        "reconciliation": [
            {
                "observed_at": datetime.combine(
                    session,
                    time(21, 0),
                    tzinfo=UTC,
                ).isoformat(),
                "mode": "alpaca-paper",
                "status": reconciliation_status,
                "unexplained_differences": (0 if reconciliation_status == "matched" else 1),
            }
            for session in sessions
        ],
        "incidents": incidents or [],
        "source_ledger_path": ledger.name,
        "source_ledger_sha256": ledger_sha256,
        "limitations": ["Synthetic fixture for pipeline testing only."],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_supporting_reports(
    tmp_path: Path,
    version: str,
    *,
    deterministic: bool = True,
    monitoring_healthy: bool = True,
) -> tuple[Path, Path, Path]:
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "replay_kind": "shared_engine_historical_replay",
                "strategy": f"champion:{version}:0.0005",
                "determinism_verified": deterministic,
                "events": {"trace_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "report_kind": "historical_replay_diagnostics",
                "strategy_mode": "champion",
                "determinism_verified": deterministic,
            }
        ),
        encoding="utf-8",
    )
    monitoring = tmp_path / "monitoring.json"
    monitoring.write_text(
        json.dumps(
            {
                "report_kind": "model_drift_and_calibration_monitoring",
                "model": {"version": version},
                "healthy": monitoring_healthy,
                "recommended_action": (
                    "continue_paper" if monitoring_healthy else "pause_and_review"
                ),
            }
        ),
        encoding="utf-8",
    )
    return replay, diagnostics, monitoring


def test_paper_graduation_reports_all_required_attribution_and_remains_paper_only(
    tmp_path: Path,
) -> None:
    registry, version = _champion(tmp_path)
    evidence = tmp_path / "paper.json"
    _write_evidence(evidence, version)
    replay, diagnostics, monitoring = _write_supporting_reports(tmp_path, version)

    report = evaluate_paper_graduation(
        registry_path=str(registry.root),
        paper_evidence_path=str(evidence),
        replay_report_path=str(replay),
        replay_diagnostics_path=str(diagnostics),
        monitoring_report_paths=[str(monitoring)],
        output_path=str(tmp_path / "graduation.json"),
        gate_config_path=str(GATE_CONFIG),
    )

    assert report["passed"] is True
    assert report["status"] == "paper_only_graduated"
    assert report["execution_scope"] == "paper_only"
    assert report["live_money_authorized"] is False
    assert report["preferred_90_day_duration_met"] is False
    performance = report["performance"]
    assert performance["directions"]["long"]["trades"] == 60
    assert performance["directions"]["short"]["trades"] == 60
    assert performance["news_split"]["news_triggered"]["trades"] == 60
    assert performance["news_split"]["non_news"]["trades"] == 60
    assert len(performance["symbols"]) == 5
    assert len(performance["sectors"]) == 5
    assert set(performance["regimes"]) == {"bear_high_vol", "bull_low_vol"}
    assert performance["financing_costs"]["borrow"] == pytest.approx(60.0)
    assert performance["financing_costs"]["funding_fx"] == pytest.approx(25.0)
    records = registry.paper_graduations(version=version)
    assert len(records) == 1
    assert records[0]["execution_scope"] == "paper_only"
    assert records[0]["live_money_authorized"] is False


def test_graduation_fails_closed_on_operational_and_evidence_gaps(
    tmp_path: Path,
) -> None:
    registry, version = _champion(tmp_path)
    evidence = tmp_path / "paper.json"
    _write_evidence(
        evidence,
        version,
        days=59,
        reconciliation_status="mismatch",
        incidents=[
            {
                "incident_id": "dq-1",
                "category": "data_quality",
                "status": "open",
                "occurred_at": "2026-01-10T12:00:00Z",
            },
            {
                "incident_id": "borrow-1",
                "category": "borrow_violation",
                "status": "resolved",
                "occurred_at": "2026-01-11T12:00:00Z",
                "resolved_at": "2026-01-11T13:00:00Z",
            },
        ],
    )
    replay, diagnostics, monitoring = _write_supporting_reports(
        tmp_path,
        version,
        deterministic=False,
        monitoring_healthy=False,
    )

    report = evaluate_paper_graduation(
        registry_path=str(registry.root),
        paper_evidence_path=str(evidence),
        replay_report_path=str(replay),
        replay_diagnostics_path=str(diagnostics),
        monitoring_report_paths=[str(monitoring)],
        output_path=str(tmp_path / "failed.json"),
    )

    assert report["passed"] is False
    failures = set(report["failures"])
    assert "insufficient_paper_trading_days" in failures
    assert "daily_broker_reconciliation_missing" in failures
    assert "unexplained_broker_reconciliation_differences" in failures
    assert "unresolved_data_quality_incidents" in failures
    assert "borrow_violations" in failures
    assert "replay_determinism_not_verified" in failures
    assert "monitoring_unhealthy" in failures
    assert registry.champion().version == version


def test_live_money_evidence_is_rejected_by_schema(tmp_path: Path) -> None:
    _, version = _champion(tmp_path)
    evidence = tmp_path / "paper.json"
    _write_evidence(evidence, version)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["execution_mode"] = "live-money"

    with pytest.raises(ValidationError):
        PaperEvidence.model_validate(payload)


def test_bundled_graduation_manifest_matches_code_defaults() -> None:
    loaded = load_graduation_gates(GATE_CONFIG)

    assert loaded == GraduationGateConfig()
    assert loaded.manifest_sha256 == GraduationGateConfig().manifest_sha256


def test_graduation_rejects_changed_source_ledger_and_duplicate_evidence(
    tmp_path: Path,
) -> None:
    registry, version = _champion(tmp_path)
    evidence = tmp_path / "paper.json"
    _write_evidence(evidence, version)
    replay, diagnostics, monitoring = _write_supporting_reports(tmp_path, version)

    first = evaluate_paper_graduation(
        registry_path=str(registry.root),
        paper_evidence_path=str(evidence),
        replay_report_path=str(replay),
        replay_diagnostics_path=str(diagnostics),
        monitoring_report_paths=[str(monitoring)],
        output_path=str(tmp_path / "first.json"),
    )
    assert first["passed"] is True

    with pytest.raises(ValueError, match="already have a graduation record"):
        evaluate_paper_graduation(
            registry_path=str(registry.root),
            paper_evidence_path=str(evidence),
            replay_report_path=str(replay),
            replay_diagnostics_path=str(diagnostics),
            monitoring_report_paths=[str(monitoring)],
            output_path=str(tmp_path / "duplicate.json"),
        )
    assert not (tmp_path / "duplicate.json").exists()

    changed = tmp_path / "changed-paper.json"
    changed_payload = json.loads(evidence.read_text(encoding="utf-8"))
    changed_payload["limitations"].append("Distinct export after ledger mutation.")
    changed.write_text(json.dumps(changed_payload), encoding="utf-8")
    (tmp_path / "events.db").write_bytes(b"changed-after-export")
    failed = evaluate_paper_graduation(
        registry_path=str(registry.root),
        paper_evidence_path=str(changed),
        replay_report_path=str(replay),
        replay_diagnostics_path=str(diagnostics),
        monitoring_report_paths=[str(monitoring)],
        output_path=str(tmp_path / "changed.json"),
    )
    assert failed["passed"] is False
    assert "source_ledger_hash_mismatch" in failed["failures"]
