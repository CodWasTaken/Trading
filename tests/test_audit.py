from datetime import UTC, datetime

from trading_app.audit import audit_ledger
from trading_app.persistence import SQLiteEventSink


def test_ledger_audit_replays_relationships(tmp_path) -> None:
    path = tmp_path / "events.db"
    sink = SQLiteEventSink(path)
    now = datetime.now(UTC)
    sink.append("proposal", {"id": "proposal-1"}, now, now)
    sink.append(
        "risk_decision",
        {"proposal_id": "proposal-1", "status": "approved"},
        now,
        now,
    )
    sink.append(
        "order",
        {"id": "order-1", "proposal_id": "proposal-1"},
        now,
        now,
    )
    sink.append("fill", {"order_id": "order-1"}, now, now)
    sink.close()

    report = audit_ledger(path)
    assert report.valid
    assert report.events == 4
    assert report.counts["fill"] == 1


def test_ledger_audit_detects_orphan_fill(tmp_path) -> None:
    path = tmp_path / "events.db"
    sink = SQLiteEventSink(path)
    now = datetime.now(UTC)
    sink.append("fill", {"order_id": "missing"}, now, now)
    sink.close()

    report = audit_ledger(path)
    assert not report.valid
    assert "unknown order" in report.errors[0]
