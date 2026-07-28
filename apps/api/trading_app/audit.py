from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .persistence import SQLiteEventSink


@dataclass(frozen=True)
class AuditReport:
    events: int
    counts: dict[str, int]
    valid: bool
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_ledger(path: str | Path) -> AuditReport:
    sink = SQLiteEventSink(path)
    try:
        events = sink.scan(limit=max(10_000, sink.count() + 1))
    finally:
        sink.close()

    counts = Counter(str(event["kind"]) for event in events)
    proposals: set[str] = set()
    decisions: dict[str, str] = {}
    orders: dict[str, str] = {}
    fills: set[str] = set()
    errors: list[str] = []

    for event in events:
        kind = str(event["kind"])
        payload = dict(event["payload"])
        event_id = event["id"]
        if kind == "proposal":
            proposal_id = str(payload.get("id", ""))
            if not proposal_id:
                errors.append(f"event {event_id}: proposal missing id")
            elif proposal_id in proposals:
                errors.append(f"event {event_id}: duplicate proposal {proposal_id}")
            proposals.add(proposal_id)
        elif kind == "risk_decision":
            proposal_id = str(payload.get("proposal_id", ""))
            if proposal_id not in proposals:
                errors.append(
                    f"event {event_id}: decision references unknown proposal {proposal_id}"
                )
            decisions[proposal_id] = str(payload.get("status", ""))
        elif kind == "order":
            order_id = str(payload.get("id", ""))
            proposal_id = str(payload.get("proposal_id", ""))
            if proposal_id not in proposals:
                errors.append(
                    f"event {event_id}: order references unknown proposal {proposal_id}"
                )
            if decisions.get(proposal_id) == "rejected":
                errors.append(
                    f"event {event_id}: order exists for rejected proposal {proposal_id}"
                )
            if proposal_id not in decisions:
                errors.append(
                    f"event {event_id}: order precedes risk decision for {proposal_id}"
                )
            if order_id in orders:
                errors.append(f"event {event_id}: duplicate order {order_id}")
            orders[order_id] = proposal_id
        elif kind == "fill":
            order_id = str(payload.get("order_id", ""))
            if order_id not in orders:
                errors.append(f"event {event_id}: fill references unknown order {order_id}")
            fills.add(order_id)

    for proposal_id, status in decisions.items():
        if status in {"approved", "resized"} and proposal_id not in orders.values():
            errors.append(f"approved proposal {proposal_id} has no order")
    return AuditReport(
        events=len(events),
        counts=dict(counts),
        valid=not errors,
        errors=errors,
    )
