from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class EventSink(Protocol):
    def append(
        self,
        kind: str,
        payload: dict[str, object],
        event_time: datetime,
        knowledge_time: datetime,
    ) -> None: ...


class SQLiteEventSink:
    """Append-only durable ledger for a private single-node deployment."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS event_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                event_time TEXT NOT NULL,
                knowledge_time TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_ledger_kind_id "
            "ON event_ledger(kind, id DESC)"
        )
        self._connection.commit()

    def append(
        self,
        kind: str,
        payload: dict[str, object],
        event_time: datetime,
        knowledge_time: datetime,
    ) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO event_ledger(kind, event_time, knowledge_time, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    event_time.astimezone(UTC).isoformat(),
                    knowledge_time.astimezone(UTC).isoformat(),
                    encoded,
                    now,
                ),
            )
            self._connection.commit()

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, object]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "event_time": row["event_time"],
            "knowledge_time": row["knowledge_time"],
            "payload": json.loads(row["payload"]),
            "created_at": row["created_at"],
        }

    def recent(self, kind: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        if limit <= 0:
            return []
        with self._lock:
            if kind is None:
                rows = self._connection.execute(
                    "SELECT * FROM event_ledger ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT * FROM event_ledger WHERE kind = ? ORDER BY id DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
        return [self._decode(row) for row in rows]

    def scan(
        self,
        *,
        kind: str | None = None,
        after_id: int = 0,
        limit: int = 10_000,
    ) -> list[dict[str, object]]:
        """Read deterministic insertion order for replay and integrity audits."""
        if limit <= 0:
            return []
        with self._lock:
            if kind is None:
                rows = self._connection.execute(
                    "SELECT * FROM event_ledger WHERE id > ? ORDER BY id ASC LIMIT ?",
                    (after_id, limit),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    """
                    SELECT * FROM event_ledger
                    WHERE id > ? AND kind = ?
                    ORDER BY id ASC LIMIT ?
                    """,
                    (after_id, kind, limit),
                ).fetchall()
        return [self._decode(row) for row in rows]

    def count(self, kind: str | None = None) -> int:
        with self._lock:
            if kind is None:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM event_ledger"
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM event_ledger WHERE kind = ?", (kind,)
                ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        with self._lock:
            self._connection.close()
