from datetime import UTC, datetime

from trading_app.persistence import SQLiteEventSink


def test_sqlite_event_sink_round_trip(tmp_path) -> None:
    sink = SQLiteEventSink(tmp_path / "events.db")
    now = datetime.now(UTC)
    sink.append("proposal", {"symbol": "AAPL", "confidence": 0.8}, now, now)
    assert sink.count() == 1
    assert sink.count("proposal") == 1
    row = sink.recent("proposal")[0]
    assert row["payload"]["symbol"] == "AAPL"
    sink.close()
