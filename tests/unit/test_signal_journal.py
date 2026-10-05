import json

from analytics.signal_journal import SignalJournal


def test_signal_journal_appends_jsonl(tmp_path):
    path = tmp_path / "signals.jsonl"
    journal = SignalJournal(path)
    journal.append_decision(
        instrument="NIFTY",
        timeframe="5m",
        source="KOTAK_NEO",
        status="WAIT",
        direction="WAIT",
        reason="test",
        confidence=52,
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["instrument"] == "NIFTY"
    assert rows[0]["direction"] == "WAIT"
    assert rows[0]["confidence"] == 52
    assert "recorded_at" in rows[0]
