import pandas as pd

from marketdata.decision_source import load_kotak_decision_snapshot


class _Disconnected:
    def connection_status(self):
        return type("Status", (), {"connected": False})()


def test_live_decision_never_falls_back_to_yahoo():
    snapshot = load_kotak_decision_snapshot(_Disconnected())
    assert snapshot.source == "KOTAK_NEO"
    assert snapshot.status == "RED"
    assert snapshot.signal.direction == "WAIT"
    assert "Kotak Neo" in snapshot.message


class _Connected:
    client = object()

    def connection_status(self):
        return type("Status", (), {"connected": True})()


def test_live_decision_blocks_when_kotak_history_is_unconfigured(monkeypatch):
    class Provider:
        def __init__(self, client):
            pass

        def get_historical_candles(self, **kwargs):
            raise RuntimeError("NEO_NIFTY_NEOSYMBOL is not configured")

    monkeypatch.setattr("marketdata.decision_source.KotakNeoProvider", Provider)
    snapshot = load_kotak_decision_snapshot(_Connected())
    assert snapshot.source == "KOTAK_NEO"
    assert snapshot.signal.direction == "WAIT"
    assert "NEO_NIFTY_NEOSYMBOL" in snapshot.message


def test_option_data_is_required_for_ce_pe(monkeypatch):
    class Provider:
        def __init__(self, client):
            pass

        def get_historical_candles(self, **kwargs):
            ts = pd.date_range("2026-10-01 09:15", periods=80, freq="5min", tz="Asia/Kolkata")
            return [
                type("Candle", (), {
                    "timestamp": t, "open": 25000.0, "high": 25010.0,
                    "low": 24990.0, "close": 25000.0, "volume": 1000.0
                })()
                for t in ts
            ]

        def get_option_chain(self, **kwargs):
            return []

    monkeypatch.setattr("marketdata.decision_source.KotakNeoProvider", Provider)
    snapshot = load_kotak_decision_snapshot(_Connected())
    assert snapshot.signal.direction == "WAIT"
    assert "option-chain" in snapshot.signal.reasons[0]
