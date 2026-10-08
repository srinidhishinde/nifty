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


def test_nifty_uses_captured_kotak_candles_not_historical_api(monkeypatch):
    class Contract:
        option_type = "CE"
        open_interest = 100.0
        volume = 10.0

    class Provider:
        def __init__(self, client):
            pass

        def get_historical_candles(self, **kwargs):
            raise AssertionError("live decision must not call Kotak historical candles")

        def get_option_chain(self, **kwargs):
            return [Contract()]

    ts = pd.date_range("2026-10-01 09:15", periods=80, freq="5min", tz="Asia/Kolkata")
    frame = pd.DataFrame({
        "timestamp": ts,
        "open": 25000.0,
        "high": 25010.0,
        "low": 24990.0,
        "close": 25000.0,
        "volume": 1000.0,
    })
    monkeypatch.setattr("marketdata.decision_source.KotakNeoProvider", Provider)
    monkeypatch.setattr(
        "marketdata.daily_store.load_captured_candles",
        lambda *args, **kwargs: (frame, "KOTAK_CAPTURED"),
    )

    snapshot = load_kotak_decision_snapshot(_Connected(), instrument="NIFTY")
    assert snapshot.source == "KOTAK_NEO"
    assert snapshot.option_count == 1
    assert snapshot.signal.direction in {"BUY", "WAIT"}
    assert "KOTAK_CAPTURED" in snapshot.message


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


def test_mcx_keeps_real_option_chain_visible_when_candles_are_missing(monkeypatch):
    class Contract:
        option_type = "CE"
        open_interest = 0.0
        volume = 0.0

    class Provider:
        def __init__(self, client):
            pass

        def get_historical_candles(self, **kwargs):
            raise AssertionError("MCX live decision must not call historical candles")

        def get_option_chain(self, **kwargs):
            assert kwargs["exchange"] == "MCX_FO"
            assert kwargs["underlying"] == "CRUDEOIL"
            return [Contract()]

    monkeypatch.setattr("marketdata.decision_source.KotakNeoProvider", Provider)
    monkeypatch.setattr(
        "marketdata.daily_store.load_captured_candles",
        lambda *args, **kwargs: (pd.DataFrame(), "KOTAK_CAPTURED"),
    )

    snapshot = load_kotak_decision_snapshot(_Connected(), instrument="CRUDEOIL")
    assert snapshot.source == "KOTAK_NEO"
    assert snapshot.status == "RED"
    assert snapshot.option_count == 1
    assert len(snapshot.option_chain) == 1
    assert snapshot.signal.direction == "WAIT"
    assert "CRUDEOIL captured SFeed candle history unavailable" in snapshot.quality_reasons
