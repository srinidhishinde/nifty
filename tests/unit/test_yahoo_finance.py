from datetime import date

import pandas as pd

from marketdata.yahoo_finance import fetch_yahoo_ohlcv


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_yahoo_adapter_normalises_timezone_and_schema(monkeypatch):
    payload = {
        "chart": {
            "result": [{
                "timestamp": [1791000000, 1791000300],
                "indicators": {
                    "quote": [{
                        "open": [25000, 25005],
                        "high": [25010, 25015],
                        "low": [24990, 25000],
                        "close": [25005, 25010],
                        "volume": [1000, 1200],
                    }]
                }
            }],
            "error": None,
        }
    }

    monkeypatch.setattr(
        "marketdata.yahoo_finance.requests.get",
        lambda *args, **kwargs: _Response(payload),
    )

    result = fetch_yahoo_ohlcv(
        date(2026, 10, 1),
        date(2026, 10, 2),
        interval="5m",
    )

    assert result.status == "OK"
    assert list(result.data.columns) == [
        "timestamp", "open", "high", "low", "close", "volume"
    ]
    assert str(result.data["timestamp"].dt.tz) == "Asia/Kolkata"


def test_yahoo_intraday_guard_prevents_unrealistic_long_request():
    result = fetch_yahoo_ohlcv(
        date(2026, 1, 1),
        date(2026, 10, 1),
        interval="5m",
    )

    assert result.status == "LIMITED"
    assert result.data.empty
