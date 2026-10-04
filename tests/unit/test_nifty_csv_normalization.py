import pandas as pd

from marketdata.nifty_csv import normalize_nifty_csv


def test_normalize_datetime_ohlcv_headers():
    frame = pd.DataFrame({
        "Datetime": ["2026-09-21 09:15:00"],
        "Open": [25000], "High": [25010], "Low": [24990],
        "Close": [25005], "Volume": [1000],
    })
    out = normalize_nifty_csv(frame)
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert out.iloc[0]["close"] == 25005


def test_normalize_daily_date_with_adj_close():
    frame = pd.DataFrame({
        "Date": ["2026-01-01"], "Open": [25000], "High": [25100],
        "Low": [24900], "Close": [25050], "Adj Close": [25050], "Volume": [500000],
    })
    out = normalize_nifty_csv(frame)
    assert len(out) == 1
    assert set(["timestamp", "open", "high", "low", "close", "volume"]).issubset(out.columns)


def test_normalize_rejects_missing_volume():
    frame = pd.DataFrame({
        "Datetime": ["2026-09-21 09:15:00"],
        "Open": [25000], "High": [25010], "Low": [24990], "Close": [25005],
    })
    try:
        normalize_nifty_csv(frame)
    except ValueError as exc:
        assert "volume" in str(exc)
    else:
        raise AssertionError("Expected missing volume to be rejected")
