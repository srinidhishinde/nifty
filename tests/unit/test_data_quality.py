import pandas as pd

from analysis.data_quality import assess_ohlcv


def test_quality_gate_flags_bad_ohlc():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 09:15", periods=2, freq="5min", tz="Asia/Kolkata"),
        "open": [100, 100],
        "high": [99, 101],
        "low": [98, 99],
        "close": [99, 100],
        "volume": [1000, 1000],
    })
    report = assess_ohlcv(frame)
    assert report.status == "RED"
    assert report.invalid_ohlc == 1


def test_quality_gate_accepts_clean_cadence():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01 09:15", periods=4, freq="5min", tz="Asia/Kolkata"),
        "open": [100, 101, 102, 103],
        "high": [101, 102, 103, 104],
        "low": [99, 100, 101, 102],
        "close": [101, 102, 103, 104],
        "volume": [1000] * 4,
    })
    report = assess_ohlcv(frame)
    assert report.status == "GREEN"
    assert report.gaps_over_expected == 0


def test_overnight_gap_is_not_an_intraday_quality_failure():
    ts=pd.to_datetime(["2026-01-05 15:35","2026-01-06 09:15"]).tz_localize("Asia/Kolkata")
    df=pd.DataFrame({"timestamp":ts,"open":[100,101],"high":[101,102],"low":[99,100],"close":[100,101],"volume":[1000,1000]})
    report=assess_ohlcv(df,expected_minutes=5)
    assert report.gaps_over_expected == 0
    assert report.status == "GREEN"


def test_real_intraday_gap_is_reported():
    ts=pd.to_datetime(["2026-01-05 10:00","2026-01-05 10:30"]).tz_localize("Asia/Kolkata")
    df=pd.DataFrame({"timestamp":ts,"open":[100,101],"high":[101,102],"low":[99,100],"close":[100,101],"volume":[1000,1000]})
    report=assess_ohlcv(df,expected_minutes=5)
    assert report.gaps_over_expected == 1
    assert report.status == "ORANGE"
