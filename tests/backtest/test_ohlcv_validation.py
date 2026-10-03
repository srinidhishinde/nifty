import pandas as pd
import pytest

from backtest.rule_engine import RuleBacktestEngine


def _candles():
    ts = pd.date_range("2026-01-05 09:15", periods=60, freq="5min")
    close = [100.0 + i * 0.1 for i in range(60)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": [x + 0.2 for x in close],
        "low": [x - 0.2 for x in close],
        "close": close,
        "volume": [1000.0] * 60,
    })


def test_backtest_requires_volume():
    data = _candles().drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        RuleBacktestEngine().run(data)


def test_backtest_rejects_invalid_ohlc():
    data = _candles()
    data.loc[10, "high"] = data.loc[10, "close"] - 1
    with pytest.raises(ValueError, match="Invalid OHLCV rows"):
        RuleBacktestEngine().run(data)
