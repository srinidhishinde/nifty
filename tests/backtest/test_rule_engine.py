import pandas as pd

from backtest.rule_engine import RuleBacktestEngine


def test_rule_backtest_hits_take_profit():
    timestamps = pd.date_range("2026-01-05 09:15", periods=4, freq="5min")
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": [100, 100, 100, 140],
        "high": [100, 100, 140, 140],
        "low": [100, 99, 100, 139],
        "close": [100, 100, 140, 140],
        "volume": [1000, 1000, 2000, 2000],
        "sentiment": [0, 0, 0, 0],
    })
    data.loc[0, "RSI"] = 25
    data.loc[0, "VOLUME_MA20"] = 500
    result = RuleBacktestEngine().run(data)
    assert not result.trades.empty
    assert "take_profit" in set(result.trades["reason"])
