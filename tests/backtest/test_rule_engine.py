import pandas as pd

from backtest.rule_engine import RuleBacktestEngine


def test_rule_backtest_hits_take_profit():
    timestamps = pd.date_range("2026-01-05 09:15", periods=21, freq="5min")
    closes = [100 - i for i in range(20)] + [81]
    highs = closes.copy()
    lows = closes.copy()
    volumes = [1000.0] * 20 + [2000.0]
    highs[-1] = 140.0
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "sentiment": [0.0] * 21,
    })
    result = RuleBacktestEngine().run(data)
    assert not result.trades.empty
    assert "take_profit" in set(result.trades["reason"])
