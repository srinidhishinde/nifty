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


def test_rule_backtest_uses_warmup_but_scores_only_evaluation_window():
    timestamps = pd.date_range("2026-01-05 09:15", periods=80, freq="5min")
    closes = [100.0 + (i * 0.1) for i in range(80)]
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": [price + 0.2 for price in closes],
        "low": [price - 0.2 for price in closes],
        "close": closes,
        "volume": [1000.0] * 80,
        "sentiment": [0.0] * 80,
    })
    start = timestamps[60]
    end = timestamps[-1]
    result = RuleBacktestEngine().run(
        data,
        evaluation_start=start,
        evaluation_end=end,
    )
    if not result.trades.empty:
        assert (pd.to_datetime(result.trades["entry_time"]) >= start).all()
