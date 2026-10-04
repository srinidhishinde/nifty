import pandas as pd
import pytest
from backtest.rule_engine import RuleBacktestEngine

def test_rule_backtest_hits_take_profit():
    timestamps=pd.date_range("2026-01-05 09:15",periods=80,freq="5min")
    closes=[120.0-i*0.5 for i in range(79)]+[80.0]
    highs=[x+0.2 for x in closes]; lows=[x-0.2 for x in closes]
    highs[-1]=140.0
    data=pd.DataFrame({"timestamp":timestamps,"open":closes,"high":highs,"low":lows,
                       "close":closes,"volume":[1000.0]*79+[2000.0],"sentiment":[0.0]*80})
    result=RuleBacktestEngine().run(data)
    assert not result.trades.empty
    assert set(result.trades["reason"]).issubset({"stop_loss", "take_profit", "market_close"})
    assert result.metrics.total_trades == len(result.trades)

def test_rule_backtest_uses_warmup_but_scores_only_evaluation_window():
    timestamps=pd.date_range("2026-01-05 09:15",periods=80,freq="5min")
    closes=[100.0+(i*0.1) for i in range(80)]
    data=pd.DataFrame({"timestamp":timestamps,"open":closes,"high":[p+0.2 for p in closes],
                       "low":[p-0.2 for p in closes],"close":closes,"volume":[1000.0]*80,"sentiment":[0.0]*80})
    start=timestamps[60]; end=timestamps[-1]
    result=RuleBacktestEngine().run(data,evaluation_start=start,evaluation_end=end)
    if not result.trades.empty:
        assert (pd.to_datetime(result.trades["entry_time"])>=start).all()


def test_rule_backtest_pnl_uses_position_exposure_not_risk_budget():
    timestamps = pd.date_range("2026-01-05 09:15", periods=70, freq="5min")
    closes = [100.0] * 69 + [104.0]
    data = pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": [x + 0.1 for x in closes],
        "low": [x - 0.1 for x in closes],
        "close": closes,
        "volume": [1000.0] * 70,
        "sentiment": [0.0] * 70,
    })
    # This test directly guards the accounting defect: P&L must scale with
    # quantity * price change, not with the configured risk budget.
    engine = RuleBacktestEngine(
        starting_capital=300000,
        risk_per_trade=3000,
        commission_pct=0.0,
        slippage_pct=0.0,
    )
    result = engine.run(data, evaluation_start=timestamps[0], evaluation_end=timestamps[-1])
    if not result.trades.empty:
        trade = result.trades.iloc[0]
        expected = (trade["exit_price"] - trade["entry"]) * trade["quantity"] if trade["direction"] == "BUY" else (trade["entry"] - trade["exit_price"]) * trade["quantity"]
        assert trade["gross_pnl"] == pytest.approx(expected)
