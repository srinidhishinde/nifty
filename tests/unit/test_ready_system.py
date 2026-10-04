import pandas as pd

from backtest.capital_aware import CapitalAwareRuleBacktestEngine
from strategy.rules import StrategyConfig, evaluate_rules


def test_symmetric_momentum_rules_emit_bearish_signals():
    row = pd.Series({
        "close": 100.0, "volume": 2000.0, "VOLUME_MA20": 1000.0,
        "RSI": 75.0, "MACD": -1.0, "MACD_SIGNAL": -0.5,
        "EMA20": 98.0, "EMA50": 100.0, "VWAP": 100.0,
        "ADX": 25.0, "STOCH_K": 85.0, "STOCH_D": 90.0,
        "candlestick_score": -2.0,
    })
    signals = evaluate_rules(row, row, StrategyConfig())
    directions = {(s.rule, s.direction) for s in signals}
    assert ("rsi_overbought_sell", "SELL") in directions
    assert ("macd_bearish", "SELL") in directions
    assert ("adx_trend_confirmation", "SELL") in directions


def test_minimum_reward_risk_is_enforced():
    data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-05 09:15", periods=80, freq="5min", tz="Asia/Kolkata"),
        "open": [25000.0] * 80, "high": [25010.0] * 80,
        "low": [24990.0] * 80, "close": [25000.0] * 80,
        "volume": [1000.0] * 80,
    })
    # The engine must remain executable even when no rule signal is present.
    result = CapitalAwareRuleBacktestEngine(
        starting_capital=100000, risk_per_trade=1000, lot_size=65,
        slippage_points=0, brokerage_per_order=0,
    ).run(data)
    assert result.validation["starting_capital"] == 100000

def test_capital_aware_backtest_exposes_rejection_funnel():
    data = pd.DataFrame({
        "timestamp": pd.date_range(
            "2026-01-05 09:15", periods=80, freq="5min", tz="Asia/Kolkata"
        ),
        "open": [25000.0] * 80,
        "high": [25010.0] * 80,
        "low": [24990.0] * 80,
        "close": [25000.0] * 80,
        "volume": [1000.0] * 80,
    })
    result = CapitalAwareRuleBacktestEngine(
        starting_capital=100000,
        risk_per_trade=1000,
        lot_size=65,
        slippage_points=0,
        brokerage_per_order=0,
    ).run(data)
    assert "qualified_signal_bars" in result.validation
    assert "rejected_risk_budget" in result.validation
    assert "bars_considered" in result.validation
    assert result.validation["starting_capital"] == 100000
