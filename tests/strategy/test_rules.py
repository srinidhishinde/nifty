import pandas as pd

from strategy.rules import StrategyConfig, evaluate_rules, generate_signal


def test_rsi_rule_requires_volume_confirmation():
    row = pd.Series({
        "close": 100.0,
        "RSI": 25.0,
        "volume": 200.0,
        "VOLUME_MA20": 100.0,
    })
    signals = evaluate_rules(row)
    assert any(s.rule == "rsi_oversold_buy" and s.direction == "BUY" for s in signals)


def test_vwap_reversion_direction():
    below = pd.Series({"close": 97.0, "VWAP": 100.0})
    above = pd.Series({"close": 103.0, "VWAP": 100.0})
    assert any(s.direction == "BUY" for s in evaluate_rules(below))
    assert any(s.direction == "SELL" for s in evaluate_rules(above))


def test_ema_cross_requires_actual_cross():
    previous = pd.Series({"EMA20": 99.0, "EMA50": 100.0})
    current = pd.Series({"close": 101.0, "EMA20": 101.0, "EMA50": 100.0})
    signals = evaluate_rules(current, previous)
    assert any(s.rule == "ema_cross" for s in signals)


def test_risk_levels_match_pseudocode_defaults():
    closes = [120.0 - i for i in range(20)] + [101.0]
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01 09:15", periods=len(closes), freq="5min"),
        "open": closes,
        "high": [price + 0.5 for price in closes],
        "low": [price - 0.5 for price in closes],
        "close": closes,
        "volume": [1000.0] * 20 + [2000.0],
    })
    signal = generate_signal(frame, StrategyConfig())

    assert signal.direction == "BUY"
    assert signal.valid
    assert signal.stop_loss < 101.0
    assert signal.target > 101.0
    assert signal.target >= round(101.0 * 1.02, 2)
