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
    frame = pd.DataFrame({
        "close": [100.0, 101.0],
        "MACD": [0.0, 0.0],
        "MACD_SIGNAL": [0.0, 0.0],
        "EMA20": [100.0, 100.0],
        "EMA50": [100.0, 100.0],
        "VOLUME_MA20": [900.0, 900.0],
        "volume": [1000.0, 2000.0],
        "RSI": [25.0, 20.0],
        "VWAP": [100.0, 101.0],
        "ATR": [1.0, 1.0],
    })
    signal = generate_signal(frame, StrategyConfig())
    assert signal.direction == "BUY"
    assert signal.stop_loss == 99.48
    assert signal.target == 141.4
