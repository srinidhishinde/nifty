import pandas as pd

from strategy.rules import StrategyConfig, evaluate_rules


def test_option_flow_bullish_confirmation_uses_pcr_pce_and_optional_greeks():
    row = pd.Series({"close": 100, "PCR_OI": 1.20, "PCE": 1.15, "NET_DELTA": 10, "GAMMA_EXPOSURE": 2})
    rules = evaluate_rules(row, config=StrategyConfig())
    assert any(r.rule == "option_flow_confirmation" and r.direction == "BUY" for r in rules)


def test_option_flow_does_not_trade_without_complete_pcr_pce():
    row = pd.Series({"close": 100, "PCR_OI": 1.20})
    rules = evaluate_rules(row, config=StrategyConfig())
    assert not any(r.rule == "option_flow_confirmation" for r in rules)
