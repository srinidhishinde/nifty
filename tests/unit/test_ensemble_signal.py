import pandas as pd
from ensemble.signal import detect_regime, combine, rule_scores
from strategy.rules import StrategyConfig

def row(**kw):
    base={"ATR":100.0,"ATR5":100.0,"ATR_PCT":0.01,"EMA_SPREAD":0.0005,"ADX":15.0,"SENTIMENT_CHANGE":0.0}
    base.update(kw); return pd.Series(base)

def test_regime_weights_match_spec():
    assert detect_regime(row(ADX=30,EMA_SPREAD=0.002)).name=="TREND"
    assert detect_regime(row(ATR=100,ATR5=150)).name=="HIGH VOLATILITY"
    assert detect_regime(row(ADX=10,EMA_SPREAD=0.0001)).name=="RANGE"

def test_regime_weights():
    assert detect_regime(row(ADX=30,EMA_SPREAD=0.002)).rule_weight == .60
    assert detect_regime(row()).ml_weight == .30
    assert detect_regime(row(ATR=100,ATR5=150)).ml_weight == .50

def test_ensemble_normalizes():
    regime=detect_regime(row(ADX=30,EMA_SPREAD=.002))
    ce,pe=combine(70,30,60,40,regime)
    assert round(ce+pe,6)==100.0
    assert ce>pe

def test_rule_scores_have_no_signal_when_rules_are_absent():
    assert rule_scores(row(close=100),row(close=100),StrategyConfig(require_option_confirmation=False)) == (0.0,0.0)
