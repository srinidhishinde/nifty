import pandas as pd
from strategy.rules import StrategyConfig, evaluate_rules, generate_signal

def test_rsi_rule_requires_volume_confirmation():
    previous=pd.Series({"close":99.0,"RSI":25.0})
    row=pd.Series({"close":100.0,"RSI":35.0,"volume":200.0,"VOLUME_MA20":100.0,"VOLUME_RATIO":2.0})
    assert any(s.rule=="rsi_oversold_buy" and s.direction=="BUY" for s in evaluate_rules(row, previous))

def test_vwap_reversion_direction():
    previous_below=pd.Series({"close":97.0,"VWAP":100.0,"ADX":10.0})
    below=pd.Series({"close":98.0,"VWAP":100.0,"ADX":10.0})
    previous_above=pd.Series({"close":103.0,"VWAP":100.0,"ADX":10.0})
    above=pd.Series({"close":102.0,"VWAP":100.0,"ADX":10.0})
    assert any(s.direction=="BUY" for s in evaluate_rules(below, previous_below))
    assert any(s.direction=="SELL" for s in evaluate_rules(above, previous_above))

def test_ema_cross_requires_actual_cross():
    previous=pd.Series({"EMA9":99.0,"EMA21":100.0})
    current=pd.Series({"close":101.0,"EMA9":101.0,"EMA21":100.0})
    assert any(s.rule=="ema_cross" for s in evaluate_rules(current,previous))

def test_technical_signal_waits_for_warmup():
    frame=pd.DataFrame({"timestamp":pd.date_range("2026-10-01 09:15",periods=20,freq="5min"),"open":[100.0]*20,"high":[101.0]*20,"low":[99.0]*20,"close":[100.0]*20,"volume":[1000.0]*20})
    signal=generate_signal(frame,StrategyConfig())
    assert signal.direction=="WAIT" and not signal.valid

def test_risk_levels_match_pseudocode_defaults():
    closes=[60.0+i*0.5 for i in range(60)]+[101.0]
    frame=pd.DataFrame({"timestamp":pd.date_range("2026-10-01 09:15",periods=len(closes),freq="5min"),"open":closes,"high":[p+0.5 for p in closes],"low":[p-0.5 for p in closes],"close":closes,"volume":[1000.0]*60+[5000.0]})
    signal=generate_signal(frame,StrategyConfig(min_rules_for_signal=1, min_evidence_groups=1, vwap_deviation_pct=100.0, min_adx=0.0, min_confidence=0.0))
    assert signal.direction=="BUY" and signal.valid
    assert signal.stop_loss<101.0<signal.target and signal.target>=round(101.0*1.02,2)


def test_precision_mode_rejects_weak_signal():
    previous=pd.Series({"EMA9":100.0,"EMA21":99.0,"ADX":25.0})
    row=pd.Series({"close":101.0,"EMA9":101.0,"EMA21":99.5,"ADX":25.0,"ATR":1.0,"ATR_PCT":0.01,
                   "RSI":55.0,"MACD":1.0,"MACD_SIGNAL":0.5,"VWAP":100.0})
    signal=generate_signal(pd.DataFrame([previous, row]), StrategyConfig(min_history_bars=2, require_option_confirmation=False))
    assert signal.direction=="WAIT" and not signal.valid


def test_precision_mode_can_be_disabled_for_legacy_research():
    closes=[60.0+i*0.5 for i in range(60)]+[101.0]
    frame=pd.DataFrame({"timestamp":pd.date_range("2026-10-01 09:15",periods=len(closes),freq="5min"),
                        "open":closes,"high":[p+0.5 for p in closes],"low":[p-0.5 for p in closes],
                        "close":closes,"volume":[1000.0]*60+[5000.0]})
    signal=generate_signal(frame, StrategyConfig(min_rules_for_signal=1, min_evidence_groups=1,
                                                  min_confidence=0.0, precision_mode=False,
                                                  vwap_deviation_pct=100.0, min_adx=0.0))
    assert signal.direction=="BUY" and signal.valid
