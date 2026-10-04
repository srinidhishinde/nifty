from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from strategy.rules import StrategyConfig, evaluate_rules

@dataclass(frozen=True)
class Regime:
    name:str
    rule_weight:float
    ml_weight:float
    reason:str

@dataclass(frozen=True)
class EnsembleRow:
    horizon_minutes:int
    rule_ce:float
    rule_pe:float
    ml_ce:float
    ml_pe:float
    final_ce:float
    final_pe:float
    stronger_side:str
    confidence_low:float
    confidence_high:float

def detect_regime(row: pd.Series, previous: pd.Series|None=None) -> Regime:
    atr_pct=float(row.get("ATR_PCT",0) or 0); ema_spread=abs(float(row.get("EMA_SPREAD",0) or 0))
    atr5=float(row.get("ATR5",0) or 0); atr=float(row.get("ATR",0) or 0)
    atr_spike=(atr>0 and atr5>atr*1.35)
    sentiment_change=abs(float(row.get("SENTIMENT_CHANGE",0) or 0))
    if atr_spike or atr_pct>=0.025 or sentiment_change>=1.0:
        return Regime("HIGH VOLATILITY",0.50,0.50,"ATR spike/volatility or sentiment anomaly")
    if float(row.get("ADX",0) or 0)>=22 and ema_spread>=0.001:
        return Regime("TREND",0.60,0.40,"ADX and EMA separation indicate directional trend")
    return Regime("RANGE",0.70,0.30,"Low directional strength / flatter EMA structure")

def rule_scores(row:pd.Series, previous:pd.Series, config:StrategyConfig) -> tuple[float,float]:
    rules=evaluate_rules(row,previous,config)
    buy=sum(max(0,r.weight) for r in rules if r.direction=="BUY")
    sell=sum(max(0,r.weight) for r in rules if r.direction=="SELL")
    total=buy+sell
    if total<=0:return 0.0,0.0
    return round(100*buy/total,2),round(100*sell/total,2)

def combine(rule_ce,rule_pe,ml_ce,ml_pe,regime):
    ce=regime.rule_weight*rule_ce+regime.ml_weight*ml_ce
    pe=regime.rule_weight*rule_pe+regime.ml_weight*ml_pe
    total=ce+pe
    if total>0: ce,pe=100*ce/total,100*pe/total
    return round(ce,2),round(pe,2)

def build_ensemble(data:pd.DataFrame,predictions,config:StrategyConfig|None=None):
    if len(data)<2: raise ValueError("At least two completed candles are required")
    c=config or StrategyConfig(require_option_confirmation=False)
    frame=data.sort_values("timestamp").copy(); row,prev=frame.iloc[-1],frame.iloc[-2]
    regime=detect_regime(row,prev); rce,rpe=rule_scores(row,prev,c); result=[]
    for p in predictions:
        ce,pe=combine(rce,rpe,p.ce_probability*100,p.pe_probability*100,regime)
        result.append(EnsembleRow(p.horizon_minutes,rce,rpe,p.ce_probability*100,p.pe_probability*100,ce,pe,"CE" if ce>pe else "PE" if pe>ce else "WAIT",p.confidence_low,p.confidence_high))
    return regime,result
