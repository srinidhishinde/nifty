from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from features.technical.indicators import add_indicators
from prediction.ensemble_model import predict_latest
from risk.risk_manager import size_position
from strategy.rules import StrategyConfig, generate_signal

@dataclass(frozen=True)
class AISignal:
    direction:str
    confidence:float
    technical_confidence:float
    ml_confidence:float
    abstain:bool
    entry:float|None
    stop_loss:float|None
    take_profit:float|None
    quantity:int
    reasons:tuple[str,...]

def generate_ai_signal(data:pd.DataFrame,capital:float,global_news_score:float=0.0,config:StrategyConfig|None=None)->AISignal:
    enriched=add_indicators(data.copy())
    technical=generate_signal(enriched,config)
    ml=predict_latest(enriched,global_news_score=global_news_score)
    if not technical.valid or technical.direction=="WAIT" or ml.direction=="UNAVAILABLE":
        return AISignal("WAIT",0.0,technical.confidence,ml.confidence,True,None,None,None,0,("Technical signal not trade-ready or ML model abstained",))
    ml_direction="BUY" if ml.direction=="UP" else "SELL"
    if ml_direction!=technical.direction or ml.abstain:
        return AISignal("WAIT",0.0,technical.confidence,ml.confidence,True,None,None,None,0,("Technical/ML consensus not strong enough",))
    row=enriched.iloc[-1]; atr=float(row.get("ATR",0.0) or 0.0)
    risk=size_position(float(row["close"]),atr,technical.direction,capital)
    if not risk.allowed:
        return AISignal("WAIT",0.0,technical.confidence,ml.confidence,True,None,None,None,0,(risk.reason,))
    confidence=round((technical.confidence+ml.confidence)/2,2)
    return AISignal(technical.direction,confidence,technical.confidence,ml.confidence,False,risk.entry,risk.stop_loss,risk.take_profit,risk.quantity,tuple(technical.reasons)+(ml.reason,))
