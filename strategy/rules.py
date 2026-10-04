from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from features.technical.indicators import add_indicators

RULE_NAMES=("rsi_oversold_buy","macd_bullish","bollinger_breakout","ema_cross","vwap_reversion","order_book_imbalance","sentiment_negative","adx_trend_confirmation","stochastic_reversal","candlestick_confirmation")

@dataclass(frozen=True)
class RuleSignal:
    rule:str
    direction:str
    reason:str
    weight:float=1.0

@dataclass(frozen=True)
class StrategySignal:
    direction:str
    rules:tuple[str,...]
    reasons:tuple[str,...]
    stop_loss:float
    target:float
    atr:float
    confidence:float
    valid:bool

@dataclass(frozen=True)
class StrategyConfig:
    rsi_oversold:float=30.0
    rsi_overbought:float=70.0
    vwap_deviation_pct:float=2.0
    bid_ask_ratio_max:float=0.5
    spread_widening_factor:float=1.25
    sentiment_negative_threshold:float=-0.5
    stop_loss_pct:float=0.02
    min_target_pct:float=0.02
    target_atr_multiple:float=2.5
    atr_stop_multiple:float=1.5
    min_confidence:float=62.0
    min_rules_for_signal:int=2
    min_adx:float=18.0
    max_atr_pct:float=0.04
    min_history_bars:int=50
    min_reward_risk:float=1.5

def _v(row,names,default=None):
    if row is None:return default
    for n in names:
        if n in row.index and pd.notna(row[n]):
            try:return float(row[n])
            except (TypeError,ValueError):pass
    return default

def evaluate_rules(row:pd.Series,previous:pd.Series|None=None,config:StrategyConfig|None=None)->list[RuleSignal]:
    c=config or StrategyConfig(); close=_v(row,("close",))
    if close is None or close<=0:return []
    out=[]
    rsi=_v(row,("RSI",)); vol=_v(row,("volume",),0); vma=_v(row,("VOLUME_MA20",))
    if rsi is not None and rsi<c.rsi_oversold and vma and vol>vma:out.append(RuleSignal("rsi_oversold_buy","BUY","RSI oversold with volume confirmation",1.2))
    if rsi is not None and rsi>c.rsi_overbought and vma and vol>vma:out.append(RuleSignal("rsi_overbought_sell","SELL","RSI overbought with volume confirmation",1.2))
    macd,sig=_v(row,("MACD",)),_v(row,("MACD_SIGNAL","Signal"))
    if macd is not None and sig is not None and macd>sig and macd>0:out.append(RuleSignal("macd_bullish","BUY","MACD above signal and zero",1.2))
    if macd is not None and sig is not None and macd<sig and macd<0:out.append(RuleSignal("macd_bearish","SELL","MACD below signal and zero",1.2))
    sentiment=_v(row,("sentiment","news_sentiment","Sentiment"),0); upper=_v(row,("BB_UPPER","UpperBand"))
    if upper is not None and close>upper and sentiment>0:out.append(RuleSignal("bollinger_breakout","BUY","Upper-band breakout with positive news",1.0))
    lower=_v(row,("BB_LOWER","LowerBand"))
    if lower is not None and close<lower and sentiment<0:out.append(RuleSignal("bollinger_breakdown","SELL","Lower-band breakdown with negative news",1.0))
    fast,slow=_v(row,("EMA20",)),_v(row,("EMA50",))
    pf,ps=_v(previous,("EMA20",)),_v(previous,("EMA50",))
    if None not in (fast,slow,pf,ps) and pf<=ps and fast>slow:out.append(RuleSignal("ema_cross","BUY","EMA20 crossed above EMA50",1.3))
    if None not in (fast,slow,pf,ps) and pf>=ps and fast<slow:out.append(RuleSignal("ema_cross","SELL","EMA20 crossed below EMA50",1.3))
    vwap=_v(row,("VWAP",))
    if vwap and vwap>0:
        dev=(close-vwap)/vwap*100
        if dev<=-c.vwap_deviation_pct:out.append(RuleSignal("vwap_reversion","BUY","Price >2% below VWAP",1.0))
        elif dev>=c.vwap_deviation_pct:out.append(RuleSignal("vwap_reversion","SELL","Price >2% above VWAP",1.0))
    bid,ask=_v(row,("bid","best_bid","Bid")),_v(row,("ask","best_ask","Ask")); spread=_v(row,("spread","Spread")); prev_spread=_v(previous,("spread","Spread"))
    if bid is not None and ask and ask>0:
        current=spread if spread is not None else ask-bid
        if bid/ask<c.bid_ask_ratio_max and prev_spread and current>prev_spread*c.spread_widening_factor:out.append(RuleSignal("order_book_imbalance","SELL","Bid/ask imbalance with widening spread",1.3))
    if sentiment<c.sentiment_negative_threshold:out.append(RuleSignal("sentiment_negative","SELL","News sentiment below -0.5",1.3))
    adx=_v(row,("ADX",))
    if adx is not None and adx>=c.min_adx:
        fast_now=_v(row,("EMA20",close)); slow_now=_v(row,("EMA50",close))
        trend="BUY" if fast_now>=slow_now else "SELL"
        out.append(RuleSignal("adx_trend_confirmation",trend,f"ADX {adx:.1f} confirms trend",0.8))
    k,d=_v(row,("STOCH_K",)),_v(row,("STOCH_D",))
    if k is not None and d is not None:
        if k<20 and k>d:out.append(RuleSignal("stochastic_reversal","BUY","Stochastic bullish reversal",0.9))
        elif k>80 and k<d:out.append(RuleSignal("stochastic_reversal","SELL","Stochastic bearish reversal",0.9))
    cs=_v(row,("candlestick_score",),0)
    if cs>=2:out.append(RuleSignal("candlestick_confirmation","BUY",f"Bullish candlestick score {cs:.0f}",1.0))
    elif cs<=-2:out.append(RuleSignal("candlestick_confirmation","SELL",f"Bearish candlestick score {cs:.0f}",1.0))
    return out

def generate_signal(data:pd.DataFrame,config:StrategyConfig|None=None)->StrategySignal:
    c=config or StrategyConfig()
    if len(data)<c.min_history_bars:
        return StrategySignal("WAIT",(),("Indicator warm-up: insufficient history",),0,0,0,0,False)
    e=add_indicators(data); row=e.iloc[-1]; prev=e.iloc[-2]
    required=("RSI","EMA20","EMA50","MACD","MACD_SIGNAL","ATR","ATR_PCT","VWAP")
    if any(pd.isna(row.get(name)) for name in required):
        return StrategySignal("WAIT",(),("Indicator warm-up: required features unavailable",),0,0,float(_v(row,("ATR",),0) or 0),0,False)
    rules=evaluate_rules(row,prev,c)
    buys=[r for r in rules if r.direction=="BUY"]; sells=[r for r in rules if r.direction=="SELL"]
    atr=float(_v(row,("ATR",),0) or 0); direction="BUY" if buys and not sells else "SELL" if sells and not buys else "WAIT"
    if direction=="WAIT":return StrategySignal(direction,tuple(r.rule for r in rules),tuple(r.reason for r in rules),0,0,atr,0,False)
    directional=sum(r.weight for r in rules if r.direction==direction); confidence=min(95.0,50+8*directional)
    atr_pct=float(_v(row,("ATR_PCT",),0) or 0)
    if atr_pct>c.max_atr_pct:return StrategySignal("WAIT",tuple(r.rule for r in rules),tuple(r.reason for r in rules),0,0,atr,round(confidence,2),False)
    close=float(row["close"]); risk=max(close*c.stop_loss_pct,atr*c.atr_stop_multiple if atr>0 else 0); reward=max(close*c.min_target_pct,atr*c.target_atr_multiple if atr>0 else 0, risk*c.min_reward_risk)
    stop=close-risk if direction=="BUY" else close+risk; target=close+reward if direction=="BUY" else close-reward
    valid=len(rules)>=c.min_rules_for_signal and confidence>=c.min_confidence
    return StrategySignal(direction,tuple(r.rule for r in rules),tuple(r.reason for r in rules),round(stop,2),round(target,2),atr,round(confidence,2),valid)

def rank_rule_performance(trades:pd.DataFrame)->pd.DataFrame:
    cols=["rule","trades","wins","losses","win_rate_pct","net_pnl","roi_pct"]
    if trades.empty:return pd.DataFrame(columns=cols)
    rows=[]
    for _,t in trades.iterrows():
        names=t.get("rules") or str(t.get("rule","")).split("|"); names=[names] if isinstance(names,str) else names
        for rule in names:
            if rule:rows.append({"rule":rule,"pnl":float(t.get("pnl",0) or 0),"roi_pct":float(t.get("roi_pct",0) or 0)})
    if not rows:return pd.DataFrame(columns=cols)
    e=pd.DataFrame(rows); out=[]
    for rule,g in e.groupby("rule"):
        pnl=pd.to_numeric(g.pnl,errors="coerce").fillna(0); wins=int((pnl>0).sum())
        out.append({"rule":rule,"trades":len(g),"wins":wins,"losses":len(g)-wins,"win_rate_pct":round(wins/len(g)*100,2),"net_pnl":round(float(pnl.sum()),2),"roi_pct":round(float(g.roi_pct.sum()),2)})
    return pd.DataFrame(out).sort_values(["net_pnl","win_rate_pct"],ascending=False).reset_index(drop=True)
