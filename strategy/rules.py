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
    # Risk levels are volatility/structure driven. Percentage values are only
    # a safety floor, never the primary stop mechanism.
    stop_loss_pct:float=0.005
    min_target_pct:float=0.02
    target_atr_multiple:float=2.5
    atr_stop_multiple:float=1.5
    min_confidence:float=62.0
    min_rules_for_signal:int=2
    min_adx:float=18.0
    max_atr_pct:float=0.04
    min_history_bars:int=50
    min_reward_risk:float=1.8
    min_volume_ratio:float=1.0
    min_signal_separation:float=0.05

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
    rsi=_v(row,("RSI",)); prev_rsi=_v(previous,("RSI",)); vol=_v(row,("volume",),0); vma=_v(row,("VOLUME_MA20",)); volume_ratio=_v(row,("VOLUME_RATIO",),1.0) or 0.0
    if rsi is not None and prev_rsi is not None and prev_rsi<=c.rsi_oversold and rsi>c.rsi_oversold and vma and volume_ratio>=c.min_volume_ratio:out.append(RuleSignal("rsi_oversold_buy","BUY","RSI oversold with volume confirmation",1.2))
    if rsi is not None and prev_rsi is not None and prev_rsi>=c.rsi_overbought and rsi<c.rsi_overbought and vma and volume_ratio>=c.min_volume_ratio:out.append(RuleSignal("rsi_overbought_sell","SELL","RSI overbought with volume confirmation",1.2))
    macd,sig=_v(row,("MACD",)),_v(row,("MACD_SIGNAL","Signal")); prev_macd,prev_sig=_v(previous,("MACD",)),_v(previous,("MACD_SIGNAL","Signal"))
    if None not in (macd,sig,prev_macd,prev_sig) and prev_macd<=prev_sig and macd>sig and macd>0:out.append(RuleSignal("macd_bullish","BUY","MACD bullish crossover above zero",1.2))
    if None not in (macd,sig,prev_macd,prev_sig) and prev_macd>=prev_sig and macd<sig and macd<0:out.append(RuleSignal("macd_bearish","SELL","MACD bearish crossover below zero",1.2))
    sentiment=_v(row,("sentiment","news_sentiment","Sentiment"),0); upper=_v(row,("BB_UPPER","UpperBand"))
    prev_close=_v(previous,("close",)); prev_upper=_v(previous,("BB_UPPER","UpperBand"))
    if upper is not None and prev_close is not None and prev_upper is not None and prev_close<=prev_upper and close>upper and sentiment>=0 and volume_ratio>=1.2:out.append(RuleSignal("bollinger_breakout","BUY","Upper-band breakout with volume/news confirmation",1.0))
    lower=_v(row,("BB_LOWER","LowerBand"))
    prev_lower=_v(previous,("BB_LOWER","LowerBand"))
    if lower is not None and prev_close is not None and prev_lower is not None and prev_close>=prev_lower and close<lower and sentiment<=0 and volume_ratio>=1.2:out.append(RuleSignal("bollinger_breakdown","SELL","Lower-band breakdown with volume/news confirmation",1.0))
    fast,slow=_v(row,("EMA9",)),_v(row,("EMA21",))
    pf,ps=_v(previous,("EMA9",)),_v(previous,("EMA21",))
    if None not in (fast,slow,pf,ps) and pf<=ps and fast>slow:out.append(RuleSignal("ema_cross","BUY","EMA9 crossed above EMA21",1.3))
    if None not in (fast,slow,pf,ps) and pf>=ps and fast<slow:out.append(RuleSignal("ema_cross","SELL","EMA9 crossed below EMA21",1.3))
    vwap=_v(row,("VWAP",))
    if vwap and vwap>0:
        dev=(close-vwap)/vwap*100
        prev_vwap=_v(previous,("VWAP",))
        prev_dev=((prev_close-prev_vwap)/prev_vwap*100) if prev_close is not None and prev_vwap else None
        adx_for_reversion=_v(row,("ADX",),99.0) or 99.0
        if prev_dev is not None and dev>prev_dev and prev_dev<=-c.vwap_deviation_pct and adx_for_reversion<c.min_adx:
            out.append(RuleSignal("vwap_reversion","BUY","VWAP deviation reverting upward in non-trending regime",1.0))
        elif prev_dev is not None and dev<prev_dev and prev_dev>=c.vwap_deviation_pct and adx_for_reversion<c.min_adx:
            out.append(RuleSignal("vwap_reversion","SELL","VWAP deviation reverting downward in non-trending regime",1.0))
    bid_size,ask_size=_v(row,("bid_size","best_bid_size","BidSize")), _v(row,("ask_size","best_ask_size","AskSize"))
    spread=_v(row,("spread","Spread")); prev_spread=_v(previous,("spread","Spread"))
    if bid_size is not None and ask_size is not None and ask_size>0:
        imbalance=bid_size/ask_size
        current=spread if spread is not None else _v(row,("ask","best_ask","Ask"),0)-_v(row,("bid","best_bid","Bid"),0)
        if imbalance<c.bid_ask_ratio_max and prev_spread is not None and current>prev_spread*c.spread_widening_factor:
            out.append(RuleSignal("order_book_imbalance","SELL","Bid-size/ask-size imbalance with widening spread",1.3))
    if sentiment<c.sentiment_negative_threshold:out.append(RuleSignal("sentiment_negative","SELL","News sentiment below -0.5",1.3))
    adx=_v(row,("ADX",))
    di_plus=_v(row,("DI_PLUS",)); di_minus=_v(row,("DI_MINUS",))
    if adx is not None and adx>=c.min_adx and di_plus is not None and di_minus is not None:
        if di_plus > di_minus:
            out.append(RuleSignal("adx_trend_confirmation","BUY",f"ADX {adx:.1f} with +DI > -DI",0.8))
        elif di_minus > di_plus:
            out.append(RuleSignal("adx_trend_confirmation","SELL",f"ADX {adx:.1f} with -DI > +DI",0.8))
    k,d=_v(row,("STOCH_K",)),_v(row,("STOCH_D",))
    pk,pd=_v(previous,("STOCH_K",)),_v(previous,("STOCH_D",))
    if None not in (k,d,pk,pd):
        if pk<=pd and k>d and k<20:out.append(RuleSignal("stochastic_reversal","BUY","Stochastic bullish crossover from oversold",0.9))
        elif pk>=pd and k<d and k>80:out.append(RuleSignal("stochastic_reversal","SELL","Stochastic bearish crossover from overbought",0.9))
    cs=_v(row,("candlestick_score",),0)
    if cs>=2:out.append(RuleSignal("candlestick_confirmation","BUY",f"Bullish candlestick score {cs:.0f}",1.0))
    elif cs<=-2:out.append(RuleSignal("candlestick_confirmation","SELL",f"Bearish candlestick score {cs:.0f}",1.0))
    return out

def generate_signal(data:pd.DataFrame,config:StrategyConfig|None=None)->StrategySignal:
    c=config or StrategyConfig()
    if len(data)<c.min_history_bars:
        return StrategySignal("WAIT",(),("Indicator warm-up: insufficient history",),0,0,0,0,False)
    e=add_indicators(data); row=e.iloc[-1]; prev=e.iloc[-2]
    required=("RSI","EMA9","EMA21","MACD","MACD_SIGNAL","ATR","ATR_PCT","VWAP")
    if any(pd.isna(row.get(name)) for name in required):
        return StrategySignal("WAIT",(),("Indicator warm-up: required features unavailable",),0,0,float(_v(row,("ATR",),0) or 0),0,False)
    rules=evaluate_rules(row,prev,c)
    buys=[r for r in rules if r.direction=="BUY"]; sells=[r for r in rules if r.direction=="SELL"]
    atr=float(_v(row,("ATR",),0) or 0); direction="BUY" if buys and not sells else "SELL" if sells and not buys else "WAIT"
    if direction=="WAIT":return StrategySignal(direction,tuple(r.rule for r in rules),tuple(r.reason for r in rules),0,0,atr,0,False)
    directional=sum(r.weight for r in rules if r.direction==direction); confidence=min(95.0,50+8*directional)
    atr_pct=float(_v(row,("ATR_PCT",),0) or 0)
    if atr_pct>c.max_atr_pct:return StrategySignal("WAIT",tuple(r.rule for r in rules),tuple(r.reason for r in rules),0,0,atr,round(confidence,2),False)
    # Avoid over-trading when opposing evidence is nearly tied.
    buy_weight=sum(r.weight for r in buys); sell_weight=sum(r.weight for r in sells)
    separation=abs(buy_weight-sell_weight)/max(buy_weight+sell_weight,1e-9)
    if separation<c.min_signal_separation:return StrategySignal("WAIT",tuple(r.rule for r in rules),tuple(r.reason for r in rules),0,0,atr,round(confidence,2),False)
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
