from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from features.technical.indicators import add_indicators

@dataclass(frozen=True)
class NiftyPrediction:
    prediction: str
    confidence: float
    reference_price: float | None
    target: float | None
    stop_loss: float | None
    global_news_score: float
    reason: str

def predict_315_340(data: pd.DataFrame, global_news_score: float = 0.0) -> NiftyPrediction:
    required = {"timestamp", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "high", "low", "close"]).sort_values("timestamp")
    times = frame["timestamp"].dt.time
    window = frame[(times >= pd.Timestamp("15:15").time()) & (times <= pd.Timestamp("15:40").time())]
    if window.empty:
        return NiftyPrediction("UNAVAILABLE", 0.0, None, None, None, global_news_score, "No candles in 15:15-15:40 window")
    enriched = add_indicators(frame) if "EMA20" not in frame.columns else frame
    sample = enriched.loc[window.index].iloc[-1]
    close = float(sample["close"])
    score, reasons = 50.0, []
    ema20, ema50 = sample.get("EMA20"), sample.get("EMA50")
    macd, sig = sample.get("MACD"), sample.get("MACD_SIGNAL")
    if pd.notna(ema20) and pd.notna(ema50):
        score += 18 if ema20 > ema50 else -18
        reasons.append("EMA20 above EMA50" if ema20 > ema50 else "EMA20 below EMA50")
    if pd.notna(macd) and pd.notna(sig):
        score += 12 if macd > sig else -12
        reasons.append("MACD bullish" if macd > sig else "MACD bearish")
    score += max(-10.0, min(10.0, global_news_score * 10.0))
    if global_news_score:
        reasons.append("Global news included")
    prediction = "UP" if score >= 55 else "DOWN" if score <= 45 else "FLAT"
    confidence = round(min(95.0, max(50.0, abs(score - 50.0) + 50.0)), 2)
    if prediction == "UP":
        target, stop = close * 1.015, close * 0.985
    elif prediction == "DOWN":
        target, stop = close * 0.985, close * 1.015
    else:
        target = stop = close
    return NiftyPrediction(prediction, confidence, close, round(target,2), round(stop,2), global_news_score, "; ".join(reasons))
