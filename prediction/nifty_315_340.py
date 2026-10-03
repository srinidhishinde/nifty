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


def _ensure_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "open" not in out.columns:
        out["open"] = pd.to_numeric(out["close"], errors="coerce")
    return out


def predict_315_340(data: pd.DataFrame, global_news_score: float = 0.0) -> NiftyPrediction:
    required = {"timestamp", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = _ensure_ohlc(data)
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
    return NiftyPrediction(prediction, confidence, close, round(target, 2), round(stop, 2), global_news_score, "; ".join(reasons))


def evaluate_next_day_accuracy(data: pd.DataFrame, global_news_score: float = 0.0) -> tuple[float, pd.DataFrame]:
    required = {"timestamp", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = _ensure_ohlc(data)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
    enriched = add_indicators(frame) if "EMA20" not in frame.columns else frame
    rows = []
    for i in range(len(enriched) - 1):
        row = enriched.iloc[i]
        close = float(row["close"])
        score = 50.0
        if pd.notna(row.get("EMA20")) and pd.notna(row.get("EMA50")):
            score += 18 if row["EMA20"] > row["EMA50"] else -18
        if pd.notna(row.get("MACD")) and pd.notna(row.get("MACD_SIGNAL")):
            score += 12 if row["MACD"] > row["MACD_SIGNAL"] else -12
        # During indicator warm-up, use only information available at the
        # current bar rather than forcing a misleading FLAT prediction.
        if score == 50.0 and i > 0:
            prior_close = float(enriched.iloc[i - 1]["close"])
            if close > prior_close:
                score += 12
            elif close < prior_close:
                score -= 12
        score += max(-10.0, min(10.0, global_news_score * 10.0))
        prediction = "UP" if score >= 55 else "DOWN" if score <= 45 else "FLAT"
        next_close = float(enriched.iloc[i + 1]["close"])
        actual = "UP" if next_close > close else "DOWN" if next_close < close else "FLAT"
        rows.append({
            "Date": row["timestamp"], "Prediction": prediction,
            "Actual": actual, "Correct": prediction == actual,
            "Confidence": round(min(95.0, max(50.0, abs(score - 50.0) + 50.0)), 2),
        })
    result = pd.DataFrame(rows)
    accuracy = float(result["Correct"].mean() * 100) if not result.empty else 0.0
    return round(accuracy, 2), result
