from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from features.technical.indicators import add_indicators


@dataclass(frozen=True)
class TrendSnapshot:
    instrument: str
    direction: str
    score: float
    strength: float
    regime: str
    momentum: str
    volatility: str
    observed_at: pd.Timestamp | None
    data_bars: int
    is_live: bool
    reason: str


def _sign_score(value: float, positive: float, negative: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return positive if value > 0 else negative if value < 0 else 0.0


def calculate_trend(
    instrument: str,
    candles: pd.DataFrame,
    *,
    is_live: bool = False,
) -> TrendSnapshot:
    required = {"open", "high", "low", "close"}
    missing = required - set(candles.columns)
    if missing:
        raise ValueError(f"Missing trend fields: {sorted(missing)}")

    frame = candles.copy()
    frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame = frame.sort_values("timestamp", kind="stable") if "timestamp" in frame else frame
    frame = add_indicators(frame)

    bars = len(frame)
    if bars < 50:
        return TrendSnapshot(
            instrument=instrument.upper(),
            direction="DATA_INSUFFICIENT",
            score=0.0,
            strength=0.0,
            regime="UNKNOWN",
            momentum="UNKNOWN",
            volatility="UNKNOWN",
            observed_at=frame["timestamp"].iloc[-1] if "timestamp" in frame and not frame.empty else None,
            data_bars=bars,
            is_live=is_live,
            reason="At least 50 completed candles are required for trend scoring.",
        )

    row = frame.iloc[-1]

    ema20 = float(row.get("EMA20", np.nan))
    ema50 = float(row.get("EMA50", np.nan))
    ema200 = float(row.get("EMA200", np.nan))
    close = float(row.get("close", np.nan))
    adx = float(row.get("ADX", np.nan))
    macd_hist = float(row.get("MACD_HIST", np.nan))
    vwap = float(row.get("VWAP", np.nan))
    roc = float(row.get("ROC", np.nan))
    volume_ratio = float(row.get("VOLUME_RATIO", np.nan))
    atr_pct = float(row.get("ATR_PCT", np.nan))

    score = 0.0
    if np.isfinite(ema20) and np.isfinite(ema50):
        score += 20.0 if ema20 > ema50 else -20.0
    if np.isfinite(ema200):
        score += 15.0 if close > ema200 else -15.0
    if np.isfinite(adx) and adx >= 18.0:
        score += _sign_score(ema20 - ema50, 15.0, -15.0)
    score += _sign_score(macd_hist, 15.0, -15.0)
    if np.isfinite(vwap) and vwap > 0:
        score += 10.0 if close > vwap else -10.0
    score += 10.0 if np.isfinite(roc) and roc > 0 else -10.0 if np.isfinite(roc) else 0.0
    if np.isfinite(volume_ratio) and volume_ratio >= 1.2:
        score += 5.0 if close >= ema20 else -5.0

    score = float(np.clip(score, -100.0, 100.0))
    if score >= 65:
        direction, regime = "STRONG_UP", "TREND_UP"
    elif score >= 30:
        direction, regime = "UP", "TREND_UP"
    elif score <= -65:
        direction, regime = "STRONG_DOWN", "TREND_DOWN"
    elif score <= -30:
        direction, regime = "DOWN", "TREND_DOWN"
    else:
        direction, regime = "RANGE", "RANGE"

    if np.isfinite(adx) and adx >= 25:
        strength_label = "STRONG"
    elif np.isfinite(adx) and adx >= 18:
        strength_label = "MEDIUM"
    else:
        strength_label = "WEAK"

    momentum = "POSITIVE" if macd_hist > 0 and roc > 0 else "NEGATIVE" if macd_hist < 0 and roc < 0 else "MIXED"
    volatility = "HIGH" if np.isfinite(atr_pct) and atr_pct > 0.04 else "ELEVATED" if np.isfinite(atr_pct) and atr_pct > 0.025 else "NORMAL"

    reason = (
        f"EMA20/50={'bullish' if ema20 > ema50 else 'bearish'}, "
        f"ADX={adx:.1f}, MACD_H={macd_hist:.4f}, VWAP={'above' if close >= vwap else 'below'}, "
        f"ROC={roc:.2f}%"
    )
    observed_at = frame["timestamp"].iloc[-1] if "timestamp" in frame.columns else None
    return TrendSnapshot(
        instrument=instrument.upper(),
        direction=direction,
        score=round(score, 1),
        strength=round(abs(score), 1),
        regime=regime,
        momentum=momentum,
        volatility=f"{volatility}/{strength_label}",
        observed_at=observed_at,
        data_bars=bars,
        is_live=is_live,
        reason=reason,
    )


def aggregate_context(
    snapshots: Mapping[str, TrendSnapshot],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    weights = weights or {"NIFTY": 0.40, "CRUDE": 0.20, "NATGAS": 0.15, "COPPER": 0.25}
    numerator = 0.0
    denominator = 0.0
    for instrument, snapshot in snapshots.items():
        weight = float(weights.get(instrument.upper(), 0.0))
        if snapshot.direction == "DATA_INSUFFICIENT" or weight <= 0:
            continue
        numerator += snapshot.score * weight
        denominator += weight
    return round(numerator / denominator, 1) if denominator else 0.0


def direction_alignment(candidate_direction: str, snapshot: TrendSnapshot) -> float:
    direction = candidate_direction.upper()
    if snapshot.direction == "DATA_INSUFFICIENT":
        return 0.0
    bullish = snapshot.direction in {"UP", "STRONG_UP"}
    bearish = snapshot.direction in {"DOWN", "STRONG_DOWN"}
    if direction in {"BUY", "BUY CE", "CE"}:
        return snapshot.strength if bullish else -snapshot.strength if bearish else 0.0
    if direction in {"SELL", "BUY PE", "PE"}:
        return snapshot.strength if bearish else -snapshot.strength if bullish else 0.0
    return 0.0
