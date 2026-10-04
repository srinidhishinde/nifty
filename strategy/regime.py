from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: str
    tradable: bool
    score: float
    reason: str


def classify_regime(
    row: pd.Series,
    news_score: float = 0.0,
    max_atr_pct: float = 0.04,
    min_adx: float = 18.0,
) -> RegimeSnapshot:
    """Classify the current market without using future bars.

    This is a gate, not a predictive model. HIGH_VOLATILITY and EVENT_DRIVEN
    are intentionally non-tradable by default; TREND/RANGE remain advisory.
    """
    def value(*names, default=None):
        for name in names:
            if name in row.index and pd.notna(row[name]):
                try:
                    return float(row[name])
                except (TypeError, ValueError):
                    pass
        return default

    adx = value("ADX", default=0.0) or 0.0
    atr_pct = value("ATR_PCT", default=0.0) or 0.0
    ema_spread = value("EMA_SPREAD", default=0.0) or 0.0
    vwap_dev = abs(value("VWAP_DEV", default=0.0) or 0.0)

    if atr_pct > max_atr_pct:
        return RegimeSnapshot("HIGH_VOLATILITY", False, 0.0, f"ATR% {atr_pct:.3f} exceeds {max_atr_pct:.3f}")
    if abs(news_score) >= 0.75:
        return RegimeSnapshot("EVENT_DRIVEN", False, 0.0, f"News sentiment {news_score:+.2f} is extreme")
    if adx >= min_adx and ema_spread > 0:
        return RegimeSnapshot("TREND_UP", True, min(100.0, 50.0 + adx), f"ADX {adx:.1f} with positive EMA spread")
    if adx >= min_adx and ema_spread < 0:
        return RegimeSnapshot("TREND_DOWN", True, min(100.0, 50.0 + adx), f"ADX {adx:.1f} with negative EMA spread")
    if vwap_dev <= 0.01:
        return RegimeSnapshot("RANGE", True, 50.0, "Price is close to VWAP and trend strength is limited")
    return RegimeSnapshot("MIXED", True, 40.0, "No dominant trend or volatility regime detected")
