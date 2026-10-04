from __future__ import annotations
import numpy as np
import pandas as pd
from features.technical.indicators import add_indicators

ALIASES = {
    "option_pcr": ("PCR_OI", "pcr_oi", "PCR"),
    "option_pcr_change": ("PCR_OI_CHG", "pcr_oi_change", "PCR_CHANGE"),
    "option_oi_imbalance": ("OI_IMBALANCE", "oi_imbalance", "NET_OI"),
    "option_delta_oi_imbalance": ("DELTA_OI_IMBALANCE", "delta_oi_imbalance", "NET_DELTA_OI"),
    "option_iv": ("ATM_IV", "atm_iv", "IV"),
    "sentiment": ("sentiment", "news_sentiment", "Sentiment"),
    "sentiment_change": ("sentiment_change", "news_sentiment_change"),
    "news_count": ("news_count", "NEWS_COUNT"),
}

def build_features(data: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    frame = add_indicators(frame)
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    frame["ema_slope_9"] = ema9.pct_change(3)
    frame["ema_slope_21"] = ema21.pct_change(3)
    frame["close_return_1"] = close.pct_change(1)
    frame["close_return_3"] = close.pct_change(3)
    frame["close_return_6"] = close.pct_change(6)
    frame["atr_percentile"] = frame["ATR_PCT"].rolling(100, min_periods=20).rank(pct=True)
    vol_mean = volume.rolling(50, min_periods=20).mean()
    vol_std = volume.rolling(50, min_periods=20).std(ddof=0).replace(0, np.nan)
    frame["volume_zscore"] = (volume-vol_mean)/vol_std
    for target, aliases in ALIASES.items():
        source = next((x for x in aliases if x in frame.columns), None)
        frame[target] = pd.to_numeric(frame[source], errors="coerce") if source else 0.0
    return frame.replace([np.inf,-np.inf],np.nan)

def feature_matrix(frame: pd.DataFrame, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    missing=[c for c in feature_columns if c not in frame.columns]
    if missing: raise ValueError(f"Missing ML feature columns: {missing}")
    return frame.loc[:,list(feature_columns)].apply(pd.to_numeric,errors="coerce")

def drop_highly_correlated(X: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, tuple[str,...]]:
    corr=X.corr().abs(); upper=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
    dropped=tuple(c for c in upper.columns if any(upper[c]>threshold))
    return X.drop(columns=list(dropped),errors="ignore"), dropped
