from __future__ import annotations

import numpy as np
import pandas as pd

from features.technical.candlestick import add_candlestick_patterns


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = add_candlestick_patterns(df.copy())
    close = pd.to_numeric(result["close"], errors="coerce")
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    volume = pd.to_numeric(result["volume"], errors="coerce") if "volume" in result else pd.Series(0.0, index=result.index)

    # Fast/slow trend stack from the research specification. Keep EMA20/50
    # for backward compatibility with existing reports/tests.
    result["EMA9"] = close.ewm(span=9, adjust=False).mean()
    result["EMA21"] = close.ewm(span=21, adjust=False).mean()
    result["EMA20"] = close.ewm(span=20, adjust=False).mean()
    result["EMA50"] = close.ewm(span=50, adjust=False).mean()
    result["SMA50"] = close.rolling(50, min_periods=50).mean()
    result["SMA200"] = close.rolling(200, min_periods=200).mean()
    result["EMA200"] = close.ewm(span=200, adjust=False).mean()
    result["RSI"] = _rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_SIGNAL"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["MACD_HIST"] = result["MACD"] - result["MACD_SIGNAL"]

    result["BB_MIDDLE"] = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    result["BB_UPPER"] = result["BB_MIDDLE"] + 2.0 * bb_std
    result["BB_LOWER"] = result["BB_MIDDLE"] - 2.0 * bb_std
    result["BB_WIDTH"] = (result["BB_UPPER"] - result["BB_LOWER"]) / result["BB_MIDDLE"].replace(0, np.nan)

    typical = (high + low + close) / 3.0
    if "timestamp" in result:
        timestamps = pd.to_datetime(result["timestamp"], errors="coerce")
        session = timestamps.dt.normalize()
        pv = typical * volume
        result["VWAP"] = pv.groupby(session).cumsum() / volume.groupby(session).cumsum().replace(0, np.nan)
    else:
        result["VWAP"] = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)

    result["VOLUME_MA20"] = volume.rolling(20, min_periods=20).mean()
    result["VOLUME_RATIO"] = volume / result["VOLUME_MA20"].replace(0, np.nan)

    prev_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    result["ATR"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    result["ATR5"] = true_range.ewm(alpha=1 / 5, adjust=False, min_periods=5).mean()
    result["ATR_PCT"] = result["ATR"] / close.replace(0, np.nan)
    # Directional components are retained so ADX cannot be misused as a
    # standalone BUY/SELL signal.
    adx_period = 14
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    plus_atr = plus_dm.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean()
    minus_atr = minus_dm.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean()
    atr_for_di = true_range.ewm(alpha=1 / adx_period, adjust=False, min_periods=adx_period).mean()
    result["DI_PLUS"] = 100 * plus_atr / atr_for_di.replace(0, np.nan)
    result["DI_MINUS"] = 100 * minus_atr / atr_for_di.replace(0, np.nan)
    result["ADX"] = _adx(high, low, close, adx_period)

    lowest = low.rolling(14, min_periods=14).min()
    highest = high.rolling(14, min_periods=14).max()
    result["STOCH_K"] = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    result["STOCH_D"] = result["STOCH_K"].rolling(3, min_periods=3).mean()
    result["ROC"] = close.pct_change(10) * 100
    result["OBV"] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    result["VWAP_DEV"] = (close - result["VWAP"]) / result["VWAP"].replace(0, np.nan)
    result["EMA_SPREAD"] = (result["EMA9"] - result["EMA21"]) / close.replace(0, np.nan)
    result["TREND_SPREAD"] = (result["SMA50"] - result["SMA200"]) / close.replace(0, np.nan)
    result["BB_POS"] = (close - result["BB_MIDDLE"]) / (result["BB_UPPER"] - result["BB_LOWER"]).replace(0, np.nan)
    return result
