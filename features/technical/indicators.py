import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators required by the rule-based strategy.

    Input columns: open, high, low, close and optionally volume.
    The function is side-effect free and preserves all input columns.
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = df.copy()
    close = pd.to_numeric(result["close"], errors="coerce")
    high = pd.to_numeric(result["high"], errors="coerce")
    low = pd.to_numeric(result["low"], errors="coerce")
    volume = (
        pd.to_numeric(result["volume"], errors="coerce")
        if "volume" in result
        else pd.Series(0.0, index=result.index)
    )

    result["EMA20"] = close.ewm(span=20, adjust=False).mean()
    result["EMA50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = ema12 - ema26
    result["MACD_SIGNAL"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["MACD_HIST"] = result["MACD"] - result["MACD_SIGNAL"]

    result["BB_MIDDLE"] = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    result["BB_UPPER"] = result["BB_MIDDLE"] + 2.0 * bb_std
    result["BB_LOWER"] = result["BB_MIDDLE"] - 2.0 * bb_std

    # Intraday VWAP. If timestamps are unavailable, use a single
    # cumulative session. The strategy/backtest engine resets it per day.
    typical_price = (high + low + close) / 3.0
    if "timestamp" in result.columns:
        timestamps = pd.to_datetime(result["timestamp"], errors="coerce")
        session = timestamps.dt.normalize()
        pv = typical_price * volume
        cumulative_pv = pv.groupby(session).cumsum()
        cumulative_volume = volume.groupby(session).cumsum()
        result["VWAP"] = cumulative_pv / cumulative_volume.replace(0, np.nan)
    else:
        result["VWAP"] = (
            (typical_price * volume).cumsum()
            / volume.cumsum().replace(0, np.nan)
        )

    result["VOLUME_MA20"] = volume.rolling(20, min_periods=20).mean()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["ATR"] = true_range.rolling(14, min_periods=14).mean()

    return result
