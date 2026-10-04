from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    take_profit_pct: float = 0.02,
    stop_loss_pct: float = 0.01,
    horizon: int = 12,
) -> pd.Series:
    """Create trade-aligned labels using intrabar barriers when OHLC is available."""
    prices = pd.to_numeric(close, errors="coerce")
    labels = pd.Series(np.nan, index=prices.index, dtype="float64")
    if take_profit_pct <= 0 or stop_loss_pct <= 0 or horizon <= 0:
        raise ValueError("Barrier percentages and horizon must be positive")

    values = prices.to_numpy(dtype=float)
    highs = pd.to_numeric(high, errors="coerce") if high is not None else prices
    lows = pd.to_numeric(low, errors="coerce") if low is not None else prices
    if len(highs) != len(prices) or len(lows) != len(prices):
        raise ValueError("high and low must have the same length as close")
    high_values = highs.to_numpy(dtype=float)
    low_values = lows.to_numpy(dtype=float)
    for i in range(len(values)):
        entry = values[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        end = min(len(values), i + 1 + horizon)
        upper = entry * (1.0 + take_profit_pct)
        lower = entry * (1.0 - stop_loss_pct)
        for j in range(i + 1, end):
            if not np.isfinite(high_values[j]) or not np.isfinite(low_values[j]):
                continue
            target_hit = high_values[j] >= upper
            stop_hit = low_values[j] <= lower
            if stop_hit:
                labels.iloc[i] = 0.0
                break
            if target_hit:
                labels.iloc[i] = 1.0
                break
    return labels
