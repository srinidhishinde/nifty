from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    *,
    take_profit_pct: float = 0.02,
    stop_loss_pct: float = 0.01,
    horizon: int = 12,
) -> pd.Series:
    """Create trade-aligned labels: +1 target first, 0 stop first, NaN time-only/insufficient."""
    prices = pd.to_numeric(close, errors="coerce")
    labels = pd.Series(np.nan, index=prices.index, dtype="float64")
    if take_profit_pct <= 0 or stop_loss_pct <= 0 or horizon <= 0:
        raise ValueError("Barrier percentages and horizon must be positive")

    values = prices.to_numpy(dtype=float)
    for i in range(len(values)):
        entry = values[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        end = min(len(values), i + 1 + horizon)
        future = values[i + 1:end]
        if len(future) == 0:
            continue
        upper = entry * (1.0 + take_profit_pct)
        lower = entry * (1.0 - stop_loss_pct)
        target_hit = np.flatnonzero(future >= upper)
        stop_hit = np.flatnonzero(future <= lower)
        if target_hit.size == 0 and stop_hit.size == 0:
            continue
        target_idx = int(target_hit[0]) if target_hit.size else 10**9
        stop_idx = int(stop_hit[0]) if stop_hit.size else 10**9
        labels.iloc[i] = 1.0 if target_idx < stop_idx else 0.0
    return labels
