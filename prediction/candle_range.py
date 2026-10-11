"""Causal next-candle high/low range forecasts with width calibration.

The forecast is made at the target candle's open using only earlier completed
candles. The calibration target controls average interval width; it does not
guarantee coverage or direction accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RangeForecastResult:
    forecasts: pd.DataFrame
    mean_width_ratio: float
    full_range_coverage_pct: float
    low_mae: float
    high_mae: float


def walk_forward_range_forecast(
    data: pd.DataFrame,
    *,
    window: int = 30,
    min_periods: int = 5,
    excursion_quantile: float = 0.80,
    target_width_ratio: float = 1.0,
) -> RangeForecastResult:
    """Forecast candle low/high bounds without target-candle high/low leakage.

    Required columns: timestamp, open, high, low. Each target candle's open is
    treated as observable at forecast time. Desired interval width is the
    rolling mean high-low range from prior completed candles times
    target_width_ratio. Historical open-to-low and high-to-open quantiles
    determine how that width is split around the target open.

    A target_width_ratio of 1.0 aims for forecast width near the recent average
    candle range. Actual test-period width ratio can differ as volatility
    changes. Narrowing intervals generally reduces coverage.
    """
    required = {"timestamp", "open", "high", "low"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if window < 1 or min_periods < 1 or min_periods > window:
        raise ValueError("Require window >= min_periods >= 1")
    if not 0.0 < excursion_quantile < 1.0:
        raise ValueError("excursion_quantile must be between 0 and 1")
    if target_width_ratio <= 0:
        raise ValueError("target_width_ratio must be positive")

    frame = data.copy().reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for col in ("open", "high", "low"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low"]).reset_index(drop=True)
    if (frame["high"] < frame["low"]).any():
        raise ValueError("Found candle with high below low")
    if ((frame["open"] < frame["low"]) | (frame["open"] > frame["high"])).any():
        raise ValueError("Each open must be inside its candle's low/high range")

    rows: list[dict] = []
    for i in range(min_periods, len(frame)):
        # Only candles strictly before target i contribute to the model.
        history = frame.iloc[max(0, i - window):i]
        if len(history) < min_periods:
            continue
        low_excursions = (history["open"] - history["low"]).clip(lower=0.0)
        high_excursions = (history["high"] - history["open"]).clip(lower=0.0)
        low_share = float(low_excursions.quantile(excursion_quantile))
        high_share = float(high_excursions.quantile(excursion_quantile))
        recent_mean_range = float((history["high"] - history["low"]).mean())
        desired_width = recent_mean_range * target_width_ratio
        excursion_total = low_share + high_share
        if not np.isfinite(desired_width) or desired_width <= 0:
            continue
        if not np.isfinite(excursion_total) or excursion_total <= 1e-12:
            low_share = high_share = 0.5
            excursion_total = 1.0
        scale = desired_width / excursion_total
        predicted_low = float(frame.iloc[i]["open"]) - low_share * scale
        predicted_high = float(frame.iloc[i]["open"]) + high_share * scale
        target = frame.iloc[i]
        rows.append({
            "timestamp": target["timestamp"],
            "reference_open": float(target["open"]),
            "predicted_low": predicted_low,
            "predicted_high": predicted_high,
            "forecast_width": predicted_high - predicted_low,
            "recent_mean_actual_range": recent_mean_range,
            "actual_low": float(target["low"]),
            "actual_high": float(target["high"]),
            "actual_range": float(target["high"] - target["low"]),
            "low_covered": predicted_low <= float(target["low"]),
            "high_covered": predicted_high >= float(target["high"]),
            "full_range_covered": (
                predicted_low <= float(target["low"])
                and predicted_high >= float(target["high"])
            ),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return RangeForecastResult(result, 0.0, 0.0, 0.0, 0.0)
    actual_mean = float(result["actual_range"].mean())
    mean_width = float(result["forecast_width"].mean())
    ratio = mean_width / actual_mean if actual_mean > 0 else 0.0
    coverage = float(result["full_range_covered"].mean() * 100.0)
    low_mae = float((result["predicted_low"] - result["actual_low"]).abs().mean())
    high_mae = float((result["predicted_high"] - result["actual_high"]).abs().mean())
    return RangeForecastResult(result, ratio, coverage, low_mae, high_mae)
