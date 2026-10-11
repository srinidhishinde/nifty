"""Index-aware, theta-aware next-minute option forecast (research only).

The index trend supplies the underlying-move estimate. Black-Scholes Greeks
translate that estimate to option premium while explicitly accounting for
calendar-time theta. Implied volatility is inferred from the last option close
unless a decimal/percent IV column is supplied. IV is held constant for the
one-minute scenario; this is a transparent baseline, not a complete market model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from math import erf, exp, log, pi, sqrt
from typing import Any

import numpy as np
import pandas as pd

_SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0
_MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0


@dataclass(frozen=True)
class IndexAwareOptionForecast:
    target_candle_time: str
    reference_index: float
    reference_premium: float
    predicted_premium: float
    predicted_low: float
    predicted_high: float
    index_direction: str
    direction: str
    expected_index_change: float
    expected_option_change: float
    theta_per_minute: float
    theta_contribution: float
    delta: float
    gamma: float
    implied_volatility: float
    iv_source: str
    aligned_bars: int
    method: str = "index_momentum_plus_black_scholes_greeks_theta"
    disclaimer: str = (
        "Research estimate only. Assumes constant IV for the next minute; "
        "not a guaranteed price, signal, or profitable trade."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _expiry_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    expiry = pd.Timestamp(value)
    if expiry.tzinfo is None:
        # Date-only expiry means 15:30 India time, the normal NIFTY session close.
        if expiry.time() == time(0, 0):
            expiry = expiry + pd.Timedelta(hours=15, minutes=30)
        expiry = expiry.tz_localize("Asia/Kolkata")
    return expiry.tz_convert("UTC")


def _bs_call(spot: float, strike: float, years: float, rate: float, iv: float) -> float:
    if years <= 0:
        return max(spot - strike, 0.0)
    sigma_root_t = max(iv, 1e-6) * sqrt(years)
    d1 = (log(spot / strike) + (rate + 0.5 * iv * iv) * years) / sigma_root_t
    d2 = d1 - sigma_root_t
    return spot * _cdf(d1) - strike * exp(-rate * years) * _cdf(d2)


def _implied_volatility(price: float, spot: float, strike: float, years: float, rate: float) -> float:
    if years <= 0:
        return 0.01
    intrinsic = max(spot - strike * exp(-rate * years), 0.0)
    target = max(float(price), intrinsic)
    low, high = 0.01, 5.0
    for _ in range(90):
        mid = (low + high) / 2.0
        if _bs_call(spot, strike, years, rate, mid) < target:
            low = mid
        else:
            high = mid
    return min(max((low + high) / 2.0, 0.01), 5.0)


def _greeks(spot: float, strike: float, years: float, rate: float, iv: float) -> tuple[float, float, float]:
    """Return call delta, gamma, and theta per calendar minute."""
    if years <= 0 or iv <= 0:
        return (1.0 if spot > strike else 0.0), 0.0, 0.0
    root_t = sqrt(years)
    sigma_root_t = iv * root_t
    d1 = (log(spot / strike) + (rate + 0.5 * iv * iv) * years) / sigma_root_t
    d2 = d1 - sigma_root_t
    delta = _cdf(d1)
    gamma = _pdf(d1) / (spot * sigma_root_t)
    theta_year = (
        -spot * _pdf(d1) * iv / (2.0 * root_t)
        - rate * strike * exp(-rate * years) * _cdf(d2)
    )
    return delta, gamma, theta_year / _MINUTES_PER_YEAR


def _normalize(frame: pd.DataFrame, name: str, *, allow_iv: bool = False) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} data missing columns: {sorted(missing)}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if allow_iv and "implied_volatility" in data:
        data["implied_volatility"] = pd.to_numeric(data["implied_volatility"], errors="coerce")
    numeric = ["open", "high", "low", "close"]
    if data["timestamp"].isna().any() or data[numeric].isna().any().any():
        raise ValueError(f"{name} data contains missing or invalid timestamps/prices")
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} data contains non-finite prices")
    if (data[numeric] <= 0).any().any():
        raise ValueError(f"{name} OHLC prices must be positive")
    invalid = (
        (data["high"] < data[["open", "low", "close"]].max(axis=1))
        | (data["low"] > data[["open", "high", "close"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError(f"{name} data contains invalid OHLC rows")
    return data.sort_values("timestamp", kind="stable").drop_duplicates(
        "timestamp", keep="last"
    ).reset_index(drop=True)


def forecast_index_aware_option(
    option_candles: pd.DataFrame,
    index_candles: pd.DataFrame,
    *,
    strike: float,
    expiry: str | pd.Timestamp,
    lookback: int = 30,
    range_quantile: float = 0.80,
    risk_free_rate: float = 0.06,
) -> IndexAwareOptionForecast:
    """Forecast the next premium using aligned NIFTY index and option candles.

    The direction is always UP or DOWN (zero expected change deterministically
    breaks UP); this deliberately does not introduce a model-based NO-TRADE gate.
    Input must contain completed candles only. The index and option timestamps
    are normalized to UTC and exact-matched, preventing accidental lookahead.
    """
    if not np.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be positive and finite")
    if lookback < 3:
        raise ValueError("lookback must be at least 3")
    if not 0.5 < range_quantile < 1.0:
        raise ValueError("range_quantile must be between 0.5 and 1.0")
    if not np.isfinite(risk_free_rate) or risk_free_rate < 0:
        raise ValueError("risk_free_rate must be non-negative and finite")

    option = _normalize(option_candles, "option", allow_iv=True)
    index = _normalize(index_candles, "index")
    keep = ["timestamp", "open", "high", "low", "close"]
    if "implied_volatility" in option:
        keep.append("implied_volatility")
    joined = option[keep].merge(
        index[["timestamp", "open", "high", "low", "close"]].rename(
            columns={c: f"index_{c}" for c in ("open", "high", "low", "close")}
        ),
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp", kind="stable").reset_index(drop=True)
    if len(joined) < 3:
        raise ValueError("At least 3 exactly timestamp-aligned index/option candles are required")

    history = joined.tail(lookback).reset_index(drop=True)
    last = history.iloc[-1]
    spot = float(last["index_close"])
    premium = float(last["close"])
    now = pd.Timestamp(last["timestamp"])
    expiry_time = _expiry_utc(expiry)
    years = max((expiry_time - now).total_seconds() / _SECONDS_PER_YEAR, 0.0)

    iv_source = "implied_volatility" if "implied_volatility" in history and pd.notna(last.get("implied_volatility")) else "inferred_from_option_close"
    if iv_source == "implied_volatility":
        iv = float(last["implied_volatility"])
        if iv > 5.0:
            iv /= 100.0
        if not np.isfinite(iv) or not 0 < iv <= 5.0:
            raise ValueError("implied_volatility must be a positive decimal or percentage")
    else:
        iv = _implied_volatility(premium, spot, float(strike), years, risk_free_rate)
    delta, gamma, theta_per_minute = _greeks(spot, float(strike), years, risk_free_rate, iv)

    # Blend 1-, 3-, and 5-minute index returns, scaled to a one-minute estimate.
    # Weights are fixed in advance; this is a baseline to benchmark, not a tuned claim.
    close = history["index_close"].astype(float)
    candidates = []
    for lag, weight in ((1, 0.50), (3, 0.30), (5, 0.20)):
        if len(close) > lag:
            candidates.append((weight, float(close.iloc[-1] / close.iloc[-1-lag] - 1.0) / lag))
    weight_sum = sum(w for w, _ in candidates)
    expected_index_change = spot * sum(w * r for w, r in candidates) / weight_sum
    index_direction = "UP" if expected_index_change >= 0 else "DOWN"
    expected_option_change = delta * expected_index_change + 0.5 * gamma * expected_index_change**2 + theta_per_minute
    # Do not forecast negative option prices; deterministic tie-break keeps a decision each candle.
    predicted_premium = max(0.05, premium + expected_option_change)
    direction = "UP" if expected_option_change >= 0 else "DOWN"

    # Use completed option candles only to estimate asymmetric next-candle excursions.
    down_excursions = (history["close"] - history["low"]).clip(lower=0.0)
    up_excursions = (history["high"] - history["close"]).clip(lower=0.0)
    floor = max(premium * 0.002, 0.05)
    down_size = max(float(down_excursions.quantile(range_quantile)), floor)
    up_size = max(float(up_excursions.quantile(range_quantile)), floor)
    target = now + pd.Timedelta(minutes=1)
    return IndexAwareOptionForecast(
        target_candle_time=target.isoformat(),
        reference_index=round(spot, 2),
        reference_premium=round(premium, 2),
        predicted_premium=round(predicted_premium, 2),
        predicted_low=round(max(0.05, predicted_premium - down_size), 2),
        predicted_high=round(predicted_premium + up_size, 2),
        index_direction=index_direction,
        direction=direction,
        expected_index_change=round(expected_index_change, 4),
        expected_option_change=round(expected_option_change, 4),
        theta_per_minute=round(theta_per_minute, 6),
        theta_contribution=round(theta_per_minute, 6),
        delta=round(delta, 6),
        gamma=round(gamma, 8),
        implied_volatility=round(iv, 6),
        iv_source=iv_source,
        aligned_bars=len(history),
    )
