"""Leakage-free online calibration and baseline comparison for low/high forecasts.

Calibration for target candle t uses only forecast errors from targets < t.
The first candle is excluded because a single-session file has no prior close
from which to issue a forecast before that candle opens.
"""
from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from prediction.low_high_forecast import walk_forward_low_high


def _finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    """Conservative order-statistic quantile (finite-sample conformal form)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0
    ordered = np.sort(values)
    rank = min(len(ordered), max(1, ceil((len(ordered) + 1) * coverage)))
    return float(ordered[rank - 1])


def _metrics(rows: pd.DataFrame, low_col: str, high_col: str, target_coverage: float) -> dict[str, Any]:
    usable = rows.dropna(subset=[low_col, high_col, "actual_low", "actual_high"])
    if usable.empty:
        return {"rows": 0, "full_range_coverage_pct": None, "mean_predicted_width": None}
    low = usable[low_col].astype(float)
    high = usable[high_col].astype(float)
    actual_low = usable["actual_low"].astype(float)
    actual_high = usable["actual_high"].astype(float)
    covered = (low <= actual_low) & (high >= actual_high)
    width = high - low
    actual_width = actual_high - actual_low
    coverage = float(covered.mean())
    mean_width = float(width.mean())
    return {
        "rows": int(len(usable)),
        "low_mae": float((low - actual_low).abs().mean()),
        "high_mae": float((high - actual_high).abs().mean()),
        "low_coverage_pct": float((low <= actual_low).mean() * 100),
        "high_coverage_pct": float((high >= actual_high).mean() * 100),
        "full_range_coverage_pct": coverage * 100,
        "mean_predicted_width": mean_width,
        "mean_actual_width": float(actual_width.mean()),
        "width_ratio": mean_width / max(float(actual_width.mean()), 1e-12),
        "coverage_shortfall_pct_points": max(0.0, (target_coverage - coverage) * 100),
    }


def tune_and_compare(
    data: pd.DataFrame,
    *,
    target_coverage: float = 0.80,
    min_train: int = 80,
    estimator: str = "hist_gradient_boosting",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Produce pre-open walk-forward forecasts and compare against baselines.

    conformal_* expands the model interval by a finite-sample quantile of
    prior out-of-sample interval misses. historical_quantile_* uses only
    completed-candle excursions available before each target candle.
    """
    if not 0.50 <= target_coverage < 1.0:
        raise ValueError("target_coverage must be in [0.50, 1.0)")
    frame = data.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    if "timestamp" not in frame.columns:
        for alias in ("datetime", "date_time", "time", "date"):
            if alias in frame.columns:
                frame = frame.rename(columns={alias: "timestamp"})
                break
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "volume" not in frame:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    valid = (
        (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
        & (frame[["open", "close", "low"]].min(axis=1) > 0)
    )
    frame = frame.loc[valid].reset_index(drop=True)
    if len(frame) < 2:
        raise ValueError("At least two valid candles are required for pre-open evaluation")

    base = walk_forward_low_high(frame, min_train=min_train, estimator=estimator, seed=seed)
    out = base.predictions.copy()
    # Explicitly mark the first candle as not forecastable before its open from
    # this file alone. Do not score its open-anchored legacy fallback.
    for col in ("predicted_low", "predicted_high"):
        out.loc[0, col] = np.nan
    out.loc[0, "forecast_method"] = "NO_PRIOR_CANDLE_NOT_PREOPEN"
    out["model_low"] = out["predicted_low"]
    out["model_high"] = out["predicted_high"]
    out["conformal_low"] = np.nan
    out["conformal_high"] = np.nan
    out["asymmetric_conformal_low"] = np.nan
    out["asymmetric_conformal_high"] = np.nan
    out["asymmetric_low_expansion"] = np.nan
    out["asymmetric_high_expansion"] = np.nan
    out["median_range_low"] = np.nan
    out["median_range_high"] = np.nan
    out["historical_quantile_low"] = np.nan
    out["historical_quantile_high"] = np.nan
    out["optimized_quantile_low"] = np.nan
    out["optimized_quantile_high"] = np.nan
    out["optimized_down_quantile"] = np.nan
    out["optimized_up_quantile"] = np.nan
    out["conformal_expansion"] = np.nan
    out["historical_down_quantile"] = np.nan
    out["historical_up_quantile"] = np.nan

    q_each_side = (1.0 + target_coverage) / 2.0
    for i in range(1, len(frame)):
        anchor = float(frame.loc[i - 1, "close"])
        model_low = float(out.loc[i, "model_low"])
        model_high = float(out.loc[i, "model_high"])
        # Only previous forecast rows have matured outcomes at this origin.
        past = out.iloc[1:i]
        scores = np.maximum(
            np.maximum(past["model_low"].to_numpy(dtype=float) - past["actual_low"].to_numpy(dtype=float), 0.0),
            np.maximum(past["actual_high"].to_numpy(dtype=float) - past["model_high"].to_numpy(dtype=float), 0.0),
        )
        expansion = _finite_sample_quantile(scores, target_coverage) if len(scores) else 0.0
        out.loc[i, "conformal_expansion"] = expansion
        out.loc[i, "conformal_low"] = max(0.01, model_low - expansion)
        out.loc[i, "conformal_high"] = max(out.loc[i, "conformal_low"], model_high + expansion)

        # Split asymmetric online calibration: estimate lower- and upper-bound
        # misses separately from prior forecast rows whose target candles have
        # already closed. Bonferroni tail allocation (1 + coverage) / 2 on each
        # side gives a conservative joint target without forcing the larger
        # miss on both sides, which can make intervals unnecessarily wide.
        past_low_miss = np.maximum(
            past["model_low"].to_numpy(dtype=float) - past["actual_low"].to_numpy(dtype=float), 0.0
        )
        past_high_miss = np.maximum(
            past["actual_high"].to_numpy(dtype=float) - past["model_high"].to_numpy(dtype=float), 0.0
        )
        low_expansion = _finite_sample_quantile(past_low_miss, q_each_side) if len(past_low_miss) else 0.0
        high_expansion = _finite_sample_quantile(past_high_miss, q_each_side) if len(past_high_miss) else 0.0
        out.loc[i, "asymmetric_low_expansion"] = low_expansion
        out.loc[i, "asymmetric_high_expansion"] = high_expansion
        out.loc[i, "asymmetric_conformal_low"] = max(0.01, model_low - low_expansion)
        out.loc[i, "asymmetric_conformal_high"] = max(
            out.loc[i, "asymmetric_conformal_low"], model_high + high_expansion
        )

        # Historical excursion samples are from completed candles only.
        hist = frame.iloc[1:i]
        hist_anchor = frame["close"].iloc[: i - 1].to_numpy(dtype=float)
        down = np.maximum(hist_anchor - hist["low"].to_numpy(dtype=float), 0.0)
        up = np.maximum(hist["high"].to_numpy(dtype=float) - hist_anchor, 0.0)
        ranges = (hist["high"] - hist["low"]).to_numpy(dtype=float)
        if len(ranges):
            half_median = max(float(np.median(ranges)) / 2.0, anchor * 0.0005, 0.01)
            out.loc[i, "median_range_low"] = max(0.01, anchor - half_median)
            out.loc[i, "median_range_high"] = anchor + half_median
        if len(down):
            down_q = _finite_sample_quantile(down, q_each_side)
            up_q = _finite_sample_quantile(up, q_each_side)
            out.loc[i, "historical_quantile_low"] = max(0.01, anchor - down_q)
            out.loc[i, "historical_quantile_high"] = anchor + up_q

            # Select the narrowest asymmetric empirical interval whose joint
            # coverage on prior completed excursions meets the requested target.
            # Require a minimum calibration sample; before that use a
            # conservative equal-tail quantile rather than overfit tiny samples.
            grid = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925,
                    0.95, 0.975, 0.99, 1.0)
            candidates = []
            if len(down) >= 20:
                for qd in grid:
                    dq = float(np.quantile(down, qd))
                    for qu in grid:
                        uq = float(np.quantile(up, qu))
                        joint_coverage = float(((down <= dq) & (up <= uq)).mean())
                        if joint_coverage >= target_coverage:
                            candidates.append((dq + uq, -joint_coverage, dq, uq, qd, qu))
            if candidates:
                _, neg_coverage, opt_down, opt_up, qd, qu = min(candidates)
            else:
                qd = qu = q_each_side
                opt_down = _finite_sample_quantile(down, qd)
                opt_up = _finite_sample_quantile(up, qu)
            out.loc[i, "optimized_quantile_low"] = max(0.01, anchor - opt_down)
            out.loc[i, "optimized_quantile_high"] = anchor + opt_up
            out.loc[i, "optimized_down_quantile"] = qd
            out.loc[i, "optimized_up_quantile"] = qu
            out.loc[i, "historical_down_quantile"] = down_q
            out.loc[i, "historical_up_quantile"] = up_q

    evaluated = out.iloc[1:].copy()
    methods = {
        "model": ("model_low", "model_high"),
        "online_conformal_model": ("conformal_low", "conformal_high"),
        "online_asymmetric_conformal_model": ("asymmetric_conformal_low", "asymmetric_conformal_high"),
        "rolling_median_range": ("median_range_low", "median_range_high"),
        "historical_excursion_quantile": ("historical_quantile_low", "historical_quantile_high"),
        "optimized_asymmetric_quantile": ("optimized_quantile_low", "optimized_quantile_high"),
    }
    summary: dict[str, Any] = {
        "status": "EVALUATED",
        "target_full_range_coverage": target_coverage,
        "target_coverage_pct": target_coverage * 100,
        "forecast_timing": "before_target_candle_open",
        "first_candle": "excluded_no_prior_candle_in_input",
        "input_candles": int(len(frame)),
        "evaluated_forecasts": int(len(evaluated)),
        "estimator": estimator,
        "min_train": min_train,
        "methods": {name: _metrics(evaluated, low, high, target_coverage) for name, (low, high) in methods.items()},
        "calibration_note": (
            "Online conformal expansion uses only earlier model forecast misses. The asymmetric conformal variant estimates lower and upper misses separately with finite-sample tail quantiles. "
            "Optimized asymmetric quantiles select the narrowest historical down/up excursion pair "
            "meeting target joint coverage on prior completed candles (minimum 20 observations; "
            "equal-tail fallback before that). All methods are evaluated on later candles only. "
            "Metrics are out-of-sample walk-forward results, not proof of future performance."
        ),
    }
    return out, summary
