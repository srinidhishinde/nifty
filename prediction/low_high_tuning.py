"""Leakage-free online calibration and baseline comparison for low/high forecasts.

Calibration for target candle t uses only forecast errors from targets < t.
The first candle's open-anchored fallback is preserved as a separate opening
forecast, but excluded from pre-open model/calibration metrics. A single-session
file cannot provide prior-session context for a genuine pre-open opening forecast.
"""
from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from prediction.low_high_forecast import walk_forward_low_high


def _adaptive_historical_quantile(values: np.ndarray, quantile: float, recent_window: int = 40) -> float:
    """Blend recent and full-history excursion quantiles without lookahead."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.maximum(values, 0.0)
    if not len(values):
        return 0.0
    recent = values[-max(1, recent_window):]
    recent_q = float(np.quantile(recent, quantile))
    history_q = float(np.quantile(values, quantile))
    return 0.75 * recent_q + 0.25 * history_q


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




def _point_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    """Score point estimates separately from interval bounds."""
    usable = rows.dropna(
        subset=[
            "point_predicted_low", "point_predicted_high", "predicted_close",
            "predicted_direction", "actual_close", "actual_direction",
            "actual_low", "actual_high",
        ]
    )
    if usable.empty:
        return {"rows": 0}
    low_error = (
        usable["point_predicted_low"].astype(float)
        - usable["actual_low"].astype(float)
    )
    high_error = (
        usable["point_predicted_high"].astype(float)
        - usable["actual_high"].astype(float)
    )
    return {
        "rows": int(len(usable)),
        "low_mae": float(low_error.abs().mean()),
        "high_mae": float(high_error.abs().mean()),
        "low_bias": float(low_error.mean()),
        "high_bias": float(high_error.mean()),
        "low_median_absolute_error": float(low_error.abs().median()),
        "high_median_absolute_error": float(high_error.abs().median()),
        "low_p90_absolute_error": float(low_error.abs().quantile(0.90)),
        "high_p90_absolute_error": float(high_error.abs().quantile(0.90)),
        "close_mae": float((usable["predicted_close"].astype(float) - usable["actual_close"].astype(float)).abs().mean()),
        "directional_accuracy_pct": float((usable["predicted_direction"] == usable["actual_direction"]).mean() * 100),
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
    # Keep the 09:15 open-anchored estimate visible as a separate forecast.
    # It is not a pre-open forecast and is excluded from model/calibration scores.
    out["opening_predicted_low"] = np.nan
    out["opening_predicted_high"] = np.nan
    out["opening_forecast_method"] = ""
    out.loc[0, "opening_predicted_low"] = out.loc[0, "predicted_low"]
    out.loc[0, "opening_predicted_high"] = out.loc[0, "predicted_high"]
    out.loc[0, "opening_forecast_method"] = "OPENING_CANDLE_OPEN_ANCHORED_FALLBACK"
    out.loc[0, "forecast_method"] = "OPENING_CANDLE_OPEN_ANCHORED_NOT_PREOPEN"
    out["model_low"] = out["predicted_low"]
    out["model_high"] = out["predicted_high"]
    out.loc[0, ["model_low", "model_high"]] = np.nan
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
    out["historical_asymmetric_quantile_low"] = np.nan
    out["historical_asymmetric_quantile_high"] = np.nan
    out["historical_down_quantile_level"] = np.nan
    out["historical_up_quantile_level"] = np.nan
    out["conformal_expansion"] = np.nan
    # Audit trail for calibration maturity and warm-up policy.
    out["calibration_model_rows"] = 0
    out["calibration_source"] = ""
    out["asymmetric_calibration_source"] = ""
    out["historical_down_quantile"] = np.nan
    out["historical_up_quantile"] = np.nan

    q_each_side = (1.0 + target_coverage) / 2.0
    for i in range(1, len(frame)):
        anchor = float(frame.loc[i - 1, "close"])
        model_low = float(out.loc[i, "model_low"])
        model_high = float(out.loc[i, "model_high"])
        # Calibrate only on matured forecasts produced by the trained model.
        # Mixing startup/range fallbacks into model residuals makes the
        # calibration distribution change artificially when the model first
        # becomes active, especially in short sessions. The filter is based
        # solely on each past row's forecast method, never its target outcome.
        prior_rows = out.iloc[1:i]
        past = prior_rows.loc[
            prior_rows["forecast_method"].astype(str).str.contains("MODEL_", regex=False)
        ]
        out.loc[i, "calibration_model_rows"] = int(len(past))

        # During calibration warm-up, zero expansion is dangerously optimistic:
        # the model has not yet accumulated enough genuine out-of-sample errors.
        # Use a predeclared historical-excursion envelope until 20 model errors
        # exist. These excursions are from completed candles only; no target
        # candle data enters the fallback interval.
        hist = frame.iloc[1:i]
        hist_anchor = frame["close"].iloc[: i - 1].to_numpy(dtype=float)
        down = np.maximum(hist_anchor - hist["low"].to_numpy(dtype=float), 0.0)
        up = np.maximum(hist["high"].to_numpy(dtype=float) - hist_anchor, 0.0)
        min_calibration_rows = 20
        warmup = len(past) < min_calibration_rows
        if warmup:
            down_q = _adaptive_historical_quantile(down, q_each_side)
            up_q = _adaptive_historical_quantile(up, q_each_side)
            fallback_low = max(0.01, anchor - down_q)
            fallback_high = anchor + up_q
            expansion = max(model_low - fallback_low, fallback_high - model_high, 0.0)
            out.loc[i, "calibration_source"] = "HISTORICAL_EXCURSION_WARMUP"
        else:
            scores = np.maximum(
                np.maximum(past["model_low"].to_numpy(dtype=float) - past["actual_low"].to_numpy(dtype=float), 0.0),
                np.maximum(past["actual_high"].to_numpy(dtype=float) - past["model_high"].to_numpy(dtype=float), 0.0),
            )
            expansion = _finite_sample_quantile(scores, target_coverage)
            out.loc[i, "calibration_source"] = "ONLINE_CONFORMAL_MODEL_RESIDUALS"
        out.loc[i, "conformal_expansion"] = expansion
        out.loc[i, "conformal_low"] = max(0.01, model_low - expansion)
        out.loc[i, "conformal_high"] = max(out.loc[i, "conformal_low"], model_high + expansion)

        # Split asymmetric online calibration. Until 20 matured model forecasts
        # are available, use the same historical warm-up envelope; after that,
        # estimate lower/upper misses separately using finite-sample tail quantiles.
        if warmup:
            low_expansion = max(model_low - fallback_low, 0.0)
            high_expansion = max(fallback_high - model_high, 0.0)
            out.loc[i, "asymmetric_calibration_source"] = "HISTORICAL_EXCURSION_WARMUP"
        else:
            past_low_miss = np.maximum(
                past["model_low"].to_numpy(dtype=float) - past["actual_low"].to_numpy(dtype=float), 0.0
            )
            past_high_miss = np.maximum(
                past["actual_high"].to_numpy(dtype=float) - past["model_high"].to_numpy(dtype=float), 0.0
            )
            low_expansion = _finite_sample_quantile(past_low_miss, q_each_side)
            high_expansion = _finite_sample_quantile(past_high_miss, q_each_side)
            out.loc[i, "asymmetric_calibration_source"] = "ONLINE_ASYMMETRIC_MODEL_RESIDUALS"
        out.loc[i, "asymmetric_low_expansion"] = low_expansion
        out.loc[i, "asymmetric_high_expansion"] = high_expansion
        out.loc[i, "asymmetric_conformal_low"] = max(0.01, model_low - low_expansion)
        out.loc[i, "asymmetric_conformal_high"] = max(
            out.loc[i, "asymmetric_conformal_low"], model_high + high_expansion
        )
        ranges = (hist["high"] - hist["low"]).to_numpy(dtype=float)
        if len(ranges):
            half_median = max(float(np.median(ranges)) / 2.0, anchor * 0.0005, 0.01)
            out.loc[i, "median_range_low"] = max(0.01, anchor - half_median)
            out.loc[i, "median_range_high"] = anchor + half_median
        if len(down):
            # Adapt to recent volatility while retaining a long-history
            # anchor. The quantile sees completed candles only.
            down_q = _adaptive_historical_quantile(down, q_each_side)
            up_q = _adaptive_historical_quantile(up, q_each_side)
            out.loc[i, "historical_quantile_low"] = max(0.01, anchor - down_q)
            out.loc[i, "historical_quantile_high"] = anchor + up_q

            # Use a fixed, predeclared equal-tail quantile rule. Do not search a
            # grid of asymmetric quantile pairs on this short rolling sample:
            # that adds selection noise and can make in-sample coverage look
            # better without reliable out-of-sample improvement.
            qd = qu = q_each_side
            hist_down = _adaptive_historical_quantile(down, qd)
            hist_up = _adaptive_historical_quantile(up, qu)
            out.loc[i, "historical_asymmetric_quantile_low"] = max(0.01, anchor - hist_down)
            out.loc[i, "historical_asymmetric_quantile_high"] = anchor + hist_up
            out.loc[i, "historical_down_quantile_level"] = qd
            out.loc[i, "historical_up_quantile_level"] = qu
            out.loc[i, "historical_down_quantile"] = down_q
            out.loc[i, "historical_up_quantile"] = up_q

    evaluated = out.iloc[1:].copy()
    methods = {
        "model": ("model_low", "model_high"),
        "online_conformal_model": ("conformal_low", "conformal_high"),
        "online_asymmetric_conformal_model": ("asymmetric_conformal_low", "asymmetric_conformal_high"),
        "rolling_median_range": ("median_range_low", "median_range_high"),
        "historical_excursion_quantile": ("historical_quantile_low", "historical_quantile_high"),
        "historical_asymmetric_quantile": ("historical_asymmetric_quantile_low", "historical_asymmetric_quantile_high"),
    }
    summary: dict[str, Any] = {
        "status": "EVALUATED",
        "target_full_range_coverage": target_coverage,
        "target_coverage_pct": target_coverage * 100,
        "forecast_timing": "before_target_candle_open",
        "first_candle": "opening_forecast_reported_separately_and_excluded_from_preopen_metrics",
        "opening_forecast": {
            "method": str(out.loc[0, "opening_forecast_method"]),
            "predicted_low": float(out.loc[0, "opening_predicted_low"]),
            "predicted_high": float(out.loc[0, "opening_predicted_high"]),
            "predicted_close": float(out.loc[0, "predicted_close"]),
            "predicted_direction": str(out.loc[0, "predicted_direction"]),
            "actual_close": float(out.loc[0, "actual_close"]),
            "actual_low": float(out.loc[0, "actual_low"]),
            "actual_high": float(out.loc[0, "actual_high"]),
            "full_range_covered": bool(out.loc[0, "opening_predicted_low"] <= out.loc[0, "actual_low"] and out.loc[0, "opening_predicted_high"] >= out.loc[0, "actual_high"]),
            "note": "Open-anchored fallback available after the 09:15 open; not a pre-open forecast and not included in the main model metrics."
        },
        "input_candles": int(len(frame)),
        "evaluated_forecasts": int(len(evaluated)),
        "estimator": estimator,
        "min_train": min_train,
        "methods": {name: _metrics(evaluated, low, high, target_coverage) for name, (low, high) in methods.items()},
        "point_forecast": _point_metrics(evaluated),
        "calibration_note": (
            "Online conformal expansion uses only earlier trained-model forecast misses after a 20-row calibration warm-up. "
            "During warm-up, a historical-excursion envelope from completed candles prevents zero-width/zero-expansion under-calibration; the source is recorded per row. "
            "The asymmetric conformal variant estimates lower and upper misses separately with finite-sample tail quantiles after warm-up. "
            "Historical asymmetric quantiles use a fixed equal-tail quantile level from completed prior excursions; no rolling grid search is used. "
            "All main methods are evaluated on later candles only. Metrics are out-of-sample walk-forward results, not proof of future performance."
        ),
    }
    return out, summary
