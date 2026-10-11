"""Leakage-aware one-candle-ahead OHLC range forecasting.

The model predicts the next candle's low and high from information available
at the close of the current candle. Walk-forward fitting uses an expanding
window: a row is predicted only by models trained on earlier rows.
This is research tooling, not a guarantee of profitable trades.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge


FEATURE_COLUMNS = (
    "ret_1", "ret_2", "ret_5", "range_pct", "body_pct",
    "upper_wick_pct", "lower_wick_pct", "volatility_5",
    "volatility_15", "momentum_5", "momentum_15",
    "ema_gap_9_21", "close_vs_sma_20", "volume_ratio",
)
REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close"}


@dataclass(frozen=True)
class LowHighForecastResult:
    predictions: pd.DataFrame
    metrics: dict[str, float | int | str]
    status: str


def _prepare(data: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "volume" not in frame:
        frame["volume"] = 0.0
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = frame.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    valid = (
        (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
        & (frame["low"] > 0)
        & (frame["close"] > 0)
        & (frame["open"] > 0)
    )
    frame = frame.loc[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid OHLC rows remain after data-quality checks")
    return frame


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    close, open_, high, low, volume = (frame[c].astype(float) for c in ("close", "open", "high", "low", "volume"))
    returns = close.pct_change()
    true_range_pct = (high - low) / close.replace(0, np.nan)
    body_pct = (close - open_) / close.replace(0, np.nan)
    feats = pd.DataFrame(index=frame.index)
    feats["ret_1"] = returns
    feats["ret_2"] = close.pct_change(2)
    feats["ret_5"] = close.pct_change(5)
    feats["range_pct"] = true_range_pct
    feats["body_pct"] = body_pct
    feats["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close.replace(0, np.nan)
    feats["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close.replace(0, np.nan)
    feats["volatility_5"] = returns.rolling(5, min_periods=3).std()
    feats["volatility_15"] = returns.rolling(15, min_periods=5).std()
    feats["momentum_5"] = close / close.shift(5) - 1
    feats["momentum_15"] = close / close.shift(15) - 1
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    feats["ema_gap_9_21"] = (ema9 - ema21) / close.replace(0, np.nan)
    feats["close_vs_sma_20"] = close / close.rolling(20, min_periods=5).mean() - 1
    volume_mean = volume.rolling(20, min_periods=5).mean()
    feats["volume_ratio"] = (volume / volume_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return feats.replace([np.inf, -np.inf], np.nan)


def _adaptive_excursion_quantiles(values: np.ndarray, recent_window: int = 40) -> tuple[float, float]:
    """Return recency-weighted q80/q95 excursion sizes using past observations only.

    A short recent window reacts to volatility regime changes faster than a
    full-session quantile, while the full history retains a stabilizing anchor.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.maximum(values, 0.0)
    if not len(values):
        return 0.0, 0.0
    recent = values[-max(1, recent_window):]
    recent_q80 = float(np.quantile(recent, 0.80))
    recent_q95 = float(np.quantile(recent, 0.95))
    global_q80 = float(np.quantile(values, 0.80))
    global_q95 = float(np.quantile(values, 0.95))
    # Recent conditions get more weight, while longer history damps noise.
    q80 = 0.75 * recent_q80 + 0.25 * global_q80
    q95 = 0.75 * recent_q95 + 0.25 * global_q95
    return q80, max(q80, q95)


def _direction_label(value: float, reference: float, tolerance: float = 0.015) -> str:
    """Label price direction; keep tiny moves neutral to avoid noisy labels."""
    delta = float(value) - float(reference)
    if delta > tolerance + 1e-12:
        return "UP"
    if delta < -tolerance - 1e-12:
        return "DOWN"
    return "FLAT"


def _model(seed: int, estimator: str) -> Any:
    if estimator == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), Ridge(alpha=2.0))
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            loss="squared_error", max_iter=80, max_leaf_nodes=7,
            learning_rate=0.06, l2_regularization=1.0,
            random_state=seed,
        ),
    )


def walk_forward_low_high(
    data: pd.DataFrame,
    *,
    min_train: int = 80,
    step: int = 1,
    estimator: str = "hist_gradient_boosting",
    seed: int = 42,
) -> LowHighForecastResult:
    """Predict each next candle's low/high with no future rows in training.

    Row t predicts candle t+1 from candle t's known features. At each forecast
    origin, training labels are restricted to candles already observed.
    """
    if min_train < 20:
        raise ValueError("min_train must be at least 20")
    if step != 1:
        raise ValueError("step must be 1; this forecaster emits a prediction for every candle")
    if estimator not in {"hist_gradient_boosting", "ridge"}:
        raise ValueError("estimator must be 'hist_gradient_boosting' or 'ridge'")
    frame = _prepare(data)
    features = _features(frame)
    # Predict relative moves from the origin close, not raw prices.
    origin_close = frame["close"].replace(0, np.nan)
    targets = pd.DataFrame({
        "low_return": frame["low"].shift(-1) / origin_close - 1.0,
        "high_return": frame["high"].shift(-1) / origin_close - 1.0,
        "actual_low": frame["low"].shift(-1),
        "actual_high": frame["high"].shift(-1),
        "close_return": frame["close"].shift(-1) / origin_close - 1.0,
        "actual_close": frame["close"].shift(-1),
        "target_timestamp": frame["timestamp"].shift(-1),
    })
    rows: list[dict[str, Any]] = []
    # Emit one row for every target candle. Before enough labelled history
    # exists, use a clearly marked range-based fallback; never use target high/
    # low to construct its prediction. The first row has no preceding candle,
    # so its open-anchored fallback is explicitly marked as not pre-open.
    for target_idx in range(len(frame)):
        target = frame.iloc[target_idx]
        if target_idx == 0:
            anchor = float(target["open"])
            band = max(anchor * 0.005, 0.01)
            predicted_low, predicted_high = anchor - band, anchor + band
            point_low, point_high = anchor - band / 2.0, anchor + band / 2.0
            predicted_close = anchor
            method = "OPEN_FALLBACK_NO_PRIOR_HISTORY"
            origin_ts = pd.NaT
        else:
            origin_idx = target_idx - 1
            anchor = float(frame.iloc[origin_idx]["close"])
            origin_ts = frame.iloc[origin_idx]["timestamp"]
            prior = frame.iloc[:target_idx]
            prior_ranges = (prior["high"] - prior["low"]).tail(20)
            band = float(prior_ranges.median()) if not prior_ranges.empty else max(anchor * 0.005, 0.01)
            band = max(band / 2.0, anchor * 0.0005, 0.01)
            method = "HISTORICAL_RANGE_FALLBACK"
            if origin_idx >= min_train and not features.iloc[[origin_idx]].isna().any(axis=None):
                x_train = features.iloc[:origin_idx]
                y_low = targets["low_return"].iloc[:origin_idx]
                y_high = targets["high_return"].iloc[:origin_idx]
                y_close = targets["close_return"].iloc[:origin_idx]
                valid = x_train.notna().all(axis=1) & y_low.notna() & y_high.notna() & y_close.notna()
                if int(valid.sum()) >= min_train:
                    x_now = features.iloc[[origin_idx]]
                    low_model = _model(seed, estimator)
                    high_model = _model(seed + 1, estimator)
                    close_model = _model(seed + 2, estimator)
                    low_model.fit(x_train.loc[valid], y_low.loc[valid])
                    high_model.fit(x_train.loc[valid], y_high.loc[valid])
                    close_model.fit(x_train.loc[valid], y_close.loc[valid])
                    raw_low = anchor * (1.0 + float(low_model.predict(x_now)[0]))
                    raw_high = anchor * (1.0 + float(high_model.predict(x_now)[0]))
                    predicted_close = anchor * (1.0 + float(close_model.predict(x_now)[0]))
                    # Preserve raw point estimates separately from the wider
                    # calibrated interval. Bounds are sorted to prevent an
                    # inverted low/high pair if regressors cross.
                    point_low, point_high = sorted((raw_low, raw_high))

                    # Calibrate the model's excursions using only completed
                    # candles. This shrinks extreme regression outputs toward
                    # robust empirical excursion quantiles and caps runaway
                    # bounds; it never reads the target candle's OHLC.
                    hist = frame.iloc[1:target_idx]
                    hist_anchor = frame["close"].iloc[:target_idx - 1].to_numpy(dtype=float)
                    down_excursions = np.maximum(hist_anchor - hist["low"].to_numpy(dtype=float), 0.0)
                    up_excursions = np.maximum(hist["high"].to_numpy(dtype=float) - hist_anchor, 0.0)
                    if len(down_excursions) >= 20:
                        # Recency-weighted quantiles adapt faster than a whole-session
                        # quantile when the option's volatility regime changes.
                        down_q80, down_q95 = _adaptive_excursion_quantiles(down_excursions)
                        up_q80, up_q95 = _adaptive_excursion_quantiles(up_excursions)
                    else:
                        down_q80 = up_q80 = band
                        down_q95 = up_q95 = max(2.0 * band, band)
                    model_down = max(0.0, anchor - raw_low)
                    model_up = max(0.0, raw_high - anchor)
                    # A 50/50 blend dampens unstable tails; quantile caps
                    # prevent the model from emitting excessively wide bounds.
                    down = min(0.5 * model_down + 0.5 * down_q80, max(down_q95, down_q80, anchor * 0.0005))
                    up = min(0.5 * model_up + 0.5 * up_q80, max(up_q95, up_q80, anchor * 0.0005))
                    predicted_low = anchor - down
                    predicted_high = anchor + up
                    method = f"CALIBRATED_MODEL_{estimator.upper()}"
                else:
                    predicted_low, predicted_high = anchor - band, anchor + band
                    point_low, point_high = anchor - band / 2.0, anchor + band / 2.0
                    predicted_close = anchor
                    method = "HISTORICAL_RANGE_FALLBACK"
            else:
                predicted_low, predicted_high = anchor - band, anchor + band
                point_low, point_high = anchor - band / 2.0, anchor + band / 2.0
                predicted_close = anchor
        predicted_low = max(0.01, min(predicted_low, predicted_high))
        predicted_high = max(predicted_low, predicted_high)
        actual_low = float(target["low"])
        actual_high = float(target["high"])
        rows.append({
            "origin_timestamp": origin_ts,
            "target_timestamp": target["timestamp"],
            "predicted_low": predicted_low,
            "predicted_high": predicted_high,
            "point_predicted_low": point_low if target_idx > 0 else np.nan,
            "point_predicted_high": point_high if target_idx > 0 else np.nan,
            "predicted_close": predicted_close,
            "predicted_direction": _direction_label(predicted_close, anchor),
            "actual_close": float(target["close"]),
            "actual_direction": _direction_label(float(target["close"]), anchor),
            "actual_low": actual_low,
            "actual_high": actual_high,
            "low_abs_error": abs(predicted_low - actual_low),
            "high_abs_error": abs(predicted_high - actual_high),
            "low_covered": predicted_low <= actual_low,
            "high_covered": predicted_high >= actual_high,
            "full_range_covered": predicted_low <= actual_low and predicted_high >= actual_high,
            "predicted_width": predicted_high - predicted_low,
            "actual_width": actual_high - actual_low,
            "forecast_method": method,
        })
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return LowHighForecastResult(predictions, {"rows": 0}, "INSUFFICIENT_DATA")
    # Score pre-open forecasts separately from the first candle's open-anchored
    # estimate. The first row is still emitted, but it must not inflate or distort
    # the main walk-forward metrics.
    scored = predictions.loc[predictions["origin_timestamp"].notna()].copy()
    opening = predictions.iloc[0]
    if scored.empty:
        return LowHighForecastResult(predictions, {
            "rows": int(len(predictions)),
            "expected_rows": int(len(frame)),
            "forecast_rows_complete": bool(len(predictions) == len(frame)),
            "evaluated_forecast_rows": 0,
            "opening_low_abs_error": float(opening["low_abs_error"]),
            "opening_high_abs_error": float(opening["high_abs_error"]),
            "opening_full_range_covered": int(bool(opening["full_range_covered"])),
            "status": "INSUFFICIENT_DATA",
        }, "INSUFFICIENT_DATA")
    actual_width = float(scored["actual_width"].mean())
    predicted_width = float(scored["predicted_width"].mean())
    point_low_rows = scored.loc[scored["point_predicted_low"].notna()]
    point_high_rows = scored.loc[scored["point_predicted_high"].notna()]
    # Score direction on every genuinely directional target candle. A FLAT
    # prediction on an UP/DOWN target is a miss, not a row to discard. Actual
    # FLAT targets are reported separately and do not enter directional accuracy.
    directional_targets = scored.loc[scored["actual_direction"] != "FLAT"]
    directional_matches = (
        directional_targets["predicted_direction"] == directional_targets["actual_direction"]
    )
    covered_directional_targets = directional_targets.loc[
        directional_targets["predicted_direction"] != "FLAT"
    ]
    active_direction_rows = covered_directional_targets
    conditional_directional_matches = (
        active_direction_rows["predicted_direction"] == active_direction_rows["actual_direction"]
    )
    exact_three_class_matches = (
        scored["predicted_direction"] == scored["actual_direction"]
    )
    metrics: dict[str, float | int | str] = {
        "rows": int(len(predictions)),
        "expected_rows": int(len(frame)),
        "forecast_rows_complete": bool(len(predictions) == len(frame)),
        "evaluated_forecast_rows": int(len(scored)),
        "model_forecast_rows": int(scored["forecast_method"].str.contains("MODEL_").sum()),
        "fallback_forecast_rows": int((~scored["forecast_method"].str.contains("MODEL_")).sum()),
        "low_mae": float(mean_absolute_error(scored["actual_low"], scored["predicted_low"])),
        "high_mae": float(mean_absolute_error(scored["actual_high"], scored["predicted_high"])),
        "point_low_mae": float(mean_absolute_error(point_low_rows["actual_low"], point_low_rows["point_predicted_low"])),
        "point_high_mae": float(mean_absolute_error(point_high_rows["actual_high"], point_high_rows["point_predicted_high"])),
        "close_mae": float(mean_absolute_error(scored["actual_close"], scored["predicted_close"])),
        "directional_accuracy_pct": float(directional_matches.mean() * 100) if len(directional_targets) else 0.0,
        "directional_target_rows": int(len(directional_targets)),
        "directional_correct_rows": int(directional_matches.sum()),
        "directional_target_coverage_pct": float((directional_targets["predicted_direction"] != "FLAT").mean() * 100) if len(directional_targets) else 0.0,
        "conditional_directional_accuracy_pct": float(conditional_directional_matches.mean() * 100) if len(active_direction_rows) else 0.0,
        "conditional_directional_scored_rows": int(len(active_direction_rows)),
        "three_class_accuracy_pct": float(exact_three_class_matches.mean() * 100),
        "directional_prediction_coverage_pct": float((scored["predicted_direction"] != "FLAT").mean() * 100),
        "directional_scored_rows": int(len(directional_targets)),
        "low_coverage_pct": float(scored["low_covered"].mean() * 100),
        "high_coverage_pct": float(scored["high_covered"].mean() * 100),
        "full_range_coverage_pct": float(scored["full_range_covered"].mean() * 100),
        "mean_predicted_width": predicted_width,
        "mean_actual_width": actual_width,
        "width_ratio": predicted_width / max(actual_width, 1e-12),
        "opening_low_abs_error": float(opening["low_abs_error"]),
        "opening_high_abs_error": float(opening["high_abs_error"]),
        "opening_full_range_covered": int(bool(opening["full_range_covered"])),
        "opening_forecast_note": "Open-anchored estimate is scored separately and is not a pre-open forecast.",
        "status": "EVALUATED",
    }
    return LowHighForecastResult(predictions, metrics, "EVALUATED")
