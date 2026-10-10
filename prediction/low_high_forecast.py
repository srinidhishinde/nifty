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
    if step < 1:
        raise ValueError("step must be >= 1")
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
        "target_timestamp": frame["timestamp"].shift(-1),
    })
    rows: list[dict[str, Any]] = []
    for i in range(min_train, len(frame) - 1, step):
        x_train = features.iloc[:i]
        y_low = targets["low_return"].iloc[:i]
        y_high = targets["high_return"].iloc[:i]
        valid = x_train.notna().all(axis=1) & y_low.notna() & y_high.notna()
        if int(valid.sum()) < min_train or features.iloc[[i]].isna().any(axis=None):
            continue
        x_now = features.iloc[[i]]
        low_model = _model(seed, estimator)
        high_model = _model(seed + 1, estimator)
        low_model.fit(x_train.loc[valid], y_low.loc[valid])
        high_model.fit(x_train.loc[valid], y_high.loc[valid])
        anchor_close = float(frame.iloc[i]["close"])
        predicted_low = anchor_close * (1.0 + float(low_model.predict(x_now)[0]))
        predicted_high = anchor_close * (1.0 + float(high_model.predict(x_now)[0]))
        predicted_low, predicted_high = min(predicted_low, predicted_high), max(predicted_low, predicted_high)
        actual_low = float(targets.iloc[i]["actual_low"])
        actual_high = float(targets.iloc[i]["actual_high"])
        rows.append({
            "origin_timestamp": frame.iloc[i]["timestamp"],
            "target_timestamp": targets.iloc[i]["target_timestamp"],
            "predicted_low": predicted_low,
            "predicted_high": predicted_high,
            "actual_low": actual_low,
            "actual_high": actual_high,
            "low_abs_error": abs(predicted_low - actual_low),
            "high_abs_error": abs(predicted_high - actual_high),
            "low_covered": predicted_low <= actual_low,
            "high_covered": predicted_high >= actual_high,
            "full_range_covered": predicted_low <= actual_low and predicted_high >= actual_high,
            "predicted_width": predicted_high - predicted_low,
            "actual_width": actual_high - actual_low,
        })
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return LowHighForecastResult(predictions, {"rows": 0}, "INSUFFICIENT_DATA")
    actual_width = float(predictions["actual_width"].mean())
    predicted_width = float(predictions["predicted_width"].mean())
    metrics: dict[str, float | int | str] = {
        "rows": int(len(predictions)),
        "low_mae": float(mean_absolute_error(predictions["actual_low"], predictions["predicted_low"])),
        "high_mae": float(mean_absolute_error(predictions["actual_high"], predictions["predicted_high"])),
        "low_coverage_pct": float(predictions["low_covered"].mean() * 100),
        "high_coverage_pct": float(predictions["high_covered"].mean() * 100),
        "full_range_coverage_pct": float(predictions["full_range_covered"].mean() * 100),
        "mean_predicted_width": predicted_width,
        "mean_actual_width": actual_width,
        "width_ratio": predicted_width / max(actual_width, 1e-12),
        "status": "EVALUATED",
    }
    return LowHighForecastResult(predictions, metrics, "EVALUATED")
