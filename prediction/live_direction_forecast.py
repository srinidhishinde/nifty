"""Live-data next-candle direction forecast; advisory only, never submits orders."""
from __future__ import annotations

from typing import Any

import pandas as pd

from prediction.low_high_forecast import _direction_label, _prepare
from scripts.benchmark_directional_classifier import _classifier, _direction_features


def predict_live_direction(
    data: pd.DataFrame,
    *,
    estimator: str = "logistic",
    min_train: int = 80,
    tolerance: float = 0.01,
    timeframe_minutes: int = 5,
) -> dict[str, Any]:
    """Predict the next candle from the latest available candle and prior outcomes.

    Training labels only use candle pairs that have already completed. The last
    candle is used only as the prediction origin, never as a training target.
    This returns a forecast, not an execution signal or a calibrated probability.
    """
    if min_train < 20:
        raise ValueError("min_train must be at least 20")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    if timeframe_minutes not in (1, 5, 15, 30, 60):
        raise ValueError("timeframe_minutes must be one of 1, 5, 15, 30, or 60")

    frame = _prepare(data)
    if len(frame) < 3:
        raise ValueError("At least three valid candles are required for a live direction forecast")

    features = _direction_features(frame)
    latest_features = features.iloc[[-1]]
    close = frame["close"].astype(float)
    labels = [
        _direction_label(float(close.iloc[i + 1]), float(close.iloc[i]), tolerance)
        for i in range(len(frame) - 1)
    ]
    train_x = features.iloc[:-1].copy()
    train_y = pd.Series(labels, index=train_x.index, dtype="object")
    valid = train_x.notna().all(axis=1) & train_y.isin(["UP", "DOWN"])
    x = train_x.loc[valid]
    y = train_y.loc[valid]

    last_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    fallback = "UP" if last_close >= previous_close else "DOWN"
    method = "MOMENTUM_FALLBACK"
    score = None
    prediction = fallback
    trained_rows = int(len(y))

    if (
        trained_rows >= min_train
        and y.nunique() == 2
        and latest_features.notna().all(axis=None)
    ):
        model = _classifier(estimator, 42)
        model.fit(x, y)
        prediction = str(model.predict(latest_features)[0])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(latest_features)[0]
            score = float(max(probabilities))
        method = f"WALK_FORWARD_TRAINED_{estimator.upper()}"

    latest_time = pd.Timestamp(frame["timestamp"].iloc[-1])
    target_time = latest_time + pd.Timedelta(minutes=timeframe_minutes)
    return {
        "direction": prediction,
        "method": method,
        "estimator": estimator if method.startswith("WALK_FORWARD_TRAINED_") else "momentum_fallback",
        "training_rows": trained_rows,
        "history_rows": int(len(frame)),
        "reference_close": last_close,
        "latest_candle_time": latest_time.isoformat(),
        "target_candle_time": target_time.isoformat(),
        "raw_model_score": score,
        "score_is_calibrated_probability": False,
        "tolerance": float(tolerance),
        "advisory_only": True,
        "live_orders_enabled": False,
    }
