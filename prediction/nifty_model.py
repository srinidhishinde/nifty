from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features.technical.indicators import add_indicators


FEATURES = ["RSI", "ema_spread", "macd_hist", "bb_pos", "vwap_dev", "vol_ratio", "global_news"]


@dataclass(frozen=True)
class WalkForwardResult:
    accuracy_pct: float
    predictions: pd.DataFrame


def _features(data: pd.DataFrame) -> pd.DataFrame:
    frame = add_indicators(data) if "EMA20" not in data.columns else data.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame.get("volume", 0.0), errors="coerce")
    frame["ema_spread"] = (frame["EMA20"] - frame["EMA50"]) / close.replace(0, np.nan)
    frame["macd_hist"] = frame["MACD"] - frame["MACD_SIGNAL"]
    frame["bb_pos"] = (close - frame["BB_MIDDLE"]) / (frame["BB_UPPER"] - frame["BB_LOWER"]).replace(0, np.nan)
    frame["vwap_dev"] = (close - frame["VWAP"]) / frame["VWAP"].replace(0, np.nan)
    frame["vol_ratio"] = volume / frame["VOLUME_MA20"].replace(0, np.nan)
    if "global_news" not in frame:
        frame["global_news"] = 0.0
    frame["global_news"] = pd.to_numeric(frame["global_news"], errors="coerce").fillna(0.0)
    return frame


def walk_forward_predict(data: pd.DataFrame, min_train: int = 30) -> WalkForwardResult:
    required = {"timestamp", "close", "high", "low"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = _features(data.copy())
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["target"] = (frame["close"].shift(-1) > frame["close"]).astype(int)
    rows = []
    for i in range(min_train, len(frame) - 1):
        train = frame.iloc[:i]
        x_train = train[FEATURES]
        y_train = train["target"]
        valid = x_train.notna().all(axis=1) & y_train.notna()
        if valid.sum() < min_train or y_train[valid].nunique() < 2:
            continue
        x_now = frame.iloc[[i]][FEATURES]
        if x_now.isna().any(axis=None):
            continue
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(x_train.loc[valid], y_train.loc[valid])
        probability = float(model.predict_proba(x_now)[0, 1])
        prediction = "UP" if probability >= 0.5 else "DOWN"
        actual = "UP" if int(frame.iloc[i]["target"]) == 1 else "DOWN"
        rows.append({
            "Timestamp": frame.iloc[i]["timestamp"],
            "Prediction": prediction,
            "Probability Up": round(probability * 100, 2),
            "Actual": actual,
            "Correct": prediction == actual,
        })
    result = pd.DataFrame(rows)
    accuracy = float(result["Correct"].mean() * 100) if not result.empty else 0.0
    return WalkForwardResult(round(accuracy, 2), result)
