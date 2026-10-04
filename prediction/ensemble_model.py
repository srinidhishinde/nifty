from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features.technical.indicators import add_indicators

FEATURES = [
    "RSI", "EMA_SPREAD", "MACD_HIST", "BB_POS", "VWAP_DEV", "VOLUME_RATIO",
    "ADX", "STOCH_K", "ROC", "ATR_PCT", "candlestick_score", "global_news",
]


@dataclass(frozen=True)
class EnsemblePrediction:
    direction: str
    probability_up: float
    confidence: float
    abstain: bool
    reason: str


@dataclass(frozen=True)
class WalkForwardEnsembleResult:
    accuracy_pct: float
    coverage_pct: float
    predictions: pd.DataFrame


def _prepare(data: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = add_indicators(data.copy())
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["global_news"] = (
        pd.to_numeric(frame["global_news"], errors="coerce").fillna(0.0)
        if "global_news" in frame.columns else 0.0
    )
    frame["target"] = (frame["close"].shift(-1) > frame["close"]).astype(float)
    return frame


def _models():
    return (
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7, learning_rate=0.05, l2_regularization=0.5, random_state=42),
    )


def _predict(models, x: pd.DataFrame) -> float:
    return float(np.mean([float(m.predict_proba(x)[0, 1]) for m in models]))


def _calibrate(models, calibration: pd.DataFrame) -> tuple[float, float]:
    if len(calibration) < 12 or calibration["target"].nunique() < 2:
        return 1.0, 0.0
    raw = np.array([_predict(models, calibration.iloc[[i]][FEATURES]) for i in range(len(calibration))])
    y = calibration["target"].astype(int).to_numpy()
    logits = np.log(np.clip(raw, 1e-5, 1-1e-5) / np.clip(1-raw, 1e-5, 1-1e-5)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    calibrator.fit(logits, y)
    return float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])


def _apply_calibration(p: float, slope: float, intercept: float) -> float:
    if slope == 1.0 and intercept == 0.0:
        return float(p)
    logit = np.log(np.clip(p, 1e-5, 1-1e-5) / np.clip(1-p, 1e-5, 1-1e-5))
    z = slope * logit + intercept
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _prediction(models, train: pd.DataFrame, current: pd.DataFrame) -> float:
    split = max(12, int(len(train) * 0.20))
    fit, calibration = train.iloc[:-split], train.iloc[-split:]
    if len(fit) < 30 or fit["target"].nunique() < 2:
        return _predict(models, current)
    # Models are fitted on the calibration-free historical portion.
    for model in models:
        model.fit(fit[FEATURES], fit["target"].astype(int))
    slope, intercept = _calibrate(models, calibration)
    return _apply_calibration(_predict(models, current[FEATURES]), slope, intercept)


def walk_forward_ensemble(data: pd.DataFrame, min_train: int = 60, confidence_threshold: float = 0.60) -> WalkForwardEnsembleResult:
    frame = _prepare(data)
    rows = []
    for i in range(min_train, len(frame) - 1):
        train = frame.iloc[:i].dropna(subset=FEATURES + ["target"])
        current = frame.iloc[[i]]
        if len(train) < min_train or train["target"].nunique() < 2 or current[FEATURES].isna().any(axis=None):
            continue
        models = _models()
        p = _prediction(models, train, current)
        direction = "UP" if p >= 0.5 else "DOWN"
        confidence = max(p, 1-p)
        abstain = confidence < confidence_threshold
        actual = "UP" if int(frame.iloc[i]["target"]) == 1 else "DOWN"
        rows.append({
            "Timestamp": frame.iloc[i]["timestamp"], "Prediction": direction,
            "Probability Up": round(p*100,2), "Confidence": round(confidence*100,2),
            "Abstain": abstain, "Actual": actual, "Correct": direction == actual,
            "Global News": float(frame.iloc[i]["global_news"]),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return WalkForwardEnsembleResult(0.0,0.0,result)
    scored = result[~result["Abstain"]]
    return WalkForwardEnsembleResult(
        round(float(scored["Correct"].mean()*100),2) if not scored.empty else 0.0,
        round(float(len(scored)/len(result)*100),2),
        result,
    )


def predict_latest(data: pd.DataFrame, global_news_score: float = 0.0, min_train: int = 60, confidence_threshold: float = 0.60) -> EnsemblePrediction:
    frame = _prepare(data)
    frame["global_news"] = float(global_news_score)
    train = frame.iloc[:-1].dropna(subset=FEATURES + ["target"])
    current = frame.iloc[[-1]]
    if len(train) < min_train or train["target"].nunique() < 2 or current[FEATURES].isna().any(axis=None):
        return EnsemblePrediction("UNAVAILABLE",0.5,0.0,True,"Insufficient clean historical training data")
    models = _models()
    p = _prediction(models, train, current)
    direction = "UP" if p >= 0.5 else "DOWN"
    confidence = max(p,1-p)
    abstain = confidence < confidence_threshold
    reason = "Calibrated ensemble consensus" if not abstain else "Calibrated model confidence below trade threshold"
    return EnsemblePrediction(direction,round(p,4),round(confidence*100,2),abstain,reason)
