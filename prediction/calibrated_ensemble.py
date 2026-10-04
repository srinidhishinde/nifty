from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CalibratedPrediction:
    probability_up: float
    probability_down: float
    uncertainty: float
    model_agreement: float
    sample_size: int


def _target(close: pd.Series, horizon: int = 3, up_threshold: float = 0.002, down_threshold: float = 0.002) -> pd.Series:
    future = close.shift(-horizon) / close - 1.0
    target = pd.Series(np.nan, index=close.index)
    target[future >= up_threshold] = 1.0
    target[future <= -down_threshold] = 0.0
    return target


def predict_calibrated(
    features: pd.DataFrame,
    *,
    feature_columns: list[str],
    horizon: int = 3,
    up_threshold: float = 0.002,
    down_threshold: float = 0.002,
    min_train: int = 80,
) -> CalibratedPrediction:
    frame = features.copy()
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = _target(pd.to_numeric(frame["close"], errors="coerce"), horizon, up_threshold, down_threshold)
    valid = x.notna().all(axis=1) & y.notna()
    train = x.loc[valid]
    labels = y.loc[valid].astype(int)
    if len(train) < min_train or labels.nunique() < 2:
        return CalibratedPrediction(0.5, 0.5, 1.0, 0.0, len(train))

    base = VotingClassifier(
        estimators=[
            ("logit", make_pipeline(StandardScaler(), LogisticRegression(max_iter=600, random_state=42))),
            ("gb", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=7, random_state=42)),
        ],
        voting="soft",
        weights=[1.0, 1.0],
    )
    folds = min(5, max(2, len(train) // 30))
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=folds),
    )
    calibrated.fit(train, labels)

    now = x.iloc[[-1]]
    if now.isna().any(axis=None):
        return CalibratedPrediction(0.5, 0.5, 1.0, 0.0, len(train))

    p_up = float(calibrated.predict_proba(now)[0, 1])
    uncertainty = float(1.0 - abs(p_up - 0.5) * 2.0)
    model_agreement = 1.0
    return CalibratedPrediction(
        probability_up=round(p_up, 4),
        probability_down=round(1.0 - p_up, 4),
        uncertainty=round(float(np.clip(uncertainty, 0.0, 1.0)), 4),
        model_agreement=round(model_agreement, 4),
        sample_size=len(train),
    )
