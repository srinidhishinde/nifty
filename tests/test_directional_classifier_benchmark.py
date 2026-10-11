from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.benchmark_directional_classifier import evaluate, load_session


def _session_csv(path, rows: int = 125) -> None:
    rng = np.random.default_rng(20261011)
    returns = rng.normal(0.00005, 0.0015, rows)
    close = 220.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.02, 0.25, rows)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(100, 5000, rows)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-01 09:15:00", periods=rows, freq="min", tz="Asia/Kolkata"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    frame.to_csv(path, index=False)


def test_load_session_normalizes_timestamp_and_validates_ohlc(tmp_path):
    path = tmp_path / "session.csv"
    _session_csv(path)
    frame = load_session(path)
    assert len(frame) == 125
    assert {"timestamp", "open", "high", "low", "close", "volume"} <= set(frame.columns)
    assert frame["timestamp"].is_monotonic_increasing


def test_walk_forward_classifier_emits_all_directional_predictions(tmp_path):
    path = tmp_path / "session.csv"
    _session_csv(path)
    result = evaluate(path, min_train=30, estimator="logistic", tolerance=0.01)
    assert result["directional_targets"] > 80
    assert result["coverage_pct"] == 100.0
    assert result["correct_predictions"] <= result["directional_targets"]
    assert result["walk_forward_classifier_predictions"] > 0
    assert result["fallback_predictions"] > 0
    assert 0.0 <= result["accuracy_pct"] <= 100.0
    assert 0.0 <= result["momentum_baseline_accuracy_pct"] <= 100.0
