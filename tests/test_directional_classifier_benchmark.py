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
    assert result["evaluated_targets"] == result["session_rows"] - 2
    assert result["correct_predictions"] <= result["evaluated_targets"]
    assert result["accuracy_pct"] == round(
        100.0 * result["correct_predictions"] / result["evaluated_targets"], 4
    )
    assert result["walk_forward_classifier_predictions"] > 0
    assert result["fallback_predictions"] > 0
    assert 0.0 <= result["accuracy_pct"] <= 100.0
    assert 0.0 <= result["momentum_baseline_accuracy_pct"] <= 100.0



def test_flat_actual_candles_are_not_removed_from_accuracy_denominator(tmp_path):
    path = tmp_path / "flat_session.csv"
    rows = 12
    pd.DataFrame({
        "timestamp": pd.date_range("2026-10-02 09:15:00", periods=rows, freq="min"),
        "open": [100.0] * rows,
        "high": [100.5] * rows,
        "low": [99.5] * rows,
        "close": [100.0] * rows,
        "volume": [1000] * rows,
    }).to_csv(path, index=False)
    result = evaluate(path, min_train=30, estimator="logistic", tolerance=0.01)
    assert result["evaluated_targets"] == rows - 2
    assert result["directional_targets"] == 0
    assert result["flat_actual_targets"] == rows - 2
    assert result["accuracy_pct"] == 0.0
    assert result["coverage_pct"] == 100.0



def test_direction_features_are_causal_and_include_regime_features(tmp_path):
    from scripts.benchmark_directional_classifier import _direction_features

    path = tmp_path / "features.csv"
    _session_csv(path, rows=150)
    frame = load_session(path)
    features = _direction_features(frame)

    required = {
        "ret_3", "ret_10", "return_acceleration", "atr_pct_14",
        "range_to_atr", "close_location", "gap_pct",
        "volatility_ratio_5_20", "volume_zscore_20", "rsi_14",
        "session_time_sin", "session_time_cos",
    }
    assert required <= set(features.columns)
    assert len(features) == len(frame)
    assert features["rsi_14"].between(0, 100).all()
    assert features["session_time_sin"].between(-1, 1).all()
    assert features["session_time_cos"].between(-1, 1).all()

    changed = frame.copy()
    changed.loc[100:, ["open", "high", "low", "close", "volume"]] *= 1.5
    changed_features = _direction_features(changed)
    pd.testing.assert_frame_equal(
        features.iloc[:100].reset_index(drop=True),
        changed_features.iloc[:100].reset_index(drop=True),
        check_dtype=False,
    )
