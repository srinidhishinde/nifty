from __future__ import annotations

"""Leakage-aware walk-forward benchmark for one-candle-ahead direction.

Each forecast for candle t uses features available at candle t-1 close.
Training labels are outcomes of candles already closed before the forecast.
This is a research benchmark; it does not guarantee accuracy or profitability.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction.low_high_forecast import FEATURE_COLUMNS, _features, _prepare, _direction_label


def load_session(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    for alias in ("datetime", "date_time", "time", "date"):
        if alias in frame.columns and "timestamp" not in frame.columns:
            frame = frame.rename(columns={alias: "timestamp"})
            break
    return _prepare(frame)


def _classifier(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(),
            LogisticRegression(C=0.25, class_weight="balanced", max_iter=500, random_state=seed),
        )
    if name == "hist_gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=80, max_leaf_nodes=7, learning_rate=0.04,
                l2_regularization=2.0, random_state=seed,
            ),
        )
    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=120, max_depth=4, min_samples_leaf=12,
                class_weight="balanced_subsample", random_state=seed, n_jobs=1,
            ),
        )
    raise ValueError(f"Unsupported classifier: {name}")


def evaluate(path: Path, *, min_train: int, estimator: str, tolerance: float) -> dict:
    frame = load_session(path)
    features = _features(frame).loc[:, list(FEATURE_COLUMNS)]
    # A training row at origin j has label from candle j+1, never candle j.
    labels = []
    for j in range(len(frame) - 1):
        labels.append(_direction_label(float(frame.loc[j + 1, "close"]), float(frame.loc[j, "close"]), tolerance))
    labels.append(None)
    label_series = pd.Series(labels, index=frame.index, dtype="object")
    actual = [
        _direction_label(float(frame.loc[i, "close"]), float(frame.loc[i - 1, "close"]), tolerance)
        for i in range(1, len(frame))
    ]
    predictions = []
    scored_actual = []
    baseline = []
    forecast_methods = []
    # Target candle i is predicted from features at i-1. At that time, the
    # latest trainable origin is i-2 because its target candle i-1 has closed.
    # Start at target index 2: the first target has no prior close-to-close
    # move from which to form a valid momentum fallback.
    for target_idx in range(2, len(frame)):
        origin_idx = target_idx - 1
        train_end = origin_idx  # exclusive; origins [0, origin_idx) only
        train_x = features.iloc[:train_end]
        train_y = label_series.iloc[:train_end]
        valid = train_x.notna().all(axis=1) & train_y.isin(["UP", "DOWN"])
        y = train_y.loc[valid]
        x = train_x.loc[valid]
        prediction = None
        method = "INSUFFICIENT_HISTORY"
        if len(y) >= min_train and y.nunique() == 2 and features.iloc[[origin_idx]].notna().all(axis=None):
            model = _classifier(estimator, 42)
            model.fit(x, y)
            prediction = str(model.predict(features.iloc[[origin_idx]])[0])
            method = "WALK_FORWARD_CLASSIFIER"
        else:
            # Deterministic fallback, explicitly included in all metrics.
            # It uses only the latest completed close-to-close move.
            if origin_idx >= 1:
                prediction = _direction_label(
                    float(frame.loc[origin_idx, "close"]),
                    float(frame.loc[origin_idx - 1, "close"]),
                    tolerance,
                )
            if prediction == "FLAT":
                prediction = "UP" if float(frame.loc[origin_idx, "close"]) >= float(frame.loc[origin_idx - 1, "close"]) else "DOWN"
            method = "MOMENTUM_FALLBACK"
        if prediction not in ("UP", "DOWN"):
            prediction = "UP" if float(frame.loc[origin_idx, "close"]) >= float(frame.loc[origin_idx - 1, "close"]) else "DOWN"
        predictions.append(prediction)
        scored_actual.append(actual[target_idx - 1])
        baseline_direction = _direction_label(
            float(frame.loc[origin_idx, "close"]),
            float(frame.loc[origin_idx - 1, "close"]),
            tolerance,
        )
        if baseline_direction == "FLAT":
            baseline_direction = "UP" if float(frame.loc[origin_idx, "close"]) >= float(frame.loc[origin_idx - 1, "close"]) else "DOWN"
        baseline.append(baseline_direction)
        forecast_methods.append(method)

    actual_arr = np.asarray(scored_actual, dtype=object)
    pred_arr = np.asarray(predictions, dtype=object)
    baseline_arr = np.asarray(baseline, dtype=object)
    directional = np.isin(actual_arr, ["UP", "DOWN"])
    actual_dir, pred_dir = actual_arr[directional], pred_arr[directional]
    classifier_rows = np.asarray(forecast_methods, dtype=object)[directional] == "WALK_FORWARD_CLASSIFIER"
    all_correct = int((pred_arr == actual_arr).sum())
    all_targets = int(len(actual_arr))
    directional_correct = int((pred_dir == actual_dir).sum())
    return {
        "file": str(path),
        "session_rows": int(len(frame)),
        "evaluated_targets": all_targets,
        "directional_targets": int(directional.sum()),
        "correct_predictions": all_correct,
        "accuracy_pct": round(float(all_correct / all_targets * 100), 4) if all_targets else 0.0,
        "directional_accuracy_pct": round(float(directional_correct / len(actual_dir) * 100), 4) if len(actual_dir) else 0.0,
        "coverage_pct": round(float(len(pred_arr) / all_targets * 100), 4) if all_targets else 0.0,
        "momentum_baseline_accuracy_pct": round(float((baseline_arr == actual_arr).mean() * 100), 4) if all_targets else 0.0,
        "momentum_baseline_directional_accuracy_pct": round(float((baseline_arr[directional] == actual_dir).mean() * 100), 4) if len(actual_dir) else 0.0,
        "walk_forward_classifier_predictions": int(classifier_rows.sum()),
        "fallback_predictions": int((~(np.asarray(forecast_methods, dtype=object) == "WALK_FORWARD_CLASSIFIER")).sum()),
        "flat_actual_targets": int((~directional).sum()),
        "label_tolerance": tolerance,
        "prediction_policy": "always UP/DOWN; flat actual outcomes count as misses for binary forecasts",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--min-train", type=int, default=80)
    parser.add_argument("--estimator", choices=("logistic", "hist_gradient_boosting", "random_forest"), default="logistic")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--min-accuracy", type=float, default=80.0)
    parser.add_argument("--min-coverage", type=float, default=80.0)
    parser.add_argument("--min-targets", type=int, default=500)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--output", default="artifacts/directional_classifier_benchmark.json")
    args = parser.parse_args()
    if args.min_train < 20:
        parser.error("--min-train must be at least 20")
    if args.tolerance < 0:
        parser.error("--tolerance cannot be negative")
    paths = [Path(p) for p in args.input]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        parser.error(f"Input CSV file(s) not found: {missing}")
    sessions = [evaluate(p, min_train=args.min_train, estimator=args.estimator, tolerance=args.tolerance) for p in paths]
    targets = sum(s["evaluated_targets"] for s in sessions)
    directional_targets = sum(s["directional_targets"] for s in sessions)
    correct = sum(s["correct_predictions"] for s in sessions)
    accuracy = 100.0 * correct / targets if targets else 0.0
    predicted = sum(round(s["coverage_pct"] * s["evaluated_targets"] / 100.0) for s in sessions)
    coverage = 100.0 * predicted / targets if targets else 0.0
    baseline_correct = sum(round(s["momentum_baseline_accuracy_pct"] * s["evaluated_targets"] / 100.0) for s in sessions)
    baseline_accuracy = 100.0 * baseline_correct / targets if targets else 0.0
    criteria = {
        "minimum_accuracy_pass": accuracy >= args.min_accuracy,
        "minimum_prediction_coverage_pass": coverage >= args.min_coverage,
        "minimum_target_count_pass": targets >= args.min_targets,
        "minimum_session_count_pass": len(sessions) >= args.min_sessions,
    }
    report = {
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "benchmark_type": "chronological_expanding_walk_forward_direction_classifier",
        "estimator": args.estimator,
        "min_train": args.min_train,
        "label_tolerance": args.tolerance,
        "thresholds": {"accuracy_pct": args.min_accuracy, "coverage_pct": args.min_coverage, "evaluated_targets": args.min_targets, "sessions": args.min_sessions},
        "aggregate": {"evaluated_targets": targets, "directional_targets": directional_targets, "correct_predictions": correct, "accuracy_pct": round(accuracy, 4), "coverage_pct": round(coverage, 4), "momentum_baseline_accuracy_pct": round(baseline_accuracy, 4)},
        "criteria": criteria,
        "sessions": sessions,
        "live_enablement": "LOCKED",
        "note": "A passing directional benchmark is not evidence of profitability and does not authorize live orders. No model can guarantee 80% accuracy.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if args.require_pass and report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
