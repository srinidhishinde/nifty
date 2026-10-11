from __future__ import annotations

"""Strict walk-forward directional benchmark for one-minute OHLCV sessions.

This is a research acceptance gate, not a live-trading switch. Every session is
evaluated independently and each forecast is made before its target candle.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction.low_high_forecast import _direction_label, walk_forward_low_high


def load_session(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    aliases = {"datetime": "timestamp", "date_time": "timestamp", "time": "timestamp", "date": "timestamp"}
    for old, new in aliases.items():
        if old in frame.columns and new not in frame.columns:
            frame = frame.rename(columns={old: new})
            break
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required OHLC columns: {sorted(missing)}")
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
    if len(frame) < 100:
        raise ValueError(f"{path}: only {len(frame)} valid candles; at least 100 required")
    return frame


def evaluate(path: Path, min_train: int, estimator: str) -> dict:
    frame = load_session(path)
    result = walk_forward_low_high(frame, min_train=min_train, estimator=estimator)
    predictions = result.predictions
    # Exclude the first opening-anchored row and genuinely flat actual moves.
    scored = predictions.loc[
        predictions["origin_timestamp"].notna() & predictions["actual_direction"].isin(["UP", "DOWN"])
    ].copy()
    target_rows = len(scored)
    correct = int((scored["predicted_direction"] == scored["actual_direction"]).sum())
    coverage_rows = int(scored["predicted_direction"].isin(["UP", "DOWN"]).sum())
    covered = scored.loc[scored["predicted_direction"].isin(["UP", "DOWN"])]
    conditional_correct = int((covered["predicted_direction"] == covered["actual_direction"]).sum())

    # A naive momentum baseline predicts the previous candle's close direction.
    # It is scored on exactly the same target rows and with the same neutral band.
    baseline_correct = 0
    baseline_rows = 0
    for idx in scored.index:
        if idx < 2:
            continue
        baseline = _direction_label(float(frame.loc[idx - 1, "close"]), float(frame.loc[idx - 2, "close"]))
        baseline_rows += 1
        baseline_correct += int(baseline == predictions.loc[idx, "actual_direction"])

    return {
        "file": str(path),
        "session_rows": int(len(frame)),
        "evaluated_directional_targets": int(target_rows),
        "covered_directional_predictions": int(coverage_rows),
        "correct_directional_predictions": correct,
        "directional_accuracy_pct": round(100.0 * correct / target_rows, 4) if target_rows else 0.0,
        "prediction_coverage_pct": round(100.0 * coverage_rows / target_rows, 4) if target_rows else 0.0,
        "conditional_accuracy_pct": round(100.0 * conditional_correct / coverage_rows, 4) if coverage_rows else 0.0,
        "momentum_baseline_rows": baseline_rows,
        "momentum_baseline_accuracy_pct": round(100.0 * baseline_correct / baseline_rows, 4) if baseline_rows else None,
        "walk_forward_model_rows": int(result.metrics.get("model_forecast_rows", 0)),
        "fallback_forecast_rows": int(result.metrics.get("fallback_forecast_rows", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Separate session OHLCV CSV files")
    parser.add_argument("--min-train", type=int, default=80)
    parser.add_argument("--estimator", choices=("ridge", "hist_gradient_boosting"), default="ridge")
    parser.add_argument("--min-accuracy", type=float, default=80.0)
    parser.add_argument("--min-coverage", type=float, default=80.0)
    parser.add_argument("--min-targets", type=int, default=500)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--require-pass", action="store_true", help="Exit non-zero unless every acceptance criterion passes")
    parser.add_argument("--output", default="artifacts/directional_benchmark.json")
    args = parser.parse_args()

    paths = [Path(name) for name in args.input]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error(f"Input CSV file(s) not found: {missing}")
    sessions = [evaluate(path, args.min_train, args.estimator) for path in paths]
    target_rows = sum(item["evaluated_directional_targets"] for item in sessions)
    correct_rows = sum(item["correct_directional_predictions"] for item in sessions)
    covered_rows = sum(item["covered_directional_predictions"] for item in sessions)
    accuracy = 100.0 * correct_rows / target_rows if target_rows else 0.0
    coverage = 100.0 * covered_rows / target_rows if target_rows else 0.0
    criteria = {
        "minimum_accuracy_pass": accuracy >= args.min_accuracy,
        "minimum_prediction_coverage_pass": coverage >= args.min_coverage,
        "minimum_target_count_pass": target_rows >= args.min_targets,
        "minimum_session_count_pass": len(sessions) >= args.min_sessions,
    }
    report = {
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "benchmark_type": "chronological_walk_forward_one_candle_ahead",
        "estimator": args.estimator,
        "min_train": args.min_train,
        "thresholds": {
            "directional_accuracy_pct": args.min_accuracy,
            "prediction_coverage_pct": args.min_coverage,
            "directional_target_rows": args.min_targets,
            "sessions": args.min_sessions,
        },
        "aggregate": {
            "directional_target_rows": target_rows,
            "correct_directional_predictions": correct_rows,
            "directional_accuracy_pct": round(accuracy, 4),
            "prediction_coverage_pct": round(coverage, 4),
        },
        "criteria": criteria,
        "sessions": sessions,
        "live_enablement": "NOT_GRANTED_BY_THIS_BENCHMARK",
        "note": "Research benchmark only. Passing does not establish profitability or authorize live orders. Validate live Kotak Neo data, freshness, instrument mapping, and execution controls separately.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.require_pass and report["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
