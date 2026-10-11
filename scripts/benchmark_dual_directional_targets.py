from __future__ import annotations

"""Evaluate NIFTY index and option-premium direction as separate targets.

This wrapper never joins unlike instruments or treats an option premium as a
proxy for index direction. Supply real, timestamped OHLCV session CSVs for each
target. Each CSV is evaluated independently by the existing chronological
walk-forward benchmark; synthetic test fixtures are not benchmark evidence.
"""
import argparse
import json
from pathlib import Path
from typing import Any

from scripts.benchmark_directional_classifier import evaluate


def summarize_target(
    paths: list[Path],
    *,
    min_train: int,
    estimator: str,
    tolerance: float,
    min_accuracy: float,
    min_coverage: float,
    min_targets: int,
    min_sessions: int,
) -> dict[str, Any]:
    sessions = [
        evaluate(path, min_train=min_train, estimator=estimator, tolerance=tolerance)
        for path in paths
    ]
    targets = sum(s["evaluated_targets"] for s in sessions)
    correct = sum(s["correct_predictions"] for s in sessions)
    baseline_correct = sum(s["momentum_baseline_correct_predictions"] for s in sessions)
    predicted = sum(
        round(s["coverage_pct"] * s["evaluated_targets"] / 100.0) for s in sessions
    )
    accuracy = 100.0 * correct / targets if targets else 0.0
    coverage = 100.0 * predicted / targets if targets else 0.0
    baseline_accuracy = 100.0 * baseline_correct / targets if targets else 0.0
    criteria = {
        "minimum_accuracy_pass": accuracy >= min_accuracy,
        "minimum_prediction_coverage_pass": coverage >= min_coverage,
        "minimum_target_count_pass": targets >= min_targets,
        "minimum_session_count_pass": len(sessions) >= min_sessions,
        "beats_momentum_baseline": accuracy > baseline_accuracy,
    }
    return {
        "status": "PASS" if all(criteria.values()) else "FAIL",
        "aggregate": {
            "evaluated_targets": targets,
            "correct_predictions": correct,
            "accuracy_pct": round(accuracy, 4),
            "coverage_pct": round(coverage, 4),
            "momentum_baseline_accuracy_pct": round(baseline_accuracy, 4),
        },
        "thresholds": {
            "accuracy_pct": min_accuracy,
            "coverage_pct": min_coverage,
            "evaluated_targets": min_targets,
            "sessions": min_sessions,
        },
        "criteria": criteria,
        "sessions": sessions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nifty-input", nargs="+", required=True,
                        help="Real NIFTY index OHLCV CSVs, one file per session")
    parser.add_argument("--option-input", nargs="+", required=True,
                        help="Real selected CE/PE premium OHLCV CSVs, one file per session")
    parser.add_argument("--option-symbol", required=True,
                        help="Exact option instrument identifier, e.g. NIFTY_22700CE")
    parser.add_argument("--min-train", type=int, default=80)
    parser.add_argument("--estimator", choices=("logistic", "hist_gradient_boosting", "random_forest"),
                        default="logistic")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--min-accuracy", type=float, default=80.0)
    parser.add_argument("--min-coverage", type=float, default=80.0)
    parser.add_argument("--min-targets", type=int, default=500)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--output", default="artifacts/dual_directional_benchmark.json")
    args = parser.parse_args()

    if args.min_train < 20:
        parser.error("--min-train must be at least 20")
    if args.tolerance < 0:
        parser.error("--tolerance cannot be negative")
    nifty_paths = [Path(p) for p in args.nifty_input]
    option_paths = [Path(p) for p in args.option_input]
    missing = [str(p) for p in nifty_paths + option_paths if not p.is_file()]
    if missing:
        parser.error(f"Input CSV file(s) not found: {missing}")
    if len(nifty_paths) < args.min_sessions or len(option_paths) < args.min_sessions:
        parser.error("Each target requires at least --min-sessions separate real session CSVs")

    common = dict(
        min_train=args.min_train,
        estimator=args.estimator,
        tolerance=args.tolerance,
        min_accuracy=args.min_accuracy,
        min_coverage=args.min_coverage,
        min_targets=args.min_targets,
        min_sessions=args.min_sessions,
    )
    nifty = summarize_target(nifty_paths, **common)
    option = summarize_target(option_paths, **common)
    report = {
        "status": "PASS" if nifty["status"] == option["status"] == "PASS" else "FAIL",
        "benchmark_type": "separate_chronological_expanding_walk_forward_targets",
        "estimator": args.estimator,
        "min_train": args.min_train,
        "label_tolerance": args.tolerance,
        "targets": {
            "nifty_index_direction": nifty,
            "option_premium_direction": {
                "instrument": args.option_symbol,
                **option,
            },
        },
        "interpretation": (
            "NIFTY index and option-premium direction are independent prediction targets. "
            "Their metrics must not be pooled. Each input file is evaluated independently; "
            "all forecasts include deterministic fallback predictions in the denominator. "
            "A PASS is a historical benchmark gate, not proof of profitability or future accuracy."
        ),
        "live_enablement": "LOCKED",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if args.require_pass and report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
