from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from prediction.low_high_tuning import tune_and_compare


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-free walk-forward interval calibration and baseline comparison. Each input file is evaluated independently."
    )
    parser.add_argument("--input", nargs="+", required=True, help="One or more OHLCV CSVs; each is evaluated as a separate session")
    parser.add_argument("--output", default="data/backtest/low_high_tuning_predictions.csv")
    parser.add_argument("--summary", default=None, help="Summary JSON (defaults beside prediction CSV)")
    parser.add_argument("--min-train", type=int, default=80)
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument("--estimator", choices=("hist_gradient_boosting", "ridge"), default="hist_gradient_boosting")
    args = parser.parse_args()

    runs = []
    prediction_frames = []
    for input_name in args.input:
        source = Path(input_name)
        if not source.is_file():
            parser.error(f"Input CSV does not exist: {source}")
        frame = pd.read_csv(source)
        predictions, summary = tune_and_compare(
            frame,
            target_coverage=args.target_coverage,
            min_train=args.min_train,
            estimator=args.estimator,
        )
        predictions.insert(0, "source_file", str(source))
        prediction_frames.append(predictions)
        runs.append({
            "input": str(source),
            "target_coverage": args.target_coverage,
            **summary,
        })

    combined = pd.concat(prediction_frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "EVALUATED",
        "forecast_timing": "before_target_candle_open",
        "target_coverage": args.target_coverage,
        "estimator": args.estimator,
        "min_train": args.min_train,
        "sessions_evaluated_independently": len(runs),
        "predictions_csv": str(output),
        "runs": runs,
        "interpretation": (
            "Each file is evaluated independently so a later session's forecasts do not silently "
            "use a prior file unless explicitly joined into a chronological dataset. Compare each "
            "method on the same evaluated rows; prefer the narrowest method near the 80% full-range "
            "coverage target, and inspect MAE and width ratio alongside coverage."
        ),
    }
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "predictions_csv": str(output),
        "summary_json": str(summary_path),
        "sessions_evaluated_independently": len(runs),
        "runs": [
            {
                "input": run["input"],
                "evaluated_forecasts": run["evaluated_forecasts"],
                "methods": run["methods"],
            }
            for run in runs
        ],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
