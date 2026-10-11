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
        description="Leakage-free walk-forward interval calibration and baseline comparison."
    )
    parser.add_argument("--input", required=True, help="OHLCV CSV: timestamp, open, high, low, close[, volume]")
    parser.add_argument("--output", default="data/backtest/low_high_tuning_predictions.csv")
    parser.add_argument("--summary", default=None, help="Summary JSON (defaults beside prediction CSV)")
    parser.add_argument("--min-train", type=int, default=80)
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument("--estimator", choices=("hist_gradient_boosting", "ridge"), default="hist_gradient_boosting")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        parser.error(f"Input CSV does not exist: {source}")
    frame = pd.read_csv(source)
    predictions, summary = tune_and_compare(
        frame,
        target_coverage=args.target_coverage,
        min_train=args.min_train,
        estimator=args.estimator,
    )
    summary["input"] = str(source)
    summary["target_coverage"] = args.target_coverage
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "predictions_csv": str(output),
        "summary_json": str(summary_path),
        "input_candles": summary["input_candles"],
        "evaluated_forecasts": summary["evaluated_forecasts"],
        "methods": summary["methods"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
