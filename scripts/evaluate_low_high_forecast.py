from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prediction.low_high_forecast import walk_forward_low_high


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate next-candle low/high forecasts with walk-forward validation."
    )
    parser.add_argument("--input", required=True, help="OHLCV CSV with timestamp, open, high, low, close columns")
    parser.add_argument("--output", default="data/backtest/low_high_predictions.csv", help="Predictions CSV output")
    parser.add_argument("--min-train", type=int, default=80, help="Minimum historical training rows")
    parser.add_argument("--estimator", choices=("hist_gradient_boosting", "ridge"), default="hist_gradient_boosting")
    parser.add_argument("--step", type=int, default=1, help="Forecast every Nth candle; 1 means every candle")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        parser.error(f"Input CSV does not exist: {source}")
    frame = pd.read_csv(source)
    # Common exports use title-case headers. Normalize case/whitespace only;
    # timestamp aliases are mapped explicitly to avoid guessing data meaning.
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    if "timestamp" not in frame.columns:
        for alias in ("datetime", "date_time", "time", "date"):
            if alias in frame.columns:
                frame = frame.rename(columns={alias: "timestamp"})
                break
    result = walk_forward_low_high(
        frame,
        min_train=args.min_train,
        estimator=args.estimator,
        step=args.step,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(output, index=False)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "status": result.status,
        "input": str(source),
        "estimator": args.estimator,
        "min_train": args.min_train,
        "step": args.step,
        **result.metrics,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result.status,
        "predictions_csv": str(output),
        "metrics_json": str(metrics_path),
        **result.metrics,
    }, indent=2))
    return 0 if result.status == "EVALUATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
