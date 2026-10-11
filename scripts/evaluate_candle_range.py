from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prediction.candle_range import walk_forward_range_forecast


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward calibrated next-candle high/low range evaluation."
    )
    parser.add_argument("--input", required=True, help="CSV with timestamp, open, high, low columns")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--min-periods", type=int, default=5)
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--target-width-ratio", type=float, default=1.0)
    parser.add_argument("--output", default="", help="Optional output CSV for per-candle forecasts")
    args = parser.parse_args()

    path = Path(args.input).expanduser()
    if not path.is_file():
        parser.error(f"Input file not found: {path}")
    data = pd.read_csv(path)
    result = walk_forward_range_forecast(
        data,
        window=args.window,
        min_periods=args.min_periods,
        excursion_quantile=args.quantile,
        target_width_ratio=args.target_width_ratio,
    )
    if result.forecasts.empty:
        print("No forecasts produced; provide more clean OHLC rows.")
        return 2

    print(f"Input candles: {len(data)}")
    print(f"Forecasts scored: {len(result.forecasts)}")
    print(f"Target width ratio: {args.target_width_ratio:.2f}x recent historical mean range")
    print(f"Observed predicted/actual width: {result.mean_width_ratio:.3f}x")
    print(f"Full candle range coverage: {result.full_range_coverage_pct:.1f}%")
    print(f"Low MAE: {result.low_mae:.4f}")
    print(f"High MAE: {result.high_mae:.4f}")
    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        result.forecasts.to_csv(out, index=False)
        print(f"Per-candle forecasts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
