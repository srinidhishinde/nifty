from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make repository-local packages importable when invoked as python scripts/...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from prediction.recursive_backtest import simulate_recursive_low_high


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a conservative recursive trade simulation from OHLC candles and next-candle forecasts."
    )
    parser.add_argument("--candles", required=True, help="Actual OHLC CSV: timestamp/open/high/low/close")
    parser.add_argument("--predictions", required=True, help="Forecast CSV with target_timestamp/predicted_low/predicted_high")
    parser.add_argument("--initial-capital", type=float, default=30000.0)
    parser.add_argument("--lot-size", type=int, required=True, help="Contract lot size applicable to the tested dates")
    parser.add_argument("--stop-loss-pct", type=float, default=0.05, help="Stop-loss fraction, default 0.05 = 5%%")
    parser.add_argument("--cost-bps-per-side", type=float, default=10.0, help="Estimated fees/taxes per side in basis points")
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0, help="Estimated slippage per side in basis points")
    parser.add_argument("--output", default="data/backtest/nifty_one_minute_trade_ledger.csv")
    args = parser.parse_args()

    candle_path, prediction_path = Path(args.candles), Path(args.predictions)
    if not candle_path.is_file():
        parser.error(f"Candles CSV does not exist: {candle_path}")
    if not prediction_path.is_file():
        parser.error(f"Predictions CSV does not exist: {prediction_path}")

    candles = pd.read_csv(candle_path)
    predictions = pd.read_csv(prediction_path)
    candles.columns = [str(c).strip().lower().replace(" ", "_") for c in candles.columns]
    predictions.columns = [str(c).strip().lower().replace(" ", "_") for c in predictions.columns]
    if "timestamp" not in candles.columns:
        for alias in ("datetime", "date_time", "time", "date"):
            if alias in candles.columns:
                candles = candles.rename(columns={alias: "timestamp"})
                break
    if "target_timestamp" not in predictions.columns:
        parser.error("Predictions CSV must contain target_timestamp. Use the walk-forward forecast output.")

    ledger, summary = simulate_recursive_low_high(
        candles,
        predictions,
        initial_capital=args.initial_capital,
        lot_size=args.lot_size,
        stop_loss_pct=args.stop_loss_pct,
        cost_bps_per_side=args.cost_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output, index=False)
    summary_path = output.with_name("nifty_recursive_backtest_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "trade_ledger_csv": str(output),
        "summary_json": str(summary_path),
        **summary,
    }, indent=2))
    return 0 if summary["status"] in {"EVALUATED", "NO_TRADES"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
