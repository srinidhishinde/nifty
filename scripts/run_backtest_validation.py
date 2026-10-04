from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.capital_aware import CapitalAwareRuleBacktestEngine
from strategy.rules import StrategyConfig


REQUIRED = {"timestamp", "open", "high", "low", "close", "volume"}


def load_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise ValueError(f"Backtest CSV missing required columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    for col in REQUIRED - {"timestamp"}:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=list(REQUIRED))
    invalid = (
        (frame["open"] <= 0) | (frame["high"] <= 0) |
        (frame["low"] <= 0) | (frame["close"] <= 0) |
        (frame["volume"] < 0) |
        (frame["high"] < frame[["open", "close"]].max(axis=1)) |
        (frame["low"] > frame[["open", "close"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError(f"CSV contains {int(invalid.sum())} invalid OHLCV rows")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return frame


def run_once(frame: pd.DataFrame, capital: float, risk: float, lot_size: int,
             slippage: float, brokerage: float, margin_per_lot: float):
    engine = CapitalAwareRuleBacktestEngine(
        starting_capital=capital,
        risk_per_trade=risk,
        instrument="NIFTY",
        config=StrategyConfig(),
        lot_size=lot_size,
        slippage_points=slippage,
        brokerage_per_order=brokerage,
        margin_per_lot=margin_per_lot,
        max_daily_loss=capital * 0.03,
        max_trades_per_day=5,
    )
    return engine.run(frame, symbol="NIFTY")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict ₹1 lakh NIFTY futures consistency backtest.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--risk", type=float, default=1000)
    parser.add_argument("--lot-size", type=int, default=65)
    parser.add_argument("--slippage", type=float, default=0.25)
    parser.add_argument("--brokerage", type=float, default=10)
    parser.add_argument("--margin-per-lot", type=float, default=0)
    parser.add_argument("--output-dir", default="data/backtest/results")
    args = parser.parse_args()

    path = Path(args.csv)
    frame = load_ohlcv(path)
    if frame.empty:
        raise SystemExit("No rows in backtest CSV.")

    result = run_once(frame, args.capital, args.risk, args.lot_size,
                      args.slippage, args.brokerage, args.margin_per_lot)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(out / "trades.csv", index=False)
    result.equity_curve.to_csv(out / "equity_curve.csv", index=False)
    result.rule_performance.to_csv(out / "rule_performance.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(result.metrics.__dict__, indent=2, default=str), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps(result.validation, indent=2, default=str), encoding="utf-8")

    print("=== STRICT BACKTEST ===")
    for key, value in result.metrics.__dict__.items():
        print(f"{key}: {value}")
    print("validation:", json.dumps(result.validation, indent=2))
    print(f"trades_csv: {out / 'trades.csv'}")
    print(f"equity_curve_csv: {out / 'equity_curve.csv'}")
    print(f"rule_performance_csv: {out / 'rule_performance.csv'}")


if __name__ == "__main__":
    main()
