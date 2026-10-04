from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.rule_engine import RuleBacktestEngine
from strategy.rules import StrategyConfig


def load_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Backtest CSV missing required columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp")
    return frame


def run_once(frame: pd.DataFrame, capital: float, risk: float):
    engine = RuleBacktestEngine(
        starting_capital=capital,
        risk_per_trade=risk,
        instrument="NIFTY",
        config=StrategyConfig(),
    )
    return engine.run(frame, symbol="NIFTY")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a capital-constrained NIFTY consistency backtest.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--risk", type=float, default=1000)
    parser.add_argument("--output-dir", default="data/backtest/results")
    args = parser.parse_args()

    path = Path(args.csv)
    frame = load_ohlcv(path)
    if frame.empty:
        raise SystemExit("No rows in backtest CSV.")

    # Use only real data. No synthetic fallback.
    result = run_once(frame, args.capital, args.risk)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.trades.to_csv(out / "trades.csv", index=False)
    metrics = result.metrics.__dict__
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    result.rule_performance.to_csv(out / "rule_performance.csv", index=False)

    print("=== BACKTEST ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"trades_csv: {out / 'trades.csv'}")
    print(f"rule_performance_csv: {out / 'rule_performance.csv'}")


if __name__ == "__main__":
    main()
