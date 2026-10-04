from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.capital_aware import CapitalAwareRuleBacktestEngine
from backtest.validation import AcceptanceCriteria, acceptance_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the evidence-grade ₹1 lakh / 100-trading-day acceptance test."
    )
    parser.add_argument("csv", type=Path, help="Real historical NIFTY OHLCV CSV")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    criteria = AcceptanceCriteria()
    result = CapitalAwareRuleBacktestEngine(
        starting_capital=criteria.starting_capital,
        risk_per_trade=1_000.0,
        instrument="NIFTY",
        max_daily_loss=2_000.0,
        max_trades_per_day=5,
        slippage_points=0.25,
        brokerage_per_order=10.0,
    ).run(data)

    report = acceptance_report(data, result.trades, result.metrics, criteria)
    report["backtest_validation"] = result.validation

    text = json.dumps(report, indent=2, default=str)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
