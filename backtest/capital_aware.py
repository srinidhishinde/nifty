from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any

import pandas as pd

from backtest.metrics import calculate_metrics
from features.technical.indicators import add_indicators
from strategy.rules import StrategyConfig, evaluate_rules, rank_rule_performance


@dataclass(frozen=True)
class StrictBacktestResult:
    trades: pd.DataFrame
    metrics: Any
    rule_performance: pd.DataFrame
    equity_curve: pd.DataFrame
    validation: dict


def _in_session(ts: pd.Timestamp, instrument: str) -> bool:
    current = ts.timetz().replace(tzinfo=None)
    if instrument.upper() == "NIFTY":
        return time(9, 15) <= current <= time(15, 30)
    if instrument.upper() == "MCX":
        return time(9, 0) <= current <= time(23, 30)
    return True


def _exit_for_bar(direction: str, stop: float, target: float, high: float, low: float):
    if direction == "BUY":
        stop_hit = low <= stop
        target_hit = high >= target
    else:
        stop_hit = high >= stop
        target_hit = low <= target
    if stop_hit:
        return stop, "stop_loss"
    if target_hit:
        return target, "take_profit"
    return None, None


class CapitalAwareRuleBacktestEngine:
    """
    Conservative, capital-aware OHLCV backtester.

    Key rules:
    - Signals are generated on bar close and executed at the NEXT bar open.
    - P&L is price movement * point_value * quantity, not ROI * risk capital.
    - Quantity is whole lots and is constrained by risk budget and optional margin.
    - Slippage and per-order brokerage are charged on both entry and exit.
    - No new entries are allowed after the evaluation end.
    - Positions are intraday and forced flat at the last session bar.
    - A daily loss limit blocks new entries after it is breached.
    """

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        risk_per_trade: float = 1_000.0,
        instrument: str = "NIFTY",
        config: StrategyConfig | None = None,
        lot_size: int = 65,
        point_value: float = 1.0,
        margin_per_lot: float = 0.0,
        slippage_points: float = 0.25,
        brokerage_per_order: float = 10.0,
        max_daily_loss: float | None = None,
        max_trades_per_day: int = 5,
    ):
        if starting_capital <= 0:
            raise ValueError("starting_capital must be > 0")
        if risk_per_trade <= 0:
            raise ValueError("risk_per_trade must be > 0")
        if lot_size <= 0:
            raise ValueError("lot_size must be > 0")
        if point_value <= 0:
            raise ValueError("point_value must be > 0")
        if margin_per_lot < 0 or slippage_points < 0 or brokerage_per_order < 0:
            raise ValueError("cost and margin parameters cannot be negative")
        self.starting_capital = float(starting_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.instrument = instrument.upper()
        self.config = config or StrategyConfig()
        self.lot_size = int(lot_size)
        self.point_value = float(point_value)
        self.margin_per_lot = float(margin_per_lot)
        self.slippage_points = float(slippage_points)
        self.brokerage_per_order = float(brokerage_per_order)
        self.max_daily_loss = float(max_daily_loss) if max_daily_loss is not None else self.starting_capital * 0.03
        self.max_trades_per_day = int(max_trades_per_day)

    def _fill_price(self, price: float, direction: str, is_entry: bool) -> float:
        # Slippage is adverse in both directions.
        if direction == "BUY":
            return price + self.slippage_points if is_entry else price - self.slippage_points
        return price - self.slippage_points if is_entry else price + self.slippage_points

    def _position_size(self, equity: float, entry: float, stop: float) -> tuple[int, float]:
        risk_per_lot = abs(entry - stop) * self.point_value * self.lot_size
        if risk_per_lot <= 0:
            return 0, 0.0
        by_risk = int(self.risk_per_trade // risk_per_lot)
        if self.margin_per_lot > 0:
            by_margin = int(max(equity, 0.0) // self.margin_per_lot)
            lots = min(by_risk, by_margin)
        else:
            lots = by_risk
        if lots <= 0:
            return 0, risk_per_lot
        return lots, risk_per_lot

    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "NIFTY",
        evaluation_start: pd.Timestamp | None = None,
        evaluation_end: pd.Timestamp | None = None,
    ) -> StrictBacktestResult:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        frame = data.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=list(required))
        invalid = (
            (frame["open"] <= 0) | (frame["high"] <= 0) |
            (frame["low"] <= 0) | (frame["close"] <= 0) |
            (frame["volume"] < 0) |
            (frame["high"] < frame[["open", "close"]].max(axis=1)) |
            (frame["low"] > frame[["open", "close"]].min(axis=1))
        )
        if invalid.any():
            raise ValueError(f"Invalid OHLCV rows: {int(invalid.sum())}")
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        frame = add_indicators(frame)

        if frame.empty:
            empty = pd.DataFrame()
            return StrictBacktestResult(
                empty,
                calculate_metrics(empty, self.starting_capital),
                pd.DataFrame(),
                pd.DataFrame(),
                {"status": "FAIL", "reason": "empty_dataset"},
            )

        eval_start = pd.Timestamp(evaluation_start) if evaluation_start is not None else frame["timestamp"].min()
        eval_end = pd.Timestamp(evaluation_end) if evaluation_end is not None else frame["timestamp"].max()

        trades: list[dict] = []
        equity_rows: list[dict] = []
        position: dict | None = None
        equity = self.starting_capital
        daily_pnl: dict = {}
        daily_trades: dict = {}

        for i in range(1, len(frame)):
            row = frame.iloc[i]
            prev = frame.iloc[i - 1]
            ts = row["timestamp"]
            if not _in_session(ts, self.instrument):
                continue
            day = ts.date()
            daily_pnl.setdefault(day, 0.0)
            daily_trades.setdefault(day, 0)

            if position is not None:
                exit_price, reason = _exit_for_bar(
                    position["direction"], position["stop_loss"], position["target"],
                    float(row["high"]), float(row["low"])
                )
                last_session_bar = (
                    i + 1 >= len(frame)
                    or frame.iloc[i + 1]["timestamp"].date() != day
                    or not _in_session(frame.iloc[i + 1]["timestamp"], self.instrument)
                )
                if exit_price is None and last_session_bar:
                    exit_price, reason = float(row["close"]), "market_close"
                if exit_price is not None:
                    filled_exit = self._fill_price(float(exit_price), position["direction"], False)
                    signed_points = (
                        filled_exit - position["entry_price"]
                        if position["direction"] == "BUY"
                        else position["entry_price"] - filled_exit
                    )
                    gross_pnl = signed_points * self.point_value * position["quantity"]
                    costs = self.brokerage_per_order * 2
                    pnl = gross_pnl - costs
                    equity += pnl
                    daily_pnl[day] += pnl
                    trades.append({
                        **position,
                        "exit_time": ts,
                        "exit_price": round(filled_exit, 4),
                        "gross_pnl": round(gross_pnl, 2),
                        "costs": round(costs, 2),
                        "pnl": round(pnl, 2),
                        "reason": reason,
                    })
                    position = None
                    equity_rows.append({"timestamp": ts, "equity": round(equity, 2), "daily_pnl": round(daily_pnl[day], 2)})
                continue

            if ts < eval_start or ts > eval_end:
                continue
            if daily_pnl[day] <= -self.max_daily_loss or daily_trades[day] >= self.max_trades_per_day:
                continue

            signals = evaluate_rules(row, prev, self.config)
            if not signals:
                continue
            buys = [s for s in signals if s.direction == "BUY"]
            sells = [s for s in signals if s.direction == "SELL"]
            if buys and sells:
                continue

            direction = "BUY" if buys else "SELL"
            if i + 1 >= len(frame):
                continue
            next_row = frame.iloc[i + 1]
            if next_row["timestamp"].date() != day or not _in_session(next_row["timestamp"], self.instrument):
                continue

            signal_close = float(row["close"])
            stop = (
                signal_close * (1 - self.config.stop_loss_pct)
                if direction == "BUY"
                else signal_close * (1 + self.config.stop_loss_pct)
            )
            target = (
                signal_close * (1 + self.config.target_roi_pct)
                if direction == "BUY"
                else signal_close * (1 - self.config.target_roi_pct)
            )
            entry_raw = float(next_row["open"])
            entry = self._fill_price(entry_raw, direction, True)
            # Keep stop/target distances relative to actual entry.
            stop = entry * (1 - self.config.stop_loss_pct) if direction == "BUY" else entry * (1 + self.config.stop_loss_pct)
            target = entry * (1 + self.config.target_roi_pct) if direction == "BUY" else entry * (1 - self.config.target_roi_pct)
            lots, risk_per_lot = self._position_size(equity, entry, stop)
            if lots <= 0:
                continue

            quantity = lots * self.lot_size
            daily_trades[day] += 1
            position = {
                "entry_time": next_row["timestamp"],
                "signal_time": ts,
                "symbol": symbol,
                "direction": direction,
                "entry_price": round(entry, 4),
                "stop_loss": round(stop, 4),
                "target": round(target, 4),
                "lots": lots,
                "quantity": quantity,
                "risk_per_lot": round(risk_per_lot, 2),
                "risk_budget": round(self.risk_per_trade, 2),
                "rule": "|".join(s.rule for s in signals),
                "rules": tuple(s.rule for s in signals),
            }

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            trades_df = pd.DataFrame(columns=[
                "entry_time", "signal_time", "exit_time", "symbol", "direction",
                "entry_price", "exit_price", "lots", "quantity", "gross_pnl", "costs", "pnl", "reason", "rule"
            ])
        metrics = calculate_metrics(
            trades_df.rename(columns={"entry_price": "entry_price"}) if not trades_df.empty else trades_df,
            self.starting_capital,
        )
        equity_df = pd.DataFrame(equity_rows)
        if not equity_df.empty:
            equity_df = equity_df.sort_values("timestamp").reset_index(drop=True)

        validation = {
            "status": "PASS",
            "rows": int(len(frame)),
            "evaluation_start": str(eval_start),
            "evaluation_end": str(eval_end),
            "trading_days": int(frame["timestamp"].dt.date.nunique()),
            "zero_volume_rows": int((frame["volume"] == 0).sum()),
            "starting_capital": self.starting_capital,
            "final_equity": round(float(equity), 2),
            "net_pnl": round(float(equity - self.starting_capital), 2),
            "lot_size": self.lot_size,
            "margin_per_lot": self.margin_per_lot,
            "slippage_points": self.slippage_points,
            "brokerage_per_order": self.brokerage_per_order,
        }
        return StrictBacktestResult(
            trades_df,
            metrics,
            rank_rule_performance(trades_df),
            equity_df,
            validation,
        )
