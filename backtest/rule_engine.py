from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from backtest.metrics import calculate_metrics
from features.technical.indicators import add_indicators
from risk.risk_manager import size_position
from strategy.rules import StrategyConfig, evaluate_rules, rank_rule_performance


@dataclass(frozen=True)
class RuleBacktestResult:
    trades: pd.DataFrame
    metrics: object
    rule_performance: pd.DataFrame


def _in_session(ts: pd.Timestamp, session: str) -> bool:
    current = ts.timetz().replace(tzinfo=None)
    if session == "NIFTY":
        return time(9, 15) <= current <= time(15, 30)
    if session == "MCX":
        return time(9, 0) <= current <= time(23, 30)
    return True


def _exit_for_bar(direction: str, entry: float, stop: float, target: float, high: float, low: float):
    if direction == "BUY":
        stop_hit, target_hit = low <= stop, high >= target
    else:
        stop_hit, target_hit = high >= stop, low <= target
    if stop_hit:
        return stop, "stop_loss"
    if target_hit:
        return target, "take_profit"
    return None, None


class RuleBacktestEngine:
    """Conservative OHLC backtest using the same ATR-aware risk model as live signals."""

    def __init__(
        self,
        starting_capital: float = 300000.0,
        risk_per_trade: float = 3000.0,
        instrument: str = "NIFTY",
        config: StrategyConfig | None = None,
        commission_pct: float = 0.0005,
        slippage_pct: float = 0.0005,
        daily_loss_limit_pct: float = 0.02,
    ):
        self.starting_capital = float(starting_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.instrument = instrument.upper()
        self.config = config or StrategyConfig()
        self.commission_pct = max(float(commission_pct), 0.0)
        self.slippage_pct = max(float(slippage_pct), 0.0)
        self.daily_loss_limit_pct = max(float(daily_loss_limit_pct), 0.0)

    def _execution_price(self, price: float, direction: str, entry: bool) -> float:
        adverse = self.slippage_pct if direction == "BUY" else -self.slippage_pct
        if not entry:
            adverse = -adverse
        return price * (1.0 + adverse)

    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "NIFTY",
        evaluation_start: pd.Timestamp | None = None,
        evaluation_end: pd.Timestamp | None = None,
    ) -> RuleBacktestResult:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if data.empty:
            empty = pd.DataFrame()
            return RuleBacktestResult(empty, calculate_metrics(empty, self.starting_capital), pd.DataFrame())

        frame = data.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
        invalid_ohlc = (
            (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame[["open", "close"]].min(axis=1))
            | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
            | (frame["volume"] < 0)
        )
        if invalid_ohlc.any():
            raise ValueError(f"Invalid OHLCV rows: {int(invalid_ohlc.sum())}")
        frame = add_indicators(frame.sort_values("timestamp").reset_index(drop=True))

        trades: list[dict] = []
        position = None
        equity = self.starting_capital
        daily_pnl: dict[object, float] = {}

        for i in range(1, len(frame)):
            row, previous = frame.iloc[i], frame.iloc[i - 1]
            ts = row["timestamp"]
            if not _in_session(ts, self.instrument):
                continue

            if position is not None:
                exit_price, reason = _exit_for_bar(
                    position["direction"], position["entry"], position["stop_loss"],
                    position["target"], float(row["high"]), float(row["low"])
                )
                if exit_price is None:
                    next_ts = frame.iloc[i + 1]["timestamp"] if i + 1 < len(frame) else None
                    if next_ts is None or next_ts.date() != ts.date() or not _in_session(next_ts, self.instrument):
                        exit_price, reason = float(row["close"]), "market_close"

                if exit_price is not None:
                    raw_exit = float(exit_price)
                    fill_exit = self._execution_price(raw_exit, position["direction"], entry=False)
                    signed_roi = (
                        (fill_exit - position["entry"]) / position["entry"]
                        if position["direction"] == "BUY"
                        else (position["entry"] - fill_exit) / position["entry"]
                    )
                    price_delta = (
                        fill_exit - position["entry"]
                        if position["direction"] == "BUY"
                        else position["entry"] - fill_exit
                    )
                    gross_pnl = price_delta * position["quantity"]
                    costs = position["quantity"] * (
                        position["entry"] * self.commission_pct
                        + fill_exit * self.commission_pct
                    )
                    pnl = gross_pnl - costs
                    day = ts.date()
                    daily_pnl[day] = daily_pnl.get(day, 0.0) + pnl
                    equity += pnl
                    trades.append({
                        **position,
                        "exit_time": ts,
                        "exit_price": round(fill_exit, 4),
                        "gross_pnl": round(float(gross_pnl), 2),
                        "costs": round(float(costs), 2),
                        "pnl": round(float(pnl), 2),
                        "roi_pct": round(float(signed_roi * 100), 4),
                        "reason": reason,
                    })
                    position = None
                continue

            if evaluation_start is not None and ts < pd.Timestamp(evaluation_start):
                continue
            if evaluation_end is not None and ts > pd.Timestamp(evaluation_end):
                continue
            if self.daily_loss_limit_pct and daily_pnl.get(ts.date(), 0.0) <= -self.starting_capital * self.daily_loss_limit_pct:
                continue

            signals = evaluate_rules(row, previous, self.config)
            if not signals:
                continue
            buys, sells = [s for s in signals if s.direction == "BUY"], [s for s in signals if s.direction == "SELL"]
            if buys and sells:
                continue
            direction = "BUY" if buys else "SELL"
            # A close-based signal is only known after this candle closes.
            # Fill on the next available bar, not on the signal candle itself.
            if i + 1 >= len(frame):
                continue
            next_row = frame.iloc[i + 1]
            next_ts = next_row["timestamp"]
            if next_ts.date() != ts.date() or not _in_session(next_ts, self.instrument):
                continue
            raw_entry = float(next_row["open"])
            entry = self._execution_price(raw_entry, direction, entry=True)
            atr = float(row.get("ATR", 0.0) or 0.0)
            risk_fraction = self.risk_per_trade / self.starting_capital
            risk = size_position(
                entry, atr, direction, self.starting_capital,
                risk_fraction=risk_fraction,
                stop_pct=self.config.stop_loss_pct,
                atr_multiple=self.config.atr_stop_multiple,
                target_multiple=self.config.target_atr_multiple,
            )
            if not risk.allowed:
                continue
            position = {
                "entry_time": ts,
                "symbol": symbol,
                "direction": direction,
                "entry": risk.entry,
                "stop_loss": risk.stop_loss,
                "target": risk.take_profit,
                "quantity": risk.quantity,
                "capital_at_risk": risk.risk_amount,
                "atr": atr,
                "reward_risk": risk.reward_risk,
                "rule": "|".join(s.rule for s in signals),
                "rules": tuple(s.rule for s in signals),
            }

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            trades_df = pd.DataFrame(columns=[
                "entry_time", "exit_time", "symbol", "direction", "entry",
                "exit_price", "quantity", "gross_pnl", "costs", "pnl", "roi_pct", "reason", "rule",
            ])
        metrics = calculate_metrics(
            trades_df.rename(columns={"entry": "entry_price"}) if "entry" in trades_df.columns else trades_df,
            self.starting_capital,
        )
        return RuleBacktestResult(trades_df, metrics, rank_rule_performance(trades_df))
