from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from backtest.metrics import calculate_metrics
from features.technical.indicators import add_indicators
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
        # MCX spans midnight in some products. This implementation treats
        # 09:00-23:30 as the active session, matching the supplied pseudocode.
        return time(9, 0) <= current <= time(23, 30)
    return True


def _exit_for_bar(direction: str, entry: float, stop: float, target: float, high: float, low: float):
    if direction == "BUY":
        stop_hit = low <= stop
        target_hit = high >= target
        if stop_hit and target_hit:
            # Conservative OHLC backtest assumption: stop is hit first when
            # both levels are inside the same candle.
            return stop, "stop_loss"
        if stop_hit:
            return stop, "stop_loss"
        if target_hit:
            return target, "take_profit"
    else:
        stop_hit = high >= stop
        target_hit = low <= target
        if stop_hit and target_hit:
            return stop, "stop_loss"
        if stop_hit:
            return stop, "stop_loss"
        if target_hit:
            return target, "take_profit"
    return None, None


class RuleBacktestEngine:
    """Backtest the seven rules on OHLCV/market-context candles."""

    def __init__(
        self,
        starting_capital: float = 300000.0,
        risk_per_trade: float = 15000.0,
        instrument: str = "NIFTY",
        config: StrategyConfig | None = None,
    ):
        self.starting_capital = float(starting_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.instrument = instrument.upper()
        self.config = config or StrategyConfig()

    def run(\n        self,\n        data: pd.DataFrame,\n        symbol: str = "NIFTY",\n        evaluation_start: pd.Timestamp | None = None,\n        evaluation_end: pd.Timestamp | None = None,\n    ) -> RuleBacktestResult:
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if data.empty:
            empty = pd.DataFrame()
            return RuleBacktestResult(empty, calculate_metrics(empty, self.starting_capital), pd.DataFrame())

        frame = data.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame = add_indicators(frame)

        trades: list[dict] = []
        position = None

        for i in range(1, len(frame)):
            row = frame.iloc[i]
            previous = frame.iloc[i - 1]
            ts = row["timestamp"]

            if not _in_session(ts, self.instrument):
                continue

            if position is not None:
                exit_price, reason = _exit_for_bar(
                    position["direction"],
                    position["entry"],
                    position["stop_loss"],
                    position["target"],
                    float(row["high"]),
                    float(row["low"]),
                )
                if exit_price is not None:
                    signed_roi = (
                        (exit_price - position["entry"]) / position["entry"]
                        if position["direction"] == "BUY"
                        else (position["entry"] - exit_price) / position["entry"]
                    )
                    pnl = signed_roi * position["capital_at_risk"]
                    trades.append({
                        **position,
                        "exit_time": ts,
                        "exit_price": round(float(exit_price), 4),
                        "pnl": round(float(pnl), 2),
                        "roi_pct": round(float(signed_roi * 100), 4),
                        "reason": reason,
                    })
                    position = None
                    continue

                # Force exit at the final in-session candle of each trading day.
                next_ts = frame.iloc[i + 1]["timestamp"] if i + 1 < len(frame) else None
                if next_ts is None or next_ts.date() != ts.date() or not _in_session(next_ts, self.instrument):
                    exit_price = float(row["close"])
                    signed_roi = (
                        (exit_price - position["entry"]) / position["entry"]
                        if position["direction"] == "BUY"
                        else (position["entry"] - exit_price) / position["entry"]
                    )
                    pnl = signed_roi * position["capital_at_risk"]
                    trades.append({
                        **position,
                        "exit_time": ts,
                        "exit_price": round(exit_price, 4),
                        "pnl": round(float(pnl), 2),
                        "roi_pct": round(float(signed_roi * 100), 4),
                        "reason": "market_close",
                    })
                    position = None
                continue

            signals = evaluate_rules(row, previous, self.config)
            if not signals:
                continue

            buys = [s for s in signals if s.direction == "BUY"]
            sells = [s for s in signals if s.direction == "SELL"]
            if buys and sells:
                continue

            direction = "BUY" if buys else "SELL"
            entry = float(row["close"])
            stop = entry * (1 - self.config.stop_loss_pct) if direction == "BUY" else entry * (1 + self.config.stop_loss_pct)
            target = entry * (1 + self.config.target_roi_pct) if direction == "BUY" else entry * (1 - self.config.target_roi_pct)
            atr = float(row.get("ATR", 0.0) or 0.0)
            risk_distance = max(abs(entry - stop), atr * self.config.atr_risk_multiplier)
            if risk_distance <= 0:
                continue

            quantity = max(1, int(self.risk_per_trade / risk_distance))
            capital_at_risk = quantity * risk_distance
            position = {
                "entry_time": ts,
                "symbol": symbol,
                "direction": direction,
                "entry": round(entry, 4),
                "stop_loss": round(stop, 4),
                "target": round(target, 4),
                "quantity": quantity,
                "capital_at_risk": capital_at_risk,
                "atr": atr,
                "rule": "|".join(s.rule for s in signals),
                "rules": tuple(s.rule for s in signals),
            }

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            trades_df = pd.DataFrame(columns=[
                "entry_time", "exit_time", "symbol", "direction", "entry",
                "exit_price", "quantity", "pnl", "roi_pct", "reason", "rule",
            ])
        metrics = calculate_metrics(
            trades_df.rename(columns={"entry": "entry_price"}) if "entry" in trades_df.columns else trades_df,
            self.starting_capital,
        )
        return RuleBacktestResult(
            trades=trades_df,
            metrics=metrics,
            rule_performance=rank_rule_performance(trades_df),
        )
