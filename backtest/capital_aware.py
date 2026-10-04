from __future__ import annotations

from dataclasses import dataclass
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


class CapitalAwareRuleBacktestEngine:
    """Conservative capital-aware futures backtest.

    Signals are generated at a completed candle and filled on the next bar open.
    P&L is price points × point value × quantity. Position size is whole lots.
    The engine is deliberately intraday and includes execution costs.
    """

    def __init__(self, starting_capital: float = 100_000.0, risk_per_trade: float = 1_000.0,
                 instrument: str = "NIFTY", config: StrategyConfig | None = None,
                 lot_size: int = 65, point_value: float = 1.0,
                 margin_per_lot: float = 0.0, slippage_points: float = 0.25,
                 brokerage_per_order: float = 10.0, max_daily_loss: float | None = None,
                 max_trades_per_day: int = 5):
        if starting_capital <= 0 or risk_per_trade <= 0 or lot_size <= 0 or point_value <= 0:
            raise ValueError("capital, risk, lot_size and point_value must be positive")
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

    def _fill(self, price: float, direction: str, entry: bool) -> float:
        slip = self.slippage_points if direction == "BUY" else -self.slippage_points
        return price + slip if entry else price - slip

    def _size(self, entry: float, stop: float, equity: float) -> tuple[int, float]:
        risk_per_lot = abs(entry - stop) * self.point_value * self.lot_size
        if risk_per_lot <= 0:
            return 0, 0.0
        by_risk = int(self.risk_per_trade // risk_per_lot)
        by_margin = int(equity // self.margin_per_lot) if self.margin_per_lot > 0 else by_risk
        return min(by_risk, by_margin), risk_per_lot

    def run(self, data: pd.DataFrame, symbol: str = "NIFTY",
            evaluation_start: pd.Timestamp | None = None,
            evaluation_end: pd.Timestamp | None = None) -> StrictBacktestResult:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        frame = data.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        for c in ("open", "high", "low", "close", "volume"):
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
        frame = frame.dropna(subset=list(required)).sort_values("timestamp")
        invalid = ((frame[["open","high","low","close"]] <= 0).any(axis=1)
                   | (frame["volume"] < 0)
                   | (frame["high"] < frame[["open","close"]].max(axis=1))
                   | (frame["low"] > frame[["open","close"]].min(axis=1)))
        if invalid.any():
            raise ValueError(f"Invalid OHLCV rows: {int(invalid.sum())}")
        frame = frame.drop_duplicates("timestamp").reset_index(drop=True)
        frame = add_indicators(frame)
        if frame.empty:
            empty = pd.DataFrame()
            return StrictBacktestResult(empty, calculate_metrics(empty, self.starting_capital), pd.DataFrame(), pd.DataFrame(), {"status":"FAIL","reason":"empty_dataset"})

        start = pd.Timestamp(evaluation_start) if evaluation_start is not None else frame.timestamp.min()
        end = pd.Timestamp(evaluation_end) if evaluation_end is not None else frame.timestamp.max()
        trades: list[dict] = []
        equity_rows: list[dict] = []
        position = None
        equity = self.starting_capital
        daily_pnl: dict = {}
        daily_trades: dict = {}

        for i in range(1, len(frame)):
            row, prev = frame.iloc[i], frame.iloc[i-1]
            ts = row.timestamp
            if self.instrument == "NIFTY" and not (ts.timetz().replace(tzinfo=None) >= pd.Timestamp("09:15").time() and ts.timetz().replace(tzinfo=None) <= pd.Timestamp("15:40").time()):
                continue
            day = ts.date()
            daily_pnl.setdefault(day, 0.0); daily_trades.setdefault(day, 0)

            if position is not None:
                direction = position["direction"]
                if direction == "BUY":
                    stop_hit, target_hit = row.low <= position["stop_loss"], row.high >= position["target"]
                else:
                    stop_hit, target_hit = row.high >= position["stop_loss"], row.low <= position["target"]
                next_day = i + 1 >= len(frame) or frame.iloc[i+1].timestamp.date() != day
                exit_price, reason = None, None
                if stop_hit:
                    exit_price, reason = position["stop_loss"], "stop_loss"
                elif target_hit:
                    exit_price, reason = position["target"], "take_profit"
                elif next_day:
                    exit_price, reason = row.close, "market_close"
                if exit_price is not None:
                    filled_exit = self._fill(float(exit_price), direction, False)
                    signed_points = filled_exit - position["entry_price"] if direction == "BUY" else position["entry_price"] - filled_exit
                    gross = signed_points * self.point_value * position["quantity"]
                    costs = self.brokerage_per_order * 2
                    pnl = gross - costs
                    equity += pnl; daily_pnl[day] += pnl
                    trades.append({**position, "exit_time":ts, "exit_price":round(filled_exit,4), "gross_pnl":round(gross,2), "costs":round(costs,2), "pnl":round(pnl,2), "reason":reason})
                    equity_rows.append({"timestamp":ts,"equity":round(equity,2),"daily_pnl":round(daily_pnl[day],2)})
                    position = None
                continue

            if ts < start or ts > end or daily_pnl[day] <= -self.max_daily_loss or daily_trades[day] >= self.max_trades_per_day:
                continue
            signals = evaluate_rules(row, prev, self.config)
            buys = [s for s in signals if s.direction == "BUY"]; sells = [s for s in signals if s.direction == "SELL"]
            if not signals or (buys and sells):
                continue
            direction = "BUY" if buys else "SELL"
            if i + 1 >= len(frame) or frame.iloc[i+1].timestamp.date() != day:
                continue
            entry = self._fill(float(frame.iloc[i+1].open), direction, True)
            risk_distance = max(entry * self.config.stop_loss_pct, float(row.get("ATR", 0) or 0) * self.config.atr_stop_multiple)
            reward_distance = max(entry * self.config.min_target_pct, float(row.get("ATR", 0) or 0) * self.config.target_atr_multiple)
            stop = entry - risk_distance if direction == "BUY" else entry + risk_distance
            target = entry + reward_distance if direction == "BUY" else entry - reward_distance
            lots, risk_per_lot = self._size(entry, stop, equity)
            if lots <= 0:
                continue
            qty = lots * self.lot_size
            daily_trades[day] += 1
            position = {"entry_time":frame.iloc[i+1].timestamp,"signal_time":ts,"symbol":symbol,"direction":direction,
                        "entry_price":round(entry,4),"stop_loss":round(stop,4),"target":round(target,4),
                        "lots":lots,"quantity":qty,"risk_per_lot":round(risk_per_lot,2),
                        "risk_budget":round(self.risk_per_trade,2),"rule":"|".join(s.rule for s in signals),
                        "rules":tuple(s.rule for s in signals)}

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            trades_df = pd.DataFrame(columns=["entry_time","signal_time","exit_time","symbol","direction","entry_price","exit_price","lots","quantity","gross_pnl","costs","pnl","reason","rule"])
        metrics = calculate_metrics(trades_df, self.starting_capital)
        equity_df = pd.DataFrame(equity_rows)
        validation = {"status":"PASS","rows":len(frame),"evaluation_start":str(start),"evaluation_end":str(end),
                      "trading_days":int(frame.timestamp.dt.date.nunique()),"zero_volume_rows":int((frame.volume==0).sum()),
                      "starting_capital":self.starting_capital,"final_equity":round(equity,2),
                      "net_pnl":round(equity-self.starting_capital,2),"lot_size":self.lot_size,
                      "slippage_points":self.slippage_points,"brokerage_per_order":self.brokerage_per_order}
        return StrictBacktestResult(trades_df, metrics, rank_rule_performance(trades_df), equity_df, validation)
