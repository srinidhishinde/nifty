from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtest.metrics import calculate_metrics
from features.technical.indicators import add_indicators
from strategy.rules import StrategyConfig, generate_signal, rank_rule_performance
from risk.daily_guard import enforce_daily_loss


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

    def __init__(self, starting_capital: float, risk_fraction: float = 0.01,
                 instrument: str = "NIFTY", config: StrategyConfig | None = None,
                 lot_size: int = 65, point_value: float = 1.0,
                 margin_per_lot: float = 0.0, slippage_points: float = 0.25,
                 brokerage_per_order: float = 10.0, max_daily_loss_fraction: float = 0.02,
                 max_trades_per_day: int = 5):
        if starting_capital <= 0 or risk_fraction <= 0 or risk_fraction >= 1 or lot_size <= 0 or point_value <= 0:
            raise ValueError("capital, risk_fraction, lot_size and point_value must be positive; risk_fraction must be < 1")
        if max_daily_loss_fraction <= 0 or max_daily_loss_fraction >= 1:
            raise ValueError("max_daily_loss_fraction must be between 0 and 1")
        self.starting_capital = float(starting_capital)
        self.risk_fraction = float(risk_fraction)
        self.max_daily_loss_fraction = float(max_daily_loss_fraction)
        self.instrument = instrument.upper()
        self.config = config or StrategyConfig()
        self.lot_size = int(lot_size)
        self.point_value = float(point_value)
        self.margin_per_lot = float(margin_per_lot)
        self.slippage_points = float(slippage_points)
        self.brokerage_per_order = float(brokerage_per_order)
        self.max_trades_per_day = int(max_trades_per_day)

    def _fill(self, price: float, direction: str, entry: bool) -> float:
        slip = self.slippage_points if direction == "BUY" else -self.slippage_points
        return price + slip if entry else price - slip

    def _size(self, entry: float, stop: float, equity: float) -> tuple[int, float]:
        risk_per_lot = abs(entry - stop) * self.point_value * self.lot_size
        if risk_per_lot <= 0:
            return 0, 0.0
        risk_budget = equity * self.risk_fraction
        by_risk = int(risk_budget // risk_per_lot)
        by_margin = int(equity // self.margin_per_lot) if self.margin_per_lot > 0 else by_risk
        return min(by_risk, by_margin), risk_per_lot

    @staticmethod
    def _resolve_exit(row: pd.Series, position: dict) -> tuple[float | None, str | None]:
        """Resolve an OHLC bar conservatively when stop and target are both touched.

        OHLC data does not reveal the intrabar path. If both protective levels
        are touched on the same bar, stop-loss wins so the backtest cannot
        manufacture an optimistic fill sequence.
        """
        direction = position["direction"]
        if direction == "BUY":
            stop_hit = float(row.low) <= float(position["stop_loss"])
            target_hit = float(row.high) >= float(position["target"])
        else:
            stop_hit = float(row.high) >= float(position["stop_loss"])
            target_hit = float(row.low) <= float(position["target"])
        if stop_hit:
            return float(position["stop_loss"]), "stop_loss"
        if target_hit:
            return float(position["target"]), "take_profit"
        return None, None

    def run(self, data: pd.DataFrame, symbol: str = "NIFTY",
            evaluation_start: pd.Timestamp | None = None,
            evaluation_end: pd.Timestamp | None = None) -> StrictBacktestResult:
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        frame = data.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        if getattr(frame["timestamp"].dt, "tz", None) is not None:
            frame["timestamp"] = frame["timestamp"].dt.tz_convert("Asia/Kolkata")
        else:
            frame["timestamp"] = frame["timestamp"].dt.tz_localize("Asia/Kolkata")
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
            return StrictBacktestResult(
                empty, calculate_metrics(empty, self.starting_capital),
                pd.DataFrame(), pd.DataFrame(),
                {"status": "FAIL", "reason": "empty_dataset"},
            )

        start = pd.Timestamp(evaluation_start) if evaluation_start is not None else frame.timestamp.min()
        end = pd.Timestamp(evaluation_end) if evaluation_end is not None else frame.timestamp.max()
        start = start.tz_localize("Asia/Kolkata") if start.tzinfo is None else start.tz_convert("Asia/Kolkata")
        end = end.tz_localize("Asia/Kolkata") if end.tzinfo is None else end.tz_convert("Asia/Kolkata")

        trades: list[dict] = []
        equity_rows: list[dict] = []
        position = None
        equity = self.starting_capital
        daily_pnl: dict = {}
        daily_trades: dict = {}
        day_start_equity: dict = {}
        cooloff_until: dict = {}

        diagnostics = {
            "bars_considered": 0, "rule_trigger_bars": 0,
            "conflicting_signal_bars": 0, "qualified_signal_bars": 0,
            "rejected_no_next_bar": 0, "rejected_risk_budget": 0,
            "rejected_daily_limit": 0, "rejected_trade_limit": 0,
            "rejected_signal_gate": 0,
        }

        session_start = pd.Timestamp("09:15").time()
        session_end = pd.Timestamp("15:40").time()

        for i in range(1, len(frame)):
            row, prev = frame.iloc[i], frame.iloc[i - 1]
            ts = row.timestamp
            clock = ts.timetz().replace(tzinfo=None)
            if self.instrument == "NIFTY" and (clock < session_start or clock > session_end):
                continue
            day = ts.date()
            daily_pnl.setdefault(day, 0.0)
            daily_trades.setdefault(day, 0)
            day_start_equity.setdefault(day, equity)

            if position is not None:
                next_day = i + 1 >= len(frame) or frame.iloc[i + 1].timestamp.date() != day
                exit_price, reason = self._resolve_exit(row, position)
                if exit_price is None and (next_day or (self.instrument == "NIFTY" and clock >= session_end)):
                    exit_price, reason = float(row.close), "market_close"
                if exit_price is not None:
                    direction = position["direction"]
                    filled_exit = self._fill(float(exit_price), direction, False)
                    signed_points = (
                        filled_exit - position["entry_price"]
                        if direction == "BUY"
                        else position["entry_price"] - filled_exit
                    )
                    gross = signed_points * self.point_value * position["quantity"]
                    costs = self.brokerage_per_order * 2
                    pnl = gross - costs
                    equity += pnl
                    daily_pnl[day] += pnl
                    trades.append({
                        **position, "exit_time": ts,
                        "exit_price": round(filled_exit, 4),
                        "gross_pnl": round(gross, 2),
                        "costs": round(costs, 2),
                        "pnl": round(pnl, 2), "reason": reason,
                    })
                    equity_rows.append({
                        "timestamp": ts, "equity": round(equity, 2),
                        "daily_pnl": round(daily_pnl[day], 2),
                    })
                    position = None
                continue

            if ts < start or ts > end:
                continue
            diagnostics["bars_considered"] += 1

            loss_limit = day_start_equity[day] * self.max_daily_loss_fraction
            volatility_band = 1.0
            atr_now = float(row.get("ATR5", 0) or 0)
            atr_base = float(row.get("ATR", 0) or 0)
            if atr_now > 0 and atr_base > 0:
                volatility_band = atr_now / atr_base
            risk_decision = enforce_daily_loss(
                max(0.0, -daily_pnl[day]),
                loss_limit,
                volatility_band,
                now=ts.to_pydatetime(),
                cooloff_until=cooloff_until.get(day),
            )
            if not risk_decision.allowed:
                diagnostics["rejected_daily_limit"] += 1
                continue
            if risk_decision.cooloff_minutes > 0 and risk_decision.reason == "first_daily_loss_breach_reduced_size":
                cooloff_until[day] = ts.to_pydatetime() + pd.Timedelta(minutes=risk_decision.cooloff_minutes)
            if daily_trades[day] >= self.max_trades_per_day:
                diagnostics["rejected_trade_limit"] += 1
                continue

            # Canonical decision path: the same generate_signal() used by
            # strategy evaluation/live logic. This prevents backtest/live drift.
            decision = generate_signal(frame.iloc[: i + 1], self.config)
            if decision.direction == "WAIT":
                continue
            diagnostics["rule_trigger_bars"] += 1
            if not decision.valid:
                diagnostics["rejected_signal_gate"] += 1
                continue

            direction = decision.direction
            diagnostics["qualified_signal_bars"] += 1
            if i + 1 >= len(frame) or frame.iloc[i + 1].timestamp.date() != day:
                diagnostics["rejected_no_next_bar"] += 1
                continue

            next_bar = frame.iloc[i + 1]
            entry = self._fill(float(next_bar.open), direction, True)

            # Keep stop/target distances determined by the completed signal bar,
            # then translate those distances to the actual next-bar fill.
            signal_close = float(row.close)
            risk_distance = abs(signal_close - float(decision.stop_loss))
            reward_distance = abs(float(decision.target) - signal_close)
            if risk_distance <= 0 or reward_distance <= 0:
                diagnostics["rejected_signal_gate"] += 1
                continue

            stop = entry - risk_distance if direction == "BUY" else entry + risk_distance
            target = entry + reward_distance if direction == "BUY" else entry - reward_distance
            reward_risk = reward_distance / risk_distance
            if reward_risk < self.config.min_reward_risk:
                diagnostics["rejected_signal_gate"] += 1
                continue

            lots, risk_per_lot = self._size(entry, stop, equity)
            if risk_decision.size_multiplier < 1.0:
                lots = int(lots * risk_decision.size_multiplier)
            if lots <= 0:
                diagnostics["rejected_risk_budget"] += 1
                continue

            qty = lots * self.lot_size
            daily_trades[day] += 1
            position = {
                "entry_time": next_bar.timestamp,
                "signal_time": ts,
                "symbol": symbol,
                "direction": direction,
                "entry_price": round(entry, 4),
                "stop_loss": round(stop, 4),
                "target": round(target, 4),
                "lots": lots,
                "quantity": qty,
                "risk_per_lot": round(risk_per_lot, 2),
                "risk_budget": round(equity * self.risk_fraction, 2),
                "confidence": decision.confidence,
                "rule": "|".join(decision.rules),
                "rules": tuple(decision.rules),
            }

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            trades_df = pd.DataFrame(columns=[
                "entry_time","signal_time","exit_time","symbol","direction",
                "entry_price","exit_price","lots","quantity","gross_pnl",
                "costs","pnl","reason","rule",
            ])
        metrics = calculate_metrics(trades_df, self.starting_capital)
        equity_df = pd.DataFrame(equity_rows)
        validation = {
            "status": "PASS",
            "rows": len(frame),
            "evaluation_start": str(start),
            "evaluation_end": str(end),
            "trading_days": int(frame.timestamp.dt.date.nunique()),
            "zero_volume_rows": int((frame.volume == 0).sum()),
            "starting_capital": self.starting_capital,
            "final_equity": round(equity, 2),
            "net_pnl": round(equity - self.starting_capital, 2),
            "lot_size": self.lot_size,
            "slippage_points": self.slippage_points,
            "brokerage_per_order": self.brokerage_per_order,
            **diagnostics,
            "risk_fraction": self.risk_fraction,
            "max_daily_loss_fraction": self.max_daily_loss_fraction,
            "point_value": self.point_value,
            "min_stop_pct": self.config.stop_loss_pct,
            "min_reward_risk": self.config.min_reward_risk,
        }
        return StrictBacktestResult(
            trades_df, metrics, rank_rule_performance(trades_df),
            equity_df, validation,
        )
