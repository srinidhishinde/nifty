"""Conservative recursive backtest for predicted next-candle low/high bounds.

Research only: limit fills, intrabar order and transaction costs are assumptions,
not broker-confirmed executions. When stop and target are both touched in one
candle, stop-loss is assumed first.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def simulate_recursive_low_high(
    candles: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    initial_capital: float = 30_000.0,
    lot_size: int = 1,
    stop_loss_pct: float = 0.05,
    cost_bps_per_side: float = 10.0,
    slippage_bps_per_side: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate one long-only limit-entry trade at a time and compound equity.

    candles require timestamp/open/high/low/close. predictions require
    target_timestamp/predicted_low/predicted_high. Quantity is whole lots,
    sized using equity at entry. Fees and slippage are approximated in bps.
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital must be > 0")
    if lot_size < 1:
        raise ValueError("lot_size must be >= 1")
    if not 0 < stop_loss_pct < 1:
        raise ValueError("stop_loss_pct must be between 0 and 1")
    if cost_bps_per_side < 0 or slippage_bps_per_side < 0:
        raise ValueError("cost and slippage bps must be >= 0")

    required_candles = {"timestamp", "open", "high", "low", "close"}
    required_preds = {"target_timestamp", "predicted_low", "predicted_high"}
    if not required_candles.issubset(candles.columns):
        raise ValueError(f"candles missing columns: {sorted(required_candles - set(candles.columns))}")
    if not required_preds.issubset(predictions.columns):
        raise ValueError(f"predictions missing columns: {sorted(required_preds - set(predictions.columns))}")

    bars = candles.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    bars = bars.dropna(subset=["timestamp", "open", "high", "low", "close"])
    bars = bars.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    preds = predictions.copy()
    preds["target_timestamp"] = pd.to_datetime(preds["target_timestamp"], errors="coerce", utc=True)
    for col in ("predicted_low", "predicted_high"):
        preds[col] = pd.to_numeric(preds[col], errors="coerce")
    preds = preds.dropna(subset=["target_timestamp", "predicted_low", "predicted_high"])
    pred_by_time = preds.drop_duplicates("target_timestamp", keep="last").set_index("target_timestamp")

    equity = float(initial_capital)
    ledger: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    cost_rate = cost_bps_per_side / 10_000
    slip_rate = slippage_bps_per_side / 10_000

    def close_position(exit_time: Any, raw_exit: float, reason: str) -> None:
        nonlocal equity, position
        assert position is not None
        qty = int(position["quantity"])
        entry_fill = float(position["entry_price"]) * (1 + slip_rate)
        exit_fill = float(raw_exit) * (1 - slip_rate)
        gross = (exit_fill - entry_fill) * qty
        costs = (entry_fill * qty + exit_fill * qty) * cost_rate
        net = gross - costs
        starting_equity = float(position["capital_before"])
        equity = max(0.0, equity + net)
        ledger.append({
            "trade": len(ledger) + 1,
            "entry_time": position["entry_time"],
            "exit_time": exit_time,
            "entry_price": float(position["entry_price"]),
            "exit_price": float(raw_exit),
            "quantity": qty,
            "lots": qty // lot_size,
            "stop_price": float(position["stop_price"]),
            "target_price": float(position["target_price"]),
            "exit_reason": reason,
            "gross_pnl": gross,
            "estimated_costs": costs,
            "net_pnl": net,
            "capital_before": starting_equity,
            "capital_after": equity,
            "return_on_starting_capital_pct": net / starting_equity * 100 if starting_equity else 0.0,
        })
        position = None

    for _, bar in bars.iterrows():
        ts = bar["timestamp"]
        op, hi, lo, cl = (float(bar[c]) for c in ("open", "high", "low", "close"))

        if position is not None:
            stop, target = float(position["stop_price"]), float(position["target_price"])
            # Opening gaps are filled at the open where it is worse than the trigger.
            if op <= stop:
                close_position(ts, op, "STOP_GAP")
            elif op >= target:
                close_position(ts, target, "TARGET_GAP")
            else:
                hit_stop = lo <= stop
                hit_target = hi >= target
                if hit_stop:  # conservative if both are touched in this bar
                    close_position(ts, stop, "STOP_LOSS" if not hit_target else "STOP_FIRST_BOTH_TOUCHED")
                elif hit_target:
                    close_position(ts, target, "TARGET")
            if position is not None:
                continue

        if ts not in pred_by_time.index or equity <= 0:
            continue
        pred = pred_by_time.loc[ts]
        entry = float(pred["predicted_low"])
        target = float(pred["predicted_high"])
        if entry <= 0 or target <= entry or op < 0:
            continue
        # A limit buy is considered filled only if the observed candle trades at/under it.
        if lo > entry:
            continue
        qty = int(equity // (entry * lot_size)) * lot_size
        if qty < lot_size:
            continue
        position = {
            "entry_time": ts,
            "entry_price": entry,
            "target_price": target,
            "stop_price": entry * (1 - stop_loss_pct),
            "quantity": qty,
            "capital_before": equity,
        }
        # Resolve same-candle exit with conservative stop-first ordering.
        hit_stop = lo <= float(position["stop_price"])
        hit_target = hi >= target
        if hit_stop:
            close_position(ts, float(position["stop_price"]), "STOP_LOSS" if not hit_target else "STOP_FIRST_BOTH_TOUCHED")
        elif hit_target:
            close_position(ts, target, "TARGET")

    if position is not None and not bars.empty:
        last = bars.iloc[-1]
        close_position(last["timestamp"], float(last["close"]), "END_OF_DATA")

    ledger_df = pd.DataFrame(ledger)
    if ledger_df.empty:
        summary: dict[str, Any] = {
            "status": "NO_TRADES",
            "initial_capital": float(initial_capital),
            "ending_capital": float(initial_capital),
            "net_profit": 0.0,
            "return_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "estimated_costs": 0.0,
            "max_drawdown_pct": 0.0,
            "lot_size": int(lot_size),
            "stop_loss_pct": float(stop_loss_pct),
            "cost_bps_per_side": float(cost_bps_per_side),
            "slippage_bps_per_side": float(slippage_bps_per_side),
        }
        return ledger_df, summary

    equity_curve = pd.concat([
        pd.Series([float(initial_capital)]),
        ledger_df["capital_after"].reset_index(drop=True),
    ], ignore_index=True)
    peaks = equity_curve.cummax()
    drawdowns = (equity_curve - peaks) / peaks.replace(0, pd.NA)
    net_profit = float(equity - initial_capital)
    summary = {
        "status": "EVALUATED",
        "initial_capital": float(initial_capital),
        "ending_capital": float(equity),
        "net_profit": net_profit,
        "return_pct": net_profit / initial_capital * 100,
        "trades": int(len(ledger_df)),
        "wins": int((ledger_df["net_pnl"] > 0).sum()),
        "losses": int((ledger_df["net_pnl"] < 0).sum()),
        "win_rate_pct": float((ledger_df["net_pnl"] > 0).mean() * 100),
        "gross_profit": float(ledger_df.loc[ledger_df["net_pnl"] > 0, "net_pnl"].sum()),
        "gross_loss": float(ledger_df.loc[ledger_df["net_pnl"] < 0, "net_pnl"].sum()),
        "estimated_costs": float(ledger_df["estimated_costs"].sum()),
        "max_drawdown_pct": float(drawdowns.min() * 100),
        "lot_size": int(lot_size),
        "stop_loss_pct": float(stop_loss_pct),
        "cost_bps_per_side": float(cost_bps_per_side),
        "slippage_bps_per_side": float(slippage_bps_per_side),
        "assumptions": "Long-only predicted-low limit entries; one position at a time; whole lots; stop-first when stop and target both touched; estimated bps costs/slippage; no broker fill guarantee.",
    }
    return ledger_df, summary
