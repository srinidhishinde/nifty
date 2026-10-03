from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class BuyTodaySellTomorrowResult:
    trades: pd.DataFrame
    accuracy_pct: float
    total_return_pct: float
    net_pnl: float

def run_buy_today_sell_tomorrow(daily_data: pd.DataFrame, stop_loss_pct: float = 0.015) -> BuyTodaySellTomorrowResult:
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(daily_data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = daily_data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    for c in ("open", "high", "low", "close"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.dropna(subset=list(required)).sort_values("timestamp").reset_index(drop=True)
    trades = []
    for i in range(len(frame) - 1):
        today, tomorrow = frame.iloc[i], frame.iloc[i + 1]
        entry = float(today["close"])
        stop = entry * (1 - stop_loss_pct)
        if float(tomorrow["low"]) <= stop:
            exit_price, reason = stop, "stop_loss"
        else:
            exit_price, reason = float(tomorrow["close"]), "next_close"
        roi = (exit_price - entry) / entry
        trades.append({
            "entry_time": today["timestamp"], "exit_time": tomorrow["timestamp"],
            "direction": "BUY", "entry": round(entry, 4), "stop_loss": round(stop, 4),
            "take_profit": round(entry * 1.40, 4), "exit_price": round(exit_price, 4),
            "roi_pct": round(roi * 100, 4), "pnl": round(roi * entry, 4),
            "reason": reason, "strategy": "buy_today_sell_tomorrow",
        })
    result = pd.DataFrame(trades)
    if result.empty:
        return BuyTodaySellTomorrowResult(result, 0.0, 0.0, 0.0)
    wins = int((result["pnl"] > 0).sum())
    return BuyTodaySellTomorrowResult(result, round(wins / len(result) * 100, 2),
                                      round(float(result["roi_pct"].sum()), 2),
                                      round(float(result["pnl"].sum()), 4))
