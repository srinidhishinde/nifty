from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from features.option_chain import OptionContract, analyze_option
from strategy.session_rules import NIFTY_CLOSING_RESEARCH, in_window


@dataclass(frozen=True)
class ClosingSessionSignal:
    decision: str
    option_type: str
    strike: float | None
    entry: float | None
    stop_loss: float | None
    target: float | None
    exit_by: str
    confidence: float
    status: str
    reasons: tuple[str, ...]


def evaluate_closing_session(
    candles: pd.DataFrame,
    contracts: Iterable[OptionContract],
    news_score: float = 0.0,
    min_confidence: float = 68.0,
    option_stop_pct: float = 0.20,
    reward_risk: float = 1.5,
) -> tuple[ClosingSessionSignal, pd.DataFrame]:
    """NIFTY 15:15–15:40 derivatives closing-session decision engine.

    It requires completed candles in the research window and a real option
    chain. It is not an NSE CAS order simulator: NSE CAS applies to the cash
    segment, while equity derivatives have a 15:40 normal-market close.
    """
    required = {"timestamp", "close"}
    if not required.issubset(candles.columns):
        raise ValueError("Closing-session candles require timestamp and close")
    frame = candles.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if getattr(frame["timestamp"].dt, "tz", None) is not None:
        frame["timestamp"] = frame["timestamp"].dt.tz_convert("Asia/Kolkata")
    else:
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("Asia/Kolkata")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
    window = frame[frame["timestamp"].map(lambda x: in_window(x, NIFTY_CLOSING_RESEARCH))]
    if len(window) < 2:
        return ClosingSessionSignal("WAIT", "NA", None, None, None, None, "15:40", 0.0, "NO_DATA",
                                    ("Need at least two completed candles inside 15:15–15:40",)), pd.DataFrame()

    last = window.iloc[-1]
    prev = window.iloc[-2]
    last_close, prev_close = float(last["close"]), float(prev["close"])
    trend = 1 if last_close > prev_close else -1 if last_close < prev_close else 0
    if "EMA20" in window.columns and "EMA50" in window.columns:
        e20, e50 = last["EMA20"], last["EMA50"]
        if pd.notna(e20) and pd.notna(e50):
            trend = 1 if float(e20) > float(e50) else -1 if float(e20) < float(e50) else 0

    rows = []
    for c in contracts:
        a = analyze_option(c)
        if c.ltp <= 0 or c.volume <= 0:
            continue
        spread_pct = ((c.ask - c.bid) / c.ltp * 100) if c.ltp else 999
        if spread_pct > 3:
            continue
        side_match = (trend > 0 and c.option_type == "CE") or (trend < 0 and c.option_type == "PE")
        score = a.score + (10 if side_match else -8)
        score += max(-8, min(8, news_score * 8 * (1 if c.option_type == "CE" else -1)))
        if c.ltp_change_pct * trend > 0:
            score += 5
        rows.append({
            "Symbol": c.symbol, "Option": c.option_type, "Strike": c.strike, "Expiry": c.expiry,
            "LTP": c.ltp, "Bid": c.bid, "Ask": c.ask, "Volume": c.volume,
            "OI": c.open_interest, "OI Change": c.oi_change, "IV": c.implied_volatility,
            "Spread %": round(spread_pct, 2), "Score": round(max(0, min(100, score)), 2),
        })
    table = pd.DataFrame(rows)
    if table.empty or trend == 0:
        return ClosingSessionSignal("WAIT", "NA", None, None, None, None, "15:40", 0.0, "NO_TRADE",
                                    ("No liquid option candidate or no directional underlying structure",)), table
    table = table.sort_values("Score", ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    if float(best["Score"]) < min_confidence:
        return ClosingSessionSignal("WAIT", str(best["Option"]), float(best["Strike"]), None, None, None, "15:40",
                                    float(best["Score"]), "NO_TRADE",
                                    (f"Top option score {best['Score']:.1f} is below {min_confidence:.1f}",)), table
    entry = float(best["Ask"]) if float(best["Ask"]) > 0 else float(best["LTP"])
    stop = entry * (1 - option_stop_pct)
    target = entry + (entry - stop) * reward_risk
    return ClosingSessionSignal(
        "BUY_CE" if best["Option"] == "CE" else "BUY_PE",
        str(best["Option"]), float(best["Strike"]), round(entry, 2), round(stop, 2), round(target, 2),
        "15:35–15:40", float(best["Score"]), "READY",
        ("Closing-session trend and option-chain alignment", f"Real option ask used for entry", "Hard exit by 15:40"),
    ), table
