from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from features.option_chain import OptionContract, analyze_option


@dataclass(frozen=True)
class OptionChainSignal:
    direction: str
    confidence: float
    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    underlying_entry: float
    underlying_stop_loss: float
    underlying_take_profit: float
    reasons: tuple[str, ...]


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _buy_entry(contract: OptionContract) -> float | None:
    """Use the ask for a BUY when a valid ask exists; LTP is a fallback only."""
    ask = float(contract.ask or 0.0)
    ltp = float(contract.ltp or 0.0)
    return ask if ask > 0 and ask >= float(contract.bid or 0.0) else ltp if ltp > 0 else None


def generate_option_chain_signal(
    contracts: Iterable[OptionContract],
    spot: float,
    global_news_score: float = 0.0,
    stop_loss_pct: float = 0.015,
    target_roi_pct: float = 0.30,
    min_premium: float = 10.0,
    max_premium: float = 500.0,
    max_spread_pct: float = 0.05,
    direction_hint: str | None = None,
    require_two_sided_quote: bool = False,
) -> tuple[OptionChainSignal, pd.DataFrame]:
    """Generate an auditable CE/PE signal.

    The target is a per-trade option-premium objective, not a daily-return guarantee.
    Entry uses the ask when available so paper/live estimates are not based on an
    unrealistically favorable LTP fill.
    """
    rows = []
    normalized_hint = str(direction_hint or "").upper().strip()
    if normalized_hint not in {"CE", "PE", "WAIT"}:
        normalized_hint = None

    for contract in contracts:
        if normalized_hint in {"CE", "PE"} and contract.option_type != normalized_hint:
            continue
        if normalized_hint == "WAIT":
            continue
        entry = _buy_entry(contract)
        if entry is None or entry < min_premium or entry > max_premium:
            continue
        bid = float(contract.bid or 0.0)
        ask = float(contract.ask or 0.0)
        if require_two_sided_quote and not (bid > 0 and ask > 0 and ask >= bid):
            continue
        spread_pct = ((ask - bid) / ask) if bid > 0 and ask > 0 and ask >= bid else 0.0
        if spread_pct > max_spread_pct:
            continue
        if float(contract.volume or 0.0) <= 0 or float(contract.open_interest or 0.0) <= 0:
            continue
        analysis = analyze_option(contract)
        side_bias = 1.0 if contract.option_type == "CE" else -1.0
        news_boost = global_news_score * 12.0 * side_bias
        momentum_boost = max(-10.0, min(10.0, float(contract.ltp_change_pct or 0.0) * 0.5))
        confidence = _clamp(analysis.score + news_boost + momentum_boost)
        effective_stop_pct = max(stop_loss_pct, min(0.10, spread_pct * 2.0))
        risk_per_unit = entry * effective_stop_pct
        target_price = entry + (risk_per_unit * 2.0)
        stop_price = entry - risk_per_unit
        rows.append({
            "Side": contract.option_type,
            "Strike": contract.strike,
            "Signal": "BUY" if confidence >= 60 else "WAIT",
            "Confidence": round(confidence, 2),
            "Entry Price": round(entry, 2),
            "Stop Loss": round(stop_price, 2),
            "Take Profit": round(target_price, 2),
            "Target Gain %": round(((target_price / entry) - 1.0) * 100.0, 2),
            "Stop Risk %": round(((entry - stop_price) / entry) * 100.0, 2),
            "Max Gain %": round(((target_price / entry) - 1.0) * 100.0, 2),
            "Max Loss %": round(((entry - stop_price) / entry) * 100.0, 2),
            "Underlying Entry": round(float(spot), 2),
            "Underlying SL": None,
            "Underlying TP": None,
            "Global News": round(global_news_score, 3),
            "Score": analysis.score,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return (OptionChainSignal("WAIT", 0.0, None, None, None, float(spot), float(spot), float(spot), ("No liquid option contracts passed premium, volume, OI and spread filters",)), frame)

    grouped = frame.groupby("Side")["Confidence"].max().to_dict()
    ce, pe = float(grouped.get("CE", 0.0)), float(grouped.get("PE", 0.0))
    edge = abs(ce - pe)
    if normalized_hint == "CE":
        direction, confidence, reason = "BUY CE", ce, "Canonical underlying signal selected CE"
    elif normalized_hint == "PE":
        direction, confidence, reason = "BUY PE", pe, "Canonical underlying signal selected PE"
    elif edge < 10.0:
        direction, confidence, reason = "WAIT", max(ce, pe), "CE/PE confidence edge is below 10 points"
    elif ce > pe:
        direction, confidence, reason = "BUY CE", ce, "CE has the strongest option-chain evidence"
    else:
        direction, confidence, reason = "BUY PE", pe, "PE has the strongest option-chain evidence"

    winner_side = "CE" if direction == "BUY CE" else "PE" if direction == "BUY PE" else None
    if winner_side is None:
        return OptionChainSignal(
            "WAIT", round(confidence, 2), None, None, None,
            float(spot), float(spot), float(spot),
            (reason, "No canonical side selected; trade is blocked."),
        ), frame

    winner = frame[frame["Side"] == winner_side].sort_values("Confidence", ascending=False).iloc[0]
    entry = winner["Entry Price"] if pd.notna(winner["Entry Price"]) else None
    sl = winner["Stop Loss"] if pd.notna(winner["Stop Loss"]) else None
    tp = winner["Take Profit"] if pd.notna(winner["Take Profit"]) else None
    return OptionChainSignal(
        direction=direction,
        confidence=round(confidence, 2),
        entry_price=float(entry) if entry is not None else None,
        stop_loss=float(sl) if sl is not None else None,
        take_profit=float(tp) if tp is not None else None,
        underlying_entry=float(winner["Underlying Entry"]),
        underlying_stop_loss=float(winner["Underlying SL"]),
        underlying_take_profit=float(winner["Underlying TP"]) if pd.notna(winner["Underlying TP"]) else float(spot),
        reasons=(reason, f"Global news sentiment={global_news_score:.2f}"),
    ), frame
