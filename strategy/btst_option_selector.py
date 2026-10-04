from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from features.option_chain import OptionContract, analyze_option


@dataclass(frozen=True)
class BTSTOptionSignal:
    decision: str
    option_type: str
    strike: float | None
    expiry: str | None
    entry: float | None
    stop_loss: float | None
    target: float | None
    confidence: float
    reasons: tuple[str, ...]


def rank_btst_options(
    contracts: Iterable[OptionContract],
    underlying_direction: str,
    min_volume: float = 1000,
    max_spread_pct: float = 3.0,
    stop_pct: float = 0.20,
    reward_risk: float = 1.5,
) -> tuple[BTSTOptionSignal, pd.DataFrame]:
    """Select a liquid option for next-session carry.

    This is deliberately conservative: it requires real chain observations,
    rejects wide/illiquid contracts, and uses the next-session risk objective.
    It never invents expiry, bid/ask, or an option premium.
    """
    direction = str(underlying_direction).upper()
    desired = "CE" if direction in {"UP", "BULLISH", "BUY", "STRONG_UP"} else "PE" if direction in {"DOWN", "BEARISH", "SELL", "STRONG_DOWN"} else ""
    rows = []
    for c in contracts:
        if desired and c.option_type != desired:
            continue
        if c.ltp <= 0 or c.volume < min_volume:
            continue
        spread_pct = ((c.ask - c.bid) / c.ltp * 100) if c.ltp > 0 else 999.0
        if spread_pct > max_spread_pct:
            continue
        a = analyze_option(c)
        # Momentum and OI are evidence, not a guarantee of next-day direction.
        score = a.score + max(-10.0, min(10.0, c.ltp_change_pct * 0.75))
        if desired == c.option_type and c.oi_change > 0:
            score += 4
        rows.append({
            "Symbol": c.symbol, "Option": c.option_type, "Strike": c.strike,
            "Expiry": c.expiry, "LTP": c.ltp, "Bid": c.bid, "Ask": c.ask,
            "Volume": c.volume, "OI": c.open_interest, "OI Change": c.oi_change,
            "IV": c.implied_volatility, "Spread %": round(spread_pct, 2),
            "Score": round(max(0.0, min(100.0, score)), 2),
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return BTSTOptionSignal("WAIT", desired or "NA", None, None, None, None, None, 0.0,
                                ("No liquid real option-chain candidate passed the BTST filters",)), table
    table = table.sort_values(["Score", "Volume"], ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    entry = float(best["Ask"]) if float(best["Ask"]) > 0 else float(best["LTP"])
    sl = entry * (1 - stop_pct)
    target = entry + (entry - sl) * reward_risk
    confidence = float(best["Score"])
    decision = "BUY_CE_BTST" if best["Option"] == "CE" else "BUY_PE_BTST"
    reasons = (
        f"Liquid {best['Option']} candidate",
        f"Option score {confidence:.1f}",
        f"Spread {best['Spread %']:.2f}%",
        "Next-session carry requires overnight gap risk acceptance",
    )
    return BTSTOptionSignal(decision, str(best["Option"]), float(best["Strike"]), str(best["Expiry"]),
                            round(entry, 2), round(sl, 2), round(target, 2), round(confidence, 2), reasons), table
