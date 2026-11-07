"""Transparent, rule-based market context analysis.

This module deliberately avoids fitted weights and treats missing/conflicting
inputs as WAIT. Option OI is interpreted together with option price change;
OI by itself cannot identify which participant initiated a position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Evidence:
    label: str
    direction: int  # +1 bullish, -1 bearish, 0 neutral/unknown
    detail: str


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def analyze_price_action(candles: pd.DataFrame) -> Evidence:
    required = {"open", "high", "low", "close"}
    if candles is None or not required.issubset(candles.columns) or len(candles) < 6:
        return Evidence("Price action", 0, "Need at least 6 OHLC candles.")
    data = candles.copy()
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    if len(data) < 6:
        return Evidence("Price action", 0, "Not enough valid OHLC candles.")
    recent = data.tail(6)
    last = recent.iloc[-1]
    prior = recent.iloc[-2]
    higher_high = last["high"] > prior["high"]
    higher_low = last["low"] > prior["low"]
    lower_high = last["high"] < prior["high"]
    lower_low = last["low"] < prior["low"]
    if higher_high and higher_low and last["close"] > last["open"]:
        return Evidence("Price action", 1, "Latest candle confirms a higher high and higher low.")
    if lower_high and lower_low and last["close"] < last["open"]:
        return Evidence("Price action", -1, "Latest candle confirms a lower high and lower low.")
    return Evidence("Price action", 0, "Recent structure is mixed; no directional confirmation.")


def analyze_option_positioning(chain: pd.DataFrame) -> tuple[Evidence, pd.DataFrame]:
    required = {"option_type", "ltp_change", "oi_change"}
    if chain is None or not required.issubset(chain.columns):
        return Evidence("Option positioning", 0, "Need option_type, ltp_change and oi_change."), pd.DataFrame()
    data = chain.copy()
    data["option_type"] = data["option_type"].astype(str).str.upper().str.strip()
    for col in ("ltp_change", "oi_change"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["ltp_change", "oi_change"])
    data = data[data["option_type"].isin(["CE", "PE"])]
    if data.empty:
        return Evidence("Option positioning", 0, "No valid CE/PE rows.")
    def classify(row: pd.Series) -> str:
        price_change = row["ltp_change"]
        oi_change = row["oi_change"]
        # A flat/unchanged premium cannot distinguish writing from buildup.
        if price_change == 0 or oi_change == 0:
            return "Mixed / unchanged"
        if price_change > 0 and oi_change > 0:
            return "Long build-up"
        if price_change < 0 and oi_change > 0:
            return "Possible writing"
        if price_change > 0 and oi_change < 0:
            return "Short covering"
        if price_change < 0 and oi_change < 0:
            return "Long unwinding"
        return "Mixed / unchanged"
    data["position_read"] = data.apply(classify, axis=1)
    # Direction is assigned only to paired price/OI behaviour, not OI alone.
    votes = []
    for _, row in data.iterrows():
        kind, side = row["position_read"], row["option_type"]
        if kind in ("Long build-up", "Short covering"):
            votes.append(1 if side == "CE" else -1)
        elif kind == "Possible writing":
            votes.append(-1 if side == "CE" else 1)
        elif kind == "Long unwinding":
            votes.append(-1 if side == "CE" else 1)
    if not votes:
        return Evidence("Option positioning", 0, "No directional price/OI combinations."), data
    balance = sum(votes)
    if balance > 0:
        return Evidence("Option positioning", 1, f"More bullish than bearish price/OI combinations ({balance:+d} net votes)."), data
    if balance < 0:
        return Evidence("Option positioning", -1, f"More bearish than bullish price/OI combinations ({balance:+d} net votes)."), data
    return Evidence("Option positioning", 0, "CE and PE positioning evidence is balanced."), data


def analyze_breadth(breadth: pd.DataFrame) -> Evidence:
    required = {"symbol", "change_pct"}
    if breadth is None or not required.issubset(breadth.columns):
        return Evidence("NIFTY 50 breadth", 0, "Need constituent symbol and change_pct columns.")
    data = breadth.copy()
    data["change_pct"] = pd.to_numeric(data["change_pct"], errors="coerce")
    data = data.dropna(subset=["change_pct"])
    if data.empty:
        return Evidence("NIFTY 50 breadth", 0, "No valid constituent changes.")
    advancing = int((data["change_pct"] > 0).sum())
    declining = int((data["change_pct"] < 0).sum())
    if advancing > declining:
        direction = 1
    elif declining > advancing:
        direction = -1
    else:
        direction = 0
    detail = f"{advancing} advancing, {declining} declining, {len(data)} valid constituents."
    if "above_vwap" in data.columns:
        above = data["above_vwap"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        detail += f" {int(above.sum())}/{len(data)} above VWAP."
    return Evidence("NIFTY 50 breadth", direction, detail)


def combine_evidence(evidence: list[Evidence]) -> tuple[str, str]:
    directional = [item.direction for item in evidence if item.direction]
    if len(directional) < 2:
        return "WAIT", "At least two independent directional inputs are required."
    bullish = sum(value == 1 for value in directional)
    bearish = sum(value == -1 for value in directional)
    if bullish >= 2 and bearish == 0:
        return "BULLISH BIAS — CE SETUP ONLY", "Price action, breadth and/or positioning agree; still wait for a valid entry trigger."
    if bearish >= 2 and bullish == 0:
        return "BEARISH BIAS — PE SETUP ONLY", "Price action, breadth and/or positioning agree; still wait for a valid entry trigger."
    return "WAIT", "Evidence conflicts. Do not force a CE/PE trade."
