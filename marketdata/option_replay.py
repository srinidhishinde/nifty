from __future__ import annotations

import pandas as pd

REQUIRED_OPTION_REPLAY_COLUMNS = {
    "timestamp", "symbol", "expiry", "strike", "option_type",
    "bid", "ask", "ltp", "volume", "oi", "iv",
}

OPTION_REPLAY_ALIASES = {
    "time": "timestamp", "datetime": "timestamp",
    "option": "option_type", "type": "option_type",
    "bid_price": "bid", "ask_price": "ask", "last_price": "ltp",
    "open_interest": "oi", "implied_volatility": "iv",
}


def normalize_option_replay(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    frame = frame.rename(columns={k: v for k, v in OPTION_REPLAY_ALIASES.items() if k in frame.columns})
    missing = REQUIRED_OPTION_REPLAY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Option replay missing required columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["expiry"] = pd.to_datetime(frame["expiry"], errors="coerce")
    for col in ("strike", "bid", "ask", "ltp", "volume", "oi", "iv"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["option_type"] = frame["option_type"].astype(str).str.upper().str.strip()
    invalid = (
        frame["timestamp"].isna() | frame["expiry"].isna() |
        ~frame["option_type"].isin(["CE", "PE"]) |
        (frame["strike"] <= 0) | (frame["bid"] < 0) |
        (frame["ask"] < frame["bid"]) | (frame["ltp"] <= 0)
    )
    if invalid.any():
        raise ValueError(f"Option replay contains {int(invalid.sum())} invalid rows")
    return frame.sort_values("timestamp").drop_duplicates(
        ["timestamp", "symbol", "expiry", "strike", "option_type"], keep="last"
    ).reset_index(drop=True)


def option_execution_price(row: pd.Series, side: str) -> float:
    side = side.upper()
    if side == "BUY":
        return float(row["ask"])
    if side == "SELL":
        return float(row["bid"])
    raise ValueError("side must be BUY or SELL")
