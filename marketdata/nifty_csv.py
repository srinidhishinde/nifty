from __future__ import annotations

import re
import pandas as pd

ALIASES = {
    "timestamp": ["timestamp", "datetime", "date", "time", "timestamp ", "datetime "],
    "open": ["open", "open "],
    "high": ["high", "high "],
    "low": ["low", "low "],
    "close": ["close", "close ", "closing price"],
    "volume": ["volume", "volume ", "shares traded", "shares traded ", "total volume"],
}


def _key(value: object) -> str:
    return re.sub(r"\\s+", " ", str(value).strip().lower())


def normalize_nifty_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common broker/exchange OHLCV headers into the canonical schema."""
    lookup = {_key(column): column for column in df.columns}
    resolved: dict[str, object] = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            if _key(alias) in lookup:
                resolved[target] = lookup[_key(alias)]
                break

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(resolved)
    if missing:
        raise ValueError(f"Missing NIFTY OHLCV fields: {sorted(missing)}")

    out = pd.DataFrame({target: df[source] for target, source in resolved.items()})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", dayfirst=True)
    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).sort_values("timestamp").reset_index(drop=True)
