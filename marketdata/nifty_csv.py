from __future__ import annotations

import re
import pandas as pd

ALIASES = {
    "timestamp": ["timestamp", "date", "date "],
    "open": ["open", "open "],
    "high": ["high", "high "],
    "low": ["low", "low "],
    "close": ["close", "close "],
    "volume": ["volume", "shares traded", "shares traded "],
}

def normalize_nifty_csv(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {re.sub(r"\s+", " ", str(c).strip().lower()): c for c in df.columns}
    resolved = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            key = re.sub(r"\s+", " ", alias.strip().lower())
            if key in lookup:
                resolved[target] = lookup[key]
                break
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(resolved)
    if missing:
        raise ValueError(f"Missing NIFTY OHLC fields: {sorted(missing)}")
    out = pd.DataFrame({target: df[source] for target, source in resolved.items()})
    if "volume" not in out:
        out["volume"] = 0.0
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", dayfirst=True)
    for c in ("open","high","low","close","volume"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").reset_index(drop=True)
