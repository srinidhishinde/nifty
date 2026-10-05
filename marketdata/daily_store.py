from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.settings import settings


class DailyMarketStore:
    """Immutable daily-partitioned storage for research/backtest datasets.

    Files are deliberately separated by asset class and date:
    data/market/NIFTY/YYYY-MM-DD/...
    data/market/MCX/YYYY-MM-DD/...
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or settings.market_data_root)

    def directory(self, instrument: str, session_date: date) -> Path:
        return self.root / instrument.upper() / session_date.isoformat()

    def save_candles(self, instrument: str, frame: pd.DataFrame, session_date: date | None = None) -> Path:
        if frame.empty:
            raise ValueError("Cannot persist an empty market-data frame.")
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

        out = frame.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
        if out.empty:
            raise ValueError("No valid timestamps remain after normalization.")

        day = session_date or out["timestamp"].iloc[0].date()
        folder = self.directory(instrument, day)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{instrument.upper()}_ohlcv.csv"
        out.to_csv(path, index=False)
        return path

    def save_option_chain(self, instrument: str, frame: pd.DataFrame, session_date: date | None = None) -> Path:
        if frame.empty:
            raise ValueError("Cannot persist an empty option-chain frame.")
        required = {"Strike", "option_type"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required option-chain columns: {sorted(missing)}")

        out = frame.copy()
        out["Strike"] = pd.to_numeric(out["Strike"], errors="coerce")
        out["option_type"] = out["option_type"].astype(str).str.upper().str.strip()
        out = out.dropna(subset=["Strike"])
        out = out[out["option_type"].isin({"CE", "PE"})]
        if out.empty:
            raise ValueError("No valid CE/PE option-chain rows remain after normalization.")
        out["Strike"] = out["Strike"].astype("float64")

        day = session_date or pd.Timestamp.now(tz="Asia/Kolkata").date()
        folder = self.directory(instrument, day)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{instrument.upper()}_option_chain.csv"
        out.to_csv(path, index=False)
        return path

    def save_metadata(self, instrument: str, session_date: date, metadata: dict) -> Path:
        folder = self.directory(instrument, session_date)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "metadata.json"
        import json
        path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return path
