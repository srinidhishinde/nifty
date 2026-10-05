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
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        if out.empty:
            raise ValueError("No valid timestamps remain after normalization.")

        day = session_date or out["timestamp"].iloc[0].date()
        folder = self.directory(instrument, day)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{instrument.upper()}_ohlcv.csv"
        if path.exists():
            existing = pd.read_csv(path)
            if "timestamp" in existing.columns:
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], errors="coerce")
                out = pd.concat([existing, out], ignore_index=True)
                out = (
                    out.dropna(subset=["timestamp"])
                    .sort_values("timestamp")
                    .drop_duplicates("timestamp", keep="last")
                    .reset_index(drop=True)
                )
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


def load_captured_candles(
    instrument: str,
    start: date | None = None,
    end: date | None = None,
    root: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load only persisted broker-capture OHLCV partitions.

    This deliberately ignores Yahoo and synthetic datasets. A caller receives
    an explicit provenance label so UI/backtest code cannot silently mix sources.
    """
    store = DailyMarketStore(root)
    base = store.root / instrument.upper()
    if not base.exists():
        return pd.DataFrame(), "KOTAK_CAPTURED"

    frames: list[pd.DataFrame] = []
    for path in sorted(base.glob("*/" + f"{instrument.upper()}_ohlcv.csv")):
        try:
            day = date.fromisoformat(path.parent.name)
        except ValueError:
            continue
        if start and day < start:
            continue
        if end and day > end:
            continue
        frame = pd.read_csv(path)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=list(required))
        if not frame.empty:
            frame["data_source"] = "KOTAK_CAPTURED"
            frames.append(frame)

    if not frames:
        return pd.DataFrame(), "KOTAK_CAPTURED"

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    return combined, "KOTAK_CAPTURED"
