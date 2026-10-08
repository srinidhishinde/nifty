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

    def save_option_chain_snapshot(
        self,
        instrument: str,
        contracts: list[object],
        captured_at: pd.Timestamp | None = None,
    ) -> Path:
        """Persist the most recent real broker option chain for failover display.

        This is a cache only: it is never used to manufacture a live decision.
        A successful broker refresh overwrites the current day's snapshot, while
        the loader below can recover the latest real snapshot after a Streamlit
        restart or a transient broker/API failure.
        """
        if not contracts:
            raise ValueError("Cannot persist an empty option-chain snapshot.")
        timestamp = pd.Timestamp(captured_at or pd.Timestamp.now(tz="Asia/Kolkata"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("Asia/Kolkata")
        rows: list[dict[str, object]] = []
        for contract in contracts:
            rows.append({
                "captured_at": timestamp.isoformat(),
                "symbol": getattr(contract, "symbol", ""),
                "exchange": getattr(contract, "exchange", ""),
                "underlying": getattr(contract, "underlying", instrument),
                "expiry": getattr(contract, "expiry", ""),
                "strike": float(getattr(contract, "strike", 0) or 0),
                "option_type": getattr(contract, "option_type", ""),
                "instrument_token": getattr(contract, "instrument_token", "") or "",
                "ltp": float(getattr(contract, "ltp", 0) or 0),
                "bid": getattr(contract, "bid", None),
                "ask": getattr(contract, "ask", None),
                "volume": float(getattr(contract, "volume", 0) or 0),
                "open_interest": float(getattr(contract, "open_interest", 0) or 0),
                "oi_change": float(getattr(contract, "oi_change", 0) or 0),
                "implied_volatility": float(getattr(contract, "implied_volatility", 0) or 0),
                "built_up": getattr(contract, "built_up", "") or "",
                "delta": getattr(contract, "delta", None),
                "theta": getattr(contract, "theta", None),
                "vega": getattr(contract, "vega", None),
                "gamma": getattr(contract, "gamma", None),
                "ltp_change_pct": float(getattr(contract, "ltp_change_pct", 0) or 0),
            })
        frame = pd.DataFrame(rows)
        day = timestamp.tz_convert("Asia/Kolkata").date()
        folder = self.directory(instrument, day)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{instrument.upper()}_option_chain_latest.csv"
        frame.to_csv(path, index=False)
        return path

    @staticmethod
    def _normalise_option_strike(instrument: str, row: dict) -> float | None:
        """Validate/recover strikes; never scale an ambiguous value silently."""
        import re
        raw = row.get("strike", row.get("Strike"))
        try:
            strike = float(raw)
        except (TypeError, ValueError):
            strike = 0.0
        symbol = str(row.get("symbol") or "").upper()
        match = re.search(r"(\\d+(?:\\.\\d+)?)(CE|PE)$", symbol)
        symbol_strike = float(match.group(1)) if match else 0.0
        if instrument.upper() == "NIFTY":
            if symbol_strike > 0:
                # Broker symbol is authoritative when the stored strike field is corrupt.
                strike = symbol_strike
            if strike <= 0 or strike > 100000 or abs((strike / 50.0) - round(strike / 50.0)) > 1e-9:
                return None
        elif strike <= 0:
            if symbol_strike > 0:
                strike = symbol_strike
            else:
                return None
        return strike

    def load_latest_option_chain_snapshot(
        self,
        instrument: str,
        before: pd.Timestamp | None = None,
    ) -> tuple[pd.DataFrame, pd.Timestamp | None]:
        """Return the newest real option-chain snapshot available before *before*.

        The loader supports both the new persistent latest-snapshot file and
        older daily option-chain files. This is intentionally a read-only
        display/recovery path: it never creates market data or marks it live.
        """
        base = self.root / instrument.upper()
        if not base.exists():
            return pd.DataFrame(), None
        cutoff = pd.Timestamp(before or pd.Timestamp.now(tz="Asia/Kolkata"))
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("Asia/Kolkata")
        else:
            cutoff = cutoff.tz_convert("Asia/Kolkata")

        candidates = list(base.glob("*/" + f"{instrument.upper()}_option_chain_latest.csv"))
        candidates += list(base.glob("*/" + f"{instrument.upper()}_option_chain.csv"))
        newest: tuple[pd.Timestamp, Path, pd.DataFrame] | None = None

        for path in candidates:
            try:
                frame = pd.read_csv(path)
                if frame.empty:
                    continue
                if "captured_at" in frame.columns:
                    captured = pd.to_datetime(frame["captured_at"], errors="coerce").dropna()
                    if captured.empty:
                        stamp = pd.Timestamp(path.stat().st_mtime, unit="s", tz="Asia/Kolkata")
                    else:
                        stamp = captured.max()
                        if stamp.tzinfo is None:
                            stamp = stamp.tz_localize("Asia/Kolkata")
                        else:
                            stamp = stamp.tz_convert("Asia/Kolkata")
                elif "timestamp" in frame.columns:
                    captured = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
                    if captured.empty:
                        stamp = pd.Timestamp(path.stat().st_mtime, unit="s", tz="Asia/Kolkata")
                    else:
                        stamp = captured.max()
                        if stamp.tzinfo is None:
                            stamp = stamp.tz_localize("Asia/Kolkata")
                        else:
                            stamp = stamp.tz_convert("Asia/Kolkata")
                else:
                    stamp = pd.Timestamp(path.stat().st_mtime, unit="s", tz="Asia/Kolkata")

                if stamp > cutoff:
                    continue
                if "option_type" not in frame.columns:
                    continue
                frame["option_type"] = frame["option_type"].astype(str).str.upper().str.strip()
                frame = frame[frame["option_type"].isin({"CE", "PE"})].copy()
                if frame.empty:
                    continue
                frame["strike"] = [
                    self._normalise_option_strike(instrument, row)
                    for row in frame.to_dict("records")
                ]
                frame = frame.dropna(subset=["strike"])
                if frame.empty:
                    continue
                if newest is None or stamp > newest[0]:
                    newest = (stamp, path, frame)
            except (OSError, ValueError, pd.errors.ParserError):
                continue

        if newest is None:
            return pd.DataFrame(), None
        return newest[2].reset_index(drop=True), newest[0]

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
