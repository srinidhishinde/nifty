from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    status: str
    rows: int
    duplicate_timestamps: int
    invalid_ohlc: int
    negative_volume: int
    zero_volume: int
    gaps_over_expected: int
    max_gap_minutes: float
    start: str | None
    end: str | None
    reasons: tuple[str, ...]


def assess_ohlcv(data: pd.DataFrame, expected_minutes: int = 5) -> DataQualityReport:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    duplicate_count = int(frame["timestamp"].duplicated().sum())
    invalid = (
        frame[["open", "high", "low", "close"]].isna().any(axis=1)
        | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
    )
    invalid_count = int(invalid.sum())
    negative_volume = int((frame["volume"].fillna(-1) < 0).sum())
    zero_volume = int((frame["volume"].fillna(0) == 0).sum())
    deltas = frame["timestamp"].diff().dt.total_seconds().div(60).dropna()
    gaps = deltas[deltas > expected_minutes * 1.5]
    max_gap = float(deltas.max()) if not deltas.empty else 0.0
    reasons: list[str] = []
    if duplicate_count:
        reasons.append(f"{duplicate_count} duplicate timestamps")
    if invalid_count:
        reasons.append(f"{invalid_count} invalid OHLC rows")
    if negative_volume:
        reasons.append(f"{negative_volume} negative-volume rows")
    if len(gaps):
        reasons.append(f"{len(gaps)} gaps exceed expected cadence")
    # Zero volume is a warning rather than an automatic failure because some
    # legitimate index-derived feeds can report zero volume.
    status = "GREEN"
    if invalid_count or negative_volume:
        status = "RED"
    elif duplicate_count or len(gaps):
        status = "ORANGE"
    elif zero_volume:
        status = "YELLOW"
        reasons.append(f"{zero_volume} zero-volume rows")
    return DataQualityReport(
        status=status,
        rows=len(frame),
        duplicate_timestamps=duplicate_count,
        invalid_ohlc=invalid_count,
        negative_volume=negative_volume,
        zero_volume=zero_volume,
        gaps_over_expected=len(gaps),
        max_gap_minutes=round(max_gap, 2),
        start=str(frame["timestamp"].min()) if not frame.empty else None,
        end=str(frame["timestamp"].max()) if not frame.empty else None,
        reasons=tuple(reasons),
    )
