from __future__ import annotations

import pandas as pd

from marketdata.option_chain_csv import parse_option_chain_csv, is_option_chain_snapshot


def replay_option_chain_csv(data: pd.DataFrame) -> list[tuple[pd.Timestamp, list]]:
    """Replay timestamped option-chain snapshots.

    A single untimestamped broker export is a snapshot, not historical replay
    data. Replay requires a timestamp column and one or more snapshots.
    """
    if "timestamp" not in {str(c).strip().lower() for c in data.columns}:
        raise ValueError(
            "Historical option-chain replay requires a timestamp column. "
            "The supplied file is a single snapshot and cannot be replayed."
        )
    timestamp_col = next(c for c in data.columns if str(c).strip().lower() == "timestamp")
    frame = data.copy()
    frame["_timestamp"] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if frame["_timestamp"].isna().all():
        raise ValueError("Historical option-chain replay requires valid timestamps.")
    snapshots = []
    for ts, group in frame.dropna(subset=["_timestamp"]).groupby("_timestamp"):
        if not is_option_chain_snapshot(group.columns):
            raise ValueError("Replay rows do not match the supported option-chain schema.")
        snapshots.append((pd.Timestamp(ts), parse_option_chain_csv(group.drop(columns=["_timestamp"]))))
    return snapshots
