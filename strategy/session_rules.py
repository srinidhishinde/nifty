from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import pandas as pd


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start: time
    end: time


# NIFTY derivatives: regular trading through 15:40. The 15:15–15:40
# window is a research/decision window, not a separate exchange auction.
NIFTY_DERIVATIVES = SessionWindow("NIFTY_DERIVATIVES", time(9, 15), time(15, 40))
NIFTY_CLOSING_RESEARCH = SessionWindow("NIFTY_CLOSING_RESEARCH", time(15, 15), time(15, 40))


def in_window(timestamp: pd.Timestamp, window: SessionWindow) -> bool:
    current = timestamp.timetz().replace(tzinfo=None)
    return window.start <= current <= window.end


def session_name(timestamp: pd.Timestamp, instrument: str) -> str:
    key = str(instrument).upper()
    if key == "NIFTY":
        if in_window(timestamp, NIFTY_CLOSING_RESEARCH):
            return "CLOSING_RESEARCH"
        if in_window(timestamp, NIFTY_DERIVATIVES):
            return "REGULAR"
    return "OUTSIDE"
