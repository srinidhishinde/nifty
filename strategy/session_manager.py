from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class SessionState(str, Enum):
    CLOSED = "CLOSED"
    ENTRY = "ENTRY"
    EXIT_ONLY = "EXIT_ONLY"


@dataclass(frozen=True)
class SessionPolicy:
    instrument: str
    exchange_open: time
    exchange_close: time
    entry_cutoff: time
    strategy_exit_cutoff: time


@dataclass(frozen=True)
class SessionContext:
    policy: SessionPolicy
    timestamp: datetime
    state: SessionState

    @property
    def entry_allowed(self) -> bool:
        return self.state == SessionState.ENTRY

    @property
    def exits_allowed(self) -> bool:
        return self.state in {SessionState.ENTRY, SessionState.EXIT_ONLY}


def policy_for(instrument: str, *, mcx_close: time = time(23, 30)) -> SessionPolicy:
    name = instrument.upper()
    if name == "NIFTY":
        return SessionPolicy(name, time(9, 15), time(15, 30), time(15, 15), time(15, 30))
    if name in {"MCX", "CRUDEOIL", "NATURALGAS", "COPPER", "SILVER", "GOLD"}:
        return SessionPolicy(name, time(9, 0), mcx_close, time(23, 0), mcx_close)
    raise ValueError(f"Unsupported instrument '{instrument}'")


def resolve_session(instrument: str, timestamp: datetime, *, mcx_close: time = time(23, 30)) -> SessionContext:
    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=IST)
    ts = ts.astimezone(IST)
    policy = policy_for(instrument, mcx_close=mcx_close)
    if ts.weekday() >= 5:
        return SessionContext(policy, ts, SessionState.CLOSED)
    current = ts.time()
    if current < policy.exchange_open or current > policy.exchange_close:
        return SessionContext(policy, ts, SessionState.CLOSED)
    if current >= policy.strategy_exit_cutoff:
        return SessionContext(policy, ts, SessionState.EXIT_ONLY)
    return SessionContext(policy, ts, SessionState.ENTRY)
