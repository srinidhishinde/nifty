from __future__ import annotations

import json
import os
import tempfile
import contextlib
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATES = ("CREATED", "SUBMITTED", "ACKNOWLEDGED", "PARTIALLY_FILLED", "FILLED", "CANCEL_PENDING", "CANCELLED", "REJECTED", "RECONCILED", "AMBIGUOUS")

@dataclass(frozen=True)
class ExecutionRecord:
    decision_id: str
    order_id: str | None
    state: str
    symbol: str
    side: str
    quantity: int
    filled_quantity: int
    broker_timestamp: str
    updated_at: str
    error: str = ""

@contextlib.contextmanager
def _file_lock(path: Path):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        if sys.platform.startswith("win"):
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

class ExecutionLedger:
    """Append-only execution state store. Never retries an ambiguous submission blindly."""
    def __init__(self, path: str | Path = "logs/execution_ledger.jsonl"):
        self.path = Path(path)

    def _records(self) -> list[ExecutionRecord]:
        if not self.path.exists():
            return []
        out=[]
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw=json.loads(line)
                out.append(ExecutionRecord(**raw))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return out

    def latest(self, decision_id: str) -> ExecutionRecord | None:
        rows=[r for r in self._records() if r.decision_id == decision_id]
        return rows[-1] if rows else None

    def can_submit(self, decision_id: str) -> bool:
        row=self.latest(decision_id)
        return row is None or row.state in {"REJECTED", "CANCELLED"}

    def _append_unlocked(self, record: ExecutionRecord) -> None:
        payload=json.dumps(asdict(record), sort_keys=True)
        fd,tmp=tempfile.mkstemp(prefix=".ledger-", dir=str(self.path.parent), text=True)
        try:
            with os.fdopen(fd,"w",encoding="utf-8") as fh:
                fh.write(payload+"\n")
                fh.flush()
                os.fsync(fh.fileno())
            with self.path.open("a",encoding="utf-8") as out:
                out.write(payload+"\n")
                out.flush()
                os.fsync(out.fileno())
        finally:
            try: os.unlink(tmp)
            except FileNotFoundError: pass

    def append(self, record: ExecutionRecord) -> None:
        if record.state not in STATES:
            raise ValueError("Invalid execution state")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(self.path):
            self._append_unlocked(record)

    def begin(self, decision_id: str, symbol: str, side: str, quantity: int, broker_timestamp: str) -> ExecutionRecord:
        if int(quantity) <= 0:
            raise ValueError("Execution quantity must be positive")
        if str(side).upper() not in {"BUY", "SELL"}:
            raise ValueError("Execution side must be BUY or SELL")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(self.path):
            if not self.can_submit(decision_id):
                raise RuntimeError("Duplicate or already-active decision cannot be submitted")
            now=datetime.now(timezone.utc).isoformat()
            row=ExecutionRecord(decision_id,None,"CREATED",symbol,side,int(quantity),0,broker_timestamp,now)
            self._append_unlocked(row)
            return row

    def reconcile(self, decision_id: str, broker_state: str, *, order_id: str | None = None,
                  filled_quantity: int | None = None, broker_timestamp: str = "",
                  error: str = "") -> ExecutionRecord:
        previous = self.latest(decision_id)
        if previous is None:
            raise RuntimeError("Unknown decision")
        state = str(broker_state).upper()
        if state not in STATES or state in {"CREATED", "SUBMITTED", "AMBIGUOUS"}:
            raise ValueError("Broker reconciliation requires a definitive execution state")
        filled = previous.filled_quantity if filled_quantity is None else int(filled_quantity)
        if filled < 0 or filled > previous.quantity:
            raise ValueError("Broker filled quantity is outside the ordered quantity")
        row = ExecutionRecord(
            decision_id, order_id or previous.order_id, state, previous.symbol,
            previous.side, previous.quantity, filled, broker_timestamp,
            datetime.now(timezone.utc).isoformat(), str(error)
        )
        self.append(row)
        return row

    def mark_ambiguous(self, decision_id: str, error: str, broker_timestamp: str) -> ExecutionRecord:
        previous=self.latest(decision_id)
        if previous is None:
            raise RuntimeError("Unknown decision")
        row=ExecutionRecord(decision_id,previous.order_id,"AMBIGUOUS",previous.symbol,previous.side,previous.quantity,previous.filled_quantity,broker_timestamp,datetime.now(timezone.utc).isoformat(),str(error))
        self.append(row)
        return row
