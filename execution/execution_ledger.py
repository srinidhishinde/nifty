from __future__ import annotations

import json
import os
import tempfile
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

    def append(self, record: ExecutionRecord) -> None:
        if record.state not in STATES:
            raise ValueError("Invalid execution state")
        self.path.parent.mkdir(parents=True, exist_ok=True)
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

    def begin(self, decision_id: str, symbol: str, side: str, quantity: int, broker_timestamp: str) -> ExecutionRecord:
        if not self.can_submit(decision_id):
            raise RuntimeError("Duplicate or already-active decision cannot be submitted")
        now=datetime.now(timezone.utc).isoformat()
        row=ExecutionRecord(decision_id,None,"CREATED",symbol,side,int(quantity),0,broker_timestamp,now)
        self.append(row)
        return row

    def mark_ambiguous(self, decision_id: str, error: str, broker_timestamp: str) -> ExecutionRecord:
        previous=self.latest(decision_id)
        if previous is None:
            raise RuntimeError("Unknown decision")
        row=ExecutionRecord(decision_id,previous.order_id,"AMBIGUOUS",previous.symbol,previous.side,previous.quantity,previous.filled_quantity,broker_timestamp,datetime.now(timezone.utc).isoformat(),str(error))
        self.append(row)
        return row
