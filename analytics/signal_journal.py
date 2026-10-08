from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SignalJournal:
    """Append-only local journal for every evaluated trading decision.

    JSONL is intentionally used so the file remains streamable and easy to
    load into pandas for weekly analysis. No order is submitted by this class.
    """

    def __init__(self, path: str | Path = "logs/signal_journal.jsonl"):
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        payload = dict(record)
        from strategy.production_hardening import idempotency_key
        payload.setdefault("decision_id", idempotency_key(payload))
        # Streamlit reruns must not create duplicate decision records.
        if self._contains_decision(payload["decision_id"]):
            return
        payload.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")

    def _contains_decision(self, decision_id: str) -> bool:
        if not self.path.exists():
            return False
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        if str(json.loads(line).get("decision_id", "")) == str(decision_id):
                            return True
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return False
        return False

    def append_decision(
        self,
        *,
        instrument: str,
        timeframe: str,
        source: str,
        status: str,
        direction: str,
        confidence: float = 0.0,
        reliability: float = 0.0,
        reason: str = "",
        regime: str = "",
        pcr: float | None = None,
        imbalance_ratio: float | None = None,
        aggressor: str = "",
        micro_weight: float | None = None,
        threshold: float | None = None,
        ml_ce: float | None = None,
        ml_pe: float | None = None,
        rule_ce: float | None = None,
        rule_pe: float | None = None,
        entry: float | None = None,
        stop_loss: float | None = None,
        target: float | None = None,
    ) -> None:
        self.append({
            "instrument": instrument,
            "timeframe": timeframe,
            "source": source,
            "status": status,
            "direction": direction,
            "confidence": confidence,
            "reliability": reliability,
            "reason": reason,
            "regime": regime,
            "pcr": pcr,
            "imbalance_ratio": imbalance_ratio,
            "aggressor": aggressor,
            "micro_weight": micro_weight,
            "threshold": threshold,
            "ml_ce": ml_ce,
            "ml_pe": ml_pe,
            "rule_ce": rule_ce,
            "rule_pe": rule_pe,
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target,
        })
