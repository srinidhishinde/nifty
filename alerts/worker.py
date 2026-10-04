from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from alerts.whatsapp import WhatsAppAlertService, build_signal_message
from config.settings import settings


class ScheduledSignalAlertWorker:
    """Five-minute signal broadcaster with duplicate suppression.

    The worker consumes a fully evaluated signal callback. It never creates
    trading decisions and never submits orders.
    """

    def __init__(
        self,
        snapshot_factory: Callable[[], object],
        interval_minutes: int | None = None,
        suppression_minutes: int | None = None,
        state_path: str | Path = "logs/whatsapp_alert_state.json",
    ):
        self.snapshot_factory = snapshot_factory
        self.interval_seconds = 60 * int(interval_minutes or settings.whatsapp_alert_interval_minutes)
        self.suppression_seconds = 60 * int(suppression_minutes or settings.whatsapp_duplicate_suppression_minutes)
        self.state_path = Path(state_path)
        self.service = WhatsAppAlertService()

    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @staticmethod
    def _fingerprint(snapshot: object) -> str:
        signal = getattr(snapshot, "signal", None)
        values = {
            "direction": getattr(signal, "direction", "WAIT"),
            "confidence": round(float(getattr(signal, "confidence", 0.0) or 0.0), 2),
            "pcr_oi": round(float(getattr(snapshot, "pcr_oi", 0.0) or 0.0), 4),
            "option_count": int(getattr(snapshot, "option_count", 0) or 0),
        }
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def run_once(self) -> list:
        if not settings.whatsapp_alerts_enabled or not self.service.configured:
            return []

        snapshot = self.snapshot_factory()
        if str(getattr(snapshot, "status", "RED")).upper() == "RED":
            return []

        signal = getattr(snapshot, "signal", None)
        if signal is None or str(getattr(signal, "direction", "WAIT")).upper() == "WAIT":
            return []

        fingerprint = self._fingerprint(snapshot)
        state = self._load_state()
        now = time.time()
        if state.get("fingerprint") == fingerprint and now - float(state.get("sent_at", 0)) < self.suppression_seconds:
            return []

        body = (
            f"AI Derivatives Signal (NIFTY, 5m)\n"
            f"Decision: {signal.direction}\n"
            f"Confidence: {float(signal.confidence):.1f}%\n"
            f"Timestamp: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
        results = self.service.broadcast(body)
        if results and any(x.ok for x in results):
            state = {"fingerprint": fingerprint, "sent_at": now}
            self._save_state(state)
        return results

    def run_forever(self) -> None:
        while True:
            started = time.monotonic()
            try:
                self.run_once()
            except Exception:
                # A failed alert cycle must not terminate the worker.
                pass
            elapsed = time.monotonic() - started
            time.sleep(max(1.0, self.interval_seconds - elapsed))
