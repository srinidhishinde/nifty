from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from alerts.recipient_store import WhatsAppRecipientStore
from config.settings import settings


@dataclass(frozen=True)
class WhatsAppResult:
    recipient: str
    ok: bool
    status_code: int | None
    message: str


class WhatsAppAlertService:
    def __init__(self, recipient_store: WhatsAppRecipientStore | None = None):
        self.recipient_store = recipient_store or WhatsAppRecipientStore()
        self.token = str(getattr(settings, "whatsapp_api_token", "") or "").strip()
        self.phone_number_id = str(getattr(settings, "whatsapp_phone_number_id", "") or "").strip()
        self.graph_version = str(getattr(settings, "whatsapp_graph_version", "v23.0") or "v23.0").strip()
        self.enabled = bool(getattr(settings, "whatsapp_alerts_enabled", False))
        self.log_path = Path("logs/whatsapp_alerts.jsonl")

    @property
    def configured(self) -> bool:
        return bool(self.enabled and self.token and self.phone_number_id and self.recipient_store.load())

    def _send(self, recipient: str, body: str) -> WhatsAppResult:
        url = f"https://graph.facebook.com/{self.graph_version}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "text",
            "text": {"body": body},
        }
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        last_error = "unknown error"
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=20)
                if response.ok:
                    return WhatsAppResult(recipient, True, response.status_code, "sent")
                last_error = response.text[:500]
                if response.status_code < 500 and response.status_code != 429:
                    break
            except requests.RequestException as exc:
                last_error = str(exc)
            if attempt < 2:
                time.sleep(2 ** attempt)
        return WhatsAppResult(recipient, False, response.status_code if "response" in locals() else None, last_error)

    def broadcast(self, body: str) -> list[WhatsAppResult]:
        results = [self._send(recipient, body) for recipient in self.recipient_store.load()]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps({
                    "recipient": result.recipient, "ok": result.ok,
                    "status_code": result.status_code, "message": result.message,
                }) + "\n")
        return results
