from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests

from config.settings import settings


@dataclass(frozen=True)
class WhatsAppResult:
    recipient: str
    ok: bool
    status_code: int | None
    message_id: str | None
    error: str | None = None


class WhatsAppAlertService:
    """Small, auditable WhatsApp Cloud API alert client.

    Credentials and recipients are configuration-only. The service never places
    broker orders and does not decide whether a trade is valid.
    """

    def __init__(
        self,
        token: str | None = None,
        phone_number_id: str | None = None,
        recipients: Iterable[str] | None = None,
        graph_version: str | None = None,
        log_path: str | Path = "logs/whatsapp_signals.jsonl",
        timeout_seconds: int = 15,
    ) -> None:
        self.token = token if token is not None else settings.whatsapp_api_token
        self.phone_number_id = (
            phone_number_id
            if phone_number_id is not None
            else settings.whatsapp_phone_number_id
        )
        configured = (
            list(recipients)
            if recipients is not None
            else settings.whatsapp_recipients
        )
        self.recipients = tuple(
            str(value).strip()
            for value in configured
            if str(value).strip()
        )
        self.graph_version = (
            graph_version
            if graph_version is not None
            else settings.whatsapp_graph_version
        )
        self.log_path = Path(log_path)
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(
            settings.whatsapp_alerts_enabled
            and self.token
            and self.phone_number_id
            and self.recipients
        )

    @property
    def endpoint(self) -> str:
        return (
            f"https://graph.facebook.com/"
            f"{self.graph_version}/{self.phone_number_id}/messages"
        )

    @staticmethod
    def format_signal_message(
        ce_reliability: float,
        pe_reliability: float,
        stronger_side: str,
        confidence_low: float | None = None,
        confidence_high: float | None = None,
        decision: str | None = None,
        instrument: str = "NIFTY",
    ) -> str:
        if confidence_low is not None and confidence_high is not None:
            ci = f"{confidence_low:.0f}%–{confidence_high:.0f}%"
        else:
            ci = "n/a"
        side = stronger_side or "WAIT"
        decision_text = decision or ("BUY " + side if side in {"CE", "PE"} else "WAIT")
        return (
            f"AI Derivatives Signal ({instrument}, 5m)\n"
            f"Decision: {decision_text}\n"
            f"CE Reliability: {ce_reliability:.1f}%\n"
            f"PE Reliability: {pe_reliability:.1f}%\n"
            f"Validation CI: {ci}\n"
            f"Stronger Side: {side}\n"
            f"Timestamp: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

    def _log(self, result: WhatsAppResult, body: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "recipient": result.recipient,
            "ok": result.ok,
            "status_code": result.status_code,
            "message_id": result.message_id,
            "error": result.error,
            "body": body,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def send_to(
        self,
        recipient: str,
        body: str,
        retries: int = 3,
    ) -> WhatsAppResult:
        if not self.token or not self.phone_number_id:
            result = WhatsAppResult(
                recipient, False, None, None,
                "WhatsApp Cloud API credentials are not configured.",
            )
            self._log(result, body)
            return result

        payload = {
            "messaging_product": "whatsapp",
            "to": recipient,
            "type": "text",
            "text": {"body": body},
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        last_error = "Unknown WhatsApp API error."
        for attempt in range(max(1, retries)):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                data = response.json() if response.content else {}
                if response.ok:
                    messages = data.get("messages") or []
                    message_id = (
                        messages[0].get("id")
                        if messages and isinstance(messages[0], dict)
                        else None
                    )
                    result = WhatsAppResult(
                        recipient, True, response.status_code, message_id
                    )
                    self._log(result, body)
                    return result

                error = data.get("error") if isinstance(data, dict) else None
                if isinstance(error, dict):
                    last_error = str(
                        error.get("message")
                        or error.get("error_data", {}).get("details")
                        or error
                    )
                else:
                    last_error = response.text[:500] or "HTTP error"

            except requests.RequestException as exc:
                last_error = str(exc)

            if attempt < max(1, retries) - 1:
                time.sleep(2 ** attempt)

        result = WhatsAppResult(
            recipient, False, None, None, last_error
        )
        self._log(result, body)
        return result

    def broadcast(
        self,
        body: str,
        retries: int = 3,
    ) -> list[WhatsAppResult]:
        return [
            self.send_to(recipient, body, retries=retries)
            for recipient in self.recipients
        ]


def build_signal_message(ensemble_row, instrument: str = "NIFTY") -> str:
    """Format an ensemble row without introducing any market values."""
    return WhatsAppAlertService.format_signal_message(
        ce_reliability=float(ensemble_row.final_ce),
        pe_reliability=float(ensemble_row.final_pe),
        stronger_side=str(ensemble_row.stronger_side),
        confidence_low=getattr(ensemble_row, "confidence_low", None),
        confidence_high=getattr(ensemble_row, "confidence_high", None),
        instrument=instrument,
    )
