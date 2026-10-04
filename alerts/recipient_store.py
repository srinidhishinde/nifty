from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_PATH = Path("config/whatsapp_recipients.json")
E164_RE = re.compile(r"^\d{8,15}$")


class WhatsAppRecipientStore:
    """Local persistent recipient list managed by the dashboard.

    Recipient numbers are stored without a leading '+' so they can be sent
    directly to the Meta WhatsApp Cloud API.
    """

    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)

    @staticmethod
    def normalize(number: str) -> str:
        value = re.sub(r"[\s()\-]", "", str(number or "").strip())
        if value.startswith("+"):
            value = value[1:]
        if not E164_RE.fullmatch(value):
            raise ValueError("Enter a WhatsApp number in international format, e.g. 919876543210.")
        return value

    def load(self) -> list[str]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        values = payload.get("recipients", []) if isinstance(payload, dict) else []
        return sorted({self.normalize(v) for v in values})

    def save(self, recipients: list[str]) -> list[str]:
        normalized = sorted({self.normalize(v) for v in recipients})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"recipients": normalized}, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        return normalized

    def add(self, number: str) -> list[str]:
        recipients = self.load()
        recipients.append(self.normalize(number))
        return self.save(recipients)

    def remove(self, number: str) -> list[str]:
        normalized = self.normalize(number)
        return self.save([v for v in self.load() if v != normalized])
