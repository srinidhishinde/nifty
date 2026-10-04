from __future__ import annotations

import json
from pathlib import Path


class WhatsAppSettingsStore:
    def __init__(self, path: str | Path = "config/whatsapp_runtime.json"):
        self.path = Path(path)

    def load(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def enabled(self) -> bool:
        return bool(self.load().get("enabled", False))

    def set_enabled(self, enabled: bool) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.load()
        payload["enabled"] = bool(enabled)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        return bool(enabled)
