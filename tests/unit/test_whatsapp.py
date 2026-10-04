from types import SimpleNamespace

import requests

from alerts.whatsapp import WhatsAppAlertService


def test_signal_message_contains_ensemble_values():
    row = SimpleNamespace(
        final_ce=72.0,
        final_pe=28.0,
        stronger_side="CE",
        confidence_low=66.0,
        confidence_high=78.0,
    )
    message = WhatsAppAlertService.format_signal_message(
        72, 28, "CE", 66, 78
    )
    assert "CE Reliability: 72.0%" in message
    assert "PE Reliability: 28.0%" in message
    assert "Stronger Side: CE" in message
    assert "66%–78%" in message


def test_broadcast_retries_and_logs(tmp_path, monkeypatch):
    calls = {"n": 0}

    class Response:
        ok = True
        status_code = 200
        content = b'{"messages":[{"id":"wamid.test"}]}'

        def json(self):
            return {"messages": [{"id": "wamid.test"}]}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.RequestException("temporary")
        return Response()

    monkeypatch.setattr("alerts.whatsapp.requests.post", fake_post)
    monkeypatch.setattr("alerts.whatsapp.time.sleep", lambda *_: None)
    service = WhatsAppAlertService(
        token="token",
        phone_number_id="123",
        recipients=["+919800000001", "+919800000002"],
        log_path=tmp_path / "signals.jsonl",
    )
    # Configuration is additionally guarded by the global enable flag, so test
    # the transport directly via send_to.
    result = service.send_to("+919800000001", "test", retries=2)
    assert result.ok
    assert result.message_id == "wamid.test"
    assert calls["n"] == 2
    assert (tmp_path / "signals.jsonl").exists()
