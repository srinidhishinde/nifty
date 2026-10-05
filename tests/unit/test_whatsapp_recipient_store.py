import json

from alerts.recipient_store import WhatsAppRecipientStore


def test_recipient_store_normalizes_deduplicates_and_persists(tmp_path):
    store = WhatsAppRecipientStore(tmp_path / "recipients.json")
    assert store.add("+91 98765-43210") == ["919876543210"]
    assert store.add("919876543210") == ["919876543210"]
    assert store.load() == ["919876543210"]


def test_recipient_store_remove(tmp_path):
    store = WhatsAppRecipientStore(tmp_path / "recipients.json")
    store.save(["919876543210", "14155552671"])
    assert store.remove("+91 98765 43210") == ["14155552671"]


def test_recipient_store_rejects_invalid_number(tmp_path):
    store = WhatsAppRecipientStore(tmp_path / "recipients.json")
    try:
        store.add("98765")
    except ValueError as exc:
        assert "international format" in str(exc)
    else:
        raise AssertionError("invalid number was accepted")


def test_recipient_file_is_json(tmp_path):
    path = tmp_path / "recipients.json"
    WhatsAppRecipientStore(path).add("919876543210")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"recipients": ["919876543210"]}
