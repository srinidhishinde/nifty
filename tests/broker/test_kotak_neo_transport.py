import sys
from types import SimpleNamespace

from broker import kotak_neo


def test_create_client_configures_bounded_connection_retries(monkeypatch):
    calls = {}

    class FakeTransport:
        def __init__(self, **kwargs):
            calls["transport"] = kwargs
            self.kwargs = kwargs

    class FakeNeoAPI:
        def __init__(self, **kwargs):
            calls["neo_api"] = kwargs

    monkeypatch.setattr(kotak_neo.settings, "neo_consumer_key", "test-consumer-key")
    monkeypatch.setattr(kotak_neo.httpx, "HTTPTransport", FakeTransport)
    monkeypatch.setitem(
        sys.modules,
        "neo_api_client",
        SimpleNamespace(NeoAPI=FakeNeoAPI),
    )

    client = kotak_neo.KotakNeoBroker().create_client()

    assert isinstance(client, FakeNeoAPI)
    transport = calls["neo_api"]["transport"]
    assert isinstance(transport, FakeTransport)
    assert calls["transport"]["retries"] == 2
    assert calls["transport"]["http2"] is True
    assert calls["transport"]["limits"].max_connections == 20
    assert calls["transport"]["limits"].max_keepalive_connections == 10
    assert calls["neo_api"]["consumer_key"] == "test-consumer-key"
    assert calls["neo_api"]["environment"] == "prod"
