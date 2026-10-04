from assistant import chatbot


def test_chatbot_requires_api_key(monkeypatch):
    monkeypatch.setattr(chatbot.settings, "openai_api_key", "")
    result = chatbot.answer("Why is the system WAIT?")
    assert "OPENAI_API_KEY" in result
