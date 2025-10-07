import json
from datetime import datetime, timedelta


class _DummyCompletion:
    class _Choice:
        class _Message:
            def __init__(self, content: str):
                self.content = content

        def __init__(self, content: str):
            self.message = self._Message(content)

    def __init__(self, content: str):
        self.choices = [self._Choice(content)]


class _DummyGroq:
    def __init__(self, api_key: str):  # pragma: no cover - trivial
        self.api_key = api_key


def _example_events():
    now = datetime.utcnow()
    return [
        {
            "event": "Economic report",
            "datetime": (now + timedelta(hours=1)).isoformat() + "Z",
            "impact": "high",
        }
    ]


def test_analyze_news_with_llm_valid_json(monkeypatch):
    import fetch_news

    monkeypatch.setattr(fetch_news, "GROQ_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(fetch_news, "Groq", _DummyGroq, raising=False)

    captured = {}

    def fake_safe_chat_completion(client, *, model, messages, **kwargs):
        captured["messages"] = messages
        payload = {"safe_decision": "no", "reason": "Volatility expected"}
        return _DummyCompletion(json.dumps(payload))

    monkeypatch.setattr(fetch_news, "safe_chat_completion", fake_safe_chat_completion, raising=False)

    result = fetch_news.analyze_news_with_llm(_example_events())

    assert result == {"safe": False, "sensitivity": 0.6, "reason": "Volatility expected"}
    assert captured["messages"][0]["role"] == "system"
    assert "`safe_decision`" in captured["messages"][0]["content"]
    assert "Events:\n" in captured["messages"][1]["content"]


def test_analyze_news_with_llm_handles_non_json(monkeypatch, caplog):
    import fetch_news

    monkeypatch.setattr(fetch_news, "GROQ_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(fetch_news, "Groq", _DummyGroq, raising=False)

    def fake_safe_chat_completion(client, *, model, messages, **kwargs):
        return _DummyCompletion("I cannot comply with that request.")

    monkeypatch.setattr(fetch_news, "safe_chat_completion", fake_safe_chat_completion, raising=False)
    warnings: list[str] = []

    def fake_warning(message, *args, **kwargs):  # pragma: no cover - trivial formatting
        warnings.append(message % args if args else message)

    monkeypatch.setattr(fetch_news.logger, "warning", fake_warning, raising=False)

    result = fetch_news.analyze_news_with_llm(_example_events())

    assert result == {"safe": True, "sensitivity": 0, "reason": "LLM non-JSON response"}
    assert any("LLM returned non-JSON response" in message for message in warnings)


def test_analyze_news_with_llm_overrides_inconsistent_decision(monkeypatch):
    import fetch_news

    monkeypatch.setattr(fetch_news, "GROQ_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(fetch_news, "Groq", _DummyGroq, raising=False)

    def fake_safe_chat_completion(client, *, model, messages, **kwargs):
        return _DummyCompletion(json.dumps({"safe_decision": "yes", "reason": "Looks calm"}))

    monkeypatch.setattr(fetch_news, "safe_chat_completion", fake_safe_chat_completion, raising=False)

    result = fetch_news.analyze_news_with_llm(_example_events())

    assert result["safe"] is False
    assert result["sensitivity"] == 0.6
    assert "Overriding" in result["reason"]
