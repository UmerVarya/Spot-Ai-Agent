import importlib
import json


def test_llm_retries_with_fallback_model(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "custom-model")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import groq_llm
    importlib.reload(groq_llm)

    calls = []

    class Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            if isinstance(self._payload, dict):
                return self._payload
            raise ValueError("No JSON payload")

    def fake_post(url, headers, json=None, **_):
        calls.append(json["model"])
        if len(calls) == 1:
            return Resp(
                400,
                {"error": {"code": "model_decommissioned", "message": "custom-model retired"}},
            )
        return Resp(200, {"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    result = groq_llm.get_llm_judgment("test prompt")

    assert result == "ok"
    assert calls == ["custom-model", groq_llm.config.DEFAULT_GROQ_MODEL]
