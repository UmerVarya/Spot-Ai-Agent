import importlib
import types


def test_llm_uses_mapped_model(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    import groq_llm
    importlib.reload(groq_llm)

    captured = {}

    def fake_post(url, headers, json):
        captured["model"] = json["model"]
        class Resp:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}
        return Resp()

    monkeypatch.setattr(groq_llm.requests, "post", fake_post)
    groq_llm.get_llm_judgment("test prompt")
    assert captured["model"] == "llama-3.1-70b"
