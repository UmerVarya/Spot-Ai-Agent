from types import SimpleNamespace

import llm_approval


def _mock_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message={"content": content})])


def test_get_llm_approval_models_from_env(monkeypatch):
    monkeypatch.setenv("LLM_APPROVAL_MODELS", "model-a, model-b , model-a ,, model-c")
    models = llm_approval.get_llm_approval_models_from_env()
    assert models == ["model-a", "model-b", "model-c"]


def test_get_llm_trade_decision_success(monkeypatch):
    monkeypatch.setenv("LLM_APPROVAL_MODELS", "primary")
    monkeypatch.setattr(llm_approval, "get_groq_client", lambda: object())
    monkeypatch.setattr(
        llm_approval,
        "safe_chat_completion",
        lambda *_args, **_kwargs: _mock_response(
            '{"approve": true, "confidence": 0.87, "reason": "looks good"}'
        ),
    )

    decision = llm_approval.get_llm_trade_decision({"symbol": "BTC"})
    assert decision.approved is True
    assert decision.confidence == 0.87
    assert decision.model == "primary"
    assert decision.decision == "approved"


def test_get_llm_trade_decision_uses_fallback(monkeypatch):
    monkeypatch.setenv("LLM_APPROVAL_MODELS", "bad,good")
    monkeypatch.setattr(llm_approval, "get_groq_client", lambda: object())

    calls: list[str] = []

    def fake_chat_completion(_client, *, model, **_kwargs):
        calls.append(model)
        if model == "bad":
            raise llm_approval.GroqAuthError("auth failed")
        return _mock_response('{"approve": false, "confidence": 0.2, "reason": "risky"}')

    monkeypatch.setattr(llm_approval, "safe_chat_completion", fake_chat_completion)

    decision = llm_approval.get_llm_trade_decision({"symbol": "ETH"})
    assert calls == ["bad", "good"]
    assert decision.approved is False
    assert decision.decision == "rejected"
    assert decision.model == "good"


def test_get_llm_trade_decision_all_fail(monkeypatch):
    monkeypatch.setenv("LLM_APPROVAL_MODELS", "bad1,bad2")
    monkeypatch.setattr(llm_approval, "get_groq_client", lambda: object())
    monkeypatch.setattr(
        llm_approval,
        "safe_chat_completion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    decision = llm_approval.get_llm_trade_decision({"symbol": "SOL"})
    assert decision.decision == "LLM unavailable"
    assert decision.approved is None
    assert decision.confidence is None
    assert decision.model is None

