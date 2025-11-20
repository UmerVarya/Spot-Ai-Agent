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
    monkeypatch.setattr(
        llm_approval,
        "call_llm_for_task",
        lambda *_args, **_kwargs: (
            _mock_response('{"approve": true, "confidence": 0.87, "reason": "looks good"}'),
            "primary",
        ),
    )

    decision = llm_approval.get_llm_trade_decision({"symbol": "BTC"})
    assert decision.approved is True
    assert decision.confidence == 0.87
    assert decision.model == "primary"
    assert decision.decision == "approved"


def test_get_llm_trade_decision_uses_fallback(monkeypatch):
    calls: list[str] = []

    def fake_call_llm(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return None, None
        return _mock_response('{"approve": false, "confidence": 0.2, "reason": "risky"}'), "good"

    monkeypatch.setattr(llm_approval, "call_llm_for_task", fake_call_llm)

    decision = llm_approval.get_llm_trade_decision({"symbol": "ETH"})
    assert calls == [1]
    assert decision.approved is None
    assert decision.decision == "LLM unavailable"
    assert decision.model is None


def test_get_llm_trade_decision_all_fail(monkeypatch):
    monkeypatch.setattr(
        llm_approval,
        "call_llm_for_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    decision = llm_approval.get_llm_trade_decision({"symbol": "SOL"})
    assert decision.decision == "LLM unavailable"
    assert decision.approved is None
    assert decision.confidence is None
    assert decision.model is None

