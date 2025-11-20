import builtins
from types import SimpleNamespace

import llm_tasks
import groq_safe
from groq_safe import GroqAuthError, reset_auth_state
from groq_http import reset_groq_key_state


def _mock_response(content: str, model: str = "model-x"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], model=model)


def test_get_model_chain_parses_env(monkeypatch):
    monkeypatch.setenv("LLM_APPROVAL_MODELS", " model-a,model-b , ,model-c ")
    models = llm_tasks.get_model_chain("LLM_APPROVAL_MODELS", ["fallback"])
    assert models == ["model-a", "model-b", "model-c"]

    monkeypatch.delenv("LLM_APPROVAL_MODELS")
    monkeypatch.setenv("GROQ_DEFAULT_MODEL", "default-model")
    models = llm_tasks.get_model_chain("LLM_APPROVAL_MODELS", ["fallback"])
    assert models == ["default-model"]


def test_iter_models_for_task_defaults(monkeypatch):
    monkeypatch.delenv("LLM_APPROVAL_MODELS", raising=False)
    models = llm_tasks.iter_models_for_task(llm_tasks.LLMTask.EXPLAIN)
    assert models == ["llama-3.3-70b-versatile"]


def test_call_llm_for_task_success(monkeypatch):
    monkeypatch.setattr(llm_tasks, "get_groq_client", lambda: object())
    monkeypatch.setattr(llm_tasks, "iter_models_for_task", lambda _task: ["primary"])
    monkeypatch.setattr(llm_tasks, "_soft_limit_check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        llm_tasks,
        "safe_chat_completion",
        lambda *_args, **_kwargs: _mock_response("{\"ok\": true}", "primary"),
    )

    response, model_used = llm_tasks.call_llm_for_task(
        llm_tasks.LLMTask.APPROVAL,
        messages=[{}],
    )
    assert response.choices[0].message.content == "{\"ok\": true}"
    assert model_used == "primary"


def test_call_llm_for_task_fallback(monkeypatch):
    monkeypatch.setattr(llm_tasks, "get_groq_client", lambda: object())
    monkeypatch.setattr(llm_tasks, "iter_models_for_task", lambda _task: ["bad", "good"])
    monkeypatch.setattr(llm_tasks, "_soft_limit_check", lambda *_args, **_kwargs: True)

    calls: list[str] = []

    def fake_chat(_client, *, model, **_kwargs):
        calls.append(model)
        if model == "bad":
            raise RuntimeError("model_not_found")
        return _mock_response("{\"ok\": true}", model)

    monkeypatch.setattr(llm_tasks, "safe_chat_completion", fake_chat)

    response, model_used = llm_tasks.call_llm_for_task(
        llm_tasks.LLMTask.NEWS,
        messages=[{}],
    )

    assert calls == ["bad", "good"]
    assert model_used == "good"
    assert response.model == "good"


def test_call_llm_for_task_all_fail(monkeypatch):
    monkeypatch.setattr(llm_tasks, "get_groq_client", lambda: object())
    monkeypatch.setattr(llm_tasks, "iter_models_for_task", lambda _task: ["only"])
    monkeypatch.setattr(llm_tasks, "_soft_limit_check", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        llm_tasks,
        "safe_chat_completion",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response, model_used = llm_tasks.call_llm_for_task(llm_tasks.LLMTask.ALT_DATA, messages=[{}])
    assert response is None
    assert model_used is None


def test_soft_limit_blocks_model(monkeypatch):
    monkeypatch.setenv("GROQ_SOFT_RPM_DEFAULT", "1")
    llm_tasks._MODEL_CALLS.clear()

    times = iter([0.0, 1.0])
    monkeypatch.setattr(llm_tasks.time, "time", lambda: next(times))

    monkeypatch.setattr(llm_tasks, "iter_models_for_task", lambda _task: ["rate-limited"])
    monkeypatch.setattr(llm_tasks, "get_groq_client", lambda: object())
    monkeypatch.setattr(llm_tasks, "safe_chat_completion", lambda *_args, **_kwargs: _mock_response("{}"))

    first_response, first_model = llm_tasks.call_llm_for_task(llm_tasks.LLMTask.EXPLAIN, messages=[{}])
    assert first_response is not None
    assert first_model == "rate-limited"

    second_response, second_model = llm_tasks.call_llm_for_task(llm_tasks.LLMTask.EXPLAIN, messages=[{}])
    assert second_response is None
    assert second_model is None


def test_call_llm_missing_key(monkeypatch, caplog):
    reset_auth_state()
    reset_groq_key_state()
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_tasks, "get_groq_client", lambda: object())
    monkeypatch.setattr(llm_tasks, "iter_models_for_task", lambda _task: ["primary"])
    caplog.set_level("WARNING")

    response, model = llm_tasks.call_llm_for_task(llm_tasks.LLMTask.APPROVAL, messages=[{}])

    assert response is None
    assert model is None
    assert any("Groq disabled for task=approval" in rec.message for rec in caplog.records)


def test_call_llm_auth_failure_disables(monkeypatch):
    reset_auth_state()
    reset_groq_key_state()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_live")
    monkeypatch.setattr(llm_tasks, "get_groq_client", lambda: object())
    monkeypatch.setattr(llm_tasks, "iter_models_for_task", lambda _task: ["primary"])

    def _auth_fail(*_args, **_kwargs):
        from groq_safe import _disable_auth

        _disable_auth({"error": {"code": "invalid_api_key"}})
        raise GroqAuthError("Groq authentication failed")

    monkeypatch.setattr(llm_tasks, "safe_chat_completion", _auth_fail)

    response, model = llm_tasks.call_llm_for_task(llm_tasks.LLMTask.NEWS, messages=[{}])

    assert response is None
    assert model is None
    assert groq_safe._AUTH_DISABLED is True
