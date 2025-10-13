import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace


def _example_events():
    now = datetime.utcnow()
    return [
        {
            "event": "Economic report",
            "datetime": (now + timedelta(hours=1)).isoformat() + "Z",
            "impact": "high",
        }
    ]


def test_analyze_news_with_llm_async_valid_json(monkeypatch):
    import fetch_news

    monkeypatch.setattr(fetch_news, "GROQ_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(fetch_news.config, "get_groq_model", lambda: "test-model", raising=False)

    async def fake_chat(messages, *, model, temperature, max_tokens):
        assert model == "test-model"
        assert "Assess the market impact" in messages[1]["content"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"safe_decision": "no", "reason": "Volatility expected"}
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(fetch_news, "_chat_completion_async", fake_chat, raising=False)

    def fake_quantify(events):
        return {
            "considered_events": len(events),
            "events_in_window": events,
            "window_hours": 24,
            "high_impact_events": 1,
            "risk_score": 3.0,
        }

    monkeypatch.setattr(fetch_news, "quantify_event_risk", fake_quantify, raising=False)

    def fake_reconcile(decision, reason, metrics):
        assert decision is False
        assert reason == "Volatility expected"
        assert metrics["high_impact_events"] == 1
        return False, 0.6, reason

    monkeypatch.setattr(
        fetch_news, "reconcile_with_quant_filters", fake_reconcile, raising=False
    )

    result = asyncio.run(fetch_news.analyze_news_with_llm_async(_example_events()))

    assert result == {"safe": False, "sensitivity": 0.6, "reason": "Volatility expected"}


def test_analyze_news_with_llm_async_retries_on_runtime_error(monkeypatch):
    import fetch_news

    monkeypatch.setattr(fetch_news, "GROQ_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(
        fetch_news.config, "get_groq_model", lambda: "custom-model", raising=False
    )

    calls = []

    async def fake_chat(messages, *, model, temperature, max_tokens):
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("Groq LLM request failed")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"safe_decision": "yes", "reason": "Fallback succeeded"}
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(fetch_news, "_chat_completion_async", fake_chat, raising=False)

    monkeypatch.setattr(
        fetch_news,
        "quantify_event_risk",
        lambda events: {
            "considered_events": len(events),
            "events_in_window": events,
            "window_hours": 24,
            "high_impact_events": 1,
            "risk_score": 2.0,
        },
        raising=False,
    )

    monkeypatch.setattr(
        fetch_news,
        "reconcile_with_quant_filters",
        lambda decision, reason, metrics: (True, 0.1, reason),
        raising=False,
    )

    result = asyncio.run(fetch_news.analyze_news_with_llm_async(_example_events()))

    assert result == {
        "safe": True,
        "sensitivity": 0.1,
        "reason": "Fallback succeeded",
    }
    assert calls == ["custom-model", fetch_news.config.DEFAULT_GROQ_MODEL]


def test_analyze_news_with_llm_async_uses_local_fallback(monkeypatch):
    import fetch_news

    monkeypatch.setattr(fetch_news, "GROQ_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(fetch_news.config, "get_groq_model", lambda: "custom-model", raising=False)

    async def fake_chat(messages, *, model, temperature, max_tokens):
        raise RuntimeError("Groq LLM request failed")

    monkeypatch.setattr(fetch_news, "_chat_completion_async", fake_chat, raising=False)

    monkeypatch.setattr(
        fetch_news,
        "quantify_event_risk",
        lambda events: {
            "considered_events": len(events),
            "events_in_window": events,
            "window_hours": 6,
            "high_impact_events": 1,
            "risk_score": 3.5,
        },
        raising=False,
    )

    monkeypatch.setattr(
        fetch_news,
        "reconcile_with_quant_filters",
        lambda decision, reason, metrics: (decision, 0.4, f"LOCAL:{reason}"),
        raising=False,
    )

    def fake_adapter():
        return (
            lambda: True,
            lambda prompt, temperature=0.1: json.dumps(
                {"safe_decision": "no", "reason": "Local fallback"}
            ),
        )

    monkeypatch.setattr(fetch_news, "_get_local_llm_adapter", fake_adapter, raising=False)

    result = asyncio.run(fetch_news.analyze_news_with_llm_async(_example_events()))

    assert result == {"safe": False, "sensitivity": 0.4, "reason": "LOCAL:Local fallback"}


def test_analyze_news_with_llm_async_handles_non_json(monkeypatch):
    import fetch_news

    monkeypatch.setattr(fetch_news, "GROQ_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(fetch_news.config, "get_groq_model", lambda: "test-model", raising=False)

    async def fake_chat(messages, *, model, temperature, max_tokens):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="I cannot comply with that request.")
                )
            ]
        )

    monkeypatch.setattr(fetch_news, "_chat_completion_async", fake_chat, raising=False)
    monkeypatch.setattr(
        fetch_news,
        "quantify_event_risk",
        lambda events: {
            "considered_events": len(events),
            "events_in_window": events,
            "window_hours": 24,
            "high_impact_events": 0,
            "risk_score": 0.0,
        },
        raising=False,
    )

    result = asyncio.run(fetch_news.analyze_news_with_llm_async(_example_events()))

    assert result == {
        "safe": True,
        "sensitivity": 0,
        "reason": "LLM non-JSON response",
    }


def test_analyze_news_with_llm_sync_wrapper(monkeypatch):
    import fetch_news

    async def fake_async(events, *, session=None):
        return {"safe": False, "sensitivity": 0.2, "reason": "Test"}

    monkeypatch.setattr(
        fetch_news, "analyze_news_with_llm_async", fake_async, raising=False
    )

    result = fetch_news.analyze_news_with_llm(_example_events())

    assert result == {"safe": False, "sensitivity": 0.2, "reason": "Test"}
