import asyncio
import json
from datetime import datetime, timedelta


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

    async def fake_post(session, payload, *, model_used):
        assert payload["model"] == "test-model"
        assert "Assess the market impact" in payload["messages"][1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"safe_decision": "no", "reason": "Volatility expected"}
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(fetch_news, "_post_groq_request", fake_post, raising=False)

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


def test_analyze_news_with_llm_async_handles_non_json(monkeypatch):
    import fetch_news

    monkeypatch.setattr(fetch_news, "GROQ_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(fetch_news.config, "get_groq_model", lambda: "test-model", raising=False)

    async def fake_post(session, payload, *, model_used):
        return {
            "choices": [
                {"message": {"content": "I cannot comply with that request."}}
            ]
        }

    monkeypatch.setattr(fetch_news, "_post_groq_request", fake_post, raising=False)
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
