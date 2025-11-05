from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

import daily_summary


def _disable_llm(monkeypatch) -> None:
    monkeypatch.setattr(daily_summary, "get_groq_client", lambda: None)


def test_generate_daily_summary_no_trades(monkeypatch) -> None:
    _disable_llm(monkeypatch)
    summary = daily_summary.generate_daily_summary(
        "2024-01-01", history=pd.DataFrame()
    )
    assert "No completed trades" in summary


def test_generate_daily_summary_fallback_highlights(monkeypatch) -> None:
    _disable_llm(monkeypatch)
    day = datetime(2024, 1, 5, tzinfo=timezone.utc)
    records = [
        {
            "exit_time": day.isoformat(),
            "symbol": "BTCUSDT",
            "direction": "long",
            "pnl": 120.5,
            "notional": 5000,
            "outcome": "tp1",
            "narrative": "Bought the breakout after reclaiming VWAP",
        },
        {
            "exit_time": day.isoformat(),
            "symbol": "ETHUSDT",
            "direction": "short",
            "pnl": -45.0,
            "notional": 2500,
            "outcome": "sl",
            "exit_reason": "Stop triggered on funding squeeze",
        },
        {
            "exit_time": day.isoformat(),
            "symbol": "SOLUSDT",
            "direction": "long",
            "llm_decision": "vetoed",
            "reason": "LLM advisor vetoed trade due to Fed announcement",
        },
    ]
    df = pd.DataFrame.from_records(records)
    summary = daily_summary.generate_daily_summary(day.date(), history=df)
    assert "2 trade" in summary
    assert "BTCUSDT" in summary
    assert "LLM vetoed 1 setup" in summary
    assert "SOLUSDT" in summary


def test_generate_daily_summary_uses_narrative_model(monkeypatch) -> None:
    day = datetime(2024, 1, 5, tzinfo=timezone.utc)
    df = pd.DataFrame.from_records(
        [
            {
                "exit_time": day.isoformat(),
                "symbol": "BTCUSDT",
                "direction": "long",
                "pnl": 50.0,
            }
        ]
    )

    captured = {}

    def fake_safe_chat_completion(_client, *, model, messages, **kwargs):
        captured["model"] = model
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Synthetic summary")
                )
            ]
        )

    monkeypatch.setattr(daily_summary, "safe_chat_completion", fake_safe_chat_completion)
    monkeypatch.setattr(daily_summary, "get_groq_client", lambda: object())
    monkeypatch.setattr(daily_summary.config, "get_narrative_model", lambda: "narrative-model")

    summary = daily_summary.generate_daily_summary(day.date(), history=df)

    assert summary == "Synthetic summary"
    assert captured["model"] == "narrative-model"


def test_daily_summary_suppresses_llm_error_reason(monkeypatch) -> None:
    _disable_llm(monkeypatch)
    day = datetime(2024, 1, 6, tzinfo=timezone.utc)
    df = pd.DataFrame.from_records(
        [
            {
                "exit_time": day.isoformat(),
                "symbol": "BTCUSDT",
                "direction": "long",
                "pnl": 42.0,
                "narrative": "⚠️ Groq client unavailable for narrative generation.",
            }
        ]
    )

    summary = daily_summary.generate_daily_summary(day.date(), history=df)

    assert "Groq client unavailable" not in summary
