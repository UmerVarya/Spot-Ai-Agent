from datetime import datetime, timezone

import pandas as pd

import daily_summary


def _disable_llm(monkeypatch) -> None:
    monkeypatch.setattr(daily_summary, "get_groq_client", lambda: None)
    monkeypatch.setattr(daily_summary, "generate_local_daily_recap", lambda *_: None)


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
