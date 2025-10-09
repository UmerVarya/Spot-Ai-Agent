"""Tests for the memory retriever fallback behaviour."""

from __future__ import annotations

import pandas as pd

import memory_retriever
from memory_retriever import _fallback_recent_trades


def _build_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T00:00:00Z",
                "symbol": "BTCUSDT",
                "pattern": "Breakout",
                "outcome": "win",
                "confidence": 0.7,
            },
            {
                "timestamp": "2024-01-03T00:00:00Z",
                "symbol": "ETHUSDT",
                "pattern": "Breakout",
                "outcome": "loss",
                "confidence": 0.4,
            },
            {
                "timestamp": "2024-01-04T00:00:00Z",
                "symbol": "BTCUSDT",
                "pattern": "Reversal",
                "outcome": "win",
                "confidence": 0.8,
            },
        ]
    )


def test_fallback_filters_by_symbol_and_pattern() -> None:
    df = _build_df()

    summary = _fallback_recent_trades(
        df,
        max_entries=5,
        symbol="BTCUSDT",
        pattern="Breakout",
    )

    assert "ETHUSDT" not in summary
    assert "Reversal" not in summary
    assert "Breakout" in summary


def test_fallback_no_matches_returns_default_message() -> None:
    df = _build_df()

    summary = _fallback_recent_trades(
        df,
        max_entries=5,
        symbol="BTCUSDT",
        pattern="DoesNotExist",
    )

    assert summary == "No prior trades on record."


def test_summary_filters_when_embeddings_unavailable(tmp_path, monkeypatch) -> None:
    df = _build_df()
    log_path = tmp_path / "trades.csv"
    df.to_csv(log_path, index=False)

    monkeypatch.setattr(memory_retriever, "LOG_FILE", str(log_path))
    monkeypatch.setattr(
        memory_retriever, "_load_trade_embeddings", lambda *_args, **_kwargs: None
    )

    summary = memory_retriever.get_recent_trade_summary(
        symbol="BTCUSDT", pattern="Breakout", max_entries=5
    )

    assert "ETHUSDT" not in summary
    assert "Breakout" in summary
