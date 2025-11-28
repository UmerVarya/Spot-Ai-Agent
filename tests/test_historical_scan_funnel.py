from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


from spot.analytics.historical_scan_funnel import (
    DailyAggregate,
    DailySymbolStats,
    aggregate_by_day,
    aggregate_by_symbol,
    filter_stats_by_date_and_symbols,
    load_all_daily_stats,
    to_funnel_dataframe,
    to_reasons_dataframe,
)


def _write_stat_file(base: Path, day: str, symbols: dict) -> None:
    payload = {"date": day, "agent_session_ids": ["sess"], "symbols": symbols}
    (base / f"scan_stats_{day}.json").write_text(json.dumps(payload))


def test_load_all_daily_stats_parses_files(tmp_path: Path) -> None:
    stats_dir = tmp_path / "live"
    stats_dir.mkdir()
    _write_stat_file(
        stats_dir,
        "2024-01-01",
        {
            "BTC": {
                "total_scans": 2,
                "scans_with_candidate": 1,
                "filters": {"passed_alpha_score_gate": 1, "passed_prob_gate": 0},
                "llm": {"llm_approved": 0, "llm_calls_total": 1, "llm_calls_failed": 0},
                "final": {"trades_entered": 0},
                "primary_no_trade_reasons": {"score_too_low": 1},
            }
        },
    )
    _write_stat_file(
        stats_dir,
        "2024-01-02",
        {
            "ETH": {
                "total_scans": 3,
                "scans_with_candidate": 2,
                "filters": {"passed_alpha_score_gate": 2, "passed_prob_gate": 1},
                "llm": {"llm_approved": 1, "llm_calls_total": 2, "llm_calls_failed": 1},
                "final": {"trades_entered": 1},
                "primary_no_trade_reasons": {"prob_too_low": 1},
            }
        },
    )

    stats = load_all_daily_stats(stats_dir)

    assert [s.date for s in stats] == [date(2024, 1, 1), date(2024, 1, 2)]
    btc = next(s for s in stats if s.symbol == "BTC")
    assert btc.candidates == 1
    assert btc.primary_reasons["score_too_low"] == 1


def test_aggregate_by_day_and_filtering() -> None:
    stats = [
        DailySymbolStats(
            date=date(2024, 1, 1),
            symbol="BTC",
            total_scans=2,
            candidates=1,
            passed_alpha=1,
            passed_prob=0,
            llm_approved=0,
            trades_entered=0,
            llm_calls_total=1,
            llm_calls_failed=0,
            primary_reasons={"score_too_low": 1},
        ),
        DailySymbolStats(
            date=date(2024, 1, 2),
            symbol="ETH",
            total_scans=3,
            candidates=2,
            passed_alpha=2,
            passed_prob=1,
            llm_approved=1,
            trades_entered=1,
            llm_calls_total=2,
            llm_calls_failed=1,
            primary_reasons={"prob_too_low": 1},
        ),
    ]

    filtered = filter_stats_by_date_and_symbols(stats, start_date=date(2024, 1, 2), end_date=None, symbols=None)
    assert len(filtered) == 1 and filtered[0].symbol == "ETH"

    aggregated = aggregate_by_day(stats)
    assert len(aggregated) == 2
    assert aggregated[0].total_scans == 2
    assert aggregated[1].primary_reasons["prob_too_low"] == 1


def test_aggregate_by_symbol_conversions() -> None:
    stats = [
        DailySymbolStats(
            date=date(2024, 1, 1),
            symbol="BTC",
            total_scans=4,
            candidates=2,
            passed_alpha=2,
            passed_prob=1,
            llm_approved=1,
            trades_entered=1,
            llm_calls_total=2,
            llm_calls_failed=0,
            primary_reasons={},
        ),
        DailySymbolStats(
            date=date(2024, 1, 2),
            symbol="BTC",
            total_scans=1,
            candidates=0,
            passed_alpha=0,
            passed_prob=0,
            llm_approved=0,
            trades_entered=0,
            llm_calls_total=0,
            llm_calls_failed=0,
            primary_reasons={},
        ),
    ]

    symbol_df = aggregate_by_symbol(stats)
    assert float(symbol_df.loc[symbol_df["symbol"] == "BTC", "conv_candidates_per_scan"].iloc[0]) == 0.4
    assert float(symbol_df.loc[symbol_df["symbol"] == "BTC", "conv_trades_per_scan"].iloc[0]) == 0.2


def test_dataframe_helpers_are_sorted() -> None:
    aggregates = [
        DailyAggregate(
            date=date(2024, 1, 2),
            total_scans=3,
            candidates=2,
            passed_alpha=2,
            passed_prob=1,
            llm_approved=1,
            trades_entered=1,
            llm_calls_total=2,
            llm_calls_failed=1,
            primary_reasons={"prob_too_low": 1},
        ),
        DailyAggregate(
            date=date(2024, 1, 1),
            total_scans=2,
            candidates=1,
            passed_alpha=1,
            passed_prob=0,
            llm_approved=0,
            trades_entered=0,
            llm_calls_total=1,
            llm_calls_failed=0,
            primary_reasons={"score_too_low": 1},
        ),
    ]

    funnel_df = to_funnel_dataframe(aggregates)
    assert list(funnel_df["date"]) == [date(2024, 1, 1), date(2024, 1, 2)]

    reasons_df = to_reasons_dataframe(aggregates)
    assert list(reasons_df.index) == [date(2024, 1, 1), date(2024, 1, 2)]
    assert set(reasons_df.columns) == {"score_too_low", "prob_too_low"}

