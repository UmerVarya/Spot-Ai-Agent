from datetime import date
from pathlib import Path

from spot.analytics.live_scan_stats import (
    DailyScanStats,
    aggregate_symbol_stats,
    load_scan_stats,
)


def _base_flags(**overrides):
    flags = {
        "session_ok": True,
        "volume_ok": True,
        "atr_ok": True,
        "macro_ok": True,
        "alpha_ok": True,
        "prob_ok": True,
    }
    flags.update(overrides)
    return flags


def test_record_scan_and_reasons(tmp_path: Path) -> None:
    stats = DailyScanStats.load_or_create(date(2025, 1, 2), "sess", base_dir=tmp_path)
    stats.record_scan(
        symbol="BTCUSDT",
        had_candidate=False,
        score=None,
        prob=None,
        filter_flags=_base_flags(),
        llm_info={},
        final_decision={},
    )
    stats.record_scan(
        symbol="BTCUSDT",
        had_candidate=True,
        score=4.2,
        prob=0.42,
        filter_flags=_base_flags(macro_ok=False, prob_ok=False),
        llm_info={"llm_called": True, "llm_success": True, "llm_approved": False},
        final_decision={"entered_trade": False, "blocked_by_risk": False, "rejected_by_llm": True},
    )
    stats.record_scan(
        symbol="ETHUSDT",
        had_candidate=True,
        score=5.5,
        prob=0.8,
        filter_flags=_base_flags(prob_ok=True),
        llm_info={"llm_called": True, "llm_success": True, "llm_approved": True},
        final_decision={"entered_trade": True, "blocked_by_risk": False},
    )

    btc = stats.data["symbols"]["BTCUSDT"]
    assert btc["total_scans"] == 2
    assert btc["scans_with_candidate"] == 1
    assert btc["filters"]["passed_macro_filter"] == 0
    assert btc["filters"]["passed_prob_gate"] == 0
    assert btc["llm"]["llm_calls_total"] == 1
    assert btc["llm"]["llm_veto"] == 1
    assert btc["final"]["rejected_by_llm"] == 1
    assert btc["primary_no_trade_reasons"]["macro_halt"] == 1

    eth = stats.data["symbols"]["ETHUSDT"]
    assert eth["final"]["trades_entered"] == 1
    assert eth["llm"]["llm_approved"] == 1


def test_flush_and_reload(tmp_path: Path) -> None:
    stats = DailyScanStats.load_or_create(date(2025, 1, 3), "sess", base_dir=tmp_path)
    stats.record_scan(
        symbol="BTCUSDT",
        had_candidate=True,
        score=3.0,
        prob=0.5,
        filter_flags=_base_flags(alpha_ok=False),
        llm_info={},
        final_decision={"entered_trade": False, "rejected_pre_llm": True},
    )
    stats.flush()
    target = tmp_path / "scan_stats_2025-01-03.json"
    assert target.exists()
    reloaded = DailyScanStats.load_or_create(date(2025, 1, 3), "sess-b", base_dir=tmp_path)
    assert "sess" in reloaded.data["agent_session_ids"]
    assert "sess-b" in reloaded.data["agent_session_ids"]
    assert reloaded.data["symbols"]["BTCUSDT"]["filters"]["passed_alpha_score_gate"] == 0


def test_aggregate_symbol_stats(tmp_path: Path) -> None:
    stats = DailyScanStats.load_or_create(date(2025, 1, 4), "sess", base_dir=tmp_path)
    stats.record_scan(
        symbol="BTCUSDT",
        had_candidate=True,
        score=5.0,
        prob=0.7,
        filter_flags=_base_flags(),
        llm_info={"llm_called": True, "llm_success": True, "llm_approved": True},
        final_decision={"entered_trade": True},
    )
    stats.record_scan(
        symbol="ETHUSDT",
        had_candidate=True,
        score=2.0,
        prob=0.3,
        filter_flags=_base_flags(alpha_ok=False, prob_ok=False),
        llm_info={"llm_called": False},
        final_decision={"entered_trade": False, "blocked_by_risk": True},
    )
    stats.flush()
    data = load_scan_stats(tmp_path / "scan_stats_2025-01-04.json")
    aggregate = aggregate_symbol_stats(data, ["BTCUSDT", "ETHUSDT"])
    assert aggregate["total_scans"] == 2
    assert aggregate["scans_with_candidate"] == 2
    assert aggregate["passed_alpha"] == 1
    assert aggregate["trades_entered"] == 1
    assert aggregate["primary_reasons"]["risk_dd_halt"] == 1
