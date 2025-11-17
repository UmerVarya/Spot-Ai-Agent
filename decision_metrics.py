"""Utilities for logging per-signal decision diagnostics and aggregates."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

_SUMMARY_INTERVAL_SECONDS = float(
    os.getenv("DECISION_METRIC_SUMMARY_INTERVAL", "900")
)

_DEFAULT_BREAKDOWN: Dict[str, Any] = {
    "symbol": None,
    "norm_score": None,
    "dyn_threshold": None,
    "setup_type": None,
    "direction_raw": None,
    "direction_final": None,
    "profile_min_score": None,
    "entry_cutoff": None,
    "alt_adj": None,
    "adjusted_score": None,
    "pos_size": None,
    "volume_gate_pass": True,
    "spread_gate_pass": True,
    "obi_gate_pass": True,
    "macro_veto": False,
    "news_veto": False,
    "profile_veto": False,
    "sr_guard_pass": True,
    "auction_guard_pass": True,
    "cooldown_block": False,
    "alt_adj_block": False,
    "volume_gate_reason": None,
    "volume_ok_for_size": True,
    "primary_skip_reason": None,
    "primary_skip_text": None,
    "forced_long_applied": False,
}

_COUNTERS: Dict[str, int] = {
    "volume_gate_fail": 0,
    "spread_gate_fail": 0,
    "obi_gate_fail": 0,
    "macro_veto": 0,
    "news_veto": 0,
    "profile_veto": 0,
    "sr_guard_fail": 0,
    "auction_guard_fail": 0,
    "alt_adj_block": 0,
    "low_score": 0,
    "pos_size_zero": 0,
    "direction_none": 0,
}

_TOTAL_SIGNALS = 0
_TRADES_OPENED = 0
_LAST_SUMMARY_TS = 0.0


def new_decision_breakdown(symbol: str | None) -> Dict[str, Any]:
    """Return a fresh breakdown dict populated with default keys."""

    breakdown = dict(_DEFAULT_BREAKDOWN)
    breakdown["symbol"] = symbol
    return breakdown


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, (time.struct_time,)):
        return tuple(value)
    return value


def log_decision_breakdown(
    symbol: str,
    snapshot: Optional[Mapping[str, Any]],
    breakdown: Mapping[str, Any] | None,
) -> None:
    """Emit a structured log line summarizing why a signal passed or failed."""

    payload: Dict[str, Any] = {
        "symbol": symbol,
    }
    if breakdown:
        for key, value in breakdown.items():
            if key is None:
                continue
            payload[key] = value
    if snapshot:
        snapshot_summary: Dict[str, Any] = {}
        for field in (
            "setup_type",
            "activation_threshold",
            "auction_state",
            "order_flow_score",
            "order_flow_flag",
            "order_book_imbalance",
            "spread_bps",
        ):
            if field in snapshot:
                snapshot_summary[field] = snapshot[field]
        if snapshot_summary:
            payload["snapshot"] = snapshot_summary
    try:
        encoded = json.dumps(payload, default=_json_default, sort_keys=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to encode decision breakdown for %s: %s", symbol, exc)
        encoded = str(payload)
    logger.info("[METRIC] DECISION_BREAKDOWN: %s", encoded)


def record_signal_evaluated() -> None:
    global _TOTAL_SIGNALS
    _TOTAL_SIGNALS += 1


def record_trade_opened() -> None:
    global _TRADES_OPENED
    _TRADES_OPENED += 1


def record_skip_reason(reason_key: Optional[str]) -> None:
    if not reason_key:
        return
    if reason_key not in _COUNTERS:
        _COUNTERS[reason_key] = 0
    _COUNTERS[reason_key] += 1


def maybe_log_summary() -> None:
    global _LAST_SUMMARY_TS
    now = time.time()
    if _SUMMARY_INTERVAL_SECONDS <= 0:
        interval = 900.0
    else:
        interval = _SUMMARY_INTERVAL_SECONDS
    if now - _LAST_SUMMARY_TS < interval:
        return
    parts: list[str] = [
        f"total_signals={_TOTAL_SIGNALS}",
        f"trades_opened={_TRADES_OPENED}",
    ]
    for key in (
        "volume_gate_fail",
        "spread_gate_fail",
        "obi_gate_fail",
        "profile_veto",
        "macro_veto",
        "news_veto",
        "sr_guard_fail",
        "auction_guard_fail",
        "alt_adj_block",
        "low_score",
        "direction_none",
        "pos_size_zero",
    ):
        parts.append(f"{key}={_COUNTERS.get(key, 0)}")
    logger.info("[METRIC] DECISION_SUMMARY: %s", ", ".join(parts))
    _LAST_SUMMARY_TS = now


def update_breakdown_reason(
    breakdown: Mapping[str, Any] | None,
    reason_key: Optional[str],
    reason_text: Optional[str],
) -> None:
    if not isinstance(breakdown, dict):
        return
    if reason_key:
        breakdown["primary_skip_reason"] = reason_key
    if reason_text:
        breakdown["primary_skip_text"] = reason_text


def ensure_breakdown_fields(
    breakdown: Optional[Mapping[str, Any]],
    required_fields: Iterable[str],
) -> Dict[str, Any]:
    result = new_decision_breakdown(str(breakdown.get("symbol")) if isinstance(breakdown, dict) else None)
    if isinstance(breakdown, Mapping):
        result.update(breakdown)
    for field in required_fields:
        result.setdefault(field, _DEFAULT_BREAKDOWN.get(field))
    return result


__all__ = [
    "log_decision_breakdown",
    "maybe_log_summary",
    "new_decision_breakdown",
    "record_signal_evaluated",
    "record_skip_reason",
    "record_trade_opened",
    "update_breakdown_reason",
]
