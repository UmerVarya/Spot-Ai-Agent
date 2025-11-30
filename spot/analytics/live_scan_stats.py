from __future__ import annotations

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

__all__ = ["DailyScanStats", "load_scan_stats", "aggregate_symbol_stats", "list_scan_stat_files"]


SCAN_STATS_DIR = Path("spot_data/live_scan_stats")


@dataclass
class DailyScanStats:
    """Container for daily live scan statistics.

    Stats are stored per symbol and persisted to JSON with atomic writes.
    """

    stats_date: date
    agent_session_id: str
    base_dir: Path = field(default_factory=lambda: SCAN_STATS_DIR)
    data: Dict[str, Any] = field(init=False)
    _dirty: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.data = {
            "date": self.stats_date.isoformat(),
            "agent_session_ids": [self.agent_session_id],
            "symbols": {},
        }

    @classmethod
    def load_or_create(
        cls, stats_date: date, agent_session_id: str, base_dir: Path | None = None
    ) -> "DailyScanStats":
        base_path = base_dir or SCAN_STATS_DIR
        instance = cls(stats_date=stats_date, agent_session_id=agent_session_id, base_dir=base_path)
        existing = instance._target_path()
        if existing.exists():
            try:
                content = json.loads(existing.read_text())
            except Exception:
                content = None
            if isinstance(content, dict):
                instance.data = content
                sessions: Iterable[str] = content.get("agent_session_ids", []) or []
                if agent_session_id not in sessions:
                    content.setdefault("agent_session_ids", [])
                    content["agent_session_ids"].append(agent_session_id)
                    instance._dirty = True
        return instance

    def _target_path(self) -> Path:
        fname = f"scan_stats_{self.stats_date.isoformat()}.json"
        return self.base_dir / fname

    def _ensure_symbol(self, symbol: str) -> Dict[str, Any]:
        symbols = self.data.setdefault("symbols", {})
        if symbol not in symbols:
            symbols[symbol] = {
                "total_scans": 0,
                "scans_with_candidate": 0,
                "score_bands": {
                    "ge_1": 0,
                    "ge_2": 0,
                    "ge_3": 0,
                    "ge_4": 0,
                    "ge_5": 0,
                },
                "filters": {
                    "passed_session_filter": 0,
                    "passed_volume_filter": 0,
                    "passed_atr_filter": 0,
                    "passed_macro_filter": 0,
                    "passed_alpha_score_gate": 0,
                    "passed_prob_gate": 0,
                },
                "llm": {
                    "llm_calls_total": 0,
                    "llm_calls_success": 0,
                    "llm_calls_failed": 0,
                    "llm_approved": 0,
                    "llm_veto": 0,
                    "llm_skipped_due_to_unavailable": 0,
                },
                "final": {
                    "trades_entered": 0,
                    "rejected_pre_llm": 0,
                    "rejected_by_llm": 0,
                    "rejected_risk_controls": 0,
                },
                "primary_no_trade_reasons": {
                    "score_too_low": 0,
                    "prob_too_low": 0,
                    "macro_halt": 0,
                    "volume_too_low": 0,
                    "session_blocked": 0,
                    "risk_dd_halt": 0,
                    "llm_veto": 0,
                    "llm_unavailable": 0,
                    "other": 0,
                    "atr_filter_failed": 0,
                },
            }
        return symbols[symbol]

    def record_scan(
        self,
        *,
        symbol: str,
        had_candidate: bool,
        score: Optional[float],
        prob: Optional[float],
        filter_flags: Dict[str, bool],
        llm_info: Dict[str, Any],
        final_decision: Dict[str, Any],
    ) -> None:
        entry = self._ensure_symbol(symbol)
        entry["total_scans"] += 1

        if had_candidate:
            entry["scans_with_candidate"] += 1
            self._update_score_bands(entry, score)
            self._update_filters(entry, filter_flags)
            self._update_llm(entry, llm_info)
            self._update_final(entry, final_decision)
            reason = self._primary_reason(filter_flags, llm_info, final_decision)
            if reason:
                entry["primary_no_trade_reasons"][reason] += 1

        self._dirty = True

    def _update_score_bands(self, entry: Dict[str, Any], score: Optional[float]) -> None:
        try:
            score_val = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_val = None
        bands = entry.setdefault("score_bands", {})
        if score_val is None:
            return
        for threshold in (1, 2, 3, 4, 5):
            if score_val >= threshold:
                bands[f"ge_{threshold}"] = bands.get(f"ge_{threshold}", 0) + 1

    def _update_filters(self, entry: Dict[str, Any], flags: Mapping[str, bool]) -> None:
        filters = entry.setdefault("filters", {})
        filters["passed_session_filter"] = filters.get("passed_session_filter", 0) + bool(
            flags.get("session_ok", False)
        )
        filters["passed_volume_filter"] = filters.get("passed_volume_filter", 0) + bool(
            flags.get("volume_ok", False)
        )
        filters["passed_atr_filter"] = filters.get("passed_atr_filter", 0) + bool(
            flags.get("atr_ok", False)
        )
        filters["passed_macro_filter"] = filters.get("passed_macro_filter", 0) + bool(
            flags.get("macro_ok", False)
        )
        filters["passed_alpha_score_gate"] = filters.get("passed_alpha_score_gate", 0) + bool(
            flags.get("alpha_ok", False)
        )
        filters["passed_prob_gate"] = filters.get("passed_prob_gate", 0) + bool(
            flags.get("prob_ok", False)
        )
        macro_reasons = flags.get("macro_reasons")
        if macro_reasons:
            sub = entry.setdefault("macro_subreasons", {})
            for reason in macro_reasons:
                if not reason:
                    continue
                sub[str(reason)] = sub.get(str(reason), 0) + 1

    def _update_llm(self, entry: Dict[str, Any], llm_info: Mapping[str, Any]) -> None:
        llm_data = entry.setdefault("llm", {})
        called = bool(llm_info.get("llm_called", False))
        success = bool(llm_info.get("llm_success", False)) if called else False
        approved_val = llm_info.get("llm_approved")
        unavailable = bool(llm_info.get("llm_unavailable", False))

        if called:
            llm_data["llm_calls_total"] = llm_data.get("llm_calls_total", 0) + 1
            if success:
                llm_data["llm_calls_success"] = llm_data.get("llm_calls_success", 0) + 1
            else:
                llm_data["llm_calls_failed"] = llm_data.get("llm_calls_failed", 0) + 1
        if unavailable:
            llm_data["llm_skipped_due_to_unavailable"] = llm_data.get(
                "llm_skipped_due_to_unavailable", 0
            ) + 1
        if approved_val is True:
            llm_data["llm_approved"] = llm_data.get("llm_approved", 0) + 1
        elif approved_val is False:
            llm_data["llm_veto"] = llm_data.get("llm_veto", 0) + 1

    def _update_final(self, entry: Dict[str, Any], final_decision: Mapping[str, Any]) -> None:
        final_data = entry.setdefault("final", {})
        if final_decision.get("entered_trade"):
            final_data["trades_entered"] = final_data.get("trades_entered", 0) + 1
        if final_decision.get("rejected_pre_llm"):
            final_data["rejected_pre_llm"] = final_data.get("rejected_pre_llm", 0) + 1
        if final_decision.get("rejected_by_llm"):
            final_data["rejected_by_llm"] = final_data.get("rejected_by_llm", 0) + 1
        if final_decision.get("blocked_by_risk"):
            final_data["rejected_risk_controls"] = final_data.get("rejected_risk_controls", 0) + 1

    def _primary_reason(
        self,
        flags: Mapping[str, bool],
        llm_info: Mapping[str, Any],
        final_decision: Mapping[str, Any],
    ) -> Optional[str]:
        if final_decision.get("entered_trade"):
            return None
        priority = (
            "risk_dd_halt",
            "macro_halt",
            "session_blocked",
            "volume_too_low",
            "atr_filter_failed",
            "score_too_low",
            "prob_too_low",
            "llm_unavailable",
            "llm_veto",
            "other",
        )
        mapped: Dict[str, bool] = {
            "macro_halt": not bool(flags.get("macro_ok", True)),
            "session_blocked": not bool(flags.get("session_ok", True)),
            "volume_too_low": not bool(flags.get("volume_ok", True)),
            "atr_filter_failed": not bool(flags.get("atr_ok", True)),
            "score_too_low": not bool(flags.get("alpha_ok", True)),
            "prob_too_low": not bool(flags.get("prob_ok", True)),
            "risk_dd_halt": bool(final_decision.get("blocked_by_risk")),
            "llm_unavailable": bool(llm_info.get("llm_unavailable", False)),
            "llm_veto": bool(llm_info.get("llm_called") and llm_info.get("llm_approved") is False),
        }
        for key in priority:
            if mapped.get(key):
                return key
        return "other"

    def flush(self) -> None:
        if not self._dirty:
            return
        target = self._target_path()
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=target.name, text=True)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, target)
            self._dirty = False
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def load_scan_stats(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_scan_stat_files(base_dir: Path | None = None) -> list[Path]:
    base_path = base_dir or SCAN_STATS_DIR
    if not base_path.exists():
        return []
    return sorted(base_path.glob("scan_stats_*.json"), reverse=True)


def aggregate_symbol_stats(data: Mapping[str, Any], symbols: Iterable[str]) -> Dict[str, Any]:
    selected = {sym.upper() for sym in symbols}
    summary = {
        "total_scans": 0,
        "scans_with_candidate": 0,
        "passed_alpha": 0,
        "passed_prob": 0,
        "trades_entered": 0,
        "llm_approved": 0,
        "primary_reasons": {},
    }
    symbols_data = data.get("symbols", {}) if isinstance(data, Mapping) else {}
    for sym, payload in symbols_data.items():
        if selected and sym.upper() not in selected:
            continue
        summary["total_scans"] += payload.get("total_scans", 0)
        summary["scans_with_candidate"] += payload.get("scans_with_candidate", 0)
        filters = payload.get("filters", {})
        summary["passed_alpha"] += filters.get("passed_alpha_score_gate", 0)
        summary["passed_prob"] += filters.get("passed_prob_gate", 0)
        final_block = payload.get("final", {})
        summary["trades_entered"] += final_block.get("trades_entered", 0)
        llm_block = payload.get("llm", {})
        summary["llm_approved"] += llm_block.get("llm_approved", 0)
        reasons = payload.get("primary_no_trade_reasons", {}) or {}
        for key, value in reasons.items():
            summary["primary_reasons"][key] = summary["primary_reasons"].get(key, 0) + int(value or 0)
    return summary


