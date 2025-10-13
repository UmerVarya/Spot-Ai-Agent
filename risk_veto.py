"""Deterministic risk veto guardrails for the trading agent.

This module centralises the pre-trade veto heuristics so they can be reused
from the agent loop and unit tests. The guard checks minimise reliance on the
LLM when the environment is clearly unsafe (e.g. imminent macro news or
conflicts with the BTC trend).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "RiskVetoResult",
    "evaluate_risk_veto",
    "minutes_until_next_event",
]

_DEFAULT_NEWS_WINDOW_MINUTES = 30.0
_DEFAULT_MAX_RR = 2.0
_VOL_SPIKE_THRESHOLD = float(os.getenv("VOLATILITY_SPIKE_THRESHOLD", "0.9"))
_NEWS_EVENTS_PATH = os.getenv("NEWS_EVENTS_PATH", "news_events.json")


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _normalise_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text not in {"long", "short"}:
        return "long"
    return text


def _normalise_trend(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"up", "bull", "bullish", "long"}:
        return "up"
    if text in {"down", "bear", "bearish", "short"}:
        return "down"
    if text in {"flat", "neutral"}:
        return "flat"
    numeric = _safe_float(value)
    if numeric is None:
        return "flat"
    if numeric > 0:
        return "up"
    if numeric < 0:
        return "down"
    return "flat"


@dataclass
class RiskVetoResult:
    """Container for the deterministic veto output."""

    enter: bool
    reasons: list[str]
    conflicts: list[str]
    max_rr: float

    def to_json(self) -> dict[str, Any]:
        return {
            "enter": self.enter,
            "reasons": list(self.reasons),
            "conflicts": list(self.conflicts),
            "max_rr": float(self.max_rr),
        }


def minutes_until_next_event(
    events: Iterable[Mapping[str, Any]] | None = None,
    *,
    path: str | None = None,
) -> float | None:
    """Return minutes until the next scheduled news event if available."""

    dataset: Sequence[Mapping[str, Any]]
    if events is None:
        target_path = path or _NEWS_EVENTS_PATH
        try:
            with open(target_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except FileNotFoundError:
            return None
        except Exception:
            return None
        if not isinstance(loaded, Sequence):
            return None
        dataset = loaded  # type: ignore[assignment]
    else:
        dataset = list(events)

    now = datetime.now(timezone.utc)
    best_minutes: float | None = None
    for event in dataset:
        dt_raw = event.get("datetime")
        if not isinstance(dt_raw, str):
            continue
        try:
            event_dt = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if event_dt.tzinfo is None:
            event_dt = event_dt.replace(tzinfo=timezone.utc)
        delta_minutes = (event_dt - now).total_seconds() / 60.0
        if delta_minutes < 0:
            continue
        if best_minutes is None or delta_minutes < best_minutes:
            best_minutes = delta_minutes
    if best_minutes is None:
        return None
    return round(best_minutes, 2)


def _base_result(max_rr: float) -> RiskVetoResult:
    return RiskVetoResult(enter=True, reasons=["baseline checks clear"], conflicts=[], max_rr=max_rr)


def evaluate_risk_veto(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate deterministic guardrails and return strict JSON output."""

    max_rr = _safe_float(payload.get("max_rr"))
    if max_rr is None:
        max_rr = _DEFAULT_MAX_RR
    result = _base_result(max_rr)

    direction = _normalise_direction(payload.get("direction"))
    btc_trend = _normalise_trend(payload.get("btc_trend"))
    if direction == "long" and btc_trend == "down":
        conflict_reason = "btc trend conflict"
        if conflict_reason not in result.reasons:
            result.reasons.append(conflict_reason)
        if conflict_reason not in result.conflicts:
            result.conflicts.append(conflict_reason)
    elif direction == "short" and btc_trend == "up":
        conflict_reason = "btc trend conflict"
        if conflict_reason not in result.reasons:
            result.reasons.append(conflict_reason)
        if conflict_reason not in result.conflicts:
            result.conflicts.append(conflict_reason)
    else:
        trend_reason = f"btc trend {btc_trend or 'flat'}"
        if trend_reason not in result.reasons:
            result.reasons.append(trend_reason)

    news_minutes = _safe_float(payload.get("time_to_news"))
    if news_minutes is None:
        news_minutes = _safe_float(payload.get("time_to_news_minutes"))
    if news_minutes is not None:
        if news_minutes < 0:
            news_minutes = 0.0
        window = _safe_float(payload.get("news_window")) or _DEFAULT_NEWS_WINDOW_MINUTES
        if news_minutes < window:
            label = "news<30m"
            if "news within window" not in result.reasons:
                result.reasons.append("news within window")
            if label not in result.conflicts:
                result.conflicts.append(label)
        else:
            result.reasons.append(f"next news in {news_minutes:.0f}m")
    else:
        result.reasons.append("no scheduled news")

    vol_percentile = _safe_float(payload.get("volatility"))
    if vol_percentile is None:
        vol_percentile = _safe_float(payload.get("volatility_percentile"))
    vol_threshold = _safe_float(payload.get("volatility_threshold")) or _VOL_SPIKE_THRESHOLD
    if vol_percentile is not None:
        if vol_percentile > 1:
            vol_percentile = vol_percentile / 100.0
        if vol_percentile >= vol_threshold:
            if "volatility spike" not in result.reasons:
                result.reasons.append("volatility spike")
            if "volatility_spike" not in result.conflicts:
                result.conflicts.append("volatility_spike")
            result.max_rr = min(result.max_rr, 1.6)
        elif vol_percentile >= 0.75:
            capped = min(result.max_rr, 1.8)
            if not math.isclose(result.max_rr, capped):
                result.max_rr = capped
                result.reasons.append("volatility elevated - max_rr capped")
        else:
            result.reasons.append(f"volatility pct {vol_percentile:.2f}")
    else:
        result.reasons.append("volatility unknown")

    if result.conflicts:
        if "deterministic conflicts present - veto" not in result.reasons:
            result.reasons.append("deterministic conflicts present - veto")
        result.enter = False
    return result.to_json()
