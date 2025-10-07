"""Helper utilities for validating Groq LLM risk assessments.

The Groq LLM is occasionally used to classify whether upcoming macro events
make trading conditions unsafe.  Historical runs have shown that the model can
hallucinate numeric values (for example, inventing arbitrary ``sensitivity``
scores) or return payloads that do not align with simple quantitative
heuristics (such as the number of high impact events in the next few hours).

This module centralises the guard-rails that harden those interactions:

* ``quantify_event_risk`` performs a lightweight, fully deterministic
  evaluation of upcoming events.
* ``parse_llm_json`` constrains the acceptable LLM response format to a
  yes/no decision with an accompanying reason.
* ``reconcile_with_quant_filters`` cross-checks the LLM output with the
  deterministic metrics and applies overrides when the assessment is
  inconsistent.

The helpers are intentionally free from network or SDK dependencies so they can
be unit tested in isolation and reused by both ``fetch_news`` and
``news_filter``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import json
from datetime import datetime


_DEFAULT_WINDOW_HOURS = 6
_DEFAULT_LOOKBACK_HOURS = 1


def _normalise_impact(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().lower()


def quantify_event_risk(
    events: Iterable[Mapping[str, Any]],
    *,
    window_hours: int = _DEFAULT_WINDOW_HOURS,
    lookback_hours: int = _DEFAULT_LOOKBACK_HOURS,
) -> dict[str, Any]:
    """Return deterministic risk metrics for ``events``.

    The function considers events occurring within ``lookback_hours`` in the
    past and ``window_hours`` into the future.  Each event contributes a weight
    based on its impact label, producing a coarse ``risk_score`` that can be
    mapped onto a sensitivity value.
    """

    now = datetime.utcnow()
    risk_score = 0.0
    high_impact_events = 0
    medium_impact_events = 0
    low_impact_events = 0
    considered_events: list[Mapping[str, Any]] = []

    for event in events:
        dt_raw = event.get("datetime")
        if not isinstance(dt_raw, str):
            continue
        try:
            event_dt = datetime.fromisoformat(dt_raw.replace("Z", ""))
        except ValueError:
            continue

        hours_until = (event_dt - now).total_seconds() / 3600.0
        if hours_until < -float(lookback_hours) or hours_until > float(window_hours):
            continue

        considered_events.append(event)

        impact = _normalise_impact(event.get("impact"))
        if impact == "high":
            weight = 3.0
            high_impact_events += 1
        elif impact == "medium":
            weight = 2.0
            medium_impact_events += 1
        elif impact == "low":
            weight = 1.0
            low_impact_events += 1
        else:
            # Unknown impact labels are treated as medium importance.
            weight = 1.5

        if hours_until < 0:
            # Events that already occurred still matter, but discount them.
            weight *= 0.5

        risk_score += weight

    return {
        "risk_score": risk_score,
        "high_impact_events": high_impact_events,
        "medium_impact_events": medium_impact_events,
        "low_impact_events": low_impact_events,
        "considered_events": len(considered_events),
        "events_in_window": considered_events,
        "window_hours": window_hours,
    }


def derive_sensitivity(risk_score: float) -> float:
    """Map ``risk_score`` onto a bounded [0, 1] sensitivity value."""

    if risk_score <= 0:
        return 0.0
    return min(1.0, round(risk_score / 5.0, 2))


def parse_llm_json(raw_reply: str, logger) -> tuple[bool | None, str | None]:
    """Parse and validate the LLM response.

    The LLM is expected to respond with a JSON object containing ``safe`` or
    ``safe_decision`` (either ``"yes"``/``"no"`` or their boolean equivalents)
    and ``reason``.  Any deviation results in ``None`` being returned so callers
    can fall back to deterministic defaults.
    """

    try:
        parsed = json.loads(raw_reply)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON response: %s", raw_reply)
        return None, "LLM non-JSON response"

    if not isinstance(parsed, Mapping):
        logger.warning("LLM JSON payload is not an object: %s", parsed)
        return None, "LLM malformed JSON response"

    decision_value = parsed.get("safe_decision", parsed.get("safe"))
    reason_value = parsed.get("reason")

    decision: bool | None
    if isinstance(decision_value, bool):
        decision = decision_value
    elif isinstance(decision_value, str):
        lowered = decision_value.strip().lower()
        if lowered in {"yes", "y", "true", "safe"}:
            decision = True
        elif lowered in {"no", "n", "false", "unsafe"}:
            decision = False
        else:
            decision = None
    else:
        decision = None

    if decision is None:
        logger.warning("LLM JSON missing or invalid safe decision: %s", parsed)
        return None, "LLM malformed JSON response"

    if not isinstance(reason_value, str) or not reason_value.strip():
        reason_value = "No reason provided."

    return decision, reason_value.strip()


def reconcile_with_quant_filters(
    safe_decision: bool,
    reason: str,
    metrics: Mapping[str, Any],
) -> tuple[bool, float, str]:
    """Enforce deterministic guard-rails on the LLM decision."""

    risk_score = float(metrics.get("risk_score", 0.0))
    high_events = int(metrics.get("high_impact_events", 0))
    window_hours = int(metrics.get("window_hours", _DEFAULT_WINDOW_HOURS))
    reason = reason.strip() or "No reason provided."

    sensitivity = derive_sensitivity(risk_score)

    if risk_score >= 3.0 and safe_decision:
        overridden_reason = (
            "Overriding LLM decision due to quantitative risk signal: "
            f"{high_events} high-impact events within the next {window_hours} hours. {reason}"
        ).strip()
        return False, sensitivity, overridden_reason

    if risk_score <= 0.5 and not safe_decision:
        overridden_reason = (
            "Quantitative filters detected no material risks in the upcoming "
            f"{window_hours} hours. Treating conditions as safe. {reason}"
        ).strip()
        return True, 0.0, overridden_reason

    return safe_decision, sensitivity, reason

