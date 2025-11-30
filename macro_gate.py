"""Macro gating decision helpers with debug visibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping

from config import macro_strict_mode


@dataclass
class MacroDecisionDebug:
    """Structured macro decision payload for observability.

    Attributes
    ----------
    macro_ok:
        Baseline macro safety assessment before applying strict/soft mode.
    strict_macro_ok:
        Final decision after applying the configured strictness policy.
    reasons:
        Human-readable reasons explaining veto/advisory signals.
    inputs:
        Minimal set of inputs that influenced the decision for logging/stats.
    """

    macro_ok: bool
    strict_macro_ok: bool
    reasons: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def evaluate_macro_gate(
    news_state: Mapping[str, Any] | None,
    *,
    skip_all: bool = False,
    skip_alt: bool = False,
    macro_filter_reasons: Iterable[str] | None = None,
    strict_mode: bool | None = None,
) -> MacroDecisionDebug:
    """Evaluate macro/news gating and return structured debug context.

    The gating combines:
    - News monitor safety/halt signals (``news_state``)
    - Macro filter skip flags (``skip_all`` / ``skip_alt``)

    It is intentionally tolerant of missing data: ``safe=None`` is treated as
    a neutral/allow state so that stale or incomplete news payloads do not act
    as a permanent kill switch.
    """

    reasons: list[str] = []
    inputs: MutableMapping[str, Any] = {}
    strict = macro_strict_mode() if strict_mode is None else bool(strict_mode)

    macro_ok = True
    news_state = news_state or {}
    news_safe = news_state.get("safe")
    news_reason = news_state.get("reason")
    if news_reason:
        inputs["news_reason"] = str(news_reason)
    inputs["news_safe"] = news_safe
    inputs["news_last_checked"] = news_state.get("last_checked")
    inputs["news_halt_mode"] = news_state.get("halt_mode")

    if news_state.get("stale"):
        _append_reason(reasons, "news_state_stale")

    if news_state.get("halt_trading"):
        macro_ok = False
        _append_reason(reasons, "news_halt_trading_flag")

    if news_safe is False:
        macro_ok = False
        _append_reason(reasons, "news_marked_unsafe")
        if news_reason:
            _append_reason(reasons, f"reason:{news_reason}")
    elif news_safe is None:
        _append_reason(reasons, "news_safe_missing_default_allow")

    if skip_all:
        macro_ok = False
        _append_reason(reasons, "macro_filter_skip_all")
    if skip_alt:
        _append_reason(reasons, "macro_filter_skip_alt")
    for reason in macro_filter_reasons or ():
        _append_reason(reasons, f"macro_filter:{reason}")

    strict_macro_ok = macro_ok if strict else True
    return MacroDecisionDebug(
        macro_ok=macro_ok,
        strict_macro_ok=strict_macro_ok,
        reasons=reasons,
        inputs=dict(inputs),
    )


__all__ = ["MacroDecisionDebug", "evaluate_macro_gate"]
