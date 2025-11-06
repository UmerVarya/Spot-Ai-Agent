"""Utilities for retrieving macro gating context with resilient fallbacks."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Tuple

from btc_dominance import get_btc_dominance
from fear_greed import get_fear_greed_index

LOG = logging.getLogger(__name__)

DEFAULT_BTC_DOMINANCE = 50.0
DEFAULT_FEAR_GREED = 50

_LOCK = threading.Lock()
_LAST_CONTEXT: Dict[str, Any] = {
    "btc_dominance": DEFAULT_BTC_DOMINANCE,
    "fear_greed": DEFAULT_FEAR_GREED,
    "macro_sentiment": "neutral",
    "stale": True,
    "timestamp": 0.0,
}


def _coerce_float(value: Any, previous: Any, default: float) -> Tuple[float, bool]:
    """Return a sane float and whether a fallback was used."""

    try:
        if value is not None:
            coerced = float(value)
            return coerced, False
    except (TypeError, ValueError):
        LOG.debug("Invalid float candidate for macro context: %s", value, exc_info=True)

    try:
        if previous is not None:
            coerced = float(previous)
            return coerced, True
    except (TypeError, ValueError):
        LOG.debug(
            "Previous macro float value unusable; falling back to default", exc_info=True
        )

    return float(default), True


def _coerce_int(value: Any, previous: Any, default: int) -> Tuple[int, bool]:
    """Return a sane integer and whether a fallback was used."""

    try:
        if value is not None:
            return int(float(value)), False
    except (TypeError, ValueError):
        LOG.debug("Invalid int candidate for macro context: %s", value, exc_info=True)

    try:
        if previous is not None:
            return int(float(previous)), True
    except (TypeError, ValueError):
        LOG.debug(
            "Previous macro int value unusable; falling back to default", exc_info=True
        )

    return int(default), True


def _evaluate_macro_sentiment(btc_d: float, fg_index: int) -> str:
    if btc_d > 52 or fg_index < 30:
        return "risk_off"
    if btc_d < 48 and fg_index > 60:
        return "risk_on"
    return "neutral"


def get_macro_context() -> Dict[str, Any]:
    """Return macro context with retries, fallbacks and last-good cache."""

    with _LOCK:
        previous = dict(_LAST_CONTEXT)

    btc_raw = None
    fg_raw = None
    try:
        btc_raw = get_btc_dominance()
    except Exception as exc:  # pragma: no cover - safety net
        LOG.warning("BTC dominance fetch failed: %s", exc)

    try:
        fg_raw = get_fear_greed_index()
    except Exception as exc:  # pragma: no cover - safety net
        LOG.warning("Fear & Greed fetch failed: %s", exc)

    btc_value, btc_fallback = _coerce_float(
        btc_raw, previous.get("btc_dominance"), DEFAULT_BTC_DOMINANCE
    )
    fg_value, fg_fallback = _coerce_int(
        fg_raw, previous.get("fear_greed"), DEFAULT_FEAR_GREED
    )
    macro_sentiment = _evaluate_macro_sentiment(btc_value, fg_value)
    stale = btc_fallback or fg_fallback
    context = {
        "btc_dominance": round(float(btc_value), 2),
        "fear_greed": int(fg_value),
        "macro_sentiment": macro_sentiment,
        "stale": stale,
        "timestamp": time.time(),
    }

    with _LOCK:
        _LAST_CONTEXT.update(context)

    return context


__all__ = ["get_macro_context"]
