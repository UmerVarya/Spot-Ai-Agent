"""Resilient macro context fetcher with cached fallbacks."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from btc_dominance import get_btc_dominance
from fear_greed import get_fear_greed_index

LOG = logging.getLogger(__name__)

DEFAULT_BTC_DOMINANCE = 50.0
DEFAULT_FEAR_GREED = 50
DEFAULT_SENTIMENT = "neutral"

MACRO_FETCH_TIMEOUT = float(os.getenv("MACRO_FETCH_TIMEOUT", "5"))
MACRO_FETCH_RETRIES = int(os.getenv("MACRO_FETCH_RETRIES", "3"))
MACRO_FETCH_BACKOFF = float(os.getenv("MACRO_FETCH_BACKOFF", "0.75"))

_LOCK = threading.Lock()
_LAST_CONTEXT: Dict[str, Any] = {
    "btc_dominance": DEFAULT_BTC_DOMINANCE,
    "fear_greed": DEFAULT_FEAR_GREED,
    "macro_sentiment": DEFAULT_SENTIMENT,
    "stale": True,
    "status": "neutral",
    "stale_for": 0.0,
    "timestamp": 0.0,
}
_LAST_GOOD_CONTEXT: Dict[str, Any] = dict(_LAST_CONTEXT)


def _invoke_with_timeout(func: Callable[..., Any], timeout: float) -> Any:
    try:
        return func(timeout=timeout)
    except TypeError:
        return func()


def _fetch_with_retry(name: str, func: Callable[..., Any]) -> Tuple[Optional[Any], bool]:
    """Return ``(result, failed)`` while retrying transient errors."""

    delay = MACRO_FETCH_BACKOFF
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, MACRO_FETCH_RETRIES) + 1):
        try:
            return _invoke_with_timeout(func, MACRO_FETCH_TIMEOUT), False
        except Exception as exc:  # pragma: no cover - defensive guard
            last_exc = exc
            if attempt >= MACRO_FETCH_RETRIES:
                break
            LOG.debug(
                "Macro fetch %s attempt %d failed: %s", name, attempt, exc, exc_info=True
            )
            time.sleep(max(0.1, delay))
            delay *= 2

    if last_exc is not None:
        LOG.warning("Macro fetch %s failed after retries: %s", name, last_exc)
    return None, True


def _coerce_float(
    value: Any, previous: Any, last_good: Any, default: float
) -> Tuple[float, bool]:
    """Return a sane float and whether a fallback was used."""

    for candidate, is_fallback in (
        (value, False),
        (previous, True),
        (last_good, True),
        (default, True),
    ):
        try:
            coerced = float(candidate)
        except (TypeError, ValueError):
            LOG.debug("Invalid float candidate for macro context: %s", candidate, exc_info=True)
            continue
        if math.isfinite(coerced):
            return coerced, is_fallback
    return float(default), True


def _coerce_int(value: Any, previous: Any, last_good: Any, default: int) -> Tuple[int, bool]:
    """Return a sane integer and whether a fallback was used."""

    for candidate, is_fallback in (
        (value, False),
        (previous, True),
        (last_good, True),
        (default, True),
    ):
        if candidate is None:
            continue
        try:
            coerced = int(float(candidate))
        except (TypeError, ValueError):
            LOG.debug("Invalid int candidate for macro context: %s", candidate, exc_info=True)
            continue
        if 0 <= coerced <= 100:
            return coerced, is_fallback
        LOG.debug("Macro int candidate out of range: %s", candidate)
    return int(default), True


def _evaluate_macro_sentiment(btc_d: float, fg_index: int) -> str:
    if btc_d > 52 or fg_index < 30:
        return "risk_off"
    if btc_d < 48 and fg_index > 60:
        return "risk_on"
    return "neutral"


def get_macro_context() -> Dict[str, Any]:
    """Return macro context with retries, fallbacks and last-good cache."""

    now = time.time()
    with _LOCK:
        previous = dict(_LAST_CONTEXT)
        last_good = dict(_LAST_GOOD_CONTEXT)

    btc_raw, btc_failed = _fetch_with_retry("btc_dominance", get_btc_dominance)
    fg_raw, fg_failed = _fetch_with_retry("fear_greed", get_fear_greed_index)

    btc_value, btc_fallback = _coerce_float(
        btc_raw, previous.get("btc_dominance"), last_good.get("btc_dominance"), DEFAULT_BTC_DOMINANCE
    )
    fg_value, fg_fallback = _coerce_int(
        fg_raw, previous.get("fear_greed"), last_good.get("fear_greed"), DEFAULT_FEAR_GREED
    )

    macro_sentiment = _evaluate_macro_sentiment(btc_value, fg_value)
    fallback_used = btc_fallback or fg_fallback
    stale = fallback_used or btc_failed or fg_failed

    status: str
    if not stale:
        status = "live"
    elif last_good.get("timestamp", 0.0) > 0:
        status = "cached"
    else:
        status = "neutral"

    last_good_ts = float(last_good.get("timestamp", 0.0))
    stale_for = now - last_good_ts if stale and last_good_ts else 0.0

    context = {
        "btc_dominance": round(float(btc_value), 2),
        "fear_greed": int(fg_value),
        "macro_sentiment": macro_sentiment,
        "stale": bool(stale),
        "status": status,
        "stale_for": float(max(0.0, stale_for)),
        "timestamp": now,
    }

    with _LOCK:
        _LAST_CONTEXT.update(context)
        if not context["stale"]:
            _LAST_GOOD_CONTEXT.update(context)

    return context


__all__ = ["get_macro_context"]
