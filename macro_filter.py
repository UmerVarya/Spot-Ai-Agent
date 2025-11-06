"""Utilities for retrieving macro gating context with resilient fallbacks."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from btc_dominance import get_btc_dominance
from fear_greed import get_fear_greed_index

LOG = logging.getLogger(__name__)

DEFAULT_BTC_DOMINANCE = 50.0
DEFAULT_FEAR_GREED = 50

MACRO_REQUIRED = os.getenv("MACRO_REQUIRED", "false").lower() in {"1", "true", "yes"}
MACRO_STALE_PENALTY = float(os.getenv("MACRO_STALE_PENALTY", "0.15"))
MACRO_MAX_STALE_SECS = float(
    os.getenv("MACRO_MAX_STALE_SECS", str(6 * 60 * 60))
)
MACRO_REFRESH_SECS = float(os.getenv("MACRO_REFRESH_SECS", "300"))
MACRO_FETCH_TIMEOUT = float(os.getenv("MACRO_FETCH_TIMEOUT", "3.0"))
MACRO_MAX_ATTEMPTS = int(os.getenv("MACRO_MAX_ATTEMPTS", "3"))
MACRO_BACKOFF_SECS = float(os.getenv("MACRO_BACKOFF_SECS", "0.75"))

_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()


def _default_context(reason: str = "bootstrap") -> Dict[str, Any]:
    now = time.time()
    return {
        "btc_dominance": DEFAULT_BTC_DOMINANCE,
        "fear_greed": DEFAULT_FEAR_GREED,
        "macro_sentiment": "neutral",
        "stale": True,
        "timestamp": now,
        "penalty": MACRO_STALE_PENALTY,
        "reason": reason,
        "last_good_timestamp": 0.0,
        "stale_for": None,
        "macro_required": MACRO_REQUIRED,
        "max_stale_seconds": MACRO_MAX_STALE_SECS,
        "refresh_interval": MACRO_REFRESH_SECS,
    }


_LAST_CONTEXT: Dict[str, Any] = _default_context()
_LAST_GOOD: Dict[str, Any] = dict(_LAST_CONTEXT)
_LAST_LOG_STATUS = {"status": "", "timestamp": 0.0}
_REFRESH_THREAD_STARTED = False
_TIMEOUT_SENTINEL = object()


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


def _call_with_timeout(func: Callable[[], Any], *, timeout: float) -> Any:
    """Execute ``func`` with a timeout, returning a sentinel on expiry."""

    finished = threading.Event()
    payload: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            payload["value"] = func()
        except Exception as exc:  # pragma: no cover - propagate to caller
            payload["error"] = exc
        finally:
            finished.set()

    thread = threading.Thread(target=_runner, name="macro-fetch", daemon=True)
    thread.start()
    if not finished.wait(timeout):
        return _TIMEOUT_SENTINEL

    if "error" in payload:
        raise payload["error"]  # type: ignore[misc]

    return payload.get("value")


def _log_status(
    status: str, message: str, *, timestamp: Optional[float] = None, level: int = logging.INFO
) -> None:
    """Log macro status transitions once per refresh cycle."""

    if timestamp is None:
        timestamp = time.time()
    with _LOG_LOCK:
        last_status = _LAST_LOG_STATUS["status"]
        last_ts = _LAST_LOG_STATUS["timestamp"]
        if status == last_status and timestamp <= last_ts:
            return
        _LAST_LOG_STATUS["status"] = status
        _LAST_LOG_STATUS["timestamp"] = timestamp

    LOG.log(level, message)


def _build_context(
    btc_value: float,
    fg_value: int,
    *,
    stale: bool,
    reason: str,
    last_good_ts: float,
) -> Dict[str, Any]:
    now = time.time()
    macro_sentiment = _evaluate_macro_sentiment(btc_value, fg_value)
    stale_for: Optional[float]
    if last_good_ts:
        stale_for = max(0.0, now - last_good_ts)
    else:
        stale_for = None

    penalty = MACRO_STALE_PENALTY if stale else 0.0
    context = {
        "btc_dominance": round(float(btc_value), 2),
        "fear_greed": int(fg_value),
        "macro_sentiment": macro_sentiment,
        "stale": stale,
        "timestamp": now,
        "penalty": penalty,
        "reason": reason,
        "last_good_timestamp": last_good_ts,
        "stale_for": stale_for,
        "macro_required": MACRO_REQUIRED,
        "max_stale_seconds": MACRO_MAX_STALE_SECS,
        "refresh_interval": MACRO_REFRESH_SECS,
    }
    return context


def _refresh_snapshot() -> Dict[str, Any]:
    """Fetch macro inputs with retries and update the shared snapshot."""

    with _LOCK:
        last_good = dict(_LAST_GOOD)

    btc_raw: Optional[float] = None
    fg_raw: Optional[int] = None
    btc_success = False
    fg_success = False
    attempts = max(1, MACRO_MAX_ATTEMPTS)
    errors: Dict[str, Exception] = {}

    for attempt in range(1, attempts + 1):
        if not btc_success:
            try:
                btc_candidate = _call_with_timeout(
                    lambda: get_btc_dominance(timeout=MACRO_FETCH_TIMEOUT),
                    timeout=MACRO_FETCH_TIMEOUT,
                )
                if btc_candidate is _TIMEOUT_SENTINEL:
                    raise TimeoutError("btc_dominance timed out")
                btc_raw = btc_candidate  # type: ignore[assignment]
                btc_success = btc_raw is not None
                if btc_success:
                    errors.pop("btc_dominance", None)
            except Exception as exc:  # pragma: no cover - safety net
                errors["btc_dominance"] = exc

        if not fg_success:
            try:
                fg_candidate = _call_with_timeout(
                    get_fear_greed_index,
                    timeout=MACRO_FETCH_TIMEOUT,
                )
                if fg_candidate is _TIMEOUT_SENTINEL:
                    raise TimeoutError("fear_greed timed out")
                fg_raw = fg_candidate  # type: ignore[assignment]
                fg_success = fg_raw is not None
                if fg_success:
                    errors.pop("fear_greed", None)
            except Exception as exc:  # pragma: no cover - safety net
                errors["fear_greed"] = exc

        if btc_success and fg_success:
            break

        if attempt < attempts:
            sleep_for = min(MACRO_BACKOFF_SECS * (2 ** (attempt - 1)), MACRO_REFRESH_SECS)
            time.sleep(sleep_for)

    btc_prev = last_good.get("btc_dominance", DEFAULT_BTC_DOMINANCE)
    fg_prev = last_good.get("fear_greed", DEFAULT_FEAR_GREED)
    btc_value, btc_fallback = _coerce_float(btc_raw, btc_prev, DEFAULT_BTC_DOMINANCE)
    fg_value, fg_fallback = _coerce_int(fg_raw, fg_prev, DEFAULT_FEAR_GREED)
    stale = btc_fallback or fg_fallback

    last_good_ts = float(last_good.get("last_good_timestamp") or last_good.get("timestamp") or 0.0)
    reason = "live"

    if stale:
        if last_good_ts <= 0.0:
            reason = "macro_missing_neutral"
        else:
            reason = "stale_from_cache"
    else:
        last_good_ts = time.time()

    context = _build_context(btc_value, fg_value, stale=stale, reason=reason, last_good_ts=last_good_ts)

    if stale and last_good_ts and MACRO_MAX_STALE_SECS > 0:
        stale_age = context.get("stale_for") or 0.0
        if stale_age > MACRO_MAX_STALE_SECS:
            context = _default_context(reason="macro_missing_neutral")
            context["timestamp"] = time.time()
            context["penalty"] = MACRO_STALE_PENALTY
            context["stale_for"] = stale_age
            context["last_good_timestamp"] = last_good_ts

    status = "macro_ok"
    level = logging.DEBUG
    error_suffix = ""
    if context["reason"] == "macro_missing_neutral":
        status = "macro_missing_neutral"
        level = logging.WARNING
    elif context["stale"]:
        status = "macro_stale"
        level = logging.INFO

    if errors:
        error_suffix = " errors=" + ", ".join(
            f"{name}:{exc}" for name, exc in errors.items()
        )

    _log_status(
        status,
        f"{status}: btc={context['btc_dominance']} fg={context['fear_greed']} reason={context['reason']}{error_suffix}",
        timestamp=context["timestamp"],
        level=level,
    )

    with _LOCK:
        _LAST_CONTEXT.update(context)
        if not context["stale"] and context["reason"] == "live":
            _LAST_GOOD.clear()
            _LAST_GOOD.update(context)

    return dict(context)


def _refresh_loop() -> None:
    """Background loop that keeps the macro snapshot up to date."""

    # Small initial delay so importers finish setting up logging handlers.
    time.sleep(min(2.0, MACRO_REFRESH_SECS / 10.0))
    while True:
        start = time.time()
        try:
            _refresh_snapshot()
        except Exception as exc:  # pragma: no cover - safety net
            _log_status(
                "macro_missing_neutral",
                f"macro refresh crash: {exc}",
                timestamp=time.time(),
                level=logging.ERROR,
            )
        elapsed = time.time() - start
        sleep_for = max(0.5, MACRO_REFRESH_SECS - elapsed)
        time.sleep(sleep_for)


def _ensure_refresh_thread() -> None:
    global _REFRESH_THREAD_STARTED
    with _LOCK:
        if _REFRESH_THREAD_STARTED:
            return
        _REFRESH_THREAD_STARTED = True

    thread = threading.Thread(target=_refresh_loop, name="macro-refresh", daemon=True)
    thread.start()


def refresh_macro_context_now() -> Dict[str, Any]:
    """Synchronously refresh the macro snapshot (used in tests and warm-up)."""

    return _refresh_snapshot()


def get_macro_context() -> Dict[str, Any]:
    """Return the latest macro snapshot without blocking the scan loop."""

    _ensure_refresh_thread()
    with _LOCK:
        snapshot = dict(_LAST_CONTEXT)

    # Guard against consumers mutating the returned dictionary.
    return dict(snapshot)


__all__ = ["get_macro_context", "refresh_macro_context_now"]
