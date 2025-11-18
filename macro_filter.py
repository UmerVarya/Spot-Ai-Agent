"""Utilities for retrieving macro gating context with resilient fallbacks."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from macro_data import get_btc_dominance_cached, get_fear_greed_cached

LOG = logging.getLogger(__name__)

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
        "btc_dominance": None,
        "fear_greed": None,
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
        "btc_timestamp": 0.0,
        "fear_greed_timestamp": 0.0,
        "btc_age_seconds": None,
        "fear_greed_age_seconds": None,
    }


_LAST_CONTEXT: Dict[str, Any] = _default_context()
_LAST_GOOD: Dict[str, Any] = dict(_LAST_CONTEXT)
_LAST_LOG_STATUS = {"status": "", "timestamp": 0.0}
_REFRESH_THREAD_STARTED = False
_TIMEOUT_SENTINEL = object()


def _coerce_float(value: Any, previous: Any) -> Tuple[Optional[float], bool]:
    """Return a sane float and whether a fallback was used."""

    for candidate_value, fallback_flag in ((value, False), (previous, True)):
        if candidate_value is None:
            continue
        try:
            coerced = float(candidate_value)
        except (TypeError, ValueError):
            LOG.debug(
                "Invalid float candidate for macro context: %s",
                candidate_value,
                exc_info=True,
            )
            continue
        if math.isfinite(coerced):
            return coerced, fallback_flag
        LOG.debug(
            "Non-finite macro float candidate: %s", candidate_value, exc_info=True
        )

    return None, True


def _coerce_int(value: Any, previous: Any) -> Tuple[Optional[int], bool]:
    """Return a sane integer and whether a fallback was used."""

    for candidate_value, fallback_flag in ((value, False), (previous, True)):
        if candidate_value is None:
            continue
        try:
            coerced = int(float(candidate_value))
        except (TypeError, ValueError):
            LOG.debug(
                "Invalid int candidate for macro context: %s",
                candidate_value,
                exc_info=True,
            )
            continue
        if 0 <= coerced <= 100:
            return coerced, fallback_flag
        LOG.debug(
            "Macro int candidate out of range (0-100): %s", candidate_value
        )

    return None, True


def _evaluate_macro_sentiment(
    btc_d: Optional[float], fg_index: Optional[int]
) -> str:
    if btc_d is None or fg_index is None:
        return "neutral"
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


def _format_age(age: Optional[float]) -> str:
    if age is None:
        return "n/a"
    if age < 90:
        return f"{age:.0f}s"
    if age < 3600:
        return f"{age / 60.0:.1f}m"
    return f"{age / 3600.0:.1f}h"


def _build_context(
    btc_value: Optional[float],
    fg_value: Optional[int],
    *,
    stale: bool,
    reason: str,
    last_good_ts: float,
    btc_ts: Optional[float],
    fg_ts: Optional[float],
) -> Dict[str, Any]:
    now = time.time()
    macro_sentiment = _evaluate_macro_sentiment(btc_value, fg_value)
    stale_for: Optional[float]
    if last_good_ts:
        stale_for = max(0.0, now - last_good_ts)
    else:
        stale_for = None

    btc_age = max(0.0, now - float(btc_ts)) if btc_ts else None
    fg_age = max(0.0, now - float(fg_ts)) if fg_ts else None
    penalty = MACRO_STALE_PENALTY if stale else 0.0
    context = {
        "btc_dominance": round(float(btc_value), 2) if btc_value is not None else None,
        "fear_greed": int(fg_value) if fg_value is not None else None,
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
        "btc_timestamp": float(btc_ts or 0.0),
        "fear_greed_timestamp": float(fg_ts or 0.0),
        "btc_age_seconds": btc_age,
        "fear_greed_age_seconds": fg_age,
    }
    return context


def _refresh_snapshot() -> Dict[str, Any]:
    """Fetch macro inputs with retries and update the shared snapshot."""

    with _LOCK:
        last_good = dict(_LAST_GOOD)

    btc_raw: Optional[float] = None
    fg_raw: Optional[int] = None
    btc_ts_raw: Optional[float] = None
    fg_ts_raw: Optional[float] = None
    btc_success = False
    fg_success = False
    attempts = max(1, MACRO_MAX_ATTEMPTS)
    errors: Dict[str, Exception] = {}

    for attempt in range(1, attempts + 1):
        if not btc_success:
            try:
                btc_candidate = _call_with_timeout(
                    lambda: get_btc_dominance_cached(),
                    timeout=MACRO_FETCH_TIMEOUT,
                )
                if btc_candidate is _TIMEOUT_SENTINEL:
                    raise TimeoutError("btc_dominance timed out")
                btc_snapshot = btc_candidate  # type: ignore[assignment]
                if btc_snapshot is not None:
                    btc_raw = btc_snapshot.value
                    btc_ts_raw = float(btc_snapshot.ts)
                btc_success = btc_raw is not None
                if btc_success:
                    errors.pop("btc_dominance", None)
            except Exception as exc:  # pragma: no cover - safety net
                errors["btc_dominance"] = exc

        if not fg_success:
            try:
                fg_candidate = _call_with_timeout(
                    get_fear_greed_cached,
                    timeout=MACRO_FETCH_TIMEOUT,
                )
                if fg_candidate is _TIMEOUT_SENTINEL:
                    raise TimeoutError("fear_greed timed out")
                fg_snapshot = fg_candidate  # type: ignore[assignment]
                if fg_snapshot is not None:
                    fg_raw = fg_snapshot.value
                    fg_ts_raw = float(fg_snapshot.ts)
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

    btc_prev = last_good.get("btc_dominance")
    fg_prev = last_good.get("fear_greed")
    btc_value, btc_fallback = _coerce_float(btc_raw, btc_prev)
    fg_value, fg_fallback = _coerce_int(fg_raw, fg_prev)
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

    if btc_fallback:
        btc_ts = float(last_good.get("btc_timestamp", 0.0))
    else:
        btc_ts = btc_ts_raw if btc_ts_raw is not None else time.time()

    if fg_fallback:
        fg_ts = float(last_good.get("fear_greed_timestamp", 0.0))
    else:
        fg_ts = fg_ts_raw if fg_ts_raw is not None else time.time()

    context = _build_context(
        btc_value,
        fg_value,
        stale=stale,
        reason=reason,
        last_good_ts=last_good_ts,
        btc_ts=btc_ts,
        fg_ts=fg_ts,
    )

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

    btc_age = _format_age(context.get("btc_age_seconds"))
    fg_age = _format_age(context.get("fear_greed_age_seconds"))
    _log_status(
        status,
        f"{status}: btc={context['btc_dominance']} (age={btc_age}) "
        f"fg={context['fear_greed']} (age={fg_age}) reason={context['reason']}{error_suffix}",
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
