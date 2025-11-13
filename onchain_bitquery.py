"""Bitquery on-chain integration for the Spot AI agent."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_BITQUERY_ENDPOINT = "https://graphql.bitquery.io/"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.debug("Invalid integer for %s=%r; using default %s", name, raw, default)
        return default


BITQUERY_REFRESH_SECONDS = max(3600, _env_int("BITQUERY_REFRESH_INTERVAL", 3600))
BITQUERY_PAUSE_AFTER_FAILURES = 3
BITQUERY_PAUSE_SECONDS = 6 * 3600

_BITQUERY_CACHE: Dict[int, Tuple[Dict[str, Any], float]] = {}
_CONSECUTIVE_FAILURES = 0
_PAUSED_UNTIL = 0.0
_PAUSE_WARNING_LOGGED = False


def _now() -> float:
    return time.time()


def _api_key_available() -> bool:
    return bool(os.getenv("BITQUERY_API_KEY"))


def _bitquery_post(query: str, variables: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("BITQUERY_API_KEY")
    if not api_key:
        logger.debug("Bitquery API key is not set; skipping on-chain request.")
        return None

    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key,
    }
    payload: Dict[str, Any] = {"query": query}
    if variables is not None:
        payload["variables"] = variables

    try:
        response = requests.post(
            _BITQUERY_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - network call
        logger.warning("Bitquery request failed: %s", exc)
        return None

    if response.status_code != 200:
        snippet = response.text[:300]
        logger.warning(
            "Bitquery returned non-200 status %s: %s", response.status_code, snippet
        )
        return None

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        logger.warning("Bitquery response was not valid JSON: %s", exc)
        return None
    except ValueError as exc:
        logger.warning("Bitquery response JSON decode error: %s", exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Bitquery response payload malformed: %r", data)
        return None

    errors = data.get("errors")
    if errors:
        logger.warning("Bitquery GraphQL errors: %s", errors)
        return None

    payload_data = data.get("data")
    if not isinstance(payload_data, dict):
        logger.warning("Bitquery response missing data field: %r", data)
        return None
    return payload_data


def _is_paused(now: float) -> bool:
    global _PAUSED_UNTIL, _CONSECUTIVE_FAILURES, _PAUSE_WARNING_LOGGED

    if _PAUSED_UNTIL and now < _PAUSED_UNTIL:
        return True

    if _PAUSED_UNTIL and now >= _PAUSED_UNTIL:
        _PAUSED_UNTIL = 0.0
        _CONSECUTIVE_FAILURES = 0
        _PAUSE_WARNING_LOGGED = False
    return False


def _register_failure(now: float) -> None:
    global _CONSECUTIVE_FAILURES, _PAUSED_UNTIL, _PAUSE_WARNING_LOGGED

    _CONSECUTIVE_FAILURES += 1
    if _CONSECUTIVE_FAILURES >= BITQUERY_PAUSE_AFTER_FAILURES:
        if not _PAUSED_UNTIL or now >= _PAUSED_UNTIL:
            _PAUSED_UNTIL = now + BITQUERY_PAUSE_SECONDS
            if not _PAUSE_WARNING_LOGGED:
                logger.warning(
                    "Bitquery refresh failed %s times; pausing on-chain updates for %s hours.",
                    _CONSECUTIVE_FAILURES,
                    BITQUERY_PAUSE_SECONDS / 3600.0,
                )
                _PAUSE_WARNING_LOGGED = True


def _reset_failures() -> None:
    global _CONSECUTIVE_FAILURES, _PAUSED_UNTIL, _PAUSE_WARNING_LOGGED

    _CONSECUTIVE_FAILURES = 0
    _PAUSED_UNTIL = 0.0
    _PAUSE_WARNING_LOGGED = False


def bitquery_health_check() -> bool:
    """Run a lightweight query to confirm the API key works."""

    query = """
    query HealthCheck {
      bitcoin {
        blocks(limit: 1) {
          count
        }
      }
    }
    """

    data = _bitquery_post(query)
    if not data:
        return False

    bitcoin = data.get("bitcoin")
    if isinstance(bitcoin, list):
        bitcoin = bitcoin[0] if bitcoin else {}
    if not isinstance(bitcoin, dict):
        return False

    blocks = bitcoin.get("blocks")
    if isinstance(blocks, dict):
        entries = [blocks]
    elif isinstance(blocks, list):
        entries = blocks
    else:
        entries = []
    if not entries:
        return False
    count = entries[0].get("count") if isinstance(entries[0], dict) else None
    return isinstance(count, (int, float))


def get_btc_onchain_signal(window_hours: int = 24) -> Optional[Dict[str, Any]]:
    """Return a lightweight BTC on-chain activity snapshot.

    The function enforces strict throttling so that the Bitquery endpoint is
    never polled more than once per hour.  Consecutive failures trigger a
    six-hour back-off window in line with the agent's safety rules.  When the
    API is unavailable a ``None`` result is returned, signalling a neutral
    on-chain bias.
    """

    try:
        window = int(window_hours)
    except (TypeError, ValueError):
        window = 24
    window = max(1, min(window, 24 * 7))

    now = _now()

    if not _api_key_available():
        _BITQUERY_CACHE.pop(window, None)
        _reset_failures()
        return None

    cached_entry = _BITQUERY_CACHE.get(window)
    if cached_entry is not None:
        payload, timestamp = cached_entry
        age = now - timestamp
        if age <= BITQUERY_REFRESH_SECONDS:
            return dict(payload)
        else:
            _BITQUERY_CACHE.pop(window, None)

    if _is_paused(now):
        return None

    since = (_dt.datetime.utcnow() - _dt.timedelta(hours=window)).isoformat(timespec="seconds") + "Z"

    query = """
    query BtcTransfers($since: ISO8601DateTime!) {
      bitcoin {
        transactions(date: {since: $since}) {
          count
        }
      }
    }
    """

    variables = {"since": since}
    data = _bitquery_post(query, variables=variables)
    if not data:
        _register_failure(now)
        return None

    bitcoin = data.get("bitcoin")
    if isinstance(bitcoin, list):
        bitcoin = bitcoin[0] if bitcoin else {}
    if not isinstance(bitcoin, dict):
        logger.warning("Bitquery BTC payload malformed: %r", bitcoin)
        _register_failure(now)
        return None

    transactions = bitcoin.get("transactions")
    if isinstance(transactions, dict):
        entries = [transactions]
    elif isinstance(transactions, list):
        entries = transactions
    else:
        entries = []

    if not entries:
        logger.warning("Bitquery BTC transactions payload missing: %r", bitcoin)
        _register_failure(now)
        return None

    first = entries[0]
    count = first.get("count") if isinstance(first, dict) else None

    if count is None:
        logger.warning("Bitquery BTC transactions count missing: %r", first)
        _register_failure(now)
        return None

    try:
        transfers = float(count)
    except (TypeError, ValueError):
        logger.warning("Bitquery BTC transactions count invalid: %r", count)
        _register_failure(now)
        return None

    payload = {
        "ok": True,
        "window_hours": window,
        "total_transfers": transfers,
        "total_volume_btc": None,
        "fetched_at": now,
    }

    _BITQUERY_CACHE[window] = (payload, now)
    _reset_failures()
    return dict(payload)


__all__ = ["bitquery_health_check", "get_btc_onchain_signal"]
