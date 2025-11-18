"""Helpers for retrieving and caching macro data feeds."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Optional, Tuple

import requests

from log_utils import setup_logger


logger = setup_logger(__name__)


DEFAULT_FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"
DEFAULT_BTC_PRIMARY_URL = "https://api.coingecko.com/api/v3/global"
DEFAULT_BTC_SECONDARY_URL = "https://open-api.coingecko.com/api/v3/global"
DEFAULT_BTC_TERTIARY_URL = "https://api.coinlore.net/api/global/"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        logger.warning("macro_data: invalid %s=%r – using default %s", name, raw, default)
        return float(default)


def _env_url(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    candidate = raw.strip()
    if not candidate:
        logger.warning("macro_data: %s is empty – using default %s", name, default)
        return default
    return candidate


HTTP_TIMEOUT = _env_float("MACRO_DATA_HTTP_TIMEOUT", 8.0)
FNG_MAX_AGE_HOURS = _env_float("FNG_MAX_AGE_HOURS", 36.0)
FNG_MAX_AGE_SECONDS = max(0.0, FNG_MAX_AGE_HOURS * 3600.0)
BTC_DOM_MAX_AGE_SECS = max(0.0, _env_float("BTC_DOM_MAX_AGE_SECS", 1800.0))
FNG_API_URL = _env_url("FNG_API_URL", DEFAULT_FNG_URL)
BTC_DOM_PRIMARY_URL = _env_url("BTC_DOM_PRIMARY_URL", DEFAULT_BTC_PRIMARY_URL)
BTC_DOM_SECONDARY_URL = _env_url("BTC_DOM_SECONDARY_URL", DEFAULT_BTC_SECONDARY_URL)
BTC_DOM_TERTIARY_URL = _env_url("BTC_DOM_TERTIARY_URL", DEFAULT_BTC_TERTIARY_URL)


@dataclass
class FearGreedSnapshot:
    value: float
    ts: int


@dataclass
class BTCDominanceSnapshot:
    value: float
    ts: int


_FEAR_GREED_CACHE: Optional[FearGreedSnapshot] = None
_BTC_DOM_CACHE: Optional[BTCDominanceSnapshot] = None


def fetch_fear_greed_raw() -> Tuple[Optional[float], Optional[int]]:
    """Return the latest Fear & Greed Index value and timestamp."""

    try:
        response = requests.get(FNG_API_URL, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not entries:
            raise ValueError("Fear & Greed payload missing 'data'")
        first = entries[0]
        value_raw = first.get("value") if isinstance(first, dict) else None
        ts_raw = first.get("timestamp") if isinstance(first, dict) else None
        value = float(value_raw)
        timestamp = int(float(ts_raw)) if ts_raw is not None else int(time.time())
        return value, timestamp
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("macro_data: failed to refresh Fear & Greed: %s", exc)
        return None, None


def _parse_coingecko_payload(payload: dict) -> Tuple[float, int]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise ValueError("CoinGecko payload missing 'data'")
    percentages = data.get("market_cap_percentage")
    if not isinstance(percentages, dict) or "btc" not in percentages:
        raise ValueError("CoinGecko payload missing BTC dominance")
    dominance = float(percentages["btc"])
    ts_candidate = data.get("updated_at") or data.get("last_updated_at")
    if ts_candidate is None:
        timestamp = int(time.time())
    else:
        try:
            timestamp = int(float(ts_candidate))
        except (TypeError, ValueError):
            timestamp = int(time.time())
    return dominance, timestamp


def fetch_btc_dominance_raw() -> Tuple[Optional[float], Optional[int]]:
    """Return BTC dominance and timestamp using CoinGecko with fallbacks."""

    for url in (BTC_DOM_PRIMARY_URL, BTC_DOM_SECONDARY_URL):
        try:
            response = requests.get(url, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            dominance, timestamp = _parse_coingecko_payload(response.json())
            return dominance, timestamp
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "macro_data: failed to refresh BTC dominance from %s: %s", url, exc
            )

    # Final fallback: use Coinlore if available.
    try:
        response = requests.get(BTC_DOM_TERTIARY_URL, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list) and payload:
            data = payload[0]
        elif isinstance(payload, dict):
            data = payload
        else:
            data = None
        if not isinstance(data, dict):
            raise ValueError("Coinlore payload missing data")
        dominance = float(data.get("btc_d"))
        return dominance, int(time.time())
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("macro_data: failed to refresh BTC dominance: %s", exc)
        return None, None


def _refresh_fear_greed_cache(now: float) -> Optional[FearGreedSnapshot]:
    value, timestamp = fetch_fear_greed_raw()
    if value is None or timestamp is None:
        return None
    snapshot = FearGreedSnapshot(value=float(value), ts=int(timestamp))
    global _FEAR_GREED_CACHE
    _FEAR_GREED_CACHE = snapshot
    age = max(0.0, now - snapshot.ts)
    logger.info(
        "macro_data: refreshed Fear & Greed value=%.2f ts=%d age=%.0fs",
        snapshot.value,
        snapshot.ts,
        age,
    )
    return snapshot


def _refresh_btc_cache(now: float) -> Optional[BTCDominanceSnapshot]:
    value, timestamp = fetch_btc_dominance_raw()
    if value is None or timestamp is None:
        return None
    snapshot = BTCDominanceSnapshot(value=float(value), ts=int(timestamp))
    global _BTC_DOM_CACHE
    _BTC_DOM_CACHE = snapshot
    age = max(0.0, now - snapshot.ts)
    logger.info(
        "macro_data: refreshed BTC dominance value=%.2f ts=%d age=%.0fs",
        snapshot.value,
        snapshot.ts,
        age,
    )
    return snapshot


def get_fear_greed_cached(now: Optional[float] = None) -> Optional[FearGreedSnapshot]:
    """Return a cached Fear & Greed snapshot, refreshing when stale."""

    if now is None:
        now = time.time()
    snapshot = _FEAR_GREED_CACHE
    if snapshot is not None:
        age = now - snapshot.ts
        if FNG_MAX_AGE_SECONDS <= 0 or age <= FNG_MAX_AGE_SECONDS:
            return snapshot
        logger.info(
            "macro_data: Fear & Greed cache stale (age=%.0fs) – refetching", age
        )
    refreshed = _refresh_fear_greed_cache(now)
    if refreshed is None:
        logger.warning("macro_data: Fear & Greed unavailable after refresh")
    return refreshed


def get_btc_dominance_cached(
    now: Optional[float] = None,
) -> Optional[BTCDominanceSnapshot]:
    """Return cached BTC dominance data, refreshing when stale."""

    if now is None:
        now = time.time()
    snapshot = _BTC_DOM_CACHE
    if snapshot is not None:
        age = now - snapshot.ts
        if BTC_DOM_MAX_AGE_SECS <= 0 or age <= BTC_DOM_MAX_AGE_SECS:
            return snapshot
        logger.info(
            "macro_data: BTC dominance cache stale (age=%.0fs) – refetching", age
        )
    refreshed = _refresh_btc_cache(now)
    if refreshed is None:
        logger.warning("macro_data: BTC dominance unavailable after refresh")
    return refreshed


__all__ = [
    "FearGreedSnapshot",
    "BTCDominanceSnapshot",
    "fetch_fear_greed_raw",
    "fetch_btc_dominance_raw",
    "get_fear_greed_cached",
    "get_btc_dominance_cached",
]
