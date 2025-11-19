"""Public futures alternative data helper for Spot AI Agent.

This module fetches funding rates, perp premiums, open interest and taker
long/short ratios from Binance Futures public endpoints.  Data is cached with
per-metric freshness windows so downstream scoring can reuse the signals
without repeatedly hitting the APIs.  No authentication is required.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests

from log_utils import setup_logger

logger = setup_logger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        logger.warning("alt_data: invalid %s=%r – using default %.2f", name, raw, default)
        return float(default)


HTTP_TIMEOUT = _env_float("ALT_DATA_HTTP_TIMEOUT", 8.0)
FUTURES_BASE_URL = os.getenv("BINANCE_FUTURES_BASE", "https://fapi.binance.com").rstrip("/")

ALT_FUNDING_MAX_AGE_SECS = _env_float("ALT_FUNDING_MAX_AGE_SECS", 900.0)
ALT_BASIS_MAX_AGE_SECS = _env_float("ALT_BASIS_MAX_AGE_SECS", 300.0)
ALT_OI_MAX_AGE_SECS = _env_float("ALT_OI_MAX_AGE_SECS", 1800.0)
ALT_TAKER_RATIO_MAX_AGE_SECS = _env_float("ALT_TAKER_RATIO_MAX_AGE_SECS", 300.0)

_CACHE_HARD_LIMIT_FACTOR = _env_float("ALT_CACHE_HARD_LIMIT_FACTOR", 3.0)


@dataclass
class FundingSnapshot:
    value: float
    ts: int


@dataclass
class BasisSnapshot:
    value: float
    ts: int


@dataclass
class OpenInterestSnapshot:
    value: float
    change_24h_pct: float
    ts: int


@dataclass
class TakerRatioSnapshot:
    long_short_ratio: float
    ts: int


class _CacheEntry:
    __slots__ = ("snapshot", "fetched_at")

    def __init__(self, snapshot, fetched_at: float) -> None:
        self.snapshot = snapshot
        self.fetched_at = fetched_at


_FUNDING_CACHE: Dict[str, _CacheEntry] = {}
_BASIS_CACHE: Dict[str, _CacheEntry] = {}
_OI_CACHE: Dict[str, _CacheEntry] = {}
_TAKER_CACHE: Dict[str, _CacheEntry] = {}


def _symbol_key(symbol: str) -> str:
    return str(symbol).upper().strip()


def _now(now: Optional[float]) -> float:
    return float(time.time() if now is None else now)


def _cache_fresh(entry: Optional[_CacheEntry], now: float, max_age: float) -> bool:
    if entry is None:
        return False
    age = max(0.0, now - entry.fetched_at)
    return max_age <= 0 or age <= max_age


def _cache_within_hard_limit(entry: Optional[_CacheEntry], now: float, max_age: float) -> bool:
    if entry is None:
        return False
    hard_age = max_age * max(1.0, _CACHE_HARD_LIMIT_FACTOR)
    age = max(0.0, now - entry.fetched_at)
    return hard_age <= 0 or age <= hard_age


def fetch_funding_raw(symbol: str) -> Tuple[Optional[float], Optional[int]]:
    symbol_key = _symbol_key(symbol)
    url = f"{FUTURES_BASE_URL}/fapi/v1/fundingRate"
    params = {"symbol": symbol_key, "limit": 1}
    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise ValueError("funding payload missing list")
        latest = payload[0]
        value_raw = latest.get("fundingRate") if isinstance(latest, dict) else None
        ts_raw = latest.get("fundingTime") if isinstance(latest, dict) else None
        value = float(value_raw)
        timestamp = int(float(ts_raw)) // 1000 if ts_raw is not None else int(time.time())
        return value, timestamp
    except Exception as exc:  # pylint: disable=broad-except
        logger.info("alt_data: failed to fetch funding for %s: %s", symbol_key, exc)
        return None, None


def fetch_basis_raw(symbol: str) -> Tuple[Optional[float], Optional[int]]:
    symbol_key = _symbol_key(symbol)
    url = f"{FUTURES_BASE_URL}/fapi/v1/premiumIndex"
    params = {"symbol": symbol_key}
    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("premium payload missing dict")
        mark = float(payload.get("markPrice"))
        index_price = float(payload.get("indexPrice"))
        if index_price == 0:
            premium = 0.0
        else:
            premium = (mark - index_price) / index_price
        ts_raw = payload.get("time")
        timestamp = int(float(ts_raw)) // 1000 if ts_raw is not None else int(time.time())
        return premium, timestamp
    except Exception as exc:  # pylint: disable=broad-except
        logger.info("alt_data: failed to fetch basis for %s: %s", symbol_key, exc)
        return None, None


def fetch_open_interest_raw(
    symbol: str,
) -> Tuple[Optional[OpenInterestSnapshot], Optional[int]]:
    symbol_key = _symbol_key(symbol)
    url = f"{FUTURES_BASE_URL}/futures/data/openInterestHist"
    params = {"symbol": symbol_key, "period": "1h", "limit": 25}
    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2:
            raise ValueError("open interest payload missing entries")
        latest = payload[-1]
        prior = payload[0]
        latest_value = float(latest.get("sumOpenInterestValue"))
        prev_value = float(prior.get("sumOpenInterestValue"))
        if prev_value == 0:
            change_pct = 0.0
        else:
            change_pct = ((latest_value - prev_value) / prev_value) * 100.0
        ts_raw = latest.get("timestamp")
        timestamp = int(float(ts_raw)) // 1000 if ts_raw is not None else int(time.time())
        snapshot = OpenInterestSnapshot(
            value=latest_value,
            change_24h_pct=change_pct,
            ts=timestamp,
        )
        return snapshot, timestamp
    except Exception as exc:  # pylint: disable=broad-except
        logger.info("alt_data: failed to fetch open interest for %s: %s", symbol_key, exc)
        return None, None


def fetch_taker_ratio_raw(symbol: str) -> Tuple[Optional[float], Optional[int]]:
    symbol_key = _symbol_key(symbol)
    url = f"{FUTURES_BASE_URL}/futures/data/takerlongshortRatio"
    params = {"symbol": symbol_key, "period": "5m", "limit": 1}
    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            logger.info("alt_data: taker ratio unavailable for %s (payload=%r)", symbol_key, payload)
            return None, None
        latest = payload[-1]
        if not isinstance(latest, dict):
            logger.info("alt_data: taker ratio unavailable for %s (payload=%r)", symbol_key, latest)
            return None, None

        ratio_raw = latest.get("longShortRatio")
        if ratio_raw in (None, ""):
            logger.info("alt_data: taker ratio unavailable for %s (payload=%r)", symbol_key, latest)
            return None, None
        try:
            ratio = float(ratio_raw)
        except (TypeError, ValueError):
            logger.info("alt_data: taker ratio unavailable for %s (payload=%r)", symbol_key, latest)
            return None, None

        ts_raw = latest.get("timestamp")
        timestamp = int(float(ts_raw)) // 1000 if ts_raw is not None else int(time.time())
        return ratio, timestamp
    except Exception as exc:  # pylint: disable=broad-except
        logger.info("alt_data: failed to fetch taker ratio for %s: %s", symbol_key, exc)
        return None, None


def _refresh_cache(
    cache: Dict[str, _CacheEntry],
    symbol: str,
    snapshot,
    now: float,
    metric: str,
) -> None:
    cache[symbol] = _CacheEntry(snapshot, now)
    if snapshot is None:
        return
    ts = getattr(snapshot, "ts", None)
    value = getattr(snapshot, "value", None)
    if metric == "taker_ratio":
        value = getattr(snapshot, "long_short_ratio", value)
    logger.info(
        "alt_data: refreshed %s symbol=%s value=%s ts=%s",
        metric,
        symbol,
        value,
        ts,
    )


def get_funding_cached(symbol: str, now: Optional[float] = None) -> Optional[FundingSnapshot]:
    symbol_key = _symbol_key(symbol)
    now_value = _now(now)
    entry = _FUNDING_CACHE.get(symbol_key)
    if not _cache_fresh(entry, now_value, ALT_FUNDING_MAX_AGE_SECS):
        if entry is not None:
            age = max(0.0, now_value - entry.fetched_at)
            logger.info(
                "alt_data: funding cache stale symbol=%s age=%.0fs – refetching",
                symbol_key,
                age,
            )
        value, ts = fetch_funding_raw(symbol_key)
        if value is not None and ts is not None:
            snapshot = FundingSnapshot(value=float(value), ts=int(ts))
            _refresh_cache(_FUNDING_CACHE, symbol_key, snapshot, now_value, "funding")
            return snapshot
        if entry and _cache_within_hard_limit(entry, now_value, ALT_FUNDING_MAX_AGE_SECS):
            return entry.snapshot
        return None
    return entry.snapshot if entry else None


def get_basis_cached(symbol: str, now: Optional[float] = None) -> Optional[BasisSnapshot]:
    symbol_key = _symbol_key(symbol)
    now_value = _now(now)
    entry = _BASIS_CACHE.get(symbol_key)
    if not _cache_fresh(entry, now_value, ALT_BASIS_MAX_AGE_SECS):
        if entry is not None:
            age = max(0.0, now_value - entry.fetched_at)
            logger.info(
                "alt_data: basis cache stale symbol=%s age=%.0fs – refetching",
                symbol_key,
                age,
            )
        value, ts = fetch_basis_raw(symbol_key)
        if value is not None and ts is not None:
            snapshot = BasisSnapshot(value=float(value), ts=int(ts))
            _refresh_cache(_BASIS_CACHE, symbol_key, snapshot, now_value, "basis")
            return snapshot
        if entry and _cache_within_hard_limit(entry, now_value, ALT_BASIS_MAX_AGE_SECS):
            return entry.snapshot
        return None
    return entry.snapshot if entry else None


def get_open_interest_cached(
    symbol: str,
    now: Optional[float] = None,
) -> Optional[OpenInterestSnapshot]:
    symbol_key = _symbol_key(symbol)
    now_value = _now(now)
    entry = _OI_CACHE.get(symbol_key)
    if not _cache_fresh(entry, now_value, ALT_OI_MAX_AGE_SECS):
        if entry is not None:
            age = max(0.0, now_value - entry.fetched_at)
            logger.info(
                "alt_data: open_interest cache stale symbol=%s age=%.0fs – refetching",
                symbol_key,
                age,
            )
        snapshot, _ = fetch_open_interest_raw(symbol_key)
        if snapshot is not None:
            _refresh_cache(_OI_CACHE, symbol_key, snapshot, now_value, "open_interest")
            return snapshot
        if entry and _cache_within_hard_limit(entry, now_value, ALT_OI_MAX_AGE_SECS):
            return entry.snapshot
        return None
    return entry.snapshot if entry else None


def get_taker_ratio_cached(
    symbol: str,
    now: Optional[float] = None,
) -> Optional[TakerRatioSnapshot]:
    symbol_key = _symbol_key(symbol)
    now_value = _now(now)
    entry = _TAKER_CACHE.get(symbol_key)
    if not _cache_fresh(entry, now_value, ALT_TAKER_RATIO_MAX_AGE_SECS):
        if entry is not None:
            age = max(0.0, now_value - entry.fetched_at)
            logger.info(
                "alt_data: taker_ratio cache stale symbol=%s age=%.0fs – refetching",
                symbol_key,
                age,
            )
        value, ts = fetch_taker_ratio_raw(symbol_key)
        if value is not None and ts is not None:
            snapshot = TakerRatioSnapshot(long_short_ratio=float(value), ts=int(ts))
            _refresh_cache(_TAKER_CACHE, symbol_key, snapshot, now_value, "taker_ratio")
            return snapshot
        if entry and _cache_within_hard_limit(entry, now_value, ALT_TAKER_RATIO_MAX_AGE_SECS):
            return entry.snapshot
        return None
    return entry.snapshot if entry else None
