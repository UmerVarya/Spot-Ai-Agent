"""Utilities for retrieving Bitcoin dominance with graceful fallbacks."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger(__name__)

# Cache file that survives process restarts so repeated failures still have
# something to fall back on.
CACHE = os.path.expanduser("/home/ubuntu/spot_data/cache/btc_dominance.json")

_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    """Return a requests session configured with retries and UA headers."""

    global _SESSION
    if _SESSION is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "SpotAI/1.0 (+https://localhost)"})
        retry = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        sess.mount("https://", HTTPAdapter(max_retries=retry))
        _SESSION = sess
    return _SESSION


def _load_cached(fresh_only: bool = True) -> Optional[float]:
    """Return the cached dominance value, optionally requiring freshness."""

    try:
        with open(CACHE, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        age = time.time() - float(payload.get("ts", 0))
        if not fresh_only or age < 1800:  # ~30 minutes
            dominance = payload.get("dominance")
            if isinstance(dominance, (int, float)):
                return float(dominance)
    except Exception:
        # Cache corruption should not break callers.
        pass
    return None


def _save_cached(value: float) -> None:
    """Persist the dominance value for future fallbacks."""

    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w", encoding="utf-8") as fh:
            json.dump({"dominance": float(value), "ts": time.time()}, fh)
    except Exception:
        # Persistence failures are non-fatal and deliberately ignored.
        pass


def get_btc_dominance(timeout: float = 10.0) -> Optional[float]:
    """Fetch BTC dominance from CoinGecko with cached/neutral fallbacks."""

    try:
        response = _session().get(
            "https://api.coingecko.com/api/v3/global", timeout=timeout
        )
        response.raise_for_status()
        payload = response.json()

        dominance = None
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                market_caps = data.get("market_cap_percentage")
                if isinstance(market_caps, dict):
                    dominance = market_caps.get("btc")

        if dominance is None:
            raise ValueError("BTC dominance missing from response payload")

        dominance = float(dominance)
        _save_cached(dominance)
        return dominance
    except Exception as exc:  # pylint: disable=broad-except
        LOG.warning("BTC dominance fetch failed: %s", exc)

        cached = _load_cached(fresh_only=False)
        if cached is not None:
            LOG.info("Using cached BTC dominance value: %s", cached)
            return cached

        # Final fallback: neutral/None so downstream checks can treat it as no veto.
        return None
