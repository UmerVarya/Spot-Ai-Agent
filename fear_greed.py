from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from http.client import RemoteDisconnected
from requests import exceptions as requests_exceptions

from log_utils import setup_logger

logger = setup_logger(__name__)


def _cache_path() -> Path:
    """Return the path for persisting the Fear & Greed cache."""

    configured = os.getenv("FEAR_GREED_CACHE_PATH", "fear_greed_cache.json")
    return Path(configured).expanduser()


class FearGreedIndexFetcher:
    """Fetch the Fear & Greed index with a disk-backed cache."""

    API_URL = "https://api.alternative.me/fng/"

    def __init__(self, cache_path: Optional[Path] = None) -> None:
        self.cache_path = cache_path or _cache_path()

    def fetch(self) -> int:
        """Return the latest Fear & Greed index, falling back to cache on errors."""

        try:
            value = self._fetch_remote()
        except RemoteDisconnected as exc:
            logger.warning(
                "Connection dropped fetching Fear & Greed Index, using cached value: %s",
                exc,
            )
            return self._cached_or_default()
        except requests_exceptions.SSLError as exc:
            logger.warning(
                "TLS error fetching Fear & Greed Index, using cached value: %s",
                exc,
            )
            return self._cached_or_default()
        except requests_exceptions.RequestException as exc:
            logger.warning(
                "Network error fetching Fear & Greed Index, using cached value: %s",
                exc,
            )
            return self._cached_or_default()
        except Exception as exc:  # pragma: no cover - unexpected failure path
            logger.warning(
                "Unexpected failure fetching Fear & Greed Index, using cached value: %s",
                exc,
                exc_info=True,
            )
            return self._cached_or_default()

        self._write_cache(value)
        return value

    def _fetch_remote(self) -> int:
        response = requests.get(self.API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return self._parse_value(data)

    @staticmethod
    def _parse_value(data: Any) -> int:
        try:
            value = int(data["data"][0]["value"])
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            raise ValueError("Fear & Greed payload missing 'data[0][\"value\"]'") from exc

        if not 0 <= value <= 100:
            raise ValueError(f"Fear & Greed Index value out of range: {value}")
        return value

    def _cached_or_default(self) -> int:
        cached = self._read_cache()
        if cached is not None:
            return cached
        return 50

    def _read_cache(self) -> Optional[int]:
        try:
            payload = json.loads(self.cache_path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Fear & Greed cache unreadable: %s", exc, exc_info=True)
            return None

        try:
            cached_value = int(payload["value"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Fear & Greed cache missing value: %s", exc, exc_info=True)
            return None

        if 0 <= cached_value <= 100:
            return cached_value
        logger.debug("Fear & Greed cache value out of range: %s", cached_value)
        return None

    def _write_cache(self, value: int) -> None:
        payload = {
            "value": int(value),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.cache_path.write_text(json.dumps(payload, indent=2))
        except OSError as exc:
            logger.debug("Failed to write Fear & Greed cache: %s", exc, exc_info=True)


_FETCHER = FearGreedIndexFetcher()


def get_fear_greed_index() -> int:
    """Public helper that returns the latest Fear & Greed Index value."""

    return _FETCHER.fetch()
