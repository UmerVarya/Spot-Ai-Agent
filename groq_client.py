"""Shared helpers for constructing Groq SDK clients.

This module centralises creation of the :class:`groq.Groq` client so that
all call sites reuse a single cached instance.  Consolidating the logic in
one place ensures that environment handling, logging and dependency
initialisation remain consistent across the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from groq import Groq

from groq_safe import get_groq_api_key
from log_utils import setup_logger

logger = setup_logger(__name__)


@lru_cache(maxsize=1)
def _build_client(api_key: Optional[str]) -> Optional[Groq]:
    """Return a cached Groq client or ``None`` when no API key is provided."""

    if not api_key:
        logger.debug("Groq API key not provided; client disabled")
        return None
    logger.debug("Initialising shared Groq client")
    return Groq(api_key=api_key)


def get_groq_client() -> Optional[Groq]:
    """Return the shared Groq SDK client if the API key is configured."""

    api_key = get_groq_api_key()
    return _build_client(api_key)


def reset_groq_client_cache() -> None:
    """Clear the cached client (primarily for use in tests)."""

    _build_client.cache_clear()

