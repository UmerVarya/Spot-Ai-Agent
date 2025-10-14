"""Utility helpers for resilient Groq LLM calls.

This module centralises common logic for interacting with the Groq API.  It
provides helpers that detect when a requested model has been decommissioned and
automatically fall back to the supported default.  The logic is shared across
both the high-level ``Groq`` SDK usage (``safe_chat_completion``) and the raw
HTTP clients used elsewhere in the codebase.
"""

from __future__ import annotations

from typing import Any, Mapping

import config
from log_utils import setup_logger

logger = setup_logger(__name__)

_DECOMMISSIONED_HINTS = (
    "model_decommissioned",
    "has been decommissioned",
    "is no longer supported",
    "has been deprecated",
    "model has been retired",
    "does not exist",
    "not found",
    "do not have access",
)


def _normalise_text(text: str | None) -> str:
    """Return a case-insensitive representation of ``text``."""

    if not text:
        return ""
    return str(text).lower()


def _extract_error_parts(error: Any) -> tuple[str, str | None]:
    """Extract a human readable message and error code from ``error``."""

    if isinstance(error, Mapping):
        inner = error.get("error")
        if isinstance(inner, Mapping):
            message = inner.get("message", "")
            code = inner.get("code")
            return str(message or ""), str(code) if code is not None else None
        message = error.get("message", "")
        code = error.get("code")
        return str(message or ""), str(code) if code is not None else None
    return str(error or ""), None


def describe_error(error: Any) -> str:
    """Return a compact description of ``error`` suitable for logging."""

    message, code = _extract_error_parts(error)
    if code and message:
        return f"{code}: {message}"
    if code:
        return str(code)
    return message


def is_model_decommissioned_error(error: Any) -> bool:
    """Return ``True`` if ``error`` indicates the requested model is retired."""

    message, code = _extract_error_parts(error)
    if code and code in {"model_decommissioned", "model_not_found"}:
        return True
    lowered = _normalise_text(message)
    return any(hint in lowered for hint in _DECOMMISSIONED_HINTS)


def safe_chat_completion(client, *, messages: list[dict[str, Any]], model: str | None = None, **kwargs: Any):
    """Invoke ``client.chat.completions.create`` with automatic model fallback."""

    requested_model = (model or config.get_groq_model()).strip()
    if not requested_model:
        requested_model = config.DEFAULT_GROQ_MODEL

    try:
        return client.chat.completions.create(
            model=requested_model,
            messages=messages,
            **kwargs,
        )
    except Exception as err:  # pragma: no cover - SDK specific exception types
        fallback_model = config.get_overflow_model()
        if fallback_model != requested_model and is_model_decommissioned_error(err):
            logger.warning(
                "Groq model %s unavailable (%s). Retrying with fallback model %s.",
                requested_model,
                describe_error(err),
                fallback_model,
            )
            return client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                **kwargs,
            )
        raise


def extract_error_payload(response: Any) -> Any:
    """Best-effort extraction of an error payload from ``response``."""

    try:
        return response.json()
    except Exception:  # pragma: no cover - requests/aiohttp differences
        return getattr(response, "text", "")
