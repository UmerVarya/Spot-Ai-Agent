"""Utility helpers for resilient Groq LLM calls.

This module centralises common logic for interacting with the Groq API.  It
provides helpers that detect when a requested model has been decommissioned and
automatically fall back to the supported default.  The logic is shared across
both the high-level ``Groq`` SDK usage (``safe_chat_completion``) and the raw
HTTP clients used elsewhere in the codebase.
"""Utility helpers for resilient Groq LLM calls.

This module centralises Groq authentication checks, error handling, and safe
fallback behaviour for Groq model interactions.
"""

from __future__ import annotations

import os
from typing import Any, List, Mapping

import config
from groq_http import http_chat_completion
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


class GroqAuthError(RuntimeError):
    """Raised when Groq authentication fails and requests must be skipped."""


class _LLMMessage:
    __slots__ = ("content",)

    def __init__(self, content: str) -> None:
        self.content = content

    def get(self, key: str, default: Any = None) -> Any:
        if key == "content":
            return self.content
        return default


class _LLMChoice:
    __slots__ = ("message",)

    def __init__(self, content: str) -> None:
        self.message = _LLMMessage(content)


class _LLMResponse:
    __slots__ = ("choices", "model")

    def __init__(self, content: str, model: str) -> None:
        self.choices: List[_LLMChoice] = [_LLMChoice(content)]
        self.model = model


_groq_auth_disabled: bool = False


def get_groq_api_key() -> str | None:
    """
    Return the Groq API key from the environment, or None if not set.

    This is the single source of truth for whether Groq auth is possible.
    """

    key = os.getenv("GROQ_API_KEY", "").strip()
    return key or None


_initial_key = get_groq_api_key()
logger.info(
    "Groq_setup: key_present=%s key_prefix=%s",
    bool(_initial_key),
    (_initial_key[:4] if _initial_key else "None"),
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


def reset_auth_state() -> None:
    """Reset authentication state (primarily for tests)."""

    global _groq_auth_disabled
    _groq_auth_disabled = False


def require_groq_api_key() -> str:
    """Return the configured Groq API key or raise when unavailable."""

    global _groq_auth_disabled

    # If we've already decided this process cannot use Groq, short-circuit
    if _groq_auth_disabled:
        raise GroqAuthError("Groq authentication disabled")

    key = get_groq_api_key()
    if not key:
        _groq_auth_disabled = True
        logger.warning("Groq disabled: no GROQ_API_KEY in environment")
        raise GroqAuthError("Groq authentication disabled")

    return key


def _build_response(content: str, model: str) -> _LLMResponse:
    return _LLMResponse(content, model)


def safe_chat_completion(client, *, messages: list[dict[str, Any]], model: str | None = None, **kwargs: Any):
    """Invoke the Groq HTTP helper with automatic model fallback."""

    if client is None:
        raise RuntimeError("Groq client unavailable")

    api_key = require_groq_api_key()

    requested_model = (model or config.get_groq_model()).strip()
    if not requested_model:
        requested_model = config.DEFAULT_GROQ_MODEL

    temperature = float(kwargs.get("temperature", 0.0))
    max_tokens = int(kwargs.get("max_tokens", 256))
    timeout = kwargs.get("timeout")

    models_to_try = [requested_model]
    fallback_model = config.get_overflow_model()
    if fallback_model and fallback_model not in models_to_try:
        models_to_try.append(fallback_model)

    for model_name in models_to_try:
        content, status_code, error_payload = http_chat_completion(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            timeout=timeout,
        )

        if content:
            return _build_response(content, model_name)

        if model_name != fallback_model and is_model_decommissioned_error(error_payload):
            logger.warning(
                "Groq model %s unavailable (%s). Retrying with fallback model %s.",
                model_name,
                describe_error(error_payload),
                fallback_model,
            )
            continue

        if isinstance(error_payload, Exception):
            raise error_payload

        message = describe_error(error_payload)
        raise RuntimeError(f"Groq HTTP request failed: {message}")


def extract_error_payload(response: Any) -> Any:
    """Best-effort extraction of an error payload from ``response``."""

    try:
        return response.json()
    except Exception:  # pragma: no cover - requests/aiohttp differences
        return getattr(response, "text", "")


__all__ = [
    "GroqAuthError",
    "describe_error",
    "is_model_decommissioned_error",
    "get_groq_api_key",
    "safe_chat_completion",
    "require_groq_api_key",
    "reset_auth_state",
]
