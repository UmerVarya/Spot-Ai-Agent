"""Shared HTTP helpers for interacting with the Groq OpenAI-compatible API."""

from __future__ import annotations

import os
from typing import Any, List, Mapping, Optional, Tuple

import requests

from log_utils import setup_logger

logger = setup_logger(__name__)

_DEFAULT_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


_HTTP_TIMEOUT = _env_float("GROQ_HTTP_TIMEOUT", 10.0, minimum=1.0, maximum=60.0)


_KEY_LOGGED = False


def reset_groq_key_state() -> None:
    """Reset logging state (used in tests)."""

    global _KEY_LOGGED
    _KEY_LOGGED = False


def get_groq_api_key() -> str | None:
    """Return the Groq API key from the environment, if present."""

    key = os.getenv("GROQ_API_KEY", "").strip()

    global _KEY_LOGGED
    if not _KEY_LOGGED:
        logger.info(
            "Groq setup: key_present=%s key_prefix=%s",
            bool(key),
            (key[:4] if key else "None"),
        )
        _KEY_LOGGED = True

    if key:
        return key

    return None


def groq_api_url() -> str:
    """Return the Groq API URL, defaulting to the official OpenAI-style endpoint."""

    return os.getenv("GROQ_API_URL", _DEFAULT_GROQ_API_URL) or _DEFAULT_GROQ_API_URL


def extract_error_payload(response: Any) -> Any:
    """Best-effort extraction of an error payload from ``response``."""

    try:
        return response.json()
    except Exception:  # pragma: no cover - requests/aiohttp differences
        return getattr(response, "text", "")


def _extract_http_content(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return ""


def is_auth_error(status_code: Optional[int], error_payload: Any) -> bool:
    """Return ``True`` if the payload describes an authentication failure."""

    if status_code == 401:
        return True
    if isinstance(error_payload, Mapping):
        payload = error_payload.get("error") if "error" in error_payload else error_payload
        if isinstance(payload, Mapping):
            code = str(payload.get("code", ""))
            if code.lower() in {"authentication_error", "invalid_api_key"}:
                return True
            message = str(payload.get("message", ""))
            lowered = message.lower()
            return "invalid api key" in lowered or "authentication" in lowered
    if isinstance(error_payload, str):
        lowered = error_payload.lower()
        return "invalid api key" in lowered or "authentication" in lowered
    return False


def http_chat_completion(
    *,
    model: str,
    messages: List[Mapping[str, str]],
    temperature: float,
    max_tokens: int,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[Optional[str], Optional[int], Any]:
    """Execute a Groq chat completion via the OpenAI-compatible HTTP API."""

    key = api_key if api_key is not None else get_groq_api_key()
    if not key:
        logger.debug("Groq API key unavailable; skipping HTTP chat completion")
        return None, None, None

    url = api_url if api_url is not None else groq_api_url()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout or _HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        return None, None, exc

    if response.status_code >= 400:
        return None, response.status_code, extract_error_payload(response)

    try:
        data = response.json()
    except ValueError:
        return None, None, None

    content = _extract_http_content(data)
    if not content:
        return None, None, data
    return content, None, data


__all__ = [
    "extract_error_payload",
    "get_groq_api_key",
    "groq_api_url",
    "http_chat_completion",
    "is_auth_error",
    "reset_groq_key_state",
]

