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
    api_key: str,
    api_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[Optional[str], Optional[int], Any]:
    """Execute a Groq chat completion via the OpenAI-compatible HTTP API."""

    from groq_safe import GroqAuthError

    key = api_key
    if not key:
        raise GroqAuthError("Groq API key missing")

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
        error_payload = extract_error_payload(response)
        if is_auth_error(response.status_code, error_payload):
            # Explicitly disable Groq usage on authentication failures.
            import groq_safe

            groq_safe._groq_auth_disabled = True
            logger.error(
                "Groq authentication failed (401). Disabling Groq requests: invalid_api_key"
            )
            raise GroqAuthError("Groq authentication disabled")

        return None, response.status_code, error_payload

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
    "groq_api_url",
    "http_chat_completion",
    "is_auth_error",
]

