"""Minimal synchronous client for the local Ollama HTTP API.

This module provides a lightweight blocking HTTP wrapper around the Ollama
REST API. It can be imported anywhere in the agent without pulling in asyncio
or the Groq SDK, and raises :class:`LLMError` so callers can gracefully fall
back to remote providers when the local service fails.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

import requests

__all__ = [
    "LLMError",
    "generate",
    "chat",
    "warm_up",
    "set_base_url",
    "get_base_url",
]


class LLMError(RuntimeError):
    """Raised when the Ollama API returns an error response."""


def _env(key: str, default: str) -> str:
    value = os.getenv(key, default)
    if not value:
        return default
    return value


_DEFAULT_BASE_URL = _env("OLLAMA_URL", "http://localhost:11434").rstrip("/")
_CURRENT_BASE_URL = _DEFAULT_BASE_URL


def get_base_url() -> str:
    """Return the base URL used for Ollama requests."""

    return _CURRENT_BASE_URL


def set_base_url(url: str | None) -> None:
    """Override the Ollama base URL for subsequent requests."""

    global _CURRENT_BASE_URL, OLLAMA_URL
    if url and url.strip():
        _CURRENT_BASE_URL = url.rstrip("/")
    else:
        _CURRENT_BASE_URL = _DEFAULT_BASE_URL
    OLLAMA_URL = _CURRENT_BASE_URL


def _resolve_default_model() -> str:
    explicit = os.getenv("OLLAMA_MODEL")
    if explicit and explicit.strip():
        return explicit.strip()
    model_id = os.getenv("MODEL_ID")
    if model_id and model_id.strip():
        return model_id.strip()
    return "llama3.2:3b"


OLLAMA_URL = get_base_url()
DEFAULT_MODEL = _resolve_default_model()
_DEFAULT_NUM_THREADS = os.getenv("OLLAMA_NUM_THREADS")
_DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))


def _post(
    path: str,
    payload: Mapping[str, Any],
    timeout: float | None = None,
    *,
    base_url: str | None = None,
) -> Mapping[str, Any]:
    url = f"{(base_url or get_base_url()).rstrip('/')}{path}"
    response = requests.post(url, json=payload, timeout=timeout or _DEFAULT_TIMEOUT)
    if response.status_code >= 400:
        raise LLMError(f"{response.status_code} {response.text}")
    try:
        return response.json()
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise LLMError(f"Invalid JSON response: {exc}") from exc


def _build_options(temperature: float, num_ctx: int | None, num_thread: int | None) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": temperature}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    thread_value = num_thread
    if thread_value is None and _DEFAULT_NUM_THREADS:
        try:
            thread_value = int(_DEFAULT_NUM_THREADS)
        except ValueError:
            thread_value = None
    if thread_value is not None:
        options["num_thread"] = thread_value
    return options


def generate(
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    num_ctx: int | None = 2048,
    num_thread: int | None = None,
    retries: int = 2,
    timeout: float | None = None,
    base_url: str | None = None,
) -> str:
    """Call ``/api/generate`` and return the ``response`` field."""

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": _build_options(temperature, num_ctx, num_thread),
    }
    for attempt in range(retries + 1):
        try:
            result = _post("/api/generate", body, timeout=timeout, base_url=base_url)
            return str(result.get("response", ""))
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.2 * (attempt + 1))
    raise LLMError("generate failed after retries")


def chat(
    messages: list[Mapping[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    num_ctx: int | None = 2048,
    num_thread: int | None = None,
    retries: int = 2,
    timeout: float | None = None,
    base_url: str | None = None,
) -> str:
    """Call ``/api/chat`` and return the assistant message content."""

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": _build_options(temperature, num_ctx, num_thread),
    }
    for attempt in range(retries + 1):
        try:
            result = _post("/api/chat", body, timeout=timeout, base_url=base_url)
            message = result.get("message", {})
            return str(message.get("content", ""))
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.2 * (attempt + 1))
    raise LLMError("chat failed after retries")


def warm_up(
    prompt: str = "OK",
    *,
    temperature: float = 0.0,
    base_url: str | None = None,
) -> bool:
    """Fire a fast request to warm the local model into memory."""

    try:
        generate(
            prompt,
            temperature=temperature,
            num_ctx=256,
            retries=0,
            timeout=15,
            base_url=base_url,
        )
        return True
    except Exception:
        return False
