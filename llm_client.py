"""Minimal synchronous client for the local Ollama HTTP API.

This module provides a lightweight blocking HTTP wrapper around the Ollama
REST API. It can be imported anywhere in the agent without pulling in asyncio
or the Groq SDK, and raises :class:`LLMError` so callers can gracefully fall
back to remote providers when the local service fails.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Mapping

import requests
from log_utils import setup_logger

__all__ = [
    "LLMError",
    "CircuitOpenError",
    "generate",
    "chat",
    "warm_up",
    "set_base_url",
    "get_base_url",
]


class LLMError(RuntimeError):
    """Raised when the Ollama API returns an error response."""


class CircuitOpenError(LLMError):
    """Raised when the Ollama circuit breaker is open."""


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
_INITIAL_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_INITIAL", "60"))
_RETRY_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_RETRY", "25"))
_BACKOFF_SCHEDULE = [0.5, 1.5, 3.0]
_MAX_FAILURES = 3
_CIRCUIT_OPEN_SECONDS = 60.0

_CALL_LOCK = threading.Lock()
_CIRCUIT_LOCK = threading.Lock()
_FAILURE_COUNT = 0
_CIRCUIT_OPEN_UNTIL = 0.0

_logger = setup_logger(__name__)


def _timeout_for_attempt(attempt: int, override: float | None) -> float:
    if override is not None:
        return override
    return _INITIAL_TIMEOUT if attempt == 0 else _RETRY_TIMEOUT


def _ensure_circuit_allows() -> None:
    global _CIRCUIT_OPEN_UNTIL
    now = time.time()
    with _CIRCUIT_LOCK:
        if _CIRCUIT_OPEN_UNTIL and now < _CIRCUIT_OPEN_UNTIL:
            remaining = int(_CIRCUIT_OPEN_UNTIL - now)
            raise CircuitOpenError(f"Ollama circuit breaker open for {remaining}s")
        if _CIRCUIT_OPEN_UNTIL and now >= _CIRCUIT_OPEN_UNTIL:
            _CIRCUIT_OPEN_UNTIL = 0.0


def _record_success() -> None:
    global _FAILURE_COUNT
    with _CIRCUIT_LOCK:
        _FAILURE_COUNT = 0


def _record_failure() -> bool:
    global _FAILURE_COUNT, _CIRCUIT_OPEN_UNTIL
    with _CIRCUIT_LOCK:
        _FAILURE_COUNT += 1
        if _FAILURE_COUNT >= _MAX_FAILURES:
            _CIRCUIT_OPEN_UNTIL = time.time() + _CIRCUIT_OPEN_SECONDS
            _FAILURE_COUNT = 0
            return True
    return False


def _log_call(path: str, model: str, status: str, latency: float, attempt: int, error: Exception | None = None) -> None:
    message = (
        f"ollama_call path={path} model={model} status={status} latency={latency:.2f}s attempt={attempt}"
    )
    if error is not None:
        message = f"{message} error={type(error).__name__}: {error}"[:500]
    _logger.info(message)


def _post(
    path: str,
    payload: Mapping[str, Any],
    timeout: float | None = None,
    *,
    base_url: str | None = None,
) -> Mapping[str, Any]:
    url = f"{(base_url or get_base_url()).rstrip('/')}{path}"
    with _CALL_LOCK:
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
    attempts = retries + 1
    for attempt in range(attempts):
        _ensure_circuit_allows()
        timeout_value = _timeout_for_attempt(attempt, timeout)
        start = time.perf_counter()
        try:
            result = _post("/api/generate", body, timeout=timeout_value, base_url=base_url)
            latency = time.perf_counter() - start
            _record_success()
            _log_call("/api/generate", model, "ok", latency, attempt + 1)
            return str(result.get("response", ""))
        except Exception as exc:
            latency = time.perf_counter() - start
            _log_call("/api/generate", model, "error", latency, attempt + 1, exc)
            if isinstance(exc, CircuitOpenError):
                raise
            circuit_tripped = _record_failure()
            if circuit_tripped:
                raise LLMError("Ollama circuit breaker tripped after repeated failures") from exc
            if attempt == attempts - 1:
                raise
            backoff = _BACKOFF_SCHEDULE[min(attempt, len(_BACKOFF_SCHEDULE) - 1)]
            time.sleep(backoff)
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
    attempts = retries + 1
    for attempt in range(attempts):
        _ensure_circuit_allows()
        timeout_value = _timeout_for_attempt(attempt, timeout)
        start = time.perf_counter()
        try:
            result = _post("/api/chat", body, timeout=timeout_value, base_url=base_url)
            latency = time.perf_counter() - start
            _record_success()
            _log_call("/api/chat", model, "ok", latency, attempt + 1)
            message = result.get("message", {})
            return str(message.get("content", ""))
        except Exception as exc:
            latency = time.perf_counter() - start
            _log_call("/api/chat", model, "error", latency, attempt + 1, exc)
            if isinstance(exc, CircuitOpenError):
                raise
            circuit_tripped = _record_failure()
            if circuit_tripped:
                raise LLMError("Ollama circuit breaker tripped after repeated failures") from exc
            if attempt == attempts - 1:
                raise
            backoff = _BACKOFF_SCHEDULE[min(attempt, len(_BACKOFF_SCHEDULE) - 1)]
            time.sleep(backoff)
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
