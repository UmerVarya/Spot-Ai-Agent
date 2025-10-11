"""Minimal synchronous client for the local Ollama HTTP API.

This module provides a lightweight blocking HTTP wrapper around the Ollama
REST API. It can be imported anywhere in the agent without pulling in asyncio
or the Groq SDK, and raises :class:`LLMError` so callers can gracefully fall
back to remote providers when the local service fails.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Mapping

import requests

from log_utils import setup_logger

__all__ = ["LLMError", "generate", "chat", "warm_up"]


class LLMError(RuntimeError):
    """Raised when the Ollama API returns an error response."""


def _env(key: str, default: str) -> str:
    value = os.getenv(key, default)
    if not value:
        return default
    return value


OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = _env("OLLAMA_MODEL", "llama3.2:3b")
_DEFAULT_NUM_THREADS = os.getenv("OLLAMA_NUM_THREADS")
_DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))
_DEFAULT_TIMEOUT = max(20.0, min(60.0, _DEFAULT_TIMEOUT))

_CALL_LOCK = threading.Lock()
_CIRCUIT_LOCK = threading.Lock()
_CIRCUIT_FAILURES = 0
_CIRCUIT_OPEN_UNTIL = 0.0

_RETRY_BACKOFF = (0.5, 1.5, 3.0)


logger = setup_logger(__name__)


def _log_call(*, path: str, latency: float, prompt_tag: str, fallback_from: str, ok: bool, err: Exception | None, model: str) -> None:
    payload = {
        "llm": "ollama",
        "model": model,
        "path": path,
        "lat_ms": int(latency * 1000),
        "prompt_tag": prompt_tag,
        "fallback_from": fallback_from,
        "ok": ok,
        "err": None if err is None else str(err),
    }
    logger.info(json.dumps(payload, sort_keys=True))


def _circuit_open(now: float | None = None) -> bool:
    global _CIRCUIT_OPEN_UNTIL, _CIRCUIT_FAILURES
    ts = now or time.time()
    with _CIRCUIT_LOCK:
        if _CIRCUIT_OPEN_UNTIL and ts < _CIRCUIT_OPEN_UNTIL:
            return True
        if _CIRCUIT_OPEN_UNTIL and ts >= _CIRCUIT_OPEN_UNTIL:
            _CIRCUIT_OPEN_UNTIL = 0.0
            _CIRCUIT_FAILURES = 0
        return False


def _record_failure() -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    with _CIRCUIT_LOCK:
        _CIRCUIT_FAILURES += 1
        if _CIRCUIT_FAILURES >= 3:
            _CIRCUIT_OPEN_UNTIL = time.time() + 60.0


def _record_success() -> None:
    global _CIRCUIT_FAILURES, _CIRCUIT_OPEN_UNTIL
    with _CIRCUIT_LOCK:
        _CIRCUIT_FAILURES = 0
        _CIRCUIT_OPEN_UNTIL = 0.0


def _post(path: str, payload: Mapping[str, Any], timeout: float | None = None) -> Mapping[str, Any]:
    url = f"{OLLAMA_URL}{path}"
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
    prompt_tag: str = "default",
    fallback_from: str = "none",
) -> str:
    """Call ``/api/generate`` and return the ``response`` field."""

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": _build_options(temperature, num_ctx, num_thread),
    }
    effective_timeout = timeout or _DEFAULT_TIMEOUT
    effective_timeout = max(15.0, min(60.0, effective_timeout))
    with _CALL_LOCK:
        if _circuit_open():
            raise LLMError("Ollama circuit breaker open; retry after cooldown")
        last_exc: Exception | None = None
        total_attempts = min(retries + 1, len(_RETRY_BACKOFF))
        for attempt in range(total_attempts):
            start = time.perf_counter()
            try:
                result = _post("/api/generate", body, timeout=effective_timeout)
                _record_success()
                latency = time.perf_counter() - start
                _log_call(
                    path="/api/generate",
                    latency=latency,
                    prompt_tag=prompt_tag,
                    fallback_from=fallback_from,
                    ok=True,
                    err=None,
                    model=model,
                )
                return str(result.get("response", ""))
            except Exception as exc:
                latency = time.perf_counter() - start
                _record_failure()
                _log_call(
                    path="/api/generate",
                    latency=latency,
                    prompt_tag=prompt_tag,
                    fallback_from=fallback_from,
                    ok=False,
                    err=exc,
                    model=model,
                )
                last_exc = exc
                if attempt >= total_attempts - 1:
                    raise
                time.sleep(_RETRY_BACKOFF[attempt])
        raise LLMError("generate failed after retries") from last_exc


def chat(
    messages: list[Mapping[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    num_ctx: int | None = 2048,
    num_thread: int | None = None,
    retries: int = 2,
    timeout: float | None = None,
    prompt_tag: str = "default",
    fallback_from: str = "none",
) -> str:
    """Call ``/api/chat`` and return the assistant message content."""

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": _build_options(temperature, num_ctx, num_thread),
    }
    effective_timeout = timeout or _DEFAULT_TIMEOUT
    effective_timeout = max(15.0, min(60.0, effective_timeout))
    with _CALL_LOCK:
        if _circuit_open():
            raise LLMError("Ollama circuit breaker open; retry after cooldown")
        last_exc: Exception | None = None
        total_attempts = min(retries + 1, len(_RETRY_BACKOFF))
        for attempt in range(total_attempts):
            start = time.perf_counter()
            try:
                result = _post("/api/chat", body, timeout=effective_timeout)
                _record_success()
                latency = time.perf_counter() - start
                _log_call(
                    path="/api/chat",
                    latency=latency,
                    prompt_tag=prompt_tag,
                    fallback_from=fallback_from,
                    ok=True,
                    err=None,
                    model=model,
                )
                message = result.get("message", {})
                return str(message.get("content", ""))
            except Exception as exc:
                latency = time.perf_counter() - start
                _record_failure()
                _log_call(
                    path="/api/chat",
                    latency=latency,
                    prompt_tag=prompt_tag,
                    fallback_from=fallback_from,
                    ok=False,
                    err=exc,
                    model=model,
                )
                last_exc = exc
                if attempt >= total_attempts - 1:
                    raise
                time.sleep(_RETRY_BACKOFF[attempt])
        raise LLMError("chat failed after retries") from last_exc


def warm_up(prompt: str = "ping", *, temperature: float = 0.0) -> bool:
    """Fire a fast request to warm the local model into memory."""

    try:
        generate(
            prompt,
            temperature=temperature,
            num_ctx=256,
            retries=0,
            timeout=60,
            prompt_tag="warmup",
        )
        return True
    except Exception:
        return False
