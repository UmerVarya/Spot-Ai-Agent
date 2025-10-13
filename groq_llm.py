"""
Safe wrapper around the Groq LLM API.

This module extends the basic LLM helper by sanitising user inputs to
mitigate prompt injection and requesting structured JSON output from
the model.  The ``get_llm_judgment`` function now instructs the model
to produce a JSON object with four fields:

* ``decision``: "Yes" or "No" indicating whether to take the trade.
* ``confidence``: a number between 0 and 10 representing the model's
  confidence in that decision.
* ``reason``: a brief explanation supporting the decision.
* ``thesis``: a 2–3 sentence trading thesis summarising the setup.

Downstream callers should attempt to parse the JSON; if parsing fails,
the raw response is returned for backward compatibility.
"""

import os
import re
import json
import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, Mapping, List, Tuple

import requests

from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

import config
from groq_client import get_groq_client
from groq_safe import (
    safe_chat_completion,
    extract_error_payload,
    is_model_decommissioned_error,
    describe_error,
)
from log_utils import setup_logger

try:  # Optional import for logging fallback details
    from llm_client import get_base_url as _get_ollama_base_url
except Exception:  # pragma: no cover - optional dependency path
    def _get_ollama_base_url() -> str:
        return "unknown"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
_HTTP_TIMEOUT_SECONDS = 10

_DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 2.0
_DEFAULT_BATCH_THROTTLE_SECONDS = 0.0


def _to_float(value: str | None, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


_RATE_LIMIT_BACKOFF_SECONDS = max(
    0.0,
    _to_float(os.getenv("GROQ_RATE_LIMIT_BACKOFF_SECONDS"), _DEFAULT_RATE_LIMIT_BACKOFF_SECONDS),
)
_BATCH_THROTTLE_SECONDS = max(
    0.0,
    _to_float(os.getenv("GROQ_BATCH_THROTTLE_SECONDS"), _DEFAULT_BATCH_THROTTLE_SECONDS),
)

logger = setup_logger(__name__)

_SINGLE_JUDGMENT_SYSTEM_MESSAGE = (
    "You are a highly experienced crypto trader assistant. Always respond with a single JSON object containing"
    " the keys decision, confidence, reason, and thesis.\n"
    "Think through the problem step by step in a private scratchpad. Before responding, silently run this "
    "private verification checklist (DO NOT OUTPUT IT):\n"
    "- Confirm you understand the market context and question.\n"
    "- Choose a decision of \"Yes\" or \"No\" consistent with the evidence.\n"
    "- Set confidence as a numeric value between 0 and 10.\n"
    "- Provide a concise reason referencing the most important factors.\n"
    "- Craft a thesis of 2-3 sentences that aligns with the decision and reason.\n"
    "- Double-check the final reply is valid JSON with keys decision, confidence, reason, and thesis."
)

_BATCH_SYSTEM_MESSAGE = (
    "You are an experienced crypto trading advisor. Respond with a single JSON object mapping each symbol"
    " to its analysis.\n"
    "Think through each symbol step by step in a private scratchpad. Before responding, silently run this "
    "private verification checklist for every symbol (DO NOT OUTPUT IT):\n"
    "- Ensure each decision is \"Yes\" or \"No\" and supported by the prompt.\n"
    "- Set confidence as a numeric value between 0 and 10.\n"
    "- Provide a concise reason highlighting the key drivers for the trade.\n"
    "- Write a thesis of 2-3 sentences consistent with the decision and reason.\n"
    "- Confirm the final reply is valid JSON mapping each symbol to an object with keys decision, confidence, reason, and thesis."
)


if GROQ_API_KEY:
    logger.info("LLM backend active: Groq (model=%s)", config.get_groq_model())
else:
    logger.info("LLM backend active: Ollama (base_url=%s)", _get_ollama_base_url())


def _normalise_header_name(name: str) -> str:
    """Return a lowercase representation that is safe for comparisons."""

    return str(name or "").lower()


def _group_rate_limit_headers(headers: Mapping[str, str]) -> Dict[str, Dict[str, str]]:
    """Organise Groq ``X-RateLimit-*`` headers by metric bucket.

    Groq returns a set of headers such as ``X-RateLimit-Limit-Requests-1m`` and
    ``X-RateLimit-Remaining-Requests-1m``.  Grouping them makes it easier to
    report how close we are to exhausting each window without hard-coding the
    available buckets.
    """

    grouped: Dict[str, Dict[str, str]] = defaultdict(dict)
    prefix = "x-ratelimit-"
    for raw_key, value in headers.items():
        key = _normalise_header_name(raw_key)
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        metric, _, bucket = remainder.partition("-")
        if not metric or not bucket:
            continue
        grouped[bucket][metric] = str(value)
    return dict(grouped)


def _parse_numeric(value: str | None) -> float | None:
    """Best-effort conversion of header values to ``float``."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_response_like(obj: Any) -> Any:
    """Best-effort extraction of an HTTP-like response object."""

    if obj is None:
        return None
    for attr in ("response", "_response"):
        candidate = getattr(obj, attr, None)
        if candidate is not None:
            return candidate
    return obj


def _extract_rate_limit_headers(obj: Any) -> Mapping[str, str] | None:
    """Return the response headers from ``obj`` if present."""

    response = _extract_response_like(obj)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None and hasattr(response, "raw"):
        headers = getattr(response.raw, "headers", None)
    if isinstance(headers, Mapping):
        return dict(headers)
    return None


def _calculate_retry_delay(headers: Mapping[str, str] | None, *, default: float | None = None) -> float:
    """Return an appropriate delay before retrying after a rate limit."""

    if default is None:
        default = _RATE_LIMIT_BACKOFF_SECONDS

    delay = max(default or 0.0, 0.0)
    if not headers:
        return delay

    normalised = {_normalise_header_name(k): str(v) for k, v in headers.items()}

    retry_after = normalised.get("retry-after") or normalised.get("retry_after")
    candidate = _parse_numeric(retry_after)
    if candidate is not None:
        delay = max(delay, candidate)

    grouped = _group_rate_limit_headers({k: str(v) for k, v in headers.items()})
    reset_candidates: list[float] = []
    for metrics in grouped.values():
        for key in ("reset", "reset-requests", "reset-tokens"):
            value = metrics.get(key)
            seconds = _parse_numeric(value)
            if seconds is not None and seconds >= 0:
                reset_candidates.append(seconds)
    if reset_candidates:
        delay = max(delay, min(reset_candidates))

    return min(delay, 60.0)


def _extract_status_code(obj: Any) -> int | None:
    """Attempt to read an HTTP status code from ``obj``."""

    response = _extract_response_like(obj)
    if response is None:
        return None
    for attr in ("status_code", "status", "statusCode"):
        value = getattr(response, attr, None)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _log_rate_limit_health(headers: Mapping[str, str] | None, status_code: int | None) -> None:
    """Log actionable information about Groq rate limit usage.

    The Groq dashboard surfaces multiple rate limit groups.  When the remaining
    quota for any bucket is almost exhausted we emit a warning so operators can
    throttle requests or request more capacity before a hard failure occurs.
    """

    if not headers:
        return

    grouped = _group_rate_limit_headers(headers)
    if not grouped:
        return

    for bucket, metrics in grouped.items():
        limit = _parse_numeric(metrics.get("limit"))
        remaining = _parse_numeric(metrics.get("remaining"))
        used = metrics.get("used")
        reset = metrics.get("reset") or metrics.get("reset-requests")

        if remaining is None or limit is None:
            continue

        message = (
            "Groq rate limit for %s — remaining=%s of %s, used=%s, resets=%s"
            % (bucket, metrics.get("remaining"), metrics.get("limit"), used or "?", reset or "?")
        )

        # Emit warnings when the window is almost consumed or a 429 is returned.
        if status_code == 429 or remaining <= 1 or remaining <= max(1.0, limit * 0.1):
            logger.warning(
                "%s. Consider reducing request concurrency or requesting higher quota.",
                message,
            )
        else:
            logger.debug(message)


def _format_exception(err: Exception) -> str:
    """Return a concise, human readable description of ``err``."""

    if isinstance(err, APIStatusError):
        status = getattr(err, "status_code", None)
        if status is not None:
            return f"{status}: {err}"
    return str(err)


def _extract_choice_content(response: Any) -> str:
    """Return the first message content from a Groq chat completion."""

    try:
        choice = response.choices[0]
        content = getattr(choice.message, "content", "")
        return str(content or "").strip()
    except Exception:
        return ""


def _extract_http_message_content(payload: Mapping[str, Any]) -> str:
    """Return the first message content from a raw HTTP payload."""

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
    return ""


def _http_chat_completion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[str | None, Any | None]:
    """Send a chat completion request via raw HTTP."""

    if not GROQ_API_KEY:
        return None, None

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=_HTTP_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.error("Groq HTTP request failed: %s", exc)
        return None, exc

    if response.status_code >= 400:
        error_payload = extract_error_payload(response)
        logger.warning(
            "Groq HTTP error for model %s: %s", model, describe_error(error_payload)
        )
        return None, error_payload

    try:
        data = response.json()
    except Exception:
        logger.warning("Groq HTTP response missing JSON body for model %s", model)
        return None, None

    content = _extract_http_message_content(data)
    if content:
        return content, None

    logger.warning("Groq HTTP response missing content for model %s", model)
    return None, data


def _http_completion_with_fallback(
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
) -> str | None:
    """Attempt an HTTP completion with automatic model fallback."""

    primary_model = config.get_groq_model()
    if not primary_model:
        primary_model = config.DEFAULT_GROQ_MODEL

    content, error = _http_chat_completion(
        primary_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if content:
        return content

    if error is not None and is_model_decommissioned_error(error):
        fallback_model = config.DEFAULT_GROQ_MODEL
        if fallback_model and fallback_model != primary_model:
            logger.warning(
                "Groq model %s unavailable (%s). Retrying with fallback model %s (HTTP path).",
                primary_model,
                describe_error(error),
                fallback_model,
            )
            content, _ = _http_chat_completion(
                fallback_model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if content:
                return content

    return None


def _sanitize_prompt(prompt: str, max_len: int = 3000) -> str:
    """Sanitise prompt text to reduce the risk of injection attacks.

    The function removes markdown code fences, braces and other potentially
    malicious characters, strips excessive whitespace and truncates the
    prompt to a maximum length.
    """
    # Remove triple backtick code blocks
    prompt = re.sub(r"```.*?```", "", prompt, flags=re.DOTALL)
    # Remove any braces or JSON‑like delimiters
    prompt = prompt.replace("{", "").replace("}", "")
    # Collapse whitespace
    prompt = re.sub(r"\s+", " ", prompt).strip()
    # Truncate if too long
    if len(prompt) > max_len:
        prompt = prompt[-max_len:]
    return prompt


def _fallback_to_local(prompt: str, *, temperature: float, max_tokens: int, reason: str) -> str | None:
    """Try serving the request via the local Ollama model."""

    try:
        from local_llm import structured_trade_judgment
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.debug("Local LLM fallback unavailable: %s", exc)
        return None

    response = structured_trade_judgment(prompt, temperature=temperature, num_ctx=None)
    if response:
        logger.info("Served LLM request via local model (%s)", reason)
    return response


async def _async_fallback_to_local(prompt: str, *, temperature: float, max_tokens: int, reason: str) -> str | None:
    try:
        from local_llm import async_structured_judgment
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.debug("Async local LLM fallback unavailable: %s", exc)
        return None

    response = await async_structured_judgment(prompt, temperature=temperature, num_ctx=None)
    if response:
        logger.info("Served async LLM request via local model (%s)", reason)
    return response


async def _async_batch_local(
    chunk: List[Tuple[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    reason: str,
) -> Dict[str, str]:
    try:
        from local_llm import async_structured_judgment, structured_trade_judgment
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.debug("Batch local fallback unavailable: %s", exc)
        return {symbol: "LLM error: local fallback unavailable" for symbol, _ in chunk}

    results: Dict[str, str] = {}
    loop = asyncio.get_running_loop()
    for symbol, prompt in chunk:
        try:
            if async_structured_judgment:
                resp = await async_structured_judgment(prompt, temperature=temperature, num_ctx=None)
            else:  # pragma: no cover - defensive branch
                resp = await loop.run_in_executor(
                    None,
                    lambda p=prompt: structured_trade_judgment(p, temperature=temperature, num_ctx=None),
                )
        except Exception as exc:  # pragma: no cover - local runtime error
            logger.debug("Local batch fallback failed for %s: %s", symbol, exc)
            resp = None
        if resp:
            results[symbol] = resp
        else:
            results[symbol] = "LLM error: local fallback unavailable"
    if results:
        logger.info("Served %d prompts via local batch fallback (%s)", len(results), reason)
    return results


def get_llm_judgment(prompt: str, temperature: float = 0.4, max_tokens: int = 500) -> str:
    """Query Groq LLM with a prompt asking for trade advice in JSON format."""

    safe_prompt = _sanitize_prompt(prompt)
    user_prompt = (
        safe_prompt
        + "\n\nPlease respond in JSON format with the following keys:"
        + " decision (Yes or No), confidence (0 to 10 as a number), reason (a short explanation),"
        + " and thesis (2-3 sentence summary)."
        + " Ensure the JSON is valid and contains no additional commentary."
    )

    if not GROQ_API_KEY:
        fallback = _fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="missing Groq API key",
        )
        if fallback:
            return fallback
        return "LLM error: Groq API key missing and local fallback unavailable."

    client = get_groq_client()
    if client is None:
        fallback = _fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="Groq client unavailable",
        )
        if fallback:
            return fallback
        return "LLM error: Groq client unavailable."

    messages = [
        {"role": "system", "content": _SINGLE_JUDGMENT_SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]

    start = time.perf_counter()
    try:
        response = safe_chat_completion(
            client,
            model=config.get_groq_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - start
        _log_rate_limit_health(
            _extract_rate_limit_headers(response),
            _extract_status_code(response),
        )
        logger.info(
            "LLM call succeeded in %.2fs (model=%s)",
            latency,
            getattr(response, "model", config.get_groq_model()),
        )
        return _extract_choice_content(response)
    except RateLimitError as err:
        latency = time.perf_counter() - start
        logger.warning("Groq rate limit after %.2fs: %s", latency, _format_exception(err))
        headers = _extract_rate_limit_headers(err)
        status_code = _extract_status_code(err)
        _log_rate_limit_health(
            headers,
            status_code,
        )
        delay = _calculate_retry_delay(headers)
        if delay > 0:
            logger.info(
                "Sleeping for %.2fs before retrying Groq fallback after rate limit (status=%s)",
                delay,
                status_code or "unknown",
            )
            time.sleep(delay)
        http_response = _http_completion_with_fallback(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if http_response:
            return http_response
        fallback = _fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="Groq rate limit",
        )
        if fallback:
            return fallback
        return "LLM error: Unable to generate response."
    except (APIStatusError, APIConnectionError, APITimeoutError, APIError) as err:
        latency = time.perf_counter() - start
        logger.error("Groq request failed in %.2fs: %s", latency, _format_exception(err))
        _log_rate_limit_health(
            _extract_rate_limit_headers(err),
            _extract_status_code(err),
        )
        http_response = _http_completion_with_fallback(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if http_response:
            return http_response
        fallback = _fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="Groq error",
        )
        if fallback:
            return fallback
        return "LLM error: Unable to generate response."
    except Exception as err:
        latency = time.perf_counter() - start
        logger.error("LLM Exception after %.2fs: %s", latency, err, exc_info=True)
        _log_rate_limit_health(
            _extract_rate_limit_headers(err),
            _extract_status_code(err),
        )
        http_response = _http_completion_with_fallback(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if http_response:
            return http_response
        fallback = _fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="Groq exception",
        )
        if fallback:
            return fallback
        return "LLM error: Exception occurred."


async def async_get_llm_judgment(prompt: str, temperature: float = 0.4, max_tokens: int = 500) -> str:
    """Asynchronous version of ``get_llm_judgment`` using aiohttp."""

    safe_prompt = _sanitize_prompt(prompt)
    user_prompt = (
        safe_prompt
        + "\n\nPlease respond in JSON format with the following keys:"
        + " decision (Yes or No), confidence (0 to 10 as a number), reason (a short explanation),"
        + " and thesis (2-3 sentence summary)."
        + " Ensure the JSON is valid and contains no additional commentary."
    )

    if not GROQ_API_KEY:
        fallback = await _async_fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="missing Groq API key",
        )
        if fallback:
            return fallback
        return "LLM error: Groq API key missing and local fallback unavailable."

    client = get_groq_client()
    if client is None:
        fallback = await _async_fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="Groq client unavailable",
        )
        if fallback:
            return fallback
        return "LLM error: Groq client unavailable."

    messages = [
        {"role": "system", "content": _SINGLE_JUDGMENT_SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]

    start = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            safe_chat_completion,
            client,
            model=config.get_groq_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = time.perf_counter() - start
        _log_rate_limit_health(
            _extract_rate_limit_headers(response),
            _extract_status_code(response),
        )
        logger.info(
            "Async LLM call succeeded in %.2fs (model=%s)",
            latency,
            getattr(response, "model", config.get_groq_model()),
        )
        return _extract_choice_content(response)
    except RateLimitError as err:
        latency = time.perf_counter() - start
        logger.warning("Async Groq rate limit after %.2fs: %s", latency, _format_exception(err))
        headers = _extract_rate_limit_headers(err)
        status_code = _extract_status_code(err)
        _log_rate_limit_health(
            headers,
            status_code,
        )
        delay = _calculate_retry_delay(headers)
        if delay > 0:
            logger.info(
                "Awaiting %.2fs before using async fallback after Groq rate limit (status=%s)",
                delay,
                status_code or "unknown",
            )
            await asyncio.sleep(delay)
        fallback = await _async_fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="Groq rate limit",
        )
        if fallback:
            return fallback
        return "LLM error: Unable to generate response."
    except (APIStatusError, APIConnectionError, APITimeoutError, APIError) as err:
        latency = time.perf_counter() - start
        logger.error("Async Groq request failed in %.2fs: %s", latency, _format_exception(err))
        _log_rate_limit_health(
            _extract_rate_limit_headers(err),
            _extract_status_code(err),
        )
        fallback = await _async_fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="Groq error",
        )
        if fallback:
            return fallback
        return "LLM error: Unable to generate response."
    except Exception as err:
        latency = time.perf_counter() - start
        logger.error("Async LLM Exception after %.2fs: %s", latency, err, exc_info=True)
        _log_rate_limit_health(
            _extract_rate_limit_headers(err),
            _extract_status_code(err),
        )
        fallback = await _async_fallback_to_local(
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="Groq exception",
        )
        if fallback:
            return fallback
        return "LLM error: Exception occurred."


async def async_batch_llm_judgment(
    prompts: Mapping[str, str],
    *,
    batch_size: int = 4,
    temperature: float = 0.4,
    max_tokens: int = 500,
) -> Dict[str, str]:
    """Send multiple prompts to Groq in batched asynchronous requests."""

    if not prompts:
        return {}
    items = [(symbol, _sanitize_prompt(prompt)) for symbol, prompt in prompts.items()]
    if not GROQ_API_KEY:
        logger.warning("Groq API key missing. Using local LLM for %d prompts", len(items))
        return await _async_batch_local(
            items,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="missing Groq API key",
        )

    client = get_groq_client()
    if client is None:
        logger.warning("Groq client unavailable. Falling back to local batch handler")
        return await _async_batch_local(
            items,
            temperature=temperature,
            max_tokens=max_tokens,
            reason="Groq client unavailable",
        )

    results: Dict[str, str] = {}
    model = config.get_groq_model()
    for batch_index, start_idx in enumerate(range(0, len(items), batch_size)):
        if batch_index and _BATCH_THROTTLE_SECONDS > 0:
            logger.debug(
                "Throttling %.2fs between Groq batch requests to avoid saturation",
                _BATCH_THROTTLE_SECONDS,
            )
            await asyncio.sleep(_BATCH_THROTTLE_SECONDS)

        chunk = items[start_idx : start_idx + batch_size]
        batch_prompt = _build_batch_prompt(chunk)
        messages = [
            {"role": "system", "content": _BATCH_SYSTEM_MESSAGE},
            {"role": "user", "content": batch_prompt},
        ]
        try:
            start = time.perf_counter()
            response = await asyncio.to_thread(
                safe_chat_completion,
                client,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens * max(1, len(chunk)),
            )
            latency = time.perf_counter() - start
            _log_rate_limit_health(
                _extract_rate_limit_headers(response),
                _extract_status_code(response),
            )
            logger.info(
                "Async batch LLM call succeeded in %.2fs (model=%s)",
                latency,
                getattr(response, "model", model),
            )
            content = _extract_choice_content(response)
        except RateLimitError as err:
            latency = time.perf_counter() - start
            logger.warning(
                "Groq batch rate limit after %.2fs: %s", latency, _format_exception(err)
            )
            headers = _extract_rate_limit_headers(err)
            status_code = _extract_status_code(err)
            _log_rate_limit_health(
                headers,
                status_code,
            )
            delay = _calculate_retry_delay(headers)
            if delay > 0:
                logger.info(
                    "Awaiting %.2fs before falling back after Groq batch rate limit (status=%s)",
                    delay,
                    status_code or "unknown",
                )
                await asyncio.sleep(delay)
            fallback = await _async_batch_local(
                chunk,
                temperature=temperature,
                max_tokens=max_tokens,
                reason="Groq rate limit",
            )
            results.update(fallback)
            continue
        except (APIStatusError, APIConnectionError, APITimeoutError, APIError) as err:
            latency = time.perf_counter() - start
            logger.error(
                "Groq batch request failed in %.2fs: %s",
                latency,
                _format_exception(err),
            )
            _log_rate_limit_health(
                _extract_rate_limit_headers(err),
                _extract_status_code(err),
            )
            fallback = await _async_batch_local(
                chunk,
                temperature=temperature,
                max_tokens=max_tokens,
                reason="Groq error",
            )
            results.update(fallback)
            continue
        except Exception as err:
            latency = time.perf_counter() - start
            logger.error("Groq batch exception after %.2fs: %s", latency, err, exc_info=True)
            _log_rate_limit_health(
                _extract_rate_limit_headers(err),
                _extract_status_code(err),
            )
            fallback = await _async_batch_local(
                chunk,
                temperature=temperature,
                max_tokens=max_tokens,
                reason="Groq exception",
            )
            results.update(fallback)
            continue

        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {}
        for symbol, _ in chunk:
            entry = parsed.get(symbol)
            if isinstance(entry, str):
                results[symbol] = entry
            elif isinstance(entry, Mapping):
                try:
                    results[symbol] = json.dumps(entry)
                except Exception:
                    results[symbol] = content
            else:
                results[symbol] = content
    return results


def _build_batch_prompt(chunk: List[Tuple[str, str]]) -> str:
    sections = []
    for symbol, prompt in chunk:
        sections.append(f"### {symbol}\n{prompt}")
    return "\n\n".join(sections)
