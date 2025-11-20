"""Task-aware Groq LLM routing with soft limits and model chains."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Iterable, Optional, Tuple

from groq_client import get_groq_client
from groq_safe import (
    GroqAuthError,
    describe_error,
    is_model_decommissioned_error,
    require_groq_api_key,
    safe_chat_completion,
)
from log_utils import setup_logger

logger = setup_logger(__name__)

_SOFT_LIMIT_WINDOW_SECONDS = 60.0
_MODEL_CALLS: dict[str, deque[float]] = defaultdict(deque)


class LLMTask(str, Enum):
    APPROVAL = "approval"
    NEWS = "news"
    ALT_DATA = "alt_data"
    EXPLAIN = "explain"


_DEFAULT_APPROVAL_MODELS = ["llama-3.1-8b-instant"]
_DEFAULT_NEWS_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
_DEFAULT_ALT_DATA_MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
_DEFAULT_EXPLAIN_MODELS = ["llama-3.3-70b-versatile"]


def _coerce_models(models: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        candidate = str(model or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def get_model_chain(env_var: str, default_models: list[str]) -> list[str]:
    raw = os.getenv(env_var, "").strip()
    if raw:
        models = _coerce_models(raw.split(","))
        if models:
            return models

    default_from_env = os.getenv("GROQ_DEFAULT_MODEL", "").strip()
    if default_from_env:
        return [default_from_env]

    return list(default_models)


def iter_models_for_task(task: LLMTask) -> list[str]:
    if task == LLMTask.APPROVAL:
        return get_model_chain("LLM_APPROVAL_MODELS", _DEFAULT_APPROVAL_MODELS)
    if task == LLMTask.NEWS:
        return get_model_chain("NEWS_LLM_MODELS", _DEFAULT_NEWS_MODELS)
    if task == LLMTask.ALT_DATA:
        return get_model_chain("ALT_DATA_LLM_MODELS", _DEFAULT_ALT_DATA_MODELS)
    if task == LLMTask.EXPLAIN:
        return get_model_chain("EXPLAIN_LLM_MODELS", _DEFAULT_EXPLAIN_MODELS)
    return get_model_chain("GROQ_DEFAULT_MODEL", _DEFAULT_APPROVAL_MODELS)


def _limit_for_model(model_name: str) -> int:
    default_limit = int(os.getenv("GROQ_SOFT_RPM_DEFAULT", "60") or 60)
    lower = model_name.lower()
    if "70b" in lower:
        return int(os.getenv("GROQ_SOFT_RPM_70B", "20") or 20)
    if "8b" in lower or "instant" in lower:
        return int(os.getenv("GROQ_SOFT_RPM_8B_INSTANT", str(default_limit)) or default_limit)
    return default_limit


def _soft_limit_check(model_name: str, *, now: Optional[float] = None) -> bool:
    allowed = max(1, _limit_for_model(model_name))
    now = time.time() if now is None else now
    window_start = now - _SOFT_LIMIT_WINDOW_SECONDS
    timestamps = _MODEL_CALLS[model_name]
    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()
    if len(timestamps) >= allowed:
        return False
    timestamps.append(now)
    return True


def _log_warning(task: LLMTask, model_name: str, error: Exception) -> None:
    logger.warning(
        "Groq error for task=%s model=%s: %s",
        task.value,
        model_name,
        describe_error(error),
    )


def call_llm_for_task(task: LLMTask, messages: list[dict[str, Any]], **kwargs: Any) -> Tuple[Any | None, str | None]:
    models = iter_models_for_task(task)

    try:
        require_groq_api_key()
    except GroqAuthError as exc:
        logger.warning(
            "Groq auth error for task=%s model=%s: %s",
            task.value,
            models[0] if models else "unknown",
            exc,
        )
        return None, None

    client = get_groq_client()
    if client is None:
        logger.warning("Groq client unavailable for task=%s", task.value)
        return None, None

    last_error: Optional[Exception] = None

    for model_name in models:
        if not _soft_limit_check(model_name):
            logger.info(
                "Soft rate limit hit for model=%s task=%s; skipping to next model",
                model_name,
                task.value,
            )
            continue

        try:
            response = safe_chat_completion(
                client,
                model=model_name,
                messages=messages,
                **kwargs,
            )
            logger.info(
                "LLM call succeeded: task=%s model=%s",
                task.value,
                getattr(response, "model", model_name),
            )
            return response, model_name
        except GroqAuthError as e:
            logger.warning(
                "Groq auth error for task=%s model=%s: %s",
                task.value,
                model_name,
                e,
            )
            last_error = e
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            if is_model_decommissioned_error(str(e)):
                _log_warning(task, model_name, e)
                continue
            _log_warning(task, model_name, e)
            continue

    logger.warning(
        "LLM unavailable for task=%s after trying models=%s. Last error: %r",
        task.value,
        models,
        last_error,
    )
    return None, None


__all__ = [
    "LLMTask",
    "call_llm_for_task",
    "get_model_chain",
    "iter_models_for_task",
]
