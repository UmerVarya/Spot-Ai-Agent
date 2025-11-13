"""Groq-powered alternative data summariser.

This module asks the Groq LLM to fuse raw social chatter and on-chain
metrics into a structured payload that mirrors the FinGPT schema used by
:mod:`alternative_data`.  The goal is to let operators rely solely on the
Groq stack for realtime alternative data when FinGPT is unavailable,
while keeping existing fallbacks intact.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from groq import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

import config
from groq_client import get_groq_client
from groq_safe import safe_chat_completion
from json_utils import parse_llm_json_response
from log_utils import setup_logger

logger = setup_logger(__name__)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


_MAX_POSTS = _env_int("GROQ_ALT_DATA_MAX_POSTS", 12, minimum=3, maximum=30)
_MAX_TOKENS = _env_int("GROQ_ALT_DATA_MAX_TOKENS", 400, minimum=200, maximum=800)

_SYSTEM_MESSAGE = (
    "You are a crypto alternative-data analyst."
    " Combine the supplied on-chain metrics and social chatter to produce"
    " a structured assessment. Always respond with compact JSON containing"
    " keys social, onchain, and sources."
    "\nSocial must include: bias (bullish/bearish/neutral), score (-1 to 1),"
    " confidence (0 to 1), posts (integer sample size), models (list of"
    " model/source identifiers), and optional rationale."
    "\nOnchain must include: composite_score (-1 to 1), net_exchange_flow,"
    " whale_ratio, large_holder_netflow, and optional commentary."
    "\nPopulate sources with identifiers describing how the analysis was"
    " generated (e.g. groq, llama3-70b)."
)


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _normalise_list(value: Any, fallback: Sequence[str]) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        items = [
            str(item)
            for item in value
            if isinstance(item, (str, bytes)) and str(item).strip()
        ]
        if items:
            return items
    if isinstance(value, (str, bytes)) and str(value).strip():
        return [str(value)]
    return [str(item) for item in fallback if str(item).strip()]


def _format_onchain_snapshot(snapshot: Mapping[str, Any] | None) -> str:
    if not isinstance(snapshot, Mapping):
        return "No explicit on-chain metrics were supplied."
    lines: List[str] = []
    for key, label in (
        ("exchange_inflow", "Exchange inflow"),
        ("exchange_outflow", "Exchange outflow"),
        ("net_exchange_flow", "Net exchange flow"),
        ("large_holder_netflow", "Large holder netflow"),
        ("whale_ratio", "Whale to exchange ratio"),
        ("composite_score", "Existing composite"),
    ):
        value = snapshot.get(key)
        if value in (None, "", []):
            continue
        number = _coerce_float(value)
        if number is None:
            continue
        lines.append(f"- {label}: {number:.4f}")
    if not lines:
        return "On-chain metrics were not provided."
    return "On-chain metrics:" + "\n" + "\n".join(lines)


def _format_posts(posts: Sequence[str] | None) -> tuple[str, int]:
    if not posts:
        return "No social posts were available.", 0
    cleaned = []
    for post in posts:
        if not isinstance(post, str):
            continue
        text = post.strip()
        if text:
            cleaned.append(text)
    if not cleaned:
        return "No social posts were available.", 0
    sample = cleaned[:_MAX_POSTS]
    formatted = "Social posts (newest first):" + "\n" + "\n".join(
        f"- {item}" for item in sample
    )
    return formatted, len(sample)


def analyze_alt_data(
    symbol: str,
    *,
    onchain_snapshot: Mapping[str, Any] | None = None,
    social_posts: Sequence[str] | None = None,
) -> Optional[Mapping[str, Any]]:
    """Return Groq alt-data payload for ``symbol``.

    The returned mapping mirrors the FinGPT schema so existing parsing
    helpers in :mod:`alternative_data` can be reused.  ``None`` is
    returned when the Groq client is unavailable or the model response
    cannot be parsed into JSON.
    """

    client = get_groq_client()
    if client is None:
        logger.debug("Groq client unavailable; skipping Groq alt-data request")
        return None

    model_name = config.get_groq_model()
    onchain_text = _format_onchain_snapshot(onchain_snapshot)
    posts_text, posts_count = _format_posts(social_posts)

    user_prompt = (
        f"Symbol: {symbol}\n\n"
        f"{onchain_text}\n\n"
        f"{posts_text}\n\n"
        "Respond with JSON only."
    )

    messages = [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = safe_chat_completion(
            client,
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=_MAX_TOKENS,
        )
    except RateLimitError as err:
        logger.warning("Groq alt-data rate limited: %s", err)
        return None
    except (APIStatusError, APIConnectionError, APITimeoutError, APIError) as err:
        logger.warning("Groq alt-data request failed: %s", err)
        return None
    except Exception as err:  # pragma: no cover - defensive guardrail
        logger.error("Unexpected Groq alt-data exception: %s", err, exc_info=True)
        return None

    content = ""
    if getattr(response, "choices", None):
        message = response.choices[0].message
        content = getattr(message, "content", "") or ""

    defaults: Dict[str, Any] = {
        "social": {
            "bias": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "posts": posts_count,
            "models": ["groq", model_name],
        },
        "onchain": {
            "composite_score": _coerce_float(
                (onchain_snapshot or {}).get("composite_score")
            )
            or 0.0,
            "net_exchange_flow": _coerce_float(
                (onchain_snapshot or {}).get("net_exchange_flow")
            ),
            "large_holder_netflow": _coerce_float(
                (onchain_snapshot or {}).get("large_holder_netflow")
            ),
            "whale_ratio": _coerce_float((onchain_snapshot or {}).get("whale_ratio")),
        },
        "sources": ["groq", model_name],
    }

    parsed, success = parse_llm_json_response(content, defaults=defaults, logger=logger)
    if not success:
        logger.debug("Groq alt-data response was not valid JSON: %s", content[:200])
        return None

    social = parsed.get("social", {})
    if not isinstance(social, Mapping):
        social = defaults["social"]
    social_map: Dict[str, Any] = dict(defaults["social"])
    social_map.update({key: value for key, value in social.items()})
    social_map["bias"] = _coerce_str(social_map.get("bias") or "neutral").strip() or "neutral"
    social_map["score"] = max(-1.0, min(1.0, _coerce_float(social_map.get("score")) or 0.0))
    social_map["confidence"] = max(
        0.0, min(1.0, _coerce_float(social_map.get("confidence")) or 0.0)
    )
    social_map["posts"] = posts_count
    social_map["models"] = _normalise_list(
        social_map.get("models") or social_map.get("sources"), ["groq", model_name]
    )

    onchain = parsed.get("onchain", {})
    if not isinstance(onchain, Mapping):
        onchain = defaults["onchain"]
    onchain_map: Dict[str, Any] = dict(defaults["onchain"])
    onchain_map.update({key: value for key, value in onchain.items()})
    for key in (
        "composite_score",
        "net_exchange_flow",
        "large_holder_netflow",
        "whale_ratio",
        "exchange_inflow",
        "exchange_outflow",
        "whale_inflow",
        "whale_outflow",
    ):
        if key not in onchain_map:
            continue
        coerced = _coerce_float(onchain_map.get(key))
        onchain_map[key] = coerced
    if onchain_map.get("composite_score") is None:
        onchain_map["composite_score"] = 0.0

    sources = parsed.get("sources", [])
    sources_list = _normalise_list(sources, ["groq", model_name])

    return {
        "social": social_map,
        "onchain": onchain_map,
        "sources": sources_list,
    }


__all__ = ["analyze_alt_data"]
