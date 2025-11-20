"""LLM trade approval helper with multi-model fallback.

This module centralises the logic for calling Groq to approve or reject trades
and normalises the result into a structured ``LLMTradeDecision`` object. The
helper is resilient to outages: if every configured model fails, it returns an
"LLM unavailable" decision while leaving ``approved``/``confidence`` unset so
the trading loop can continue without blocking.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from groq_client import get_groq_client
from groq_safe import GroqAuthError, safe_chat_completion
from log_utils import setup_logger


logger = setup_logger(__name__)


@dataclass
class LLMTradeDecision:
    """Structured representation of an LLM approval response."""

    decision: str
    approved: Optional[bool]
    confidence: Optional[float]
    model: Optional[str]
    rationale: Optional[str]


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


def get_llm_approval_models_from_env() -> list[str]:
    """Return the ordered list of Groq models to try for trade approval."""

    env_value = os.getenv("LLM_APPROVAL_MODELS")
    if env_value:
        return _coerce_models(env_value.split(","))

    defaults = [
        os.getenv("GROQ_MODEL_TRADE"),
        os.getenv("TRADE_LLM_MODEL"),
        os.getenv("GROQ_MODEL_FALLBACK"),
        os.getenv("GROQ_OVERFLOW_MODEL"),
        "llama-3.3-70b-versatile",
    ]
    return _coerce_models(defaults)


def _build_trade_prompt(trade_context: dict[str, Any]) -> str:
    """Create a compact JSON-style prompt from ``trade_context``."""

    summary = json.dumps(trade_context, indent=2, ensure_ascii=False)
    return (
        "You are an experienced crypto trade risk reviewer."
        " Review the following context and respond ONLY with valid JSON containing"
        " keys approve (true/false), confidence (0-1 float), and reason (short string).\n\n"
        f"Context:\n{summary}\n\n"
        "Respond with a single JSON object and no extra text."
    )


def _parse_llm_json(content: str) -> tuple[Optional[bool], Optional[float], str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON response: {exc}") from exc

    approve_value = data.get("approve")
    approved: Optional[bool]
    if isinstance(approve_value, bool):
        approved = approve_value
    elif isinstance(approve_value, str):
        lowered = approve_value.strip().lower()
        if lowered in {"true", "yes", "y"}:
            approved = True
        elif lowered in {"false", "no", "n"}:
            approved = False
        else:
            approved = None
    else:
        approved = None

    confidence_raw = data.get("confidence")
    confidence: Optional[float] = None
    try:
        if confidence_raw is not None:
            confidence = max(0.0, min(float(confidence_raw), 1.0))
    except (TypeError, ValueError):
        confidence = None

    reason = str(data.get("reason", "")).strip()
    return approved, confidence, reason


def get_llm_trade_decision(trade_context: dict[str, Any]) -> LLMTradeDecision:
    """Return an LLM-backed approval decision with multi-model fallback."""

    client = get_groq_client()
    models = get_llm_approval_models_from_env()
    prompt = _build_trade_prompt(trade_context)
    messages = [
        {"role": "system", "content": "Trade approval analyst"},
        {"role": "user", "content": prompt},
    ]

    if not models:
        logger.warning("No LLM approval models configured; skipping LLM gate")
        return LLMTradeDecision(
            decision="LLM unavailable",
            approved=None,
            confidence=None,
            model=None,
            rationale=None,
        )

    errors: list[str] = []
    for model_name in models:
        try:
            response = safe_chat_completion(
                client,
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=300,
                timeout=10,
            )
            content = response.choices[0].message.get("content", "") if response else ""
            approved, confidence, reason = _parse_llm_json(content)
            decision_label = "approved" if approved else "rejected"
            if approved is None:
                decision_label = "skipped"
            logger.info(
                "LLM approval: model=%s decision=%s approved=%s conf=%s reason=%s for %s",
                model_name,
                decision_label,
                approved,
                f"{confidence:.3f}" if confidence is not None else None,
                reason,
                trade_context.get("symbol"),
            )
            return LLMTradeDecision(
                decision=decision_label,
                approved=approved,
                confidence=confidence if confidence is None else round(confidence, 3),
                model=model_name,
                rationale=reason or None,
            )
        except GroqAuthError as auth_err:
            errors.append(f"{model_name}: auth error {auth_err}")
            logger.warning("LLM approval auth failed for %s: %s", model_name, auth_err)
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model_name}: {exc}")
            logger.warning("LLM approval via %s failed: %s", model_name, exc)
            continue

    logger.warning(
        "LLM approval unavailable after trying models=%s: %s",
        ",".join(models),
        "; ".join(errors) if errors else "no models responded",
    )
    return LLMTradeDecision(
        decision="LLM unavailable",
        approved=None,
        confidence=None,
        model=None,
        rationale=None,
    )


__all__ = [
    "LLMTradeDecision",
    "get_llm_trade_decision",
    "get_llm_approval_models_from_env",
]
