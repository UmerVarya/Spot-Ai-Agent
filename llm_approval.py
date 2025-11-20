"""LLM trade approval helper with multi-model fallback.

This module centralises the logic for calling Groq to approve or reject trades
and normalises the result into a structured ``LLMTradeDecision`` object. The
helper is resilient to outages: if every configured model fails, it returns an
"LLM unavailable" decision while leaving ``approved``/``confidence`` unset so
the trading loop can continue without blocking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from groq_safe import GroqAuthError
from llm_tasks import LLMTask, call_llm_for_task, get_model_chain
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


def get_llm_approval_models_from_env() -> list[str]:
    """Return the ordered list of Groq models to try for trade approval."""

    return get_model_chain("LLM_APPROVAL_MODELS", ["llama-3.1-8b-instant"])


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
    prompt = _build_trade_prompt(trade_context)
    messages = [
        {"role": "system", "content": "Trade approval analyst"},
        {"role": "user", "content": prompt},
    ]

    response, model_used = call_llm_for_task(
        LLMTask.APPROVAL,
        messages=messages,
        temperature=0.2,
        max_tokens=300,
        timeout=10,
    )

    if response is None:
        logger.warning("LLM approval unavailable for %s", trade_context.get("symbol"))
        return LLMTradeDecision(
            decision="LLM unavailable",
            approved=None,
            confidence=None,
            model=None,
            rationale=None,
        )

    try:
        content = response.choices[0].message.get("content", "") if response else ""
        approved, confidence, reason = _parse_llm_json(content)
    except GroqAuthError:
        logger.warning("LLM approval auth failed for %s", trade_context.get("symbol"))
        return LLMTradeDecision(
            decision="LLM unavailable",
            approved=None,
            confidence=None,
            model=None,
            rationale=None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM approval parsing failed: %s", exc)
        return LLMTradeDecision(
            decision="LLM unavailable",
            approved=None,
            confidence=None,
            model=model_used,
            rationale=None,
        )

    decision_label = "approved" if approved else "rejected"
    if approved is None:
        decision_label = "skipped"
    logger.info(
        "LLM approval: task=approval model=%s decision=%s approved=%s conf=%s reason=%s for %s",
        model_used,
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
        model=model_used,
        rationale=reason or None,
    )


__all__ = [
    "LLMTradeDecision",
    "get_llm_trade_decision",
    "get_llm_approval_models_from_env",
]
