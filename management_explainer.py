"""Generate natural-language explanations for trailing stop management events."""

from __future__ import annotations

from typing import Any, Mapping

import config
from groq_client import get_groq_client
from groq_safe import safe_chat_completion
from log_utils import setup_logger

logger = setup_logger(__name__)


def _format_context(context: Mapping[str, Any] | None) -> str:
    """Render ``context`` as bullet points for the LLM prompt."""

    if not context:
        return "None"

    lines = []
    for key in sorted(context):
        value = context[key]
        if isinstance(value, float):
            formatted = f"{value:.6f}"
        else:
            formatted = str(value)
        lines.append(f"- {key}: {formatted}")
    return "\n".join(lines)


def explain_trailing_action(
    event: str,
    trade: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> str:
    """Return a short natural language explanation for a trailing action."""

    client = get_groq_client()
    if client is None:
        return ""

    symbol = str(trade.get("symbol", "Unknown")).upper()
    direction = str(trade.get("direction", "")).lower() or "long"
    system_prompt = (
        "You are an expert trading assistant. Explain trailing stop updates "
        "clearly using concise natural language. Focus on the market signals "
        "that justified the adjustment."
    )

    context_block = _format_context(context)
    user_prompt = (
        "Provide a one to two sentence explanation for the trading desk.\n"
        f"Symbol: {symbol}\n"
        f"Direction: {direction}\n"
        f"Event: {event}\n"
        "Key signals:\n"
        f"{context_block}\n"
        "Respond in plain English describing why the trailing stop decision "
        "was sensible."
    )

    try:
        response = safe_chat_completion(
            client,
            model=config.get_narrative_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=180,
        )
    except Exception:  # pragma: no cover - SDK level failures
        logger.debug("Trailing explanation generation failed", exc_info=True)
        return ""

    message = getattr(response, "choices", [None])[0]
    if not message or not getattr(message, "message", None):
        return ""

    content = message.message.get("content")  # type: ignore[index]
    return content.strip() if isinstance(content, str) else ""


__all__ = ["explain_trailing_action"]
