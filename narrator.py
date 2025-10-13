"""Compatibility wrapper for the consolidated trade narrative builder."""

from __future__ import annotations

from typing import Any, Mapping

from log_utils import setup_logger
from narrative_builder import generate_trade_narrative

logger = setup_logger(__name__)


def generate_narrative(trade: Mapping[str, Any] | None) -> str:
    """Generate an explanation for a trade using the central narrative builder.

    ``narrator`` previously duplicated the Groq prompt logic that now lives in
    :mod:`narrative_builder`.  To keep the narrative format consistent across
    the agent, this wrapper simply delegates to :func:`generate_trade_narrative`
    while preserving the legacy public interface.
    """

    if not trade:
        logger.warning("Narrative requested without trade data")
        return "⚠️ Narrative generation failed: no trade data provided"

    try:
        return generate_trade_narrative(trade)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Narrative generation wrapper failed")
        return f"⚠️ Narrative generation failed: {exc}"
