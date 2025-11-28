from __future__ import annotations

from typing import Optional

from config import DEFAULT_MIN_PROB_FOR_TRADE

__all__ = ["should_veto_on_probability"]


def should_veto_on_probability(ml_prob: Optional[float]) -> bool:
    """Return ``True`` when the ML probability is below the shared threshold."""

    return ml_prob is not None and ml_prob < DEFAULT_MIN_PROB_FOR_TRADE

