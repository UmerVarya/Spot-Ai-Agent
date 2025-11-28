from __future__ import annotations

from typing import Optional

from config import DEFAULT_MIN_PROB_FOR_TRADE
from risk_profiles.btc_profile import SymbolProfile

__all__ = ["get_effective_min_prob_for_symbol", "should_veto_on_probability"]


def get_effective_min_prob_for_symbol(
    symbol: str,
    profile: Optional[SymbolProfile],
    override_min_prob: Optional[float] = None,
) -> float:
    """Return the active probability threshold for ``symbol``.

    Precedence:
    1. ``override_min_prob`` when provided (e.g. BacktestConfig.min_prob)
    2. ``profile.min_prob_for_trade`` if configured on the symbol profile
    3. ``DEFAULT_MIN_PROB_FOR_TRADE`` global fallback
    """

    if override_min_prob is not None:
        return override_min_prob
    if profile is not None and profile.min_prob_for_trade is not None:
        return profile.min_prob_for_trade
    return DEFAULT_MIN_PROB_FOR_TRADE


def should_veto_on_probability(
    ml_prob: Optional[float], *, min_prob: Optional[float] = None
) -> bool:
    """Return ``True`` when the ML probability is below the configured threshold."""

    threshold = DEFAULT_MIN_PROB_FOR_TRADE if min_prob is None else min_prob
    return ml_prob is not None and ml_prob < threshold

