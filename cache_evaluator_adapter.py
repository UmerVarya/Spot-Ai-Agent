"""Adapter that makes :func:`evaluate_signal` compatible with the cache."""

from __future__ import annotations

from typing import Optional, Tuple, Sequence, Mapping, Any

import pandas as pd

from trade_utils import evaluate_signal as _evaluate_signal


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    """Best-effort conversion of ``value`` to ``float`` with fallback."""

    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN check without importing math
        return default
    return number


def evaluator_for_cache(
    symbol: str,
    df: pd.DataFrame,
    **kwargs: Any,
) -> Tuple[float, Optional[str], float, Optional[str]]:
    """Adapter for :class:`RealTimeSignalCache`.

    This wrapper ensures our real evaluator can be invoked with keyword-only
    overrides (e.g. ``sentiment_bias``) and always returns the normalized
    ``(score, direction, position_size, pattern)`` tuple expected by the cache.
    """

    result = _evaluate_signal(price_data=df, symbol=symbol, **kwargs)

    score: float
    direction: Optional[str]
    position_size: float
    pattern: Optional[str]

    if isinstance(result, Mapping):
        score = _coerce_float(result.get("score"))
        direction = result.get("direction") or result.get("bias")  # type: ignore[assignment]
        position_size = _coerce_float(result.get("position_size"), default=1.0)
        pattern = result.get("pattern")  # type: ignore[assignment]
    elif isinstance(result, Sequence) and len(result) >= 4:
        score = _coerce_float(result[0])
        direction = result[1] if isinstance(result[1], str) or result[1] is None else str(result[1])
        position_size = _coerce_float(result[2], default=1.0)
        pattern_val = result[3]
        pattern = pattern_val if isinstance(pattern_val, str) or pattern_val is None else str(pattern_val)
    else:
        score = _coerce_float(result)
        direction = None
        position_size = 1.0
        pattern = None

    return score, direction, position_size, pattern
