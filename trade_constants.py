"""Shared configuration constants for trade sizing and management."""

from __future__ import annotations

import os

ATR_STOP_MULTIPLIER = 1.5
"""Multiple of ATR used to position default stop-loss distances."""

TP_ATR_MULTIPLIERS = (3.0, 4.0, 5.0)
"""ATR multiples for management thresholds after entry."""

TRAIL_LOCK_IN_RATIO = 0.75
"""Fraction of the initial TP move to preserve when activating trailing mode."""

TRAIL_INITIAL_ATR = 1.5
"""ATR multiple for the first trailing-stop distance after TP1 is tagged."""

TRAIL_TIGHT_ATR = 1.0
"""ATR multiple applied once price extends to the second management threshold."""

TRAIL_FINAL_ATR = 0.5
"""ATR multiple used when momentum weakens or the final threshold is reached."""

DEFAULT_TAKE_PROFIT_STRATEGY = (
    os.getenv("TAKE_PROFIT_STRATEGY", "atr_trailing").strip().lower() or "atr_trailing"
)
"""Environment-controlled default take-profit strategy label."""

TP1_TRAILING_ONLY_STRATEGY = "tp1_trailing_only"
"""Identifier for the TP1-triggered trailing-only strategy."""
