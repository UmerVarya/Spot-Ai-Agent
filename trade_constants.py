"""Shared configuration constants for trade sizing and management."""

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
