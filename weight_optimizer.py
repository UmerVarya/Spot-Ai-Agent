import os
from typing import Optional

import pandas as pd

from log_utils import setup_logger
from trade_schema import normalise_history_columns
from trade_storage import TRADE_HISTORY_FILE

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_LOG_FILE = os.getenv("SIGNAL_LOG_FILE", os.path.join(_MODULE_DIR, "signal_log.csv"))

logger = setup_logger(__name__)

def optimize_indicator_weights(
    base_weights: dict, lookback: int = 200, ema_span: Optional[int] = None
) -> dict:
    """Return adjusted indicator weights based on recent performance.

    Reads the signal log and completed trade log, joins them on timestamp and symbol,
    and calculates a simple win rate for each indicator. The base weight is scaled
    by a factor between 0.5 and 1.0 depending on the win rate (0..1). When
    ``ema_span`` is provided the win rate uses an exponentially weighted mean so
    that more recent trades have a larger impact while still considering a deeper
    history. If insufficient data is available, the original weights are returned.
    """
    if not (os.path.exists(SIGNAL_LOG_FILE) and os.path.exists(TRADE_HISTORY_FILE)):
        return base_weights
    try:
        if ema_span is not None and ema_span <= 0:
            logger.warning(
                "Invalid ema_span %s supplied to optimize_indicator_weights; ignoring.",
                ema_span,
            )
            ema_span = None

        signals = pd.read_csv(SIGNAL_LOG_FILE).tail(lookback)
        trades = pd.read_csv(TRADE_HISTORY_FILE).tail(lookback)
        trades = normalise_history_columns(trades)

        # ``normalise_history_columns`` only renames headers – it does not
        # guarantee that the important ones actually exist.  Older trade logs
        # (or partially written rows) sometimes omit ``timestamp`` or
        # ``symbol`` which would make the ``merge`` below raise a ``KeyError``.
        # Gracefully bail out instead of letting the exception bubble up so the
        # caller keeps the default weights.
        required_trade_cols = {"timestamp", "symbol", "outcome"}
        missing_trade_cols = required_trade_cols - set(trades.columns)
        if missing_trade_cols:
            logger.warning(
                "Skipping weight optimisation – missing trade columns: %s",
                ", ".join(sorted(missing_trade_cols)),
            )
            return base_weights

        required_signal_cols = {"timestamp", "symbol"}
        missing_signal_cols = required_signal_cols - set(signals.columns)
        if missing_signal_cols:
            logger.warning(
                "Skipping weight optimisation – missing signal columns: %s",
                ", ".join(sorted(missing_signal_cols)),
            )
            return base_weights

        # Normalize keys consistently on both frames
        for df in (signals, trades):
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(
                    df["timestamp"], errors="coerce", utc=True
                )

        if trades.empty or "outcome" not in trades.columns:
            return base_weights

        merged = pd.merge(
            signals,
            trades[["timestamp", "symbol", "outcome"]],
            on=["timestamp", "symbol"],
            how="inner",
        )
        if merged.empty:
            return base_weights
        new_weights = base_weights.copy()
        for key in base_weights.keys():
            col = f"{key}_trigger"
            if col not in merged.columns:
                continue
            triggered = merged[merged[col] > 0]
            if triggered.empty:
                continue
            outcome_score = (
                triggered["outcome"].astype(str).str.lower() == "win"
            ).astype(float)
            if ema_span:
                win_rate = outcome_score.ewm(span=ema_span, adjust=False).mean().iloc[-1]
            else:
                win_rate = outcome_score.mean()
            if win_rate == win_rate:  # check for NaN
                new_weights[key] = round(base_weights[key] * (0.5 + win_rate / 2), 3)
        return new_weights
    except Exception as exc:
        logger.exception("Failed to optimize indicator weights: %s", exc)
        return base_weights
