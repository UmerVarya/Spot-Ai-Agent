import os
import pandas as pd
from log_utils import setup_logger
from trade_storage import TRADE_HISTORY_FILE

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNAL_LOG_FILE = os.getenv("SIGNAL_LOG_FILE", os.path.join(_MODULE_DIR, "signal_log.csv"))

logger = setup_logger(__name__)

def optimize_indicator_weights(base_weights: dict, lookback: int = 200) -> dict:
    """Return adjusted indicator weights based on recent performance.

    Reads the signal log and completed trade log, joins them on timestamp and symbol,
    and calculates a simple win rate for each indicator. The base weight is scaled
    by a factor between 0.5 and 1.0 depending on the win rate (0..1).
    If insufficient data is available, the original weights are returned.
    """
    if not (os.path.exists(SIGNAL_LOG_FILE) and os.path.exists(TRADE_HISTORY_FILE)):
        return base_weights
    try:
        signals = pd.read_csv(SIGNAL_LOG_FILE).tail(lookback)
        trades = pd.read_csv(TRADE_HISTORY_FILE).tail(lookback)

        # Normalize keys consistently on both frames
        for df in (signals, trades):
            if "symbol" in df.columns:
                df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

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
            win_rate = (triggered["outcome"] == "win").mean()
            if win_rate == win_rate:  # check for NaN
                new_weights[key] = round(base_weights[key] * (0.5 + win_rate / 2), 3)
        return new_weights
    except Exception as exc:
        logger.exception("Failed to optimize indicator weights: %s", exc)
        return base_weights
