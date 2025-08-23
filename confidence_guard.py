"""
Adaptive confidence threshold guard based on recent trade outcomes.

This module reads the unified completed trades log to compute a dynamic
confidence threshold that adapts to your strategy's performance.  If the log file is
missing, contains too few entries, or has malformed lines, a conservative
default threshold is returned.  You can use the returned value in your
brain/decision logic to calibrate how strict the bot should be.
"""

import pandas as pd
import os

from trade_storage import COMPLETED_TRADES_FILE

# Path to the completed trades log used for adaptive thresholding
LEARNING_LOG = COMPLETED_TRADES_FILE


def get_adaptive_conf_threshold() -> float:
    """
    Calculate an adaptive confidence threshold based on recent performance.

    The function reads the last 25 entries of the completed trades log and
    computes the win rate and average confidence.  If the win rate is
    exceptionally high (>=70%), the threshold is lowered slightly; if the
    win rate is low (<=40%), the threshold is raised slightly.  Otherwise,
    the average confidence is used directly.

    Returns
    -------
    float
        The adapted confidence threshold.  Defaults to 5.5 if insufficient
        history or the file is missing.  The result is clamped to a
        reasonable range (4.5 to 7.5).
    """
    if not os.path.exists(LEARNING_LOG):
        return 5.5  # Default if no data

    try:
        # Use python engine and skip bad lines to handle inconsistent log entries
        df = pd.read_csv(LEARNING_LOG, engine="python", on_bad_lines="skip", encoding="utf-8")
    except Exception:
        return 5.5

    if len(df) < 10:
        return 5.5  # Not enough history to adapt

    recent = df.tail(25)
    wins = recent[recent.get("outcome") == "win"]
    win_rate = len(wins) / len(recent)
    try:
        avg_conf = float(recent.get("confidence").mean())
    except Exception:
        avg_conf = 5.5

    # Base threshold is the average confidence; adapt up or down
    if win_rate >= 0.7:
        threshold = max(4.5, avg_conf - 0.5)
    elif win_rate <= 0.4:
        threshold = min(7.5, avg_conf + 0.5)
    else:
        threshold = avg_conf

    return round(threshold, 2)
