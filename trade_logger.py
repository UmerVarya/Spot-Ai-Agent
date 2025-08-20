"""
Logger for final trade outcomes (updated).

This module defines ``log_trade_result`` which appends a record to
``trade_learning_log.csv`` whenever a trade closes.  The log captures key
information such as the symbol, session, trade score, direction, outcome,
macro context (including sentiment bias) and final confidence.  The CSV is appended with headers if
absent.  Caller should ensure that each trade dict contains the expected
keys (see below).  If optional fields are missing, sensible defaults are
used.

Unlike the original version, this update allows overriding the path of
the learning log via the ``TRADE_LEARNING_LOG_FILE`` environment
variable and resolves the default path relative to this module.  This
ensures consistent file locations across different processes.
"""

import csv
import os
from datetime import datetime


def _ensure_symlink(target: str, link: str) -> None:
    try:
        if os.path.islink(link) or os.path.exists(link):
            return
        os.symlink(target, link)
    except OSError:
        pass


def log_trade_result(trade: dict, outcome: str, **kwargs) -> None:
    """
    Append the result of a completed trade to the learning log.

    Parameters
    ----------
    trade : dict
        A dictionary describing the trade.  Required keys include:
        ``symbol``, ``direction``, ``confidence``, ``score``, ``session``,
        ``btc_dominance``, ``fear_greed``, ``sentiment_bias`` and
        ``sentiment_confidence``.  Optional keys include ``pattern``,
        ``support_zone``, ``resistance_zone`` and ``volume``.
    outcome : str
        The result of the trade (e.g., "win", "loss", "breakeven").
    **kwargs : Any
        Additional keyword arguments are accepted for compatibility (e.g.,
        ``exit_price``) but are ignored by the logger.
    """
    # Determine log file path: environment variable takes precedence,
    # otherwise use the module directory.  Using ``os.path.abspath``
    # ensures relative paths are resolved correctly.
    log_file = os.environ.get(
        "TRADE_LEARNING_LOG_FILE",
        "/home/ubuntu/spot_data/trades/trade_logs.csv",
    )

    _ensure_symlink(
        log_file,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_learning_log.csv"),
    )
    fields = [
        "timestamp",
        "symbol",
        "session",
        "score",
        "direction",
        "outcome",
        "btc_dominance",
        "fear_greed",
        "sentiment_bias",
        "pattern",
        "support_zone",
        "resistance_zone",
        "volume",
        "confidence",
    ]

    # Compose the row with sensible defaults for missing fields
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": trade.get("symbol"),
        "session": trade.get("session", "unknown"),
        # Score reflects the normalized technical score at entry time
        "score": trade.get("score", trade.get("confidence", 0)),
        "direction": trade.get("direction", "long"),
        "outcome": outcome,
        "btc_dominance": trade.get("btc_dominance", 0),
        "fear_greed": trade.get("fear_greed", 0),
        "sentiment_bias": trade.get("sentiment_bias", "unknown"),
        # Use the chart/candlestick pattern detected when the trade was opened
        "pattern": trade.get("pattern", trade.get("pattern_name", "none")),
        "support_zone": trade.get("support_zone", False),
        "resistance_zone": trade.get("resistance_zone", False),
        "volume": trade.get("volume", 0),
        # Confidence is the final blended confidence at entry time
        "confidence": trade.get("confidence", trade.get("sentiment_confidence", 0)),
    }

    file_exists = os.path.isfile(log_file)
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
