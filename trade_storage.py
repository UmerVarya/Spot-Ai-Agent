"""
Enhanced trade storage for Spot AI Super Agent (updated).

This module extends the original ``trade_storage.py`` by capturing
additional metadata when trades open and close.  In addition, it
standardises the locations of the active and historical trade logs so
that both the trading engine and the Streamlit dashboard read and
write the same files.  Paths are now resolved relative to the
module’s own directory, unless overridden via environment variables.
"""

import json
import os
import csv
from datetime import datetime
from typing import Optional

# Determine the directory where this file resides.  All log files
# default to this directory so they remain consistent across
# processes, regardless of the current working directory.
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# File locations; these can be overridden via environment variables if desired.
# ``ACTIVE_TRADES_FILE`` stores open trades in JSON format.  ``TRADE_LOG_FILE``
# stores completed trades in CSV format.  Expose these constants so other
# modules (e.g. ``trade_manager`` and ``dashboard``) can import them, ensuring
# all components read and write the exact same files.
ACTIVE_TRADES_FILE = os.environ.get(
    "ACTIVE_TRADES_FILE",
    os.path.join(_MODULE_DIR, "active_trades.json"),
)
TRADE_LOG_FILE = os.environ.get(
    "TRADE_LOG_FILE",
    os.path.join(_MODULE_DIR, "trade_log.csv"),
)


def load_active_trades() -> list:
    """Return the list of currently active trades from disk."""
    if os.path.exists(ACTIVE_TRADES_FILE):
        try:
            with open(ACTIVE_TRADES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_active_trades(trades: list) -> None:
    """Persist the list of active trades to disk."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(ACTIVE_TRADES_FILE), exist_ok=True)
    with open(ACTIVE_TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=4)


def is_trade_active(symbol: str) -> bool:
    """Return True if a trade with ``symbol`` exists in the active file."""
    trades = load_active_trades()
    return any(t.get("symbol") == symbol for t in trades)


def store_trade(trade: dict) -> None:
    """
    Append a new trade to the active trades list.

    The trade dict can include optional metadata such as ``entry_time``,
    ``size``, ``strategy`` and ``session``.  If not provided,
    ``entry_time`` will default to the current UTC time.
    """
    trades = load_active_trades()
    # Ensure entry_time is set
    if "entry_time" not in trade:
        trade["entry_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    # Remove leverage field (spot only)
    trade.pop("leverage", None)
    trades.append(trade)
    save_active_trades(trades)


def remove_trade(symbol: str) -> None:
    """Remove a trade with a given symbol from the active list."""
    trades = load_active_trades()
    updated = [t for t in trades if t.get("symbol") != symbol]
    save_active_trades(updated)


def log_trade_result(
    trade: dict,
    outcome: str,
    exit_price: float,
    *,
    exit_time: Optional[str] = None,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> None:
    """
    Append the result of a completed trade to ``trade_log.csv``.

    Parameters
    ----------
    trade : dict
        The trade dictionary that was opened.  Expected keys include
        ``symbol``, ``direction``, ``entry``, ``entry_time``, ``size``,
        ``strategy``, ``session`` and optional ``confidence``,
        ``btc_dominance``, ``fear_greed``, ``score``, ``pattern`` and
        ``narrative``.
    outcome : str
        A human-readable outcome label (e.g., "tp1", "tp2", "sl", "manual exit").
    exit_price : float
        The price at which the trade was closed.
    exit_time : str, optional
        ISO timestamp when the trade exited.  Defaults to the current UTC time.
    fees : float, default 0.0
        Total commissions paid on exit.
    slippage : float, default 0.0
        Slippage incurred on exit.
    """
    # Build headers if the file is empty / non-existent
    headers = [
        "timestamp",
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "entry",
        "exit",
        "size",
        "fees",
        "slippage",
        "outcome",
        "strategy",
        "session",
        "confidence",
        "btc_dominance",
        "fear_greed",
        "score",
        "pattern",
        "narrative",
    ]
    # Compose row
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": trade.get("symbol"),
        "direction": trade.get("direction"),
        "entry_time": trade.get("entry_time"),
        "exit_time": exit_time or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "entry": trade.get("entry"),
        "exit": exit_price,
        "size": trade.get("size", trade.get("position_size", 0)),
        "fees": fees,
        "slippage": slippage,
        "outcome": outcome,
        "strategy": trade.get("strategy", "unknown"),
        "session": trade.get("session", "unknown"),
        "confidence": trade.get("confidence", 0),
        "btc_dominance": trade.get("btc_dominance", 0),
        "fear_greed": trade.get("fear_greed", 0),
        "score": trade.get("score", trade.get("strength", 0)),
        "pattern": trade.get("pattern", "None"),
        "narrative": trade.get("narrative", "No explanation"),
    }
    file_exists = os.path.exists(TRADE_LOG_FILE)
    # Ensure directory exists
    os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
    # Write the row to CSV
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
