"""
Enhanced trade storage for Spot AI Super Agent.

This module extends the original ``trade_storage.py`` by capturing
additional metadata when trades open and close.  In particular it adds
``entry_time``, ``exit_time``, ``size``, ``leverage``, ``fees``,
``slippage``, ``strategy``, ``session`` and ``duration`` fields to the
CSV log, allowing richer analytics in the dashboard.  The API is
backwards‑compatible: existing callers can still invoke
``store_trade()`` and ``log_trade_result()`` with the old parameters,
and missing fields will be filled with sensible defaults.
"""

import json
import os
import csv
from datetime import datetime
from typing import Optional

# File locations; these can be overridden via environment variables if desired
ACTIVE_FILE = os.environ.get("ACTIVE_TRADES_FILE", "active_trades.json")
LOG_FILE = os.environ.get("TRADE_LOG_FILE", "trade_log.csv")


def load_active_trades() -> list:
    """Return the list of currently active trades from disk."""
    if os.path.exists(ACTIVE_FILE):
        try:
            with open(ACTIVE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_active_trades(trades: list) -> None:
    """Persist the list of active trades to disk."""
    with open(ACTIVE_FILE, "w") as f:
        json.dump(trades, f, indent=4)


def is_trade_active(symbol: str) -> bool:
    """Return True if a trade with ``symbol`` exists in the active file."""
    trades = load_active_trades()
    return any(t.get("symbol") == symbol for t in trades)


def store_trade(trade: dict) -> None:
    """Append a new trade to the active trades list.

    The trade dict can include optional metadata such as ``entry_time``,
    ``size``, ``leverage``, ``strategy`` and ``session``.  If not provided,
    ``entry_time`` will default to the current UTC time.
    """
    trades = load_active_trades()
    # Ensure entry_time is set
    if "entry_time" not in trade:
        trade["entry_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
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
        ``leverage``, ``strategy``, ``session`` and optional ``confidence``,
        ``btc_dominance``, ``fear_greed``, ``score``, ``pattern`` and
        ``narrative``.
    outcome : str
        A human‑readable outcome label (e.g., "tp1", "tp2", "sl", "manual exit").
    exit_price : float
        The price at which the trade was closed.
    exit_time : str, optional
        ISO timestamp when the trade exited.  Defaults to the current UTC time.
    fees : float, default 0.0
        Total commissions paid on exit.
    slippage : float, default 0.0
        Slippage incurred on exit.
    """
    # Build headers if the file is empty / non‑existent
    headers = [
        "timestamp", "symbol", "direction", "entry_time", "exit_time",
        "entry", "exit", "size", "leverage", "fees", "slippage",
        "outcome", "strategy", "session", "confidence", "btc_dominance",
        "fear_greed", "score", "pattern", "narrative"
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
        "leverage": trade.get("leverage", 1),
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
    file_exists = os.path.exists(LOG_FILE)
    # Write the row to CSV
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
