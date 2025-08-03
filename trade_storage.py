"""
Enhanced trade storage for Spot AI Super Agent (updated).

This module extends the original ``trade_storage.py`` by capturing
additional metadata when trades open and close.  It also centralises
where trade data are written so that restarts do not wipe the agent’s
memory.  Paths now default to a persistent data directory (configurable
via the ``DATA_DIR`` environment variable) instead of the repository
root, allowing containers such as Render services to mount a volume and
retain trade history across process restarts.
"""

import json
import os
import csv
from datetime import datetime
from typing import Optional

# Optional PostgreSQL support -------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_CONN = DB_CURSOR = None
Json = None

if DATABASE_URL:
    try:
        import psycopg2
        from psycopg2.extras import Json

        DB_CONN = psycopg2.connect(DATABASE_URL)
        DB_CONN.autocommit = True
        DB_CURSOR = DB_CONN.cursor()
        DB_CURSOR.execute(
            """
            CREATE TABLE IF NOT EXISTS active_trades (
                symbol TEXT PRIMARY KEY,
                data   JSONB NOT NULL
            )
            """
        )
        DB_CURSOR.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_log (
                id   SERIAL PRIMARY KEY,
                data JSONB NOT NULL
            )
            """
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"Database initialisation failed: {exc}. Falling back to file storage.")
        DB_CONN = DB_CURSOR = None

# ---------------------------------------------------------------------------
# Storage locations
# ---------------------------------------------------------------------------

# ``DATA_DIR`` can point to a mounted volume (e.g. /var/data on Render) to
# ensure logs persist across restarts.  By default we use a hidden directory in
# the user's home folder.
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.expanduser("~"), ".spot_ai_agent"),
)
os.makedirs(DATA_DIR, exist_ok=True)

# File locations; these can be overridden individually via environment
# variables if desired. ``ACTIVE_TRADES_FILE`` stores open trades in JSON
# format. ``TRADE_LOG_FILE`` stores completed trades in CSV format.  Expose
# these constants so other modules (e.g. ``trade_manager`` and ``dashboard``)
# can import them, ensuring all components read and write the exact same files.
ACTIVE_TRADES_FILE = os.environ.get(
    "ACTIVE_TRADES_FILE",
    os.path.join(DATA_DIR, "active_trades.json"),
)
TRADE_LOG_FILE = os.environ.get(
    "TRADE_LOG_FILE",
    os.path.join(DATA_DIR, "trade_log.csv"),
)


def load_active_trades() -> list:
    """Return the list of currently active trades."""
    if DB_CURSOR:
        DB_CURSOR.execute("SELECT data FROM active_trades")
        return [row[0] for row in DB_CURSOR.fetchall()]
    if os.path.exists(ACTIVE_TRADES_FILE):
        try:
            with open(ACTIVE_TRADES_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_active_trades(trades: list) -> None:
    """Persist the list of active trades."""
    if DB_CURSOR:
        DB_CURSOR.execute("DELETE FROM active_trades")
        for trade in trades:
            DB_CURSOR.execute(
                "INSERT INTO active_trades (symbol, data) VALUES (%s, %s)",
                (trade.get("symbol"), Json(trade)),
            )
        return
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(ACTIVE_TRADES_FILE), exist_ok=True)
    with open(ACTIVE_TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=4)


def is_trade_active(symbol: str) -> bool:
    """Return True if a trade with ``symbol`` exists."""
    if DB_CURSOR:
        DB_CURSOR.execute(
            "SELECT 1 FROM active_trades WHERE symbol = %s LIMIT 1",
            (symbol,),
        )
        return DB_CURSOR.fetchone() is not None
    trades = load_active_trades()
    return any(t.get("symbol") == symbol for t in trades)


def store_trade(trade: dict) -> None:
    """Append a new trade to the active trades list."""
    # Ensure entry_time is set
    if "entry_time" not in trade:
        trade["entry_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    # Remove leverage field (spot only)
    trade.pop("leverage", None)
    if DB_CURSOR:
        DB_CURSOR.execute(
            """
            INSERT INTO active_trades (symbol, data)
            VALUES (%s, %s)
            ON CONFLICT (symbol) DO UPDATE SET data = EXCLUDED.data
            """,
            (trade.get("symbol"), Json(trade)),
        )
        return
    trades = load_active_trades()
    trades.append(trade)
    save_active_trades(trades)


def remove_trade(symbol: str) -> None:
    """Remove a trade with a given symbol from the active list."""
    if DB_CURSOR:
        DB_CURSOR.execute("DELETE FROM active_trades WHERE symbol = %s", (symbol,))
        return
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
    Append the result of a completed trade to storage.

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
    if DB_CURSOR:
        DB_CURSOR.execute(
            "INSERT INTO trade_log (data) VALUES (%s)",
            (Json(row),),
        )
        return
    file_exists = os.path.exists(TRADE_LOG_FILE)
    # Ensure directory exists
    os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
    # Write the row to CSV
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
