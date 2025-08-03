"""
Enhanced trade storage for Spot AI Super Agent (updated).

This module extends the original ``trade_storage.py`` by capturing
additional metadata when trades open and close. It centralises where
trade data are written so that restarts do not wipe the agent's memory.
Paths default to a persistent data directory (configurable via the
``DATA_DIR`` environment variable), but if a ``DATABASE_URL``
environment variable is provided the module will store both active
trades and the trade log in a PostgreSQL database.
"""

import json
import os
import csv
from datetime import datetime
from typing import Optional

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    import psycopg2
    from psycopg2.extras import Json

    _conn = psycopg2.connect(DATABASE_URL)
    _conn.autocommit = True

    def _init_db() -> None:
        with _conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS active_trades (
                    symbol TEXT PRIMARY KEY,
                    data   JSONB NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_log (
                    id   SERIAL PRIMARY KEY,
                    data JSONB NOT NULL
                );
                """
            )

    _init_db()

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


if DATABASE_URL:
    def load_active_trades() -> list:
        with _conn.cursor() as cur:
            cur.execute("SELECT data FROM active_trades")
            return [row[0] for row in cur.fetchall()]


    def save_active_trades(trades: list) -> None:
        with _conn.cursor() as cur:
            cur.execute("TRUNCATE active_trades")
            cur.executemany(
                "INSERT INTO active_trades(symbol, data) VALUES (%s, %s)",
                [(t.get("symbol"), Json(t)) for t in trades],
            )


    def is_trade_active(symbol: str) -> bool:
        with _conn.cursor() as cur:
            cur.execute("SELECT 1 FROM active_trades WHERE symbol=%s", (symbol,))
            return cur.fetchone() is not None


    def store_trade(trade: dict) -> None:
        if "entry_time" not in trade:
            trade["entry_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        trade.pop("leverage", None)
        with _conn.cursor() as cur:
            cur.execute(
                "INSERT INTO active_trades(symbol, data) VALUES (%s,%s) "
                "ON CONFLICT (symbol) DO UPDATE SET data = EXCLUDED.data",
                (trade.get("symbol"), Json(trade)),
            )


    def remove_trade(symbol: str) -> None:
        with _conn.cursor() as cur:
            cur.execute("DELETE FROM active_trades WHERE symbol=%s", (symbol,))


    def log_trade_result(
        trade: dict,
        outcome: str,
        exit_price: float,
        *,
        exit_time: Optional[str] = None,
        fees: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
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
        with _conn.cursor() as cur:
            cur.execute("INSERT INTO trade_log(data) VALUES (%s)", (Json(row),))

else:
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
        os.makedirs(os.path.dirname(ACTIVE_TRADES_FILE), exist_ok=True)
        with open(ACTIVE_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=4)


    def is_trade_active(symbol: str) -> bool:
        trades = load_active_trades()
        return any(t.get("symbol") == symbol for t in trades)


    def store_trade(trade: dict) -> None:
        trades = load_active_trades()
        if "entry_time" not in trade:
            trade["entry_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        trade.pop("leverage", None)
        trades.append(trade)
        save_active_trades(trades)


    def remove_trade(symbol: str) -> None:
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
        os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
        with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
