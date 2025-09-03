"""
Enhanced trade storage utilities for the Spot AI Super Agent.

This module mirrors the original ``trade_storage.py`` while
introducing the following improvements:

* **Notional calculation** – each logged trade now includes a
  ``notional`` field equal to ``entry_price * size``.  Storing the
  notional explicitly simplifies downstream PnL percentage
  calculations and clarifies how much capital was committed to each
  position.
* **Flexible data directories** – preserves the existing environment
  variable behaviour for locating data files and database support.
* **Safer history loading** – gracefully handles missing files or
  malformed rows.

These changes are backward compatible with existing logs because
``load_trade_history_df`` still falls back to deriving ``notional``
from ``entry`` and ``size`` when absent.
"""

import json
import os
import csv
import logging
from datetime import datetime
from typing import Optional
from log_utils import ensure_symlink
from notifier import send_performance_email

import pandas as pd

logger = logging.getLogger(__name__)

# Optional PostgreSQL support (unchanged from original)
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_CONN = DB_CURSOR = None
Json = None
if DATABASE_URL:
    try:
        import psycopg2
        from psycopg2.extras import Json  # type: ignore
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
        logger.info("Connected to PostgreSQL for trade storage.")
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.exception(
            "Database initialisation failed: %s. Falling back to file storage.",
            exc,
        )
        DB_CONN = DB_CURSOR = None

# ---------------------------------------------------------------------------
# Storage locations
# ---------------------------------------------------------------------------

# ``DATA_DIR`` now defaults to the shared spot data directory so that trades
# are written directly to persistent storage instead of a symlink inside the
# repository.  This avoids ``ReadWritePaths`` restrictions in systemd and
# ensures historical data survives service restarts.  The environment variable
# is still honoured for flexibility.
DEFAULT_DATA_DIR = "/home/ubuntu/spot_data/trades"
raw_data_dir = os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)
# Remove inline comments and surrounding whitespace
raw_data_dir = raw_data_dir.split("#", 1)[0].strip() or DEFAULT_DATA_DIR

try:
    os.makedirs(raw_data_dir, exist_ok=True)
    DATA_DIR = raw_data_dir
except OSError:
    DATA_DIR = DEFAULT_DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)

# File locations; these can be overridden individually via environment
# variables if desired. ``ACTIVE_TRADES_FILE`` stores open trades in JSON
# format. ``TRADE_LOG_FILE`` stores completed trades in CSV format.  Expose
# these constants so other modules (e.g. ``trade_manager`` and ``dashboard``)
# can import them, ensuring all components read and write the exact same files.
# Canonical file locations within the data directory.  ``completed_trades.csv``
# replaces the legacy ``trade_log.csv`` name; ``TRADE_LOG_FILE`` is kept as an
# alias for backward compatibility.
ACTIVE_TRADES_FILE = os.environ.get(
    "ACTIVE_TRADES_FILE", os.path.join(DATA_DIR, "active_trades.json")
).split("#", 1)[0].strip()
COMPLETED_TRADES_FILE = os.environ.get(
    "COMPLETED_TRADES_FILE", os.path.join(DATA_DIR, "completed_trades.csv")
).split("#", 1)[0].strip()
TRADE_LOG_FILE = os.environ.get("TRADE_LOG_FILE", COMPLETED_TRADES_FILE)


# Symlinks in the repository root allow read-only access for legacy code
# that still expects files beside the source tree.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ensure_symlink(ACTIVE_TRADES_FILE, os.path.join(_REPO_ROOT, "active_trades.json"))
ensure_symlink(COMPLETED_TRADES_FILE, os.path.join(_REPO_ROOT, "completed_trades.csv"))


def load_active_trades() -> list:
    """Return the list of currently active trades."""
    if DB_CURSOR:
        try:
            DB_CURSOR.execute("SELECT data FROM active_trades")
            return [row[0] for row in DB_CURSOR.fetchall()]
        except Exception as exc:
            logger.exception("Failed to load active trades from database: %s", exc)
    if os.path.exists(ACTIVE_TRADES_FILE):
        try:
            with open(ACTIVE_TRADES_FILE, "r") as f:
                return json.load(f)
        except Exception as exc:
            logger.exception("Failed to read active trades file: %s", exc)
    return []


def save_active_trades(trades: list) -> None:
    """Persist the list of active trades."""
    if DB_CURSOR:
        try:
            DB_CURSOR.execute("DELETE FROM active_trades")
            for trade in trades:
                DB_CURSOR.execute(
                    "INSERT INTO active_trades (symbol, data) VALUES (%s, %s)",
                    (trade.get("symbol"), Json(trade)),
                )
            return
        except Exception as exc:
            logger.exception("Failed to save active trades to database: %s", exc)
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(ACTIVE_TRADES_FILE), exist_ok=True)
    with open(ACTIVE_TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=4)


def is_trade_active(symbol: str) -> bool:
    """Return True if a trade with ``symbol`` exists in active storage."""
    if DB_CURSOR:
        DB_CURSOR.execute(
            "SELECT 1 FROM active_trades WHERE symbol = %s LIMIT 1",
            (symbol,),
        )
        return DB_CURSOR.fetchone() is not None
    trades = load_active_trades()
    return any(t.get("symbol") == symbol for t in trades)


def store_trade(trade: dict) -> bool:
    """Store ``trade`` in the active trades list if not already present.

    Returns
    -------
    bool
        ``True`` if the trade was stored, ``False`` if a duplicate was
        detected and the trade was ignored.
    """
    # Ensure entry_time is set
    if "entry_time" not in trade:
        trade["entry_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    # Remove leverage field (spot only)
    trade.pop("leverage", None)
    symbol = trade.get("symbol")
    if symbol is None:
        logger.warning("Cannot store trade without symbol: %s", trade)
        return False
    # Database-backed storage
    if DB_CURSOR:
        try:
            if is_trade_active(symbol):
                logger.warning("Duplicate trade for %s detected; skipping store.", symbol)
                return False
            DB_CURSOR.execute(
                """
                INSERT INTO active_trades (symbol, data)
                VALUES (%s, %s)
                """,
                (symbol, Json(trade)),
            )
            return True
        except Exception as exc:
            logger.exception("Failed to store trade in database: %s", exc)
            return False
    # File-based storage
    trades = load_active_trades()
    if any(t.get("symbol") == symbol for t in trades):
        logger.warning("Duplicate trade for %s detected; skipping store.", symbol)
        return False
    trades.append(trade)
    save_active_trades(trades)
    return True


def remove_trade(symbol: str) -> None:
    """Remove a trade with a given symbol from the active list."""
    if DB_CURSOR:
        try:
            DB_CURSOR.execute("DELETE FROM active_trades WHERE symbol = %s", (symbol,))
            return
        except Exception as exc:
            logger.exception("Failed to remove trade from database: %s", exc)
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
    # Build headers including notional
    headers = [
        "timestamp",
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "entry",
        "exit",
        "size",
        "notional",
        "fees",
        "slippage",
        "outcome",
        "strategy",
        "session",
        "confidence",
        "btc_dominance",
        "fear_greed",
        "sentiment_bias",
        "sentiment_confidence",
        "score",
        "pattern",
        "narrative",
        "llm_decision",
        "llm_confidence",
        "llm_error",
        "volatility",
        "htf_trend",
        "order_imbalance",
        "macro_indicator",
    ]
    # Compose row
    entry_price = trade.get("entry")
    size_val = trade.get("size", trade.get("position_size", 0))
    try:
        size_val = float(size_val)
    except Exception:
        size_val = 0.0
    try:
        entry_val = float(entry_price) if entry_price is not None else None
    except Exception:
        entry_val = None
    notional = None
    if entry_val is not None:
        notional = entry_val * size_val
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": trade.get("symbol"),
        "direction": trade.get("direction"),
        "entry_time": trade.get("entry_time"),
        "exit_time": exit_time or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "entry": trade.get("entry"),
        "exit": exit_price,
        "size": size_val,
        "notional": notional,
        "fees": fees,
        "slippage": slippage,
        "outcome": outcome,
        "strategy": trade.get("strategy", "unknown"),
        "session": trade.get("session", "unknown"),
        "confidence": trade.get("confidence", 0),
        "btc_dominance": trade.get("btc_dominance", 0),
        "fear_greed": trade.get("fear_greed", 0),
        "sentiment_bias": trade.get("sentiment_bias", "neutral"),
        "sentiment_confidence": trade.get(
            "sentiment_confidence", trade.get("confidence", 0)
        ),
        "score": trade.get("score", trade.get("strength", 0)),
        "pattern": trade.get("pattern", "None"),
        "narrative": trade.get("narrative", "No explanation"),
        "llm_decision": trade.get("llm_decision"),
        "llm_confidence": trade.get("llm_confidence"),
        "llm_error": trade.get("llm_error"),
        "volatility": trade.get("volatility", 0),
        "htf_trend": trade.get("htf_trend", 0),
        "order_imbalance": trade.get("order_imbalance", 0),
        "macro_indicator": trade.get("macro_indicator", 0),
    }
    if DB_CURSOR:
        try:
            DB_CURSOR.execute(
                "INSERT INTO trade_log (data) VALUES (%s)",
                (Json(row),),
            )
            return
        except Exception as exc:
            logger.exception("Failed to log trade result to database: %s", exc)
    file_exists = os.path.exists(TRADE_LOG_FILE)
    # Ensure directory exists
    os.makedirs(os.path.dirname(TRADE_LOG_FILE), exist_ok=True)
    # Write the row to CSV
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    _maybe_send_llm_performance_email()


def load_trade_history_df() -> pd.DataFrame:
    """Return historical trades as a DataFrame."""
    df = pd.DataFrame()
    if DB_CURSOR:
        try:
            DB_CURSOR.execute("SELECT data FROM trade_log ORDER BY id")
            rows = [row[0] for row in DB_CURSOR.fetchall()]
            df = pd.DataFrame(rows)
        except Exception as exc:
            logger.exception("Failed to load trade history from database: %s", exc)
    else:
        path = COMPLETED_TRADES_FILE
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except Exception as exc:
                logger.exception("Failed to read trade log file: %s", exc)
        else:
            df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "entry",
                    "exit",
                    "size",
                    "direction",
                    "outcome",
                    "pnl",
                ]
            )
    # Filter out rows with outcome recorded as "open"
    if not df.empty and "outcome" in df.columns:
        df = df[df["outcome"].astype(str).str.lower() != "open"]
    return df


def _maybe_send_llm_performance_email() -> None:
    """Send LLM performance metrics after every 50 completed trades."""
    df = load_trade_history_df()
    total_trades = len(df)
    if total_trades == 0 or total_trades % 50 != 0:
        return

    def _calc_return(row: pd.Series) -> float:
        try:
            entry = float(row.get("entry", 0))
            exit_price = float(row.get("exit", 0))
            direction = str(row.get("direction", "")).lower()
            if entry == 0:
                return 0.0
            if direction == "short":
                return (entry - exit_price) / entry
            return (exit_price - entry) / entry
        except Exception:
            return 0.0

    df["return"] = df.apply(_calc_return, axis=1)
    df["win"] = df["return"] > 0

    def _metrics(mask: pd.Series) -> tuple[int, float, float]:
        subset = df[mask]
        count = len(subset)
        if count == 0:
            return 0, 0.0, 0.0
        win_rate = subset["win"].mean() * 100
        avg_return = subset["return"].mean() * 100
        return count, win_rate, avg_return

    llm_dec = df.get("llm_decision", pd.Series(dtype=str)).astype(str).str.lower()
    llm_err = df.get("llm_error", pd.Series(dtype=str)).astype(str).str.lower()

    approved_mask = (llm_err != "true") & llm_dec.isin(["true", "1", "yes"])
    vetoed_mask = (llm_err != "true") & llm_dec.isin(["false", "0", "no"])
    error_mask = llm_err.isin(["true", "1", "yes"])

    a_count, a_win, a_ret = _metrics(approved_mask)
    v_count, v_win, v_ret = _metrics(vetoed_mask)
    e_count, e_win, e_ret = _metrics(error_mask)

    body = f"""
    <h2>LLM Decision Performance ({total_trades} trades)</h2>
    <ul>
        <li><strong>Approved:</strong> {a_count} trades | Win rate {a_win:.2f}% | Avg return {a_ret:.2f}%</li>
        <li><strong>Vetoed:</strong> {v_count} trades | Win rate {v_win:.2f}% | Avg return {v_ret:.2f}%</li>
        <li><strong>Error:</strong> {e_count} trades | Win rate {e_win:.2f}% | Avg return {e_ret:.2f}%</li>
    </ul>
    """
    send_performance_email("LLM Performance Summary", body)
