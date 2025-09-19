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
import uuid
import re
from datetime import datetime, timezone
from typing import Optional
from log_utils import ensure_symlink
from notifier import send_performance_email

import pandas as pd

logger = logging.getLogger(__name__)
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# When ``SIZE_AS_NOTIONAL`` is true (default), the ``size`` field provided to
# :func:`log_trade_result` is interpreted as the notional dollar amount of the
# trade.  When false, ``size`` represents the asset quantity.  This mirrors the
# behaviour used by the live dashboard.
SIZE_AS_NOTIONAL = os.getenv("SIZE_AS_NOTIONAL", "true").lower() == "true"


# Human-readable descriptions for outcome codes used across the system.  These
# labels are stored alongside each logged trade so downstream CSV exports do not
# have to map shorthand codes to friendly text.
OUTCOME_DESCRIPTIONS = {
    "tp1_partial": "Exited 50% at TP1",
    "tp2_partial": "Exited additional 30% at TP2",
    "tp4": "Final Exit (TP4 ride)",
    "tp4_sl": "Stopped out after TP3",
    "sl": "Stopped Out (SL)",
    "early_exit": "Early Exit",
    # Fallbacks for other potential outcomes
    "tp1": "Take Profit 1",
    "tp2": "Take Profit 2",
    "tp3": "Take Profit 3",
    "time_exit": "Time-based Exit",
}


# Canonical column order used for trade history CSV files.  This list is
# shared by ``log_trade_result`` when writing new rows and
# ``load_trade_history_df`` when falling back to manual header assignment for
# malformed files.
TRADE_HISTORY_HEADERS = [
    "trade_id",
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
    "pnl",
    "pnl_pct",
    "win",
    "size_tp1",
    "notional_tp1",
    "pnl_tp1",
    "size_tp2",
    "notional_tp2",
    "pnl_tp2",
    "outcome",
    "outcome_desc",
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


def _to_utc_iso(ts: Optional[str] = None) -> str:
    """Return an ISO-8601 UTC timestamp with ``Z`` suffix.

    Parameters
    ----------
    ts : str, optional
        Timestamp string to convert. If ``None`` or parsing fails, the current
        UTC time is used.
    """
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

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

# File locations; ``ACTIVE_TRADES_FILE`` stores open trades in JSON format and
# ``TRADE_HISTORY_FILE`` stores completed trades in CSV format.  The latter is
# the canonical location for historical trade data and is configurable via the
# ``TRADE_HISTORY_FILE`` or legacy ``COMPLETED_TRADES_FILE`` environment
# variables.  Legacy constant names remain as aliases for backward
# compatibility.
_active_default = os.path.join(DATA_DIR, "active_trades.json")
ACTIVE_TRADES_FILE = (
    os.environ.get("ACTIVE_TRADES_FILE", _active_default).split("#", 1)[0].strip()
    or _active_default
)
_history_default = os.path.join(DATA_DIR, "completed_trades.csv")
_history_override = os.environ.get("TRADE_HISTORY_FILE")
if _history_override is None:
    _history_override = os.environ.get("COMPLETED_TRADES_FILE")
_history_override = (_history_override or "").split("#", 1)[0].strip()
_HISTORY_ENV_OVERRIDE = bool(_history_override)
TRADE_HISTORY_FILE = _history_override or _history_default
# Backwards-compatible aliases
COMPLETED_TRADES_FILE = TRADE_HISTORY_FILE
TRADE_LOG_FILE = TRADE_HISTORY_FILE

# Legacy trade history files that may still contain data from earlier
# deployments where the CSV lived alongside the source tree. These are read in
# addition to the primary history file when no explicit override is supplied.
_LEGACY_HISTORY_FILENAMES = ("completed_trades.csv", "trade_history.csv")
_LEGACY_HISTORY_FILES = [
    os.path.join(_REPO_ROOT, name) for name in _LEGACY_HISTORY_FILENAMES
]
_EXPECTED_HISTORY_KEYS = ("timestamp", "entry_time", "exit_time", "symbol")


# Symlinks in the repository root allow read-only access for legacy code
# that still expects files beside the source tree.
ensure_symlink(ACTIVE_TRADES_FILE, os.path.join(_REPO_ROOT, "active_trades.json"))
ensure_symlink(TRADE_HISTORY_FILE, os.path.join(_REPO_ROOT, "completed_trades.csv"))


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
    # Ensure entry_time is set and normalised to UTC ISO string
    trade["entry_time"] = _to_utc_iso(trade.get("entry_time"))
    # Normalise additional timestamp fields that may be present in incoming trade data
    for ts_field in ("open_time", "close"):
        if ts_field in trade and trade[ts_field]:
            dt = pd.to_datetime(trade[ts_field], errors="coerce", utc=True)
            if pd.notna(dt):
                trade[ts_field] = dt.isoformat()
    # Assign unique trade ID if missing
    trade.setdefault("trade_id", str(uuid.uuid4()))
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
    # Normalise any timestamp fields that may appear in the incoming trade
    trade = trade.copy()
    for ts_field in ("open_time", "close"):
        if ts_field in trade and trade[ts_field]:
            dt = pd.to_datetime(trade[ts_field], errors="coerce", utc=True)
            if pd.notna(dt):
                trade[ts_field] = dt.isoformat()

    # Build headers including notional.  Using a shared constant keeps the
    # writer and reader in sync.
    headers = TRADE_HISTORY_HEADERS
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

    if SIZE_AS_NOTIONAL:
        notional = size_val
        quantity = (size_val / entry_val) if entry_val else 0.0
    else:
        quantity = size_val
        notional = entry_val * size_val if entry_val else None

    # Compute net PnL and percentage
    pnl_val = 0.0
    if entry_val is not None:
        if str(trade.get("direction", "")).lower() == "short":
            pnl_val = (entry_val - exit_price) * quantity
        else:
            pnl_val = (exit_price - entry_val) * quantity
    pnl_val -= fees
    pnl_val -= slippage
    pnl_pct = (pnl_val / notional * 100) if (notional and notional != 0) else 0.0
    row = {
        "trade_id": trade.get("trade_id", str(uuid.uuid4())),
        "timestamp": _to_utc_iso(),
        "symbol": trade.get("symbol"),
        "direction": trade.get("direction"),
        "entry_time": _to_utc_iso(trade.get("entry_time")),
        "exit_time": _to_utc_iso(exit_time),
        "entry": entry_val,
        "exit": exit_price,
        "size": size_val,
        "notional": notional,
        "fees": fees,
        "slippage": slippage,
        "pnl": pnl_val,
        "pnl_pct": pnl_pct,
        "win": pnl_val > 0,
        "size_tp1": 0.0,
        "notional_tp1": 0.0,
        "pnl_tp1": 0.0,
        "size_tp2": 0.0,
        "notional_tp2": 0.0,
        "pnl_tp2": 0.0,
        "outcome": outcome,
        "outcome_desc": OUTCOME_DESCRIPTIONS.get(outcome, outcome),
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
    if "tp1_partial" in outcome:
        row["size_tp1"] = quantity
        row["notional_tp1"] = notional or 0.0
        row["pnl_tp1"] = pnl_val
    if "tp2_partial" in outcome:
        row["size_tp2"] = quantity
        row["notional_tp2"] = notional or 0.0
        row["pnl_tp2"] = pnl_val
    float_cols = [
        "entry",
        "exit",
        "size",
        "notional",
        "fees",
        "confidence",
        "size_tp1",
        "notional_tp1",
        "pnl_tp1",
        "size_tp2",
        "notional_tp2",
        "pnl_tp2",
    ]
    for col in float_cols:
        try:
            row[col] = float(row.get(col, 0))
        except Exception:
            row[col] = 0.0
    if DB_CURSOR:
        try:
            DB_CURSOR.execute(
                "INSERT INTO trade_log (data) VALUES (%s)",
                (Json(row),),
            )
        except Exception as exc:
            logger.exception("Failed to log trade result to database: %s", exc)
    # Determine whether the history file already contains data.  Simply
    # checking ``os.path.exists`` is insufficient because the file may
    # exist yet be zero bytes long (for example after a crash or manual
    # truncation).  In that case ``csv.DictWriter`` would append rows
    # without a header and subsequent reads would treat the first trade
    # as the header, effectively hiding it from the dashboard.  We treat
    # an empty file the same as a missing file so the header is written
    # on the next append.
    file_exists = os.path.exists(TRADE_HISTORY_FILE) and os.path.getsize(
        TRADE_HISTORY_FILE
    ) > 0

    header_needed = True
    if file_exists:
        with open(TRADE_HISTORY_FILE, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line and first_line.lower().startswith("trade_id,"):
                header_needed = False
            else:
                remainder = f.read()
        if header_needed:
            with open(TRADE_HISTORY_FILE, "w", encoding="utf-8") as f:
                f.write(",".join(headers) + "\n")
                f.write(first_line)
                f.write(remainder)
            header_needed = False
            file_exists = True

    os.makedirs(os.path.dirname(TRADE_HISTORY_FILE), exist_ok=True)
    df_row = pd.DataFrame([row], columns=headers)
    df_row.to_csv(
        TRADE_HISTORY_FILE,
        mode="a",
        header=header_needed,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    _maybe_send_llm_performance_email()


def _deduplicate_history(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse partial exits so each trade occupies a single row.

    Rows are grouped by ``trade_id`` when present, otherwise by ``entry_time``,
    ``symbol`` and ``strategy``. All PnL values within a group are summed and the
    last non-partial row is used as the representative record. Boolean columns
    ``tp1_partial`` and ``tp2_partial`` flag whether those partial exits
    occurred. Additional ``pnl_tp1``/``pnl_tp2`` fields (with corresponding size
    and notional columns) expose the contribution of each stage.
    """

    if df.empty:
        return df
    df = df.drop_duplicates()
    if "trade_id" in df.columns:
        key_cols = ["trade_id"]
    else:
        key_cols = [c for c in ["entry_time", "symbol", "strategy"] if c in df.columns]
    if not key_cols:
        return df

    def _calc_fields(row: pd.Series) -> pd.Series:
        try:
            entry = float(row.get("entry", 0))
            exit_price = float(row.get("exit", 0))
            qty = float(row.get("position_size", row.get("quantity", row.get("size", 0))))
            if "position_size" not in row and "quantity" not in row:
                size_field = float(row.get("size", 0))
                if SIZE_AS_NOTIONAL:
                    qty = size_field / entry if entry != 0 else size_field
                else:
                    qty = size_field
            direction = str(row.get("direction", "")).lower()
            pnl = (exit_price - entry) * qty
            if direction == "short":
                pnl = (entry - exit_price) * qty
            pnl -= float(row.get("fees", 0) or 0)
            pnl -= float(row.get("slippage", 0) or 0)
            notional = float(row.get("notional", entry * qty if entry and qty else 0))
            return pd.Series({"_pnl": pnl, "_size": qty, "_notional": notional})
        except Exception:
            return pd.Series({"_pnl": 0.0, "_size": 0.0, "_notional": 0.0})

    df = df.join(df.apply(_calc_fields, axis=1))

    def _collapse(group: pd.DataFrame) -> pd.Series:
        pnl_total = group["_pnl"].sum()
        size_total = group["_size"].sum()
        notional_total = group["_notional"].sum()
        final = group[~group["outcome"].astype(str).str.contains("_partial", na=False)]
        if final.empty:
            final = group.tail(1)
        row = final.iloc[0].copy()
        row["pnl"] = pnl_total
        if "position_size" in row:
            row["position_size"] = size_total
        if "size" in row:
            if SIZE_AS_NOTIONAL:
                row["size"] = notional_total
            else:
                row["size"] = size_total
        else:
            row["size"] = size_total
        row["notional"] = notional_total if notional_total else row.get("notional")
        notional_val = row.get("notional")
        if notional_val:
            try:
                row["pnl_pct"] = pnl_total / float(notional_val) * 100
            except Exception:
                row["pnl_pct"] = None
        tp1_rows = group[group["outcome"].astype(str).str.contains("tp1_partial", na=False)]
        tp2_rows = group[group["outcome"].astype(str).str.contains("tp2_partial", na=False)]
        row["tp1_partial"] = not tp1_rows.empty
        row["tp2_partial"] = not tp2_rows.empty
        row["pnl_tp1"] = tp1_rows["_pnl"].sum()
        row["pnl_tp2"] = tp2_rows["_pnl"].sum()
        row["size_tp1"] = tp1_rows["_size"].sum()
        row["size_tp2"] = tp2_rows["_size"].sum()
        row["notional_tp1"] = tp1_rows["_notional"].sum()
        row["notional_tp2"] = tp2_rows["_notional"].sum()
        return row

    collapsed = (
        df.groupby(key_cols, group_keys=False).apply(_collapse).reset_index(drop=True)
    )
    return collapsed.drop(columns=["_pnl", "_size", "_notional"], errors="ignore")


def _read_history_frame(path: str) -> pd.DataFrame:
    """Read a trade history CSV from ``path`` using tolerant parsing."""

    if not path or not (os.path.exists(path) and os.path.getsize(path) > 0):
        return pd.DataFrame()

    try:
        try:  # pandas >= 1.3
            df = pd.read_csv(
                path,
                encoding="utf-8",
                on_bad_lines="skip",
                engine="python",
            )
        except TypeError:  # pragma: no cover - older pandas
            df = pd.read_csv(
                path,
                encoding="utf-8",
                engine="python",
                error_bad_lines=False,
                warn_bad_lines=False,
            )
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.exception("Failed to read trade log file %s: %s", path, exc)
        return pd.DataFrame()

    if df.empty or not any(
        any(key in str(col).lower() for key in _EXPECTED_HISTORY_KEYS)
        for col in df.columns
    ):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            if rows:
                df = pd.DataFrame(rows)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.exception("Fallback CSV parsing failed for %s: %s", path, exc)

    if not df.empty:
        cols_lower = [str(col).lower() for col in df.columns]
        if not any(
            any(key in col for key in _EXPECTED_HISTORY_KEYS) for col in cols_lower
        ):
            try:
                try:  # pandas >= 1.3
                    df = pd.read_csv(
                        path,
                        header=None,
                        names=TRADE_HISTORY_HEADERS,
                        encoding="utf-8",
                        on_bad_lines="skip",
                        engine="python",
                    )
                except TypeError:  # pragma: no cover - older pandas
                    df = pd.read_csv(
                        path,
                        header=None,
                        names=TRADE_HISTORY_HEADERS,
                        encoding="utf-8",
                        engine="python",
                        error_bad_lines=False,
                        warn_bad_lines=False,
                    )
            except Exception as exc:  # pragma: no cover - diagnostic only
                logger.exception("Failed to recover trade log file %s: %s", path, exc)

    return df


def load_trade_history_df(path: Optional[str] = None) -> pd.DataFrame:
    """Return historical trades as a DataFrame.

    Parameters
    ----------
    path : str, optional
        When provided, load data from this CSV path instead of the configured
        ``TRADE_HISTORY_FILE`` (and legacy fallbacks).  Passing ``None``
        preserves the original behaviour which also checks the optional
        database cursor.

    Notes
    -----
    The loader is intentionally tolerant so that header or format changes do
    not silently discard valid rows.  Column names are normalised and mapped to
    a canonical set used throughout the dashboard.  Timestamp fields are parsed
    with ``errors='coerce'`` and only rows lacking *all* parseable timestamps
    are dropped.  Optional fields are preserved even when missing so that the
    dashboard can still display partial information.
    """

    df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Load from database or CSV
    # ------------------------------------------------------------------
    use_db = path is None and DB_CURSOR
    if use_db:
        try:
            DB_CURSOR.execute("SELECT data FROM trade_log ORDER BY id")
            rows = [row[0] for row in DB_CURSOR.fetchall()]
            df = pd.DataFrame(rows)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.exception("Failed to load trade history from database: %s", exc)
    else:
        if path:
            candidate_paths = [path]
        else:
            candidate_paths = [TRADE_HISTORY_FILE]
            if not _HISTORY_ENV_OVERRIDE:
                candidate_paths.extend(_LEGACY_HISTORY_FILES)

        frames = []
        seen = set()
        for candidate in candidate_paths:
            if not candidate:
                continue
            resolved = os.path.abspath(candidate)
            if resolved in seen:
                continue
            seen.add(resolved)
            frame = _read_history_frame(candidate)
            if not frame.empty:
                frames.append(frame)

        if frames:
            df = pd.concat(frames, ignore_index=True, sort=False)
        elif path:
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(
                columns=[
                    "trade_id",
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

    if df.empty:
        return df

    # ------------------------------------------------------------------
    # Normalise headers
    # ------------------------------------------------------------------
    def _norm(col: str) -> str:
        return re.sub(r"[\s_]+", "", str(col).strip().lower())

    # Map of normalised legacy names to canonical forms
    synonyms = {
        "tradeid": "trade_id",
        "time": "timestamp",
        "timestamp": "timestamp",
        "symbol": "symbol",
        "pair": "symbol",
        "ticker": "symbol",
        "entryprice": "entry",
        "entry": "entry",
        "exitprice": "exit",
        "exit": "exit",
        "positionsize": "position_size",
        "position_size": "position_size",
        "qty": "position_size",
        "quantity": "position_size",
        "usd_size": "size",
        "side": "direction",
        "position": "direction",
        "direction": "direction",
        "tradeoutcome": "outcome",
        "result": "outcome",
        "traderesult": "outcome",
        "entrytimestamp": "entry_time",
        "entrytime": "entry_time",
        "exittimestamp": "exit_time",
        "exittime": "exit_time",
        "pnlusd": "pnl",
        "pnl$": "pnl",
        "netpnl": "net_pnl",
        "pnl": "pnl",
        "pnlpercent": "pnl_pct",
        "pnl%": "pnl_pct",
        "pnlpct": "pnl_pct",
        "notionalvalue": "notional",
        "notionalusd": "notional",
        "notional": "notional",
    }

    rename_map = {}
    for col in df.columns:
        key = _norm(col)
        if key in synonyms:
            rename_map[col] = synonyms[key]
        else:
            rename_map[col] = str(col).strip().lower().replace(" ", "_")
    df = df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # Parse timestamps and drop rows that cannot be parsed at all
    # ------------------------------------------------------------------
    time_cols = [c for c in ["timestamp", "entry_time", "exit_time"] if c in df.columns]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    if time_cols:
        time_mask = pd.Series(False, index=df.index)
        for col in time_cols:
            time_mask |= df[col].notna()
        dropped = len(df) - int(time_mask.sum())
        if dropped:
            logger.warning("Dropped %d rows with unparseable timestamps", dropped)
        df = df[time_mask]

    # ------------------------------------------------------------------
    # Validate symbol/direction but keep other rows intact
    # ------------------------------------------------------------------
    if "symbol" in df.columns:
        symbol_mask = df["symbol"].astype(str).str.match(r"^[A-Za-z0-9_]+$", na=False)
    else:
        symbol_mask = pd.Series([True] * len(df))

    if "direction" in df.columns:
        dir_series = df["direction"].astype(str).str.lower()
        dir_series = dir_series.replace({"buy": "long", "sell": "short"})
        df["direction"] = dir_series
        direction_mask = dir_series.isin(["long", "short"])
    else:
        direction_mask = pd.Series([True] * len(df))

    mask = symbol_mask & direction_mask
    dropped = len(df) - int(mask.sum())
    if dropped:
        logger.warning("Dropped %d malformed trade rows", dropped)
    df = df[mask]

    # ------------------------------------------------------------------
    # Ensure notional column for backward compatibility
    # ------------------------------------------------------------------
    if "notional" not in df.columns and {"entry", "size"}.issubset(df.columns):
        if SIZE_AS_NOTIONAL:
            df["notional"] = pd.to_numeric(df["size"], errors="coerce")
        else:
            df["notional"] = pd.to_numeric(df["entry"], errors="coerce") * pd.to_numeric(
                df["size"], errors="coerce"
            )

    # ------------------------------------------------------------------
    # De-duplicate partial exits and filter out open trades
    # ------------------------------------------------------------------
    df = _deduplicate_history(df)

    if not df.empty and "outcome" in df.columns:
        df = df[df["outcome"].astype(str).str.lower() != "open"]

    # ------------------------------------------------------------------
    # Compute PnL percentage when possible
    # ------------------------------------------------------------------
    if "pnl_pct" not in df.columns:
        pnl_source = None
        if {"net_pnl", "notional"}.issubset(df.columns):
            pnl_source = df["net_pnl"]
        elif {"pnl", "notional"}.issubset(df.columns):
            pnl_source = df["pnl"]
        if pnl_source is not None:
            notional = pd.to_numeric(df["notional"], errors="coerce").replace(0, pd.NA)
            df["pnl_pct"] = (
                pd.to_numeric(pnl_source, errors="coerce") / notional
            ) * 100
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct"), errors="coerce")
    if "pnl_pct" in df.columns:
        df["PnL (%)"] = df["pnl_pct"]

    # ------------------------------------------------------------------
    # Determine win/loss classification
    # ------------------------------------------------------------------
    if "win" in df.columns:
        df["win"] = df["win"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    elif "pnl" in df.columns:
        df["win"] = pd.to_numeric(df["pnl"], errors="coerce") > 0

    # ------------------------------------------------------------------
    # Ensure human readable outcome descriptions
    # ------------------------------------------------------------------
    if "outcome_desc" in df.columns:
        df = df.rename(columns={"outcome_desc": "Outcome Description"})
    elif "outcome" in df.columns:
        df["Outcome Description"] = df["outcome"].map(
            lambda x: OUTCOME_DESCRIPTIONS.get(str(x), str(x))
        )

    return df.reset_index(drop=True)


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
