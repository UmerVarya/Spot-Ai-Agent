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
from datetime import datetime, timezone
from typing import Optional, Sequence
from log_utils import ensure_symlink
from notifier import send_performance_email

import pandas as pd
from trade_schema import (
    TRADE_HISTORY_COLUMNS,
    build_rename_map,
    normalise_history_columns,
)

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


# Canonical column order used for trade history CSV files.  The list lives in
# :mod:`trade_schema` so that every component consumes the same definition.
# ``TRADE_HISTORY_HEADERS`` is kept as an alias for backwards compatibility.
TRADE_HISTORY_HEADERS = TRADE_HISTORY_COLUMNS
MISSING_VALUE = "N/A"
ERROR_VALUE = "error"


def _normalise_header_line(header_line: str) -> list[str]:
    """Return a lower-cased token list for the CSV header line."""

    if not header_line:
        return []
    cleaned = header_line.strip().lstrip("\ufeff")
    if not cleaned:
        return []
    return [segment.strip().strip('"').lower() for segment in cleaned.split(",")]


def _parse_header_columns(header_line: str) -> list[str]:
    """Return the raw header columns from ``header_line`` preserving casing."""

    if not header_line:
        return []
    cleaned = header_line.strip().lstrip("\ufeff")
    if not cleaned:
        return []
    return [segment.strip().strip('"') for segment in cleaned.split(",")]


def _header_is_compatible(
    header_line: str, headers: Sequence[str], *, require_essential: bool = False
) -> bool:
    """Return ``True`` when ``header_line`` resembles a supported schema."""

    tokens = _normalise_header_line(header_line)
    if not tokens:
        return False

    rename_map = build_rename_map(tokens)
    recognised = {
        rename_map.get(token, "") for token in tokens if rename_map.get(token, "")
    }
    recognised &= {str(col) for col in headers}
    if not recognised:
        return False

    essential = set(_EXPECTED_HISTORY_KEYS)
    if require_essential:
        return essential.issubset(recognised)

    if recognised & essential:
        return True

    return len(recognised) >= 3


def _archive_legacy_history_file(path: str) -> Optional[str]:
    """Move a legacy history file aside so a fresh log can be created."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = f"{path}.legacy-{timestamp}"
    try:
        os.replace(path, backup_path)
    except OSError as exc:  # pragma: no cover - filesystem specific
        logger.warning("Unable to archive legacy trade log %s: %s", path, exc)
        return None
    logger.info("Archived legacy trade log %s -> %s", path, backup_path)
    return backup_path


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
_history_default = os.path.join(DATA_DIR, "historical_trades.csv")
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
_LEGACY_HISTORY_FILENAMES = (
    "historical_trades.csv",
    "completed_trades.csv",
    "trade_history.csv",
)
_LEGACY_HISTORY_FILES = [
    os.path.join(_REPO_ROOT, name) for name in _LEGACY_HISTORY_FILENAMES
]
_EXPECTED_HISTORY_KEYS = ("timestamp", "entry_time", "exit_time", "symbol")


# Symlinks in the repository root allow read-only access for legacy code
# that still expects files beside the source tree.
ensure_symlink(ACTIVE_TRADES_FILE, os.path.join(_REPO_ROOT, "active_trades.json"))
ensure_symlink(TRADE_HISTORY_FILE, os.path.join(_REPO_ROOT, "historical_trades.csv"))
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
    headers = list(TRADE_HISTORY_HEADERS)
    existing_headers: list[str] = []
    header_rename_map: dict[str, str] = {}

    def _optional_text_field(value: Optional[str]) -> str:
        if value is None:
            return MISSING_VALUE
        if isinstance(value, str):
            return value if value.strip() else MISSING_VALUE
        return str(value)

    def _optional_numeric_field(value: Optional[object]):
        if value is None or value == "":
            return MISSING_VALUE
        try:
            return float(value)
        except Exception:
            return ERROR_VALUE

    def _optional_bool_field(value: Optional[object]):
        if value is None or value == "":
            return MISSING_VALUE
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if not token:
                return MISSING_VALUE
            if token in {"true", "1", "yes", "y", "approved"}:
                return True
            if token in {"false", "0", "no", "n", "vetoed"}:
                return False
        return ERROR_VALUE

    def _serialise_extra_field(value: Optional[object]):
        if value is None:
            return MISSING_VALUE
        if isinstance(value, (str, int, float)):
            return value
        if isinstance(value, bool):
            return value
        try:
            return json.dumps(value)
        except Exception:
            return ERROR_VALUE

    entry_price = trade.get("entry")
    try:
        exit_price_val = float(exit_price)
    except Exception:
        try:
            exit_price_val = float(trade.get("exit_price"))
        except Exception:
            exit_price_val = 0.0

    raw_size = trade.get("size", trade.get("position_size", 0))
    try:
        size_val = float(raw_size)
    except Exception:
        size_val = 0.0
    try:
        entry_val = float(entry_price) if entry_price is not None else None
    except Exception:
        entry_val = None

    try:
        initial_qty = float(trade.get("initial_size"))
    except Exception:
        initial_qty = None

    raw_notional = trade.get("notional")
    try:
        notional = float(raw_notional)
    except Exception:
        notional = None

    quantity = None
    if initial_qty is not None:
        quantity = initial_qty
    else:
        try:
            quantity = float(trade.get("position_size"))
        except Exception:
            quantity = None

    if SIZE_AS_NOTIONAL:
        if quantity is not None and entry_val is not None:
            notional = entry_val * quantity
            size_val = notional
        elif notional is None:
            notional = size_val
    else:
        if quantity is not None:
            size_val = quantity
        if entry_val is not None and quantity is not None:
            notional = entry_val * quantity
        elif notional is None and entry_val is not None:
            notional = entry_val * size_val

    if quantity is None:
        if SIZE_AS_NOTIONAL and entry_val:
            quantity = size_val / entry_val if entry_val else 0.0
        else:
            quantity = size_val

    if SIZE_AS_NOTIONAL:
        notional = size_val if notional is None else notional
    else:
        if notional is None and entry_val is not None:
            notional = entry_val * quantity if quantity is not None else None

    try:
        fees_val = float(fees)
    except Exception:
        fees_val = 0.0
    try:
        slippage_val = float(slippage)
    except Exception:
        slippage_val = 0.0

    fees_override = trade.get("total_fees", trade.get("realized_fees"))
    if fees_override is not None:
        try:
            fees_val = float(fees_override)
        except Exception:
            pass
    slippage_override = trade.get("total_slippage", trade.get("realized_slippage"))
    if slippage_override is not None:
        try:
            slippage_val = float(slippage_override)
        except Exception:
            pass

    # Compute net PnL and percentage
    pnl_val = 0.0
    if entry_val is not None and quantity is not None:
        if str(trade.get("direction", "")).lower() == "short":
            pnl_val = (entry_val - exit_price_val) * quantity
        else:
            pnl_val = (exit_price_val - entry_val) * quantity
    pnl_val -= fees_val
    pnl_val -= slippage_val

    pnl_override = trade.get("total_pnl", trade.get("realized_pnl"))
    if pnl_override is not None:
        try:
            pnl_val = float(pnl_override)
        except Exception:
            pass

    pnl_pct = None
    if isinstance(notional, (int, float)) and notional not in (0, 0.0):
        pnl_pct = (pnl_val / notional) * 100

    def _parse_timestamp(value: Optional[object]) -> Optional[pd.Timestamp]:
        """Return a timezone-aware timestamp when ``value`` resembles a date."""

        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or cleaned.upper() == MISSING_VALUE.upper():
                return None
            value = cleaned
        try:
            dt = pd.to_datetime(value, errors="coerce", utc=True)
        except Exception:
            return None
        if pd.isna(dt):
            return None
        return dt

    strategy_raw = trade.get("strategy")
    session_raw = trade.get("session")
    strategy_ts = _parse_timestamp(strategy_raw)
    session_ts = _parse_timestamp(session_raw)

    entry_time_source = trade.get("entry_time") or trade.get("open_time")
    if not entry_time_source and strategy_ts is not None:
        entry_time_source = strategy_raw
    entry_time_value = MISSING_VALUE
    entry_dt = _parse_timestamp(entry_time_source)
    if entry_dt is not None:
        entry_time_value = entry_dt.isoformat().replace("+00:00", "Z")

    exit_time_source = exit_time or trade.get("close")
    if exit_time_source is None and session_ts is not None:
        exit_time_source = session_raw
    exit_dt = _parse_timestamp(exit_time_source)
    if exit_dt is not None:
        exit_time_value = exit_dt.isoformat().replace("+00:00", "Z")
    else:
        exit_time_value = _to_utc_iso()

    def _display_value(
        raw_value: Optional[object],
        parsed_ts: Optional[pd.Timestamp],
        fallback_keys: Sequence[str],
    ) -> Optional[object]:
        """Return a human-readable value, avoiding timestamps in string fields."""

        if parsed_ts is None:
            return raw_value
        for key in fallback_keys:
            alt = trade.get(key)
            if alt and _parse_timestamp(alt) is None:
                return alt
        return None

    strategy_value = _display_value(
        strategy_raw,
        strategy_ts,
        ("strategy_name", "strategy_label", "pattern"),
    )
    session_value = _display_value(
        session_raw,
        session_ts,
        ("session_name", "session_label"),
    )

    sentiment_conf = trade.get("sentiment_confidence")
    if sentiment_conf is None:
        sentiment_conf = trade.get("confidence")
    score_val = trade.get("score", trade.get("strength"))
    outcome_text = str(outcome or "")
    outcome_lower = outcome_text.lower()
    tp1_flag = bool(trade.get("tp1_partial")) or "tp1_partial" in outcome_lower
    tp2_flag = bool(trade.get("tp2_partial")) or "tp2_partial" in outcome_lower

    row = {
        "trade_id": trade.get("trade_id", str(uuid.uuid4())),
        "timestamp": _to_utc_iso(),
        "symbol": _optional_text_field(trade.get("symbol")),
        "direction": _optional_text_field(trade.get("direction")),
        "entry_time": entry_time_value,
        "exit_time": exit_time_value,
        "entry": _optional_numeric_field(entry_val),
        "exit": _optional_numeric_field(exit_price),
        "size": _optional_numeric_field(size_val),
        "notional": _optional_numeric_field(notional),
        "fees": _optional_numeric_field(fees_val),
        "slippage": _optional_numeric_field(slippage_val),
        "pnl": _optional_numeric_field(pnl_val),
        "pnl_pct": _optional_numeric_field(pnl_pct),
        "outcome": _optional_text_field(outcome_text or None),
        "outcome_desc": _optional_text_field(
            OUTCOME_DESCRIPTIONS.get(outcome, outcome)
        ),
        "exit_reason": _optional_text_field(trade.get("exit_reason")),
        "strategy": _optional_text_field(strategy_value),
        "session": _optional_text_field(session_value),
        "confidence": _optional_numeric_field(trade.get("confidence")),
        "btc_dominance": _optional_numeric_field(trade.get("btc_dominance")),
        "fear_greed": _optional_numeric_field(trade.get("fear_greed")),
        "sentiment_bias": _optional_text_field(trade.get("sentiment_bias")),
        "sentiment_confidence": _optional_numeric_field(sentiment_conf),
        "score": _optional_numeric_field(score_val),
        "pattern": _optional_text_field(trade.get("pattern")),
        "narrative": _optional_text_field(trade.get("narrative")),
        "llm_decision": _optional_text_field(trade.get("llm_decision")),
        "llm_approval": _optional_bool_field(trade.get("llm_approval")),
        "llm_confidence": _optional_numeric_field(trade.get("llm_confidence")),
        "llm_error": _optional_text_field(trade.get("llm_error")),
        "technical_indicator_score": _optional_numeric_field(
            trade.get("technical_indicator_score")
        ),
        "volatility": _optional_numeric_field(trade.get("volatility")),
        "htf_trend": _optional_numeric_field(trade.get("htf_trend")),
        "order_imbalance": _optional_numeric_field(trade.get("order_imbalance")),
        "order_flow_score": _optional_numeric_field(trade.get("order_flow_score")),
        "order_flow_flag": _optional_numeric_field(trade.get("order_flow_flag")),
        "order_flow_state": _optional_text_field(trade.get("order_flow_state")),
        "cvd": _optional_numeric_field(trade.get("cvd")),
        "cvd_change": _optional_numeric_field(trade.get("cvd_change")),
        "taker_buy_ratio": _optional_numeric_field(trade.get("taker_buy_ratio")),
        "trade_imbalance": _optional_numeric_field(trade.get("trade_imbalance")),
        "aggressive_trade_rate": _optional_numeric_field(trade.get("aggressive_trade_rate")),
        "spoofing_intensity": _optional_numeric_field(trade.get("spoofing_intensity")),
        "spoofing_alert": _optional_numeric_field(trade.get("spoofing_alert")),
        "volume_ratio": _optional_numeric_field(trade.get("volume_ratio")),
        "price_change_pct": _optional_numeric_field(trade.get("price_change_pct")),
        "spread_bps": _optional_numeric_field(trade.get("spread_bps")),
        "macro_indicator": _optional_numeric_field(trade.get("macro_indicator")),
        "tp1_partial": tp1_flag,
        "tp2_partial": tp2_flag,
        "pnl_tp1": _optional_numeric_field(0.0),
        "pnl_tp2": _optional_numeric_field(0.0),
        "size_tp1": _optional_numeric_field(0.0),
        "size_tp2": _optional_numeric_field(0.0),
        "notional_tp1": _optional_numeric_field(0.0),
        "notional_tp2": _optional_numeric_field(0.0),
    }

    if tp1_flag:
        pnl_tp1_val = trade.get("pnl_tp1")
        if pnl_tp1_val is None:
            pnl_tp1_val = pnl_val
        size_tp1_val = trade.get("size_tp1")
        if size_tp1_val is None:
            size_tp1_val = quantity
        notional_tp1_val = trade.get("notional_tp1")
        if notional_tp1_val is None and entry_val is not None and size_tp1_val is not None:
            try:
                notional_tp1_val = float(size_tp1_val) * entry_val
            except Exception:
                notional_tp1_val = None
        if notional_tp1_val is None:
            notional_tp1_val = notional
        size_tp1_display = size_tp1_val
        row["pnl_tp1"] = _optional_numeric_field(pnl_tp1_val)
        row["size_tp1"] = _optional_numeric_field(size_tp1_display)
        row["notional_tp1"] = _optional_numeric_field(notional_tp1_val)
    if tp2_flag:
        pnl_tp2_val = trade.get("pnl_tp2")
        if pnl_tp2_val is None:
            pnl_tp2_val = pnl_val
        size_tp2_val = trade.get("size_tp2")
        if size_tp2_val is None:
            size_tp2_val = quantity
        notional_tp2_val = trade.get("notional_tp2")
        if notional_tp2_val is None and entry_val is not None and size_tp2_val is not None:
            try:
                notional_tp2_val = float(size_tp2_val) * entry_val
            except Exception:
                notional_tp2_val = None
        if notional_tp2_val is None:
            notional_tp2_val = notional
        size_tp2_display = size_tp2_val
        row["pnl_tp2"] = _optional_numeric_field(pnl_tp2_val)
        row["size_tp2"] = _optional_numeric_field(size_tp2_display)
        row["notional_tp2"] = _optional_numeric_field(notional_tp2_val)
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
        try:
            with open(TRADE_HISTORY_FILE, "r", encoding="utf-8") as f:
                first_line = f.readline()
        except OSError as exc:  # pragma: no cover - filesystem specific
            logger.warning(
                "Unable to inspect trade history header %s: %s", TRADE_HISTORY_FILE, exc
            )
            first_line = ""
        if _header_is_compatible(first_line, headers, require_essential=True):
            header_needed = False
            existing_headers = _parse_header_columns(first_line)
            if existing_headers:
                headers = existing_headers
                try:
                    header_rename_map = build_rename_map(existing_headers)
                except Exception:
                    header_rename_map = {col: col for col in existing_headers}
                canonical_headers = {
                    header_rename_map.get(col, col) for col in existing_headers
                }
                if "pattern" not in canonical_headers:
                    try:
                        _consolidate_trade_history_file()
                    except Exception:  # pragma: no cover - best effort repair
                        logger.exception(
                            "Failed to backfill missing pattern column in %s", TRADE_HISTORY_FILE
                        )
                    else:
                        try:
                            with open(TRADE_HISTORY_FILE, "r", encoding="utf-8") as f:
                                first_line = f.readline()
                        except OSError as exc:  # pragma: no cover - filesystem specific
                            logger.warning(
                                "Unable to re-read trade history header %s after consolidation: %s",
                                TRADE_HISTORY_FILE,
                                exc,
                            )
                            first_line = ""
                        if _header_is_compatible(first_line, TRADE_HISTORY_HEADERS):
                            existing_headers = _parse_header_columns(first_line)
                            headers = existing_headers
                            try:
                                header_rename_map = build_rename_map(existing_headers)
                            except Exception:
                                header_rename_map = {col: col for col in existing_headers}
                            canonical_headers = {
                                header_rename_map.get(col, col) for col in existing_headers
                            }
                        else:
                            canonical_headers = set()
                    if "pattern" not in canonical_headers:
                        backup_path = _archive_legacy_history_file(TRADE_HISTORY_FILE)
                        if backup_path is None:
                            try:
                                with open(TRADE_HISTORY_FILE, "w", encoding="utf-8") as f:
                                    f.truncate(0)
                            except OSError as exc:  # pragma: no cover - filesystem specific
                                logger.exception(
                                    "Failed to reset legacy trade log %s: %s", TRADE_HISTORY_FILE, exc
                                )
                                raise
                        file_exists = False
                        header_needed = True
                        headers = list(TRADE_HISTORY_HEADERS)
                        existing_headers = []
                        header_rename_map = {}
        else:
            backup_path = _archive_legacy_history_file(TRADE_HISTORY_FILE)
            if backup_path is None:
                try:
                    with open(TRADE_HISTORY_FILE, "w", encoding="utf-8") as f:
                        f.truncate(0)
                except OSError as exc:  # pragma: no cover - filesystem specific
                    logger.exception(
                        "Failed to reset legacy trade log %s: %s", TRADE_HISTORY_FILE, exc
                    )
                    raise
            file_exists = False
            header_needed = True

    if header_needed:
        header_rename_map = {col: col for col in headers}
        for key in trade.keys():
            if key not in header_rename_map:
                headers.append(key)
                header_rename_map[key] = key

    if not header_rename_map:
        header_rename_map = {col: col for col in headers}

    os.makedirs(os.path.dirname(TRADE_HISTORY_FILE), exist_ok=True)
    final_row: dict[str, object] = {}
    for column in headers:
        canonical = header_rename_map.get(column, column)
        value = None
        if canonical in row:
            value = row[canonical]
        elif column in row:
            value = row[column]
        elif canonical in trade:
            value = trade.get(canonical)
        elif column in trade:
            value = trade.get(column)

        if value is None:
            final_row[column] = MISSING_VALUE
        elif isinstance(value, (int, float, bool)):
            final_row[column] = value
        elif column in row or canonical in row:
            final_row[column] = value
        else:
            final_row[column] = _serialise_extra_field(value)

    df_row = pd.DataFrame([{col: final_row.get(col, MISSING_VALUE) for col in headers}], columns=headers)
    df_row.to_csv(
        TRADE_HISTORY_FILE,
        mode="a",
        header=header_needed,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    try:
        _consolidate_trade_history_file()
    except Exception:  # pragma: no cover - consolidation best effort
        logger.exception("Failed to consolidate trade history after append")

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

    df = df.replace(MISSING_VALUE, pd.NA)
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
            notional = float(row.get("notional", entry * qty if entry and qty else 0))
            return pd.Series(
                {
                    "_gross_pnl": pnl,
                    "_size": qty,
                    "_notional": notional,
                }
            )
        except Exception:
            return pd.Series({"_gross_pnl": 0.0, "_size": 0.0, "_notional": 0.0})

    df = df.join(df.apply(_calc_fields, axis=1))

    def _collapse(group: pd.DataFrame) -> pd.Series:
        def _cost_increments(values: pd.Series) -> pd.Series:
            if values is None or values.empty:
                return pd.Series(0.0, index=group.index)

            numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)

            has_final_row = False
            if "outcome" in group:
                outcomes = group["outcome"].astype(str)
                has_final_row = (~outcomes.str.contains("_partial", na=False)).any()

            if not has_final_row:
                # Only partial rows are present; treat recorded values as per-leg costs.
                return pd.Series(numeric.to_numpy(dtype=float), index=values.index)

            increments: list[float] = []
            cumulative_max = 0.0
            for idx, val in numeric.items():
                if val >= cumulative_max:
                    inc = val - cumulative_max
                    cumulative_max = val
                else:
                    inc = val
                    cumulative_max += inc
                increments.append(inc)

            return pd.Series(increments, index=values.index, dtype=float)

        fees_increments = (
            _cost_increments(group["fees"]) if "fees" in group else pd.Series(0.0, index=group.index)
        )
        slippage_increments = (
            _cost_increments(group["slippage"])
            if "slippage" in group
            else pd.Series(0.0, index=group.index)
        )

        net_pnl_rows = group["_gross_pnl"] - fees_increments - slippage_increments
        pnl_total = net_pnl_rows.sum()
        size_total = group["_size"].sum()
        notional_total = group["_notional"].sum()
        final = group[~group["outcome"].astype(str).str.contains("_partial", na=False)]
        if final.empty:
            final = group.tail(1)
        row = final.iloc[0].copy()
        row["pnl"] = pnl_total
        total_fees = float(fees_increments.sum()) if not fees_increments.empty else 0.0
        total_slippage = (
            float(slippage_increments.sum()) if not slippage_increments.empty else 0.0
        )
        if "fees" in row or total_fees:
            row["fees"] = total_fees
        if "slippage" in row or total_slippage:
            row["slippage"] = total_slippage
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
        row["pnl_tp1"] = net_pnl_rows.loc[tp1_rows.index].sum()
        row["pnl_tp2"] = net_pnl_rows.loc[tp2_rows.index].sum()
        row["size_tp1"] = tp1_rows["_size"].sum()
        row["size_tp2"] = tp2_rows["_size"].sum()
        row["notional_tp1"] = tp1_rows["_notional"].sum()
        row["notional_tp2"] = tp2_rows["_notional"].sum()
        return row

    collapsed = (
        df.groupby(key_cols, group_keys=False).apply(_collapse).reset_index(drop=True)
    )
    return collapsed.drop(columns=["_gross_pnl", "_size", "_notional"], errors="ignore")


def _consolidate_trade_history_file(path: Optional[str] = None) -> None:
    """Rewrite the trade history file with de-duplicated rows.

    The live system occasionally records multiple rows for the same ``trade_id``
    when partial exits or repeated status updates are logged. Dashboard readers
    already collapse those duplicates during :func:`load_trade_history_df`, but
    opening the raw CSV still shows the redundant entries. By reloading the file
    with the same normalisation pipeline and writing it back we ensure the
    persisted history matches what downstream tools display without requiring
    every consumer to run de-duplication logic.
    """

    history_path = path or TRADE_HISTORY_FILE
    if not history_path:
        return

    df = load_trade_history_df(path=history_path)
    # ``load_trade_history_df`` returns an empty frame with the expected columns
    # when the file is missing or unreadable. In that case we still want to
    # ensure the on-disk file contains a header so future appends remain
    # aligned.
    ordered_cols = [col for col in TRADE_HISTORY_HEADERS if col in df.columns]
    extra_cols = [col for col in df.columns if col not in ordered_cols]
    all_columns = ordered_cols + extra_cols

    time_columns = [
        col
        for col in ("timestamp", "entry_time", "exit_time")
        if col in df.columns
    ]
    for col in time_columns:
        try:
            df[col] = df[col].apply(
                lambda ts: (
                    ts.isoformat().replace("+00:00", "Z")
                    if hasattr(ts, "isoformat")
                    else str(ts)
                )
                if pd.notna(ts)
                else MISSING_VALUE
            )
        except Exception:
            # If the column is not datetime-like we leave it untouched.
            pass

    tmp_path = f"{history_path}.tmp"
    try:
        if df.empty:
            with open(tmp_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=all_columns or TRADE_HISTORY_HEADERS)
                writer.writeheader()
        else:
            df = df.reindex(columns=all_columns)
            df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, history_path)
    except FileNotFoundError:
        # Nothing to consolidate yet; clean up any partial temp file.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_history_frame(path: str) -> pd.DataFrame:
    """Read a trade history CSV from ``path`` using tolerant parsing."""

    if not path or not (os.path.exists(path) and os.path.getsize(path) > 0):
        return pd.DataFrame()

    try:
        with open(path, "r", encoding="utf-8") as fh:
            header_line = fh.readline()
    except OSError as exc:  # pragma: no cover - filesystem specific
        logger.warning("Unable to inspect trade log %s: %s", path, exc)
        return pd.DataFrame()

    if not _header_is_compatible(header_line, TRADE_HISTORY_HEADERS):
        logger.warning(
            "Skipping legacy trade log with unexpected header: %s", path
        )
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
            df = pd.DataFrame(columns=TRADE_HISTORY_COLUMNS)

    if df.empty:
        return df

    # ------------------------------------------------------------------
    # Normalise headers using the shared schema helpers
    # ------------------------------------------------------------------
    df = normalise_history_columns(df)

    if "llm_approval" not in df.columns and "llm_decision" in df.columns:
        decisions = df["llm_decision"].astype(str).str.lower()
        positive = {"true", "1", "yes", "approved", "y"}
        negative = {"false", "0", "no", "vetoed", "n"}
        bool_mask = decisions.isin(positive | negative)
        if bool_mask.any():
            approved_mask = decisions.isin(positive)
            df.loc[bool_mask, "llm_approval"] = approved_mask.loc[bool_mask]
            df.loc[bool_mask, "llm_decision"] = approved_mask.loc[bool_mask].map(
                {True: "approved", False: "vetoed"}
            )

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

    for col in ("tp1_partial", "tp2_partial"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes", "y"])
            )
        else:
            df[col] = False

    if "llm_approval" in df.columns:
        approval_raw = df["llm_approval"].astype(str).str.lower()
        positive = {"true", "1", "yes", "approved", "y"}
        negative = {"false", "0", "no", "vetoed", "n"}
        approval_series = pd.Series(pd.NA, index=df.index, dtype="object")
        approval_series.loc[approval_raw.isin(positive)] = True
        approval_series.loc[approval_raw.isin(negative)] = False
        df["llm_approval"] = approval_series

    numeric_columns = [
        "entry",
        "exit",
        "size",
        "notional",
        "fees",
        "slippage",
        "pnl",
        "pnl_pct",
        "confidence",
        "btc_dominance",
        "fear_greed",
        "sentiment_confidence",
        "score",
        "technical_indicator_score",
        "llm_confidence",
        "volatility",
        "htf_trend",
        "order_imbalance",
        "order_flow_score",
        "order_flow_flag",
        "cvd",
        "cvd_change",
        "taker_buy_ratio",
        "trade_imbalance",
        "aggressive_trade_rate",
        "spoofing_intensity",
        "spoofing_alert",
        "volume_ratio",
        "price_change_pct",
        "spread_bps",
        "macro_indicator",
        "pnl_tp1",
        "pnl_tp2",
        "size_tp1",
        "size_tp2",
        "notional_tp1",
        "notional_tp2",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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

    llm_err = df.get("llm_error", pd.Series(dtype=str)).astype(str).str.lower()

    if "llm_approval" in df.columns:
        approval_series = df["llm_approval"]
        approved_mask = (llm_err != "true") & approval_series.eq(True).fillna(False)
        vetoed_mask = (llm_err != "true") & approval_series.eq(False).fillna(False)
    else:
        llm_dec = df.get("llm_decision", pd.Series(dtype=str)).astype(str).str.lower()
        approved_mask = (llm_err != "true") & llm_dec.isin(["true", "1", "yes", "approved"])
        vetoed_mask = (llm_err != "true") & llm_dec.isin(["false", "0", "no", "vetoed"])
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
