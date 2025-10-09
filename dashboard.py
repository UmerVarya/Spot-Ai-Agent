"""
Enhanced Streamlit dashboard for the Spot AI Super Agent.

This version extends the original ``dashboard.py`` by fixing several
calculation inconsistencies and adding additional context for open
positions.  In particular, the unrealised profit/loss (PnL) is now
computed against the notional value of each trade rather than the raw
position size.  The dashboard introduces an optional ``SIZE_AS_NOTIONAL``
environment variable which, when set to ``true``, treats the ``size``
field on each trade as a dollar (or quote currency) notional rather
than a token quantity.  If this flag is false (the default), ``size``
continues to represent the quantity of the asset purchased.  In either
case the application calculates a derived quantity and notional for
each trade to ensure consistent PnL accounting.

To aid in debugging and performance analysis, the live table includes
the notional column by default.  Historical analytics are left
unchanged except for improved handling of missing ``notional`` values.
Additional summary metrics such as profit factor and average trade PnL
could be added in future iterations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import csv
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from log_utils import setup_logger, LOG_FILE
from trade_schema import TRADE_HISTORY_COLUMNS, normalise_history_columns
from backtest import compute_buy_and_hold_pnl, generate_trades_from_ohlcv
from ml_model import train_model
import requests

BINANCE_FEE_RATE = 0.00075

try:
    import altair as alt  # type: ignore
except Exception:  # pragma: no cover
    alt = None


def numcol(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    """Return numeric Series for column `name` (or an aligned NaN Series if missing)."""

    if name in df.columns:
        col = df[name]
        if isinstance(col, pd.DataFrame):  # duplicate headers
            col = col.iloc[:, 0]
        s = pd.to_numeric(col, errors="coerce")
    else:
        s = pd.Series(default, index=df.index, dtype="float64")
    # sanitize weird values without using .replace on a scalar
    s = s.where(np.isfinite(s))  # turn inf/-inf into NaN
    return s

# Ensure environment variables are loaded once
import config

# Optional Binance client for live prices
try:
    from binance.client import Client  # type: ignore
except Exception:
    class Client:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "python-binance library not installed; cannot fetch live prices."
            )

from streamlit_autorefresh import st_autorefresh


def _arrow_safe_scalar(value):
    """Return a JSON/Arrow serialisable representation of ``value``.

    ``pyarrow`` (used internally by ``st.dataframe``) relies on json encoding for
    certain pieces of pandas metadata.  Numpy scalar types such as
    :class:`numpy.int64` are not JSON serialisable which results in runtime
    errors when Streamlit attempts to display the data frame.  Converting those
    scalars to their native Python equivalents keeps the values intact while
    satisfying Arrow's encoder.
    """

    if isinstance(value, np.generic):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _arrow_safe_index(index: pd.Index) -> pd.Index:
    """Return ``index`` with numpy scalar labels converted to Python scalars."""

    if isinstance(index, pd.MultiIndex):
        tuples = [
            tuple(_arrow_safe_scalar(level) for level in labels)
            for labels in index.tolist()
        ]
        names = [
            (_arrow_safe_scalar(name) if name is not None else None)
            for name in index.names
        ]
        return pd.MultiIndex.from_tuples(tuples, names=names)
    if isinstance(index, pd.RangeIndex):
        start = _arrow_safe_scalar(index.start)
        stop = _arrow_safe_scalar(index.stop)
        step = _arrow_safe_scalar(index.step)
        name = index.name
        if name is not None:
            name = _arrow_safe_scalar(name)
        return pd.RangeIndex(start=start, stop=stop, step=step, name=name)
    name = index.name
    if name is not None:
        name = _arrow_safe_scalar(name)
    return pd.Index([_arrow_safe_scalar(x) for x in index.tolist()], name=name)


def arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy of ``df`` safe for Streamlit/Arrow display."""

    if df is None:
        return df
    safe = df.copy()
    if isinstance(safe.columns, pd.MultiIndex):
        tuples = [
            tuple(_arrow_safe_scalar(level) for level in labels)
            for labels in safe.columns.tolist()
        ]
        names = [
            (_arrow_safe_scalar(name) if name is not None else None)
            for name in safe.columns.names
        ]
        safe.columns = pd.MultiIndex.from_tuples(tuples, names=names)
    else:
        safe.columns = [_arrow_safe_scalar(col) for col in safe.columns]
    safe.index = _arrow_safe_index(safe.index)
    if not safe.empty:
        safe = safe.applymap(_arrow_safe_scalar)
    return safe


def _fmt_money(val) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"${float(val):,.2f}"
    except Exception:
        return str(val)


def _fmt_price(val) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"{float(val):,.6f}"
    except Exception:
        return str(val)


def _fmt_quantity(val) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"{float(val):,.4f}"
    except Exception:
        return str(val)


def _fmt_percent(val) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"{float(val):.2f}%"
    except Exception:
        return str(val)


def _fmt_score(val) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"{float(val):.2f}"
    except Exception:
        return str(val)


def _fmt_duration(val) -> str:
    if pd.isna(val):
        return ""
    try:
        return f"{float(val):.1f}"
    except Exception:
        return str(val)


def _highlight_position_limit(val) -> str:
    if pd.isna(val) or not np.isfinite(MAX_POSITION_USD):
        return ""
    try:
        return (
            "background-color: rgba(255, 87, 51, 0.25);"
            if float(val) > MAX_POSITION_USD + 1e-6
            else ""
        )
    except Exception:
        return ""


PRIMARY = os.getenv(
    "TRADE_HISTORY_FILE",
    "/home/ubuntu/spot_data/trades/historical_trades.csv",
)

# legacy fallbacks that may contain older rows
LEGACY = [
    os.getenv("COMPLETED_TRADES_FILE", ""),          # legacy env alias
    "/home/ubuntu/spot_data/trades/completed_trades.csv",  # previous default path
    "/home/ubuntu/spot_data/completed_trades.csv",   # very old default path
]

PARSE_ERROR_TOKEN = "[parse error]"
_RECOVER_TEXT_COLUMNS = {"llm_error", "narrative", "outcome_desc"}


def _repair_bad_row(fields: list[str], headers: list[str]) -> list[str]:
    """Return ``fields`` padded to ``headers`` while flagging parse issues."""

    expected = len(headers)
    if expected == 0:
        return list(fields)
    result = [PARSE_ERROR_TOKEN] * expected
    total_fields = len(fields)
    pointer = 0
    idx = 0
    while idx < expected and pointer < total_fields:
        column = headers[idx]
        remaining_cols = expected - idx
        remaining_fields = total_fields - pointer
        if column in _RECOVER_TEXT_COLUMNS and remaining_fields > remaining_cols:
            tail_cols = expected - idx - 1
            value_end = total_fields - tail_cols
            if value_end <= pointer:
                value_end = pointer + 1
            value_parts = fields[pointer:value_end]
            recovered = ",".join(value_parts).strip()
            result[idx] = (
                f"{PARSE_ERROR_TOKEN}: {recovered}"
                if recovered
                else PARSE_ERROR_TOKEN
            )
            pointer = value_end
        else:
            result[idx] = fields[pointer]
            pointer += 1
        idx += 1
    while idx < expected and pointer < total_fields:
        result[idx] = fields[pointer]
        pointer += 1
        idx += 1
    return result


def _relaxed_split(line: str) -> list[str]:
    """Split ``line`` treating stray quotes as literal characters."""

    fields: list[str] = []
    current: list[str] = []
    in_quotes = False
    i = 0
    length = len(line)
    while i < length:
        ch = line[i]
        if ch == '"':
            if in_quotes:
                next_char = line[i + 1] if i + 1 < length else ""
                if next_char == '"':
                    current.append('"')
                    i += 1
                elif next_char in {",", "\n", "\r"}:
                    in_quotes = False
                else:
                    current.append('"')
            else:
                in_quotes = True
        elif ch == "," and not in_quotes:
            fields.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    fields.append("".join(current).rstrip("\r\n"))
    return fields


def _read_csv_with_recovery(path: Path) -> tuple[pd.DataFrame, int]:
    """Read ``path`` tolerating malformed rows.

    Returns the DataFrame and the count of recovered rows.
    """

    recovered_rows = 0
    headers: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            header_line = handle.readline()
        if header_line:
            headers = next(csv.reader([header_line]))
    except Exception:
        headers = []

    def _handle_bad_line(fields: list[str]) -> list[str]:
        nonlocal recovered_rows
        recovered_rows += 1
        return _repair_bad_row(fields, headers)

    on_bad_lines: object
    if headers:
        on_bad_lines = _handle_bad_line
    else:
        on_bad_lines = "error"

    try:
        df = pd.read_csv(
            str(path),
            engine="python",
            on_bad_lines=on_bad_lines,
        )
    except pd.errors.ParserError:
        if not headers:
            return pd.DataFrame(), recovered_rows
        df, manual_recovered = _manual_csv_recovery(path, headers)
        recovered_rows += manual_recovered
        return df, recovered_rows

    if headers and df.empty:
        with path.open("r", encoding="utf-8") as handle:
            next(handle, "")
            has_data = any(line.strip() for line in handle)
        if has_data:
            df, manual_recovered = _manual_csv_recovery(path, headers)
            recovered_rows += manual_recovered
    return df, recovered_rows


def _manual_csv_recovery(path: Path, headers: list[str]) -> tuple[pd.DataFrame, int]:
    rows: list[list[str]] = []
    manual_recovered = 0
    with path.open("r", encoding="utf-8") as handle:
        next(handle, "")
        for raw in handle:
            if not raw.strip():
                continue
            try:
                parsed = next(csv.reader([raw]))
            except csv.Error:
                parsed = _relaxed_split(raw)
            manual_recovered += 1
            rows.append(_repair_bad_row(parsed, headers))
    if not rows:
        return pd.DataFrame(columns=headers), manual_recovered
    return pd.DataFrame(rows, columns=headers), manual_recovered


@st.cache_data(ttl=30)
def _read_history_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return pd.DataFrame()
    try:
        df, recovered_rows = _read_csv_with_recovery(p)
    except Exception as e:
        st.warning(f"Could not read {p}: {e}")
        return pd.DataFrame()
    if recovered_rows:
        st.warning(
            f"Recovered {recovered_rows} malformed row(s) while reading {p.name}."
        )
    df = normalise_history_columns(df)
    # normalize types that downstream filters expect
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # many dashboards expect a simple OUTCOME field; normalize to upper strings
    if "outcome" in df.columns:
        df["outcome"] = df["outcome"].astype(str).str.upper()
    # keep only the columns we actually use in charts/tables (optional but safer)
    preferred = list(dict.fromkeys(TRADE_HISTORY_COLUMNS + ["entry_price", "exit_price"]))
    ordered_cols = [c for c in preferred if c in df.columns]
    if ordered_cols:
        remainder = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + remainder]
    return df

@st.cache_data(ttl=30)
def load_trade_history_df() -> pd.DataFrame:
    frames = [_read_history_frame(PRIMARY)]
    for lp in LEGACY:
        if lp and Path(lp).as_posix() != Path(PRIMARY).as_posix():
            frames.append(_read_history_frame(lp))
    # merge, drop dupes on trade_id+exit_time (tweak if your schema differs)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ("trade_id", "exit_time"):
        if col not in df.columns:
            df[col] = ""
    df = df.drop_duplicates(subset=["trade_id","exit_time"], keep="last")
    try:
        df = _deduplicate_history(df)
    except Exception:
        pass
    if "outcome" in df.columns:
        df = df[df["outcome"].astype(str).str.lower() != "open"]
    sort_cols = [col for col in ("exit_time", "timestamp") if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.reset_index(drop=True)

# Import risk metrics from trade_utils for historical risk analysis
try:
    # risk_metrics defines sharpe_ratio, calmar_ratio, max_drawdown, value_at_risk, expected_shortfall
    from risk_metrics import (
        sharpe_ratio,
        calmar_ratio,
        max_drawdown,
        value_at_risk,
        expected_shortfall,
    )  # type: ignore
except Exception:
    # Fallback simple implementations in case risk_metrics is unavailable
    def sharpe_ratio(returns, risk_free=0.0):  # type: ignore
        import numpy as _np
        r = _np.asarray(list(returns), dtype=float)
        if len(r) < 2:
            return float("nan")
        excess = r - risk_free
        mean = excess.mean()
        std = excess.std(ddof=1)
        return float(_np.sqrt(252) * mean / std) if std != 0 else float("nan")

    def max_drawdown(equity_curve):  # type: ignore
        import numpy as _np
        equity = _np.asarray(list(equity_curve), dtype=float)
        if len(equity) == 0:
            return float("nan")
        cum_max = _np.maximum.accumulate(equity)
        drawdowns = (equity - cum_max) / cum_max
        return float(drawdowns.min())

    def calmar_ratio(returns):  # type: ignore
        import numpy as _np
        r = _np.asarray(list(returns), dtype=float)
        if len(r) < 2:
            return float("nan")
        total_return = _np.prod(1 + r) - 1
        annualised_return = (1 + total_return) ** (252 / len(r)) - 1
        mdd = abs(max_drawdown(_np.cumprod(1 + r)))
        return float(annualised_return / mdd) if mdd != 0 else float("nan")

    def value_at_risk(returns, alpha=0.05):  # type: ignore
        import numpy as _np
        r = _np.sort(_np.asarray(list(returns), dtype=float))
        if len(r) == 0:
            return float("nan")
        idx = int(alpha * len(r))
        return float(r[idx])

    def expected_shortfall(returns, alpha=0.05):  # type: ignore
        import numpy as _np
        r = _np.sort(_np.asarray(list(returns), dtype=float))
        if len(r) == 0:
            return float("nan")
        idx = int(alpha * len(r))
        tail = r[:idx]
        if len(tail) == 0:
            return float("nan")
        return float(tail.mean())

# Load API keys from environment
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
try:
    client = Client(api_key, api_secret)
except Exception:
    client = None

# Import shared paths after environment variables are loaded
from trade_storage import (
    load_active_trades,
    log_trade_result,
    TRADE_HISTORY_FILE,
    ACTIVE_TRADES_FILE,
    _deduplicate_history,
)
from notifier import REJECTED_TRADES_FILE
from trade_logger import TRADE_LEARNING_LOG_FILE

# Determine how to interpret the trade "size" field.  When set to true,
# each trade's ``size`` is assumed to represent notional in quote
# currency (e.g., USDT).  When false, ``size`` represents the token
# quantity.  See ``format_active_row`` for details.
#
# Default to ``true`` because most users allocate a fixed dollar amount per
# trade rather than specifying a raw token quantity.  You can override
# this behaviour by setting the ``SIZE_AS_NOTIONAL`` environment variable to
# "false".
SIZE_AS_NOTIONAL = os.getenv("SIZE_AS_NOTIONAL", "true").lower() == "true"

# Endpoint for live position data.  ``LIVE_POSITIONS_ENDPOINT`` takes
# precedence, otherwise we attempt to build the endpoint from several
# commonly used base URL variables.
LIVE_POSITIONS_ENDPOINT = os.getenv("LIVE_POSITIONS_ENDPOINT")
if not LIVE_POSITIONS_ENDPOINT:
    for env_name in (
        "TRADE_API_BASE_URL",
        "TRADE_API_URL",
        "BACKEND_API_URL",
        "BACKEND_URL",
        "API_BASE_URL",
    ):
        base_url = os.getenv(env_name)
        if base_url:
            LIVE_POSITIONS_ENDPOINT = base_url.rstrip("/") + "/api/trade/positions/live"
            break

try:
    LIVE_POSITIONS_TIMEOUT = float(os.getenv("LIVE_POSITIONS_TIMEOUT", "5"))
except (TypeError, ValueError):
    LIVE_POSITIONS_TIMEOUT = 5.0


def _normalise_live_positions_payload(payload) -> list[dict]:
    """Return a list of trade dictionaries from ``payload``.

    The live positions API can return the positions directly as a list or
    wrapped under a variety of keys (``data``, ``positions``, ``items``).
    Some backends expose a mapping of ``symbol -> data``.  This helper
    normalises those cases into a uniform ``list[dict]``.
    """

    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        # Common nesting patterns: {"data": {...}}, {"positions": [...]}, etc.
        possible_keys = ("positions", "data", "items", "results")
        for key in possible_keys:
            if key not in payload:
                continue
            value = payload[key]
            # Some APIs wrap the actual list under ``data`` again
            if isinstance(value, dict):
                nested = _normalise_live_positions_payload(value)
                if nested:
                    return nested
                continue
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        # If the response itself is a mapping of symbol -> trade data.
        dict_values = [v for v in payload.values() if isinstance(v, dict)]
        if dict_values:
            return dict_values
    return []


def _status_token_is_open(value) -> bool:
    """Return ``True`` when ``value`` represents an open trade state."""

    if value is None:
        return False
    if isinstance(value, bool):
        return False
    token = str(value).strip().lower()
    if not token:
        return False
    open_tokens = {"open", "active", "running", "live", "entered"}
    if token in open_tokens:
        return True
    for marker in open_tokens:
        if marker in token:
            return True
    return False


def _status_token_is_closed(value) -> bool:
    """Return ``True`` when ``value`` represents a closed trade state."""

    if value is None:
        return False
    if isinstance(value, bool):
        # Boolean flags alone do not convey whether a trade is closed.  Some backends
        # expose ``status.open`` or ``status.active`` booleans where ``True`` means the
        # trade is running, while others provide ``status.closed``.  The surrounding
        # heuristics inspect the key names to interpret these values, so treat the bare
        # boolean as "unknown" here to avoid false positives.
        return False
    token = str(value).strip().lower()
    if not token:
        return False
    try:
        numeric_token = float(token)
    except ValueError:
        pass
    else:
        if numeric_token == 0.0:
            return True
    closed_tokens = {
        "closed",
        "complete",
        "completed",
        "exited",
        "exit",
        "flat",
        "inactive",
        "inactive_position",
        "inactive-trade",
        "stopped",
        "cancelled",
        "canceled",
        "finished",
        "done",
        "closed_out",
        "hit_sl",
        "stop_hit",
    }
    if token in closed_tokens:
        return True
    if _status_token_is_open(token):
        return False
    for marker in closed_tokens:
        if marker in token:
            return True
    return False


def _status_value_indicates_closed(value, key_hint: str | None = None) -> bool:
    """Return ``True`` when ``value`` conveys a closed state in context."""

    if value is None:
        return False
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            if _status_value_indicates_closed(nested_value, nested_key):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_status_value_indicates_closed(item, key_hint) for item in value)
    if isinstance(value, bool):
        if not value:
            return False
        if not key_hint:
            return False
        token = str(key_hint).strip().lower()
        if not token:
            return False
        if token.endswith("_closed") or token.startswith("is_closed"):
            return True
        if _status_token_is_closed(token):
            return True
        return False
    if isinstance(value, (int, float)):
        if float(value) != 0.0:
            return False
        if not key_hint:
            return False
        token = str(key_hint).strip().lower()
        if not token:
            return False
        numeric_closure_tokens = {
            "status",
            "state",
            "trade_status",
            "position_status",
            "status_code",
            "code",
            "exit_code",
            "position_state",
        }
        if (
            token in numeric_closure_tokens
            or token.endswith("_status")
            or token.endswith("_state")
            or token.endswith("_code")
            or _status_token_is_closed(token)
        ):
            return True
        return False
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return False
        try:
            numeric = float(token)
        except ValueError:
            return _status_token_is_closed(token)
        return _status_value_indicates_closed(numeric, key_hint)
    return _status_token_is_closed(value)


def _status_value_is_truthy(value) -> bool:
    """Return ``True`` when ``value`` affirms a closed status."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return False
        return True
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return False
        lower = token.lower()
        explicit_negatives = {"false", "no", "off"}
        if lower in explicit_negatives:
            return False
        try:
            numeric_token = float(token)
        except ValueError:
            return True
        else:
            if numeric_token != numeric_token:
                return False
            return True
    return bool(value)


def _status_value_is_explicitly_false(value) -> bool:
    """Return ``True`` when ``value`` explicitly negates an open status."""

    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return False
        lower = token.lower()
        explicit_negatives = {"false", "no", "off"}
        if lower in explicit_negatives:
            return True
        try:
            return float(token) == 0.0
        except ValueError:
            return False
    return False


def _is_trade_closed(trade: dict) -> bool:
    """Return ``True`` if ``trade`` should be considered closed."""

    if not isinstance(trade, dict):
        return True
    # Direct boolean hints
    for key in ("is_open", "open", "active"):
        if key in trade:
            hint = trade[key]
            if isinstance(hint, str):
                hint_bool = to_bool(hint)
                # ``to_bool`` only returns True for positive affirmations.  When
                # it returns False we need to differentiate between an explicit
                # negative and "unknown".  A non-empty token implies an explicit
                # negation.
                if not hint_bool and hint.strip():
                    return True
                if hint_bool:
                    return False
            elif hint is False:
                return True
            elif hint is True:
                return False
    status_field = trade.get("status")
    if isinstance(status_field, dict):
        for key, value in status_field.items():
            if _status_token_is_open(key) and _status_value_is_explicitly_false(value):
                return True
            if _status_token_is_closed(key) and _status_value_is_truthy(value):
                return True
            if _status_value_indicates_closed(value, key):
                return True
        # Nested state hints
        for nested_key in ("state", "status", "trade_status"):
            if nested_key in status_field:
                nested_value = status_field[nested_key]
                if _status_value_indicates_closed(nested_value, nested_key):
                    return True
    elif _status_value_indicates_closed(status_field, "status"):
        return True
    # Additional explicit status fields commonly emitted by the agent
    for alt_key in ("state", "position_status", "trade_status", "status_name"):
        alt_value = trade.get(alt_key)
        if _status_value_indicates_closed(alt_value, alt_key):
            return True
    return False


def _trade_identity(trade: dict) -> str:
    """Return a stable identifier for ``trade`` used to track UI changes."""

    if not isinstance(trade, dict):
        return "unknown"
    for key in ("trade_id", "id", "uuid", "order_id"):
        if trade.get(key):
            return str(trade[key])
    symbol = str(trade.get("symbol", "?"))
    entry_time = trade.get("entry_time") or trade.get("timestamp") or "?"
    return f"{symbol}|{entry_time}"


def fetch_live_positions() -> tuple[list[dict], str, str | None]:
    """Return live position data, preferring the API over local storage."""

    if LIVE_POSITIONS_ENDPOINT:
        try:
            response = requests.get(LIVE_POSITIONS_ENDPOINT, timeout=LIVE_POSITIONS_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            trades = _normalise_live_positions_payload(payload)
            if trades:
                return trades, "api", None
            return trades, "api", None
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Live positions API unavailable: %s", exc)
            error = (
                "Live positions API unavailable; falling back to local cache. "
                "Check TRADE_API_BASE_URL or LIVE_POSITIONS_ENDPOINT."
            )
            return load_active_trades(), "storage", error
    return load_active_trades(), "storage", None


def load_open_trades() -> tuple[list[dict], str, str | None, int]:
    """Return active trades with closed entries filtered out."""

    trades, source, error = fetch_live_positions()
    open_trades: list[dict] = []
    filtered_closed = 0
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        if _is_trade_closed(trade):
            filtered_closed += 1
            continue
        open_trades.append(trade)
    return open_trades, source, error, filtered_closed


def _env_float(name: str, default: float) -> float:
    """Return environment variable ``name`` parsed as ``float`` with fallback."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


MAX_POSITION_USD = _env_float("MAX_TRADE_USD", 200.0)

# Page configuration
st.set_page_config(
    page_title="Spot AI Super Agent Dashboard",
    page_icon="",
    layout="wide",
)
logger = setup_logger(__name__)
logger.info(
    "Paths: LOG_FILE=%s TRADE_HISTORY=%s ACTIVE_TRADES=%s REJECTED_TRADES=%s LEARNING_LOG=%s",
    LOG_FILE,
    TRADE_HISTORY_FILE,
    ACTIVE_TRADES_FILE,
    REJECTED_TRADES_FILE,
    TRADE_LEARNING_LOG_FILE,
)

st.title(" Spot AI Super Agent – Live Trade Dashboard")

def get_live_price(symbol: str) -> float:
    """Fetch the current price for a symbol from Binance.  Returns None on failure."""
    try:
        res = client.get_symbol_ticker(symbol=symbol)
        return float(res["price"])
    except Exception:
        return None


def get_price_history(
    symbol: str, interval: str = "15m", limit: int = 50
) -> pd.DataFrame | None:
    """Fetch recent price history for a symbol.

    Returns a dataframe with ``time`` and ``close`` columns or ``None`` if
    data cannot be retrieved.  Requires the Binance client to be available.
    """
    if client is None:
        return None
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except Exception:
        return None
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["open_time", "close"]].rename(columns={"open_time": "time"})


def to_bool(val) -> bool:
    """Convert ``val`` into a boolean, handling loose string inputs."""

    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        token = val.strip().lower()
        if not token:
            return False
        truthy_tokens = {
            "true",
            "1",
            "yes",
            "y",
            "on",
            "enabled",
            "active",
            "hit",
            "triggered",
            "reached",
            "filled",
            "executed",
            "done",
            "achieved",
            "target hit",
            "target reached",
        }
        if token in truthy_tokens:
            return True
        # Normalise composite status strings such as ``tp1_hit`` or ``TP Hit``
        words = {w for w in re.split(r"[\s_:\-]+", token) if w}
        keyword_hits = {
            "hit",
            "triggered",
            "reached",
            "filled",
            "executed",
            "done",
            "achieved",
        }
        if not words & keyword_hits:
            return False
        negation_terms = {
            "not",
            "no",
            "never",
            "none",
            "without",
            "pending",
            "await",
            "awaited",
            "awaiting",
            "wait",
            "waited",
            "waiting",
            "miss",
            "missed",
            "missing",
            "fail",
            "failed",
            "failing",
            "cancel",
            "canceled",
            "cancelled",
            "void",
            "voided",
            "skip",
            "skipped",
            "hold",
            "holding",
            "inactive",
            "halt",
            "halted",
            "blocked",
            "stopped",
            "pause",
            "paused",
            "unfilled",
            "untriggered",
            "unhit",
            "unreached",
            "unmet",
            "undo",
            "undone",
            "notfilled",
            "nottriggered",
            "nothit",
            "notreached",
            "partial",
            "partially",
            "partialfill",
            "partialfilled",
            "tbd",
        }
        if words & negation_terms:
            return False
        negation_pattern = re.compile(
            r"(^|[^a-z])(" +
            "|".join(
                sorted((re.escape(term) for term in negation_terms), key=len, reverse=True)
            ) +
            r")([^a-z]|$)"
        )
        if negation_pattern.search(token):
            return False
        return True
    return bool(val)

def format_active_row(symbol: str, data: dict) -> dict | None:
    """
    Format a single active trade dictionary into a row for display.

    This helper derives the position quantity and notional based on the
    environment variable ``SIZE_AS_NOTIONAL``.  If true, the ``size``
    field is treated as the notional value of the position and the
    quantity is computed as ``size / entry_price``.  If false, ``size``
    is interpreted as the quantity and the notional is ``entry_price * size``.
    Returns ``None`` if mandatory fields are missing or a live price
    cannot be retrieved.
    """
    entry = data.get("entry", data.get("entry_price"))
    direction_raw = data.get("direction")
    direction = str(direction_raw).lower() if direction_raw is not None else None
    sl = data.get("sl")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")
    tp3 = data.get("tp3")
    size_field = data.get("size", data.get("position_size", 0))
    status = data.get("status", {})
    profit_riding = data.get("profit_riding", False)
    current_price = get_live_price(symbol)
    llm_signal = data.get("llm_decision")
    if isinstance(llm_signal, bool):
        llm_signal = "approved" if llm_signal else "vetoed"
    llm_approval = data.get("llm_approval")
    if llm_approval is None and llm_signal is not None:
        token = str(llm_signal).strip().lower()
        if token in {"approved", "true", "yes", "1"}:
            llm_approval = True
        elif token in {"vetoed", "false", "no", "0"}:
            llm_approval = False
    # Validate required fields
    if entry is None or direction is None:
        return None
    try:
        entry_price = float(entry)
    except Exception:
        return None
    if current_price is None:
        # Fall back to any price snapshot stored with the trade data.  The agent
        # currently persists only the entry price for active trades when the
        # Binance client is unavailable (e.g. in offline or testing
        # environments).  Showing the trade with a stale price is preferable to
        # hiding it entirely from the dashboard, so we degrade gracefully by
        # using the last known price and defaulting to the entry price when
        # nothing else is provided.
        fallback_fields = (
            data.get("last_price"),
            data.get("current_price"),
            data.get("price"),
        )
        for candidate in fallback_fields:
            if candidate is None:
                continue
            try:
                current_price = float(candidate)
                break
            except Exception:
                continue
        if current_price is None:
            current_price = entry_price
    try:
        size_val = float(size_field)
    except Exception:
        size_val = 0.0
    # Derive quantity and notional based on configuration
    if SIZE_AS_NOTIONAL:
        # ``size`` represents the dollar notional; derive quantity using the entry price
        qty = size_val / entry_price if entry_price else 0.0
        notional = size_val
    else:
        # ``size`` represents the asset quantity
        qty = size_val
        notional = entry_price * size_val
    # Compute unrealised PnL in dollars and percent
    is_short = direction == "short"
    if not is_short:
        pnl_dollars = (current_price - entry_price) * qty
    else:
        pnl_dollars = (entry_price - current_price) * qty
    # Avoid division by zero when computing percentage PnL
    if notional:
        pnl_percent = (pnl_dollars / abs(notional)) * 100
    else:
        pnl_percent = 0.0
    # Time in trade (minutes)
    entry_time_str = data.get("entry_time") or data.get("timestamp")
    entry_dt = pd.to_datetime(entry_time_str, errors="coerce", utc=True)
    if pd.notna(entry_dt):
        now = pd.Timestamp.utcnow()
        time_delta_min = (now - entry_dt).total_seconds() / 60
    else:
        time_delta_min = None
    # Status flags for each take-profit level and SL
    status_flags: list[str] = []

    def tp_hit(key: str, target_price) -> bool:
        """Determine whether a TP level has been reached or executed."""

        if target_price in (None, ""):
            return False
        try:
            target_value = float(target_price)
        except Exception:
            # Fall back to the stored status flag when the target price cannot
            # be parsed (should not normally happen but keeps the UI resilient)
            return to_bool(status.get(key))
        # Explicit status/partial flags take precedence so the indicator stays
        # green once a partial execution is logged even if price retraces.
        if to_bool(status.get(key)):
            return True
        partial_flag = data.get(f"{key}_partial")
        if partial_flag is not None and to_bool(partial_flag):
            return True
        # Use the current price to infer live status when no explicit flag is
        # present.  A small tolerance keeps float precision noise from causing
        # flickering indicators.
        if current_price is None:
            return False
        try:
            price_value = float(current_price)
        except Exception:
            return False
        tolerance = max(abs(target_value) * 1e-6, 1e-8)
        if not is_short:
            return price_value >= target_value - tolerance
        return price_value <= target_value + tolerance

    def tp_flag(hit: bool | None, label: str) -> str:
        """Return an emoji tag indicating whether a TP level was hit."""

        return ("🟢" if hit else "🔵") + f" {label}"

    for key, label in [("tp1", "TP1"), ("tp2", "TP2"), ("tp3", "TP3")]:
        # Only show flags for targets that exist in the trade data
        if data.get(key) is not None:
            hit = tp_hit(key, data.get(key))
            status_flags.append(tp_flag(hit, label))

    if to_bool(status.get("sl")):
        status_flags.append("🔴 SL")

    # Indicate TP4 profit-riding mode when enabled
    if profit_riding:
        status_flags.append("🚀 TP4 mode (Trailing)")

    status_str = " | ".join(status_flags) if status_flags else "Running"
    approval_label = ""
    if llm_approval is True:
        approval_label = "Approved"
    elif llm_approval is False:
        approval_label = "Vetoed"
    try:
        llm_conf_val = float(data.get("llm_confidence"))
    except Exception:
        llm_conf_val = None
    try:
        tech_score_val = float(data.get("technical_indicator_score"))
    except Exception:
        tech_score_val = None
    exit_reason = data.get("exit_reason", "")

    def _round_price_or_none(value):
        try:
            if value is None or value == "":
                return None
            return round(float(value), 4)
        except Exception:
            return None

    lvn_entry_val = _round_price_or_none(data.get("lvn_entry_level"))
    lvn_stop_val = _round_price_or_none(data.get("lvn_stop"))
    poc_target_val = _round_price_or_none(data.get("poc_target"))
    auction_state_val = str(data.get("auction_state") or "").strip()
    return {
        "Symbol": symbol,
        "Direction": direction_raw if direction_raw is not None else ("short" if is_short else "long"),
        "Entry": round(entry_price, 4),
        "Price": round(current_price, 4),
        "SL": round(sl, 4) if sl else None,
        "TP1": round(tp1, 4) if tp1 else None,
        "TP2": round(tp2, 4) if tp2 else None,
        "TP3": round(tp3, 4) if tp3 else None,
        "Quantity": round(qty, 6),
        "Position Size (USDT)": round(notional, 2),
        "PnL ($)": round(pnl_dollars, 2),
        "PnL (%)": round(pnl_percent, 2),
        "Time in Trade (min)": round(time_delta_min, 1) if time_delta_min is not None else None,
        "Strategy": data.get("strategy", data.get("pattern", "")),
        "Session": data.get("session", ""),
        "Status": status_str,
        "LLM Decision": llm_signal or "",
        "LLM Approval": approval_label,
        "LLM Confidence Score": llm_conf_val,
        "Technical Indicators": tech_score_val,
        "Exit Reason": exit_reason,
        "Auction State": auction_state_val,
        "LVN Entry": lvn_entry_val,
        "LVN Stop": lvn_stop_val,
        "POC Target": poc_target_val,
    }


def compute_llm_decision_stats(active_trades: list[dict]) -> tuple[int, int, int]:
    """Return counts of LLM-approved, vetoed and error trades."""
    approved = vetoed = errors = 0
    error_tokens = {"true", "1", "yes"}
    # Active trades
    for t in active_trades:
        error_flag = str(t.get("llm_error")).lower() in error_tokens
        if error_flag:
            errors += 1
            continue
        approval = t.get("llm_approval")
        if approval is True:
            approved += 1
        elif approval is False:
            vetoed += 1
        else:
            decision_token = str(t.get("llm_decision", "")).strip().lower()
            if decision_token in {"vetoed", "false", "0", "no"}:
                vetoed += 1
            else:
                approved += 1
    # Completed trades
    df = load_trade_history_df()
    if not df.empty:
        err_series = df.get("llm_error", pd.Series(dtype=str)).astype(str).str.lower()
        approval_series = df.get("llm_approval")
        if approval_series is not None:
            ok_mask = ~err_series.isin(error_tokens)
            approved += int((ok_mask & approval_series.eq(True).fillna(False)).sum())
            vetoed += int((ok_mask & approval_series.eq(False).fillna(False)).sum())
            errors += int(err_series.isin(error_tokens).sum())
        else:
            decisions = df.get("llm_decision", pd.Series(dtype=str)).astype(str).str.lower()
            ok_mask = ~err_series.isin(error_tokens)
            errors += int(err_series.isin(error_tokens).sum())
            vetoed += int((ok_mask & decisions.isin({"false", "0", "no", "vetoed"})).sum())
            approved += int((ok_mask & ~decisions.isin({"false", "0", "no", "vetoed"})).sum())
    # Rejected trades (LLM veto or error)
    if os.path.exists(REJECTED_TRADES_FILE) and os.path.getsize(REJECTED_TRADES_FILE) > 0:
        rej_df = pd.read_csv(REJECTED_TRADES_FILE)
        for reason in rej_df.get("reason", []):
            r = str(reason).lower()
            if "llm advisor vetoed trade" in r:
                vetoed += 1
            elif "llm" in r and "error" in r:
                errors += 1
    return approved, vetoed, errors


def render_live_tab() -> None:
    """Render the live trade dashboard tab."""
    # Sidebar controls
    refresh_interval = st.sidebar.slider(
        "⏱️ Refresh Interval (seconds)", 10, 60, 30
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built for  **Spot AI Super Agent**")
    # Auto refresh
    st_autorefresh(interval=refresh_interval * 1000, key="refresh")
    # Load active trades and format into rows
    trades, source, error_message, filtered_closed = load_open_trades()
    if error_message:
        if st.session_state.get("live_positions_error") != error_message:
            st.warning(error_message)
        st.session_state["live_positions_error"] = error_message
    else:
        st.session_state.pop("live_positions_error", None)
    trade_entries: list[tuple[str, dict]] = []
    for data in trades:
        sym = data.get("symbol", "")
        row = format_active_row(sym, data)
        if row:
            trade_entries.append((_trade_identity(data), row))
    active_rows = [row for _, row in trade_entries]
    # Detect closures for realtime feedback
    previous_rows: dict[str, dict] = st.session_state.get("prev_active_rows", {})
    current_rows = {key: row for key, row in trade_entries}
    closed_keys = sorted(set(previous_rows) - set(current_rows))
    for key in closed_keys:
        last_snapshot = previous_rows.get(key, {})
        symbol = last_snapshot.get("Symbol", key)
        pnl_pct = last_snapshot.get("PnL (%)")
        pnl_note = ""
        if pnl_pct is not None and not pd.isna(pnl_pct):
            pnl_note = f" ({pnl_pct:+.2f}%)"
        status_note = last_snapshot.get("Status")
        if status_note and status_note != "Running":
            pnl_note += f" – {status_note}"
        st.success(f"Trade {symbol} closed{pnl_note}.")
    st.session_state["prev_active_rows"] = current_rows
    prev_count = st.session_state.get("prev_active_count")
    st.session_state["prev_active_count"] = len(active_rows)
    # LLM decision statistics
    approved, vetoed, errors = compute_llm_decision_stats(trades)
    total_decisions = approved + vetoed + errors
    if total_decisions:
        st.subheader("🤖 LLM Decision Outcomes")
        c1, c2, c3 = st.columns(3)
        c1.metric("Approved", f"{approved} ({approved / total_decisions:.0%})")
        c2.metric("Vetoed", f"{vetoed} ({vetoed / total_decisions:.0%})")
        c3.metric("Errors", f"{errors} ({errors / total_decisions:.0%})")
    # Display live PnL section
    st.subheader("📈 Live PnL – Active Trades")
    source_label = "live API" if source == "api" else "local cache"
    caption_parts = [f"Active positions source: {source_label}."]
    if filtered_closed:
        caption_parts.append(f"Filtered {filtered_closed} closed trade(s) from feed.")
    st.caption(" ".join(caption_parts))
    if active_rows:
        df_active = pd.DataFrame(active_rows)
        # Summary metrics for active trades
        col1, col2, col3, col4, col5 = st.columns(5)
        metric_kwargs = {}
        if prev_count is not None:
            metric_kwargs["delta"] = len(df_active) - prev_count
        col1.metric("Active Trades", len(df_active), **metric_kwargs)
        avg_pnl_pct = df_active["PnL (%)"].mean() if not df_active.empty else 0.0
        prev_avg = st.session_state.get("prev_avg_pnl", avg_pnl_pct)
        delta_avg = avg_pnl_pct - prev_avg
        col2.metric(
            "Average PnL (%)",
            f"{avg_pnl_pct:.2f}%",
            delta=f"{delta_avg:+.2f}%",
        )
        st.session_state["prev_avg_pnl"] = avg_pnl_pct
        total_unrealised = df_active["PnL ($)"].sum() if not df_active.empty else 0.0
        col3.metric("Total Unrealised PnL", f"${total_unrealised:,.2f}")
        wins_active = (df_active["PnL ($)"] > 0).sum()
        col4.metric("Winning Trades", wins_active)
        total_notional = (
            df_active["Position Size (USDT)"].sum() if not df_active.empty else 0.0
        )
        col5.metric("Total Position Size", f"${total_notional:,.2f}")
        if np.isfinite(MAX_POSITION_USD) and not df_active.empty:
            breaches_active = df_active["Position Size (USDT)"] > (MAX_POSITION_USD + 1e-6)
            if breaches_active.any():
                st.error(
                    f"{breaches_active.sum()} active trade(s) exceed the configured "
                    f"{MAX_POSITION_USD:.0f} USDT cap."
                )
        df_display = arrow_safe_dataframe(df_active.copy())
        formatters = {
            "Entry": _fmt_price,
            "Price": _fmt_price,
            "SL": _fmt_price,
            "TP1": _fmt_price,
            "TP2": _fmt_price,
            "TP3": _fmt_price,
            "LVN Entry": _fmt_price,
            "LVN Stop": _fmt_price,
            "POC Target": _fmt_price,
            "Quantity": _fmt_quantity,
            "Position Size (USDT)": _fmt_money,
            "PnL ($)": _fmt_money,
            "PnL (%)": _fmt_percent,
            "Time in Trade (min)": _fmt_duration,
            "LLM Confidence Score": _fmt_score,
            "Technical Indicators": _fmt_score,
        }
        fmt_subset = {k: v for k, v in formatters.items() if k in df_display.columns}
        df_style = df_display.style.format(fmt_subset)
        if "Position Size (USDT)" in df_display.columns:
            df_style = df_style.applymap(
                _highlight_position_limit, subset=["Position Size (USDT)"]
            )
        st.dataframe(df_style, use_container_width=True)

        # Optional price chart for a selected trade
        selected_symbol = st.selectbox(
            "Select trade for price chart", df_active["Symbol"]
        )
        price_hist = get_price_history(selected_symbol)
        if price_hist is not None:
            trade_info = next(
                (t for t in trades if t.get("symbol") == selected_symbol), {}
            )
            entry_source = trade_info.get("entry") or trade_info.get("entry_price")
            try:
                entry_val = float(entry_source)
            except Exception:
                entry_val = 0.0
            levels = {
                "Entry": entry_val,
                "SL": trade_info.get("sl"),
                "TP1": trade_info.get("tp1"),
                "TP2": trade_info.get("tp2"),
                "TP3": trade_info.get("tp3"),
                "LVN Entry": trade_info.get("lvn_entry_level"),
                "LVN Stop": trade_info.get("lvn_stop"),
                "POC Target": trade_info.get("poc_target"),
            }
            if alt is not None:
                base = (
                    alt.Chart(price_hist)
                    .mark_line(color="white")
                    .encode(x="time:T", y="close:Q")
                )
                rules = []
                for name, val in levels.items():
                    if val is not None:
                        rules.append(
                            alt.Chart(pd.DataFrame({"y": [float(val)], "lbl": [name]}))
                            .mark_rule()
                            .encode(y="y", color=alt.value("red"), tooltip=["lbl", "y"])
                        )
                chart = alt.layer(base, *rules)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.line_chart(price_hist.set_index("time")["close"], use_container_width=True)
            tv_url = f"https://www.tradingview.com/chart?symbol={selected_symbol}"
            st.markdown(f"[Open {selected_symbol} on TradingView]({tv_url})")
        else:
            st.info("Price history unavailable.")
    else:
        st.info("No active trades found.")
    # Load trade history and compute summary statistics
    st.subheader("📊 Historical Performance – Completed Trades")
    hist = load_trade_history_df()
    fallback_sources = sum(1 for p in LEGACY if p)
    sizing_note = (
        "Position sizes are shown as USDT notionals."
        if SIZE_AS_NOTIONAL
        else "Position sizes reflect entry price multiplied by filled quantity."
    )
    limit_note = (
        f" Maximum configured notional per trade: {MAX_POSITION_USD:,.0f} USDT."
        if np.isfinite(MAX_POSITION_USD)
        else ""
    )
    st.caption(
        f"Loaded {len(hist)} completed trades (partial exits collapsed) from "
        f"{Path(PRIMARY).as_posix()} and {fallback_sources} fallback file(s). "
        f"{sizing_note}{limit_note}"
    )

    if hist.empty:
        st.info("No completed trades logged yet.")
    else:
        hist_df = hist.copy()
        entry_col = next((c for c in ("entry", "entry_price") if c in hist_df.columns), None)
        exit_col = next((c for c in ("exit", "exit_price") if c in hist_df.columns), None)
        # Ensure date columns are parsed with timezone awareness
        for col in ["entry_time", "exit_time"]:
            if col in hist_df.columns:
                hist_df[col] = pd.to_datetime(hist_df[col], errors="coerce", utc=True)
        # Derive size column
        if "size" not in hist_df.columns and "position_size" in hist_df.columns:
            hist_df["size"] = hist_df["position_size"].astype(float)
        # Derive notional column if missing
        if "notional" not in hist_df.columns and entry_col and "size" in hist_df.columns:
            size_numeric = pd.to_numeric(hist_df["size"], errors="coerce")
            if SIZE_AS_NOTIONAL:
                hist_df["notional"] = size_numeric
            else:
                entry_numeric = pd.to_numeric(hist_df[entry_col], errors="coerce")
                hist_df["notional"] = entry_numeric * size_numeric
        # Compute PnL absolute and percent
        if "direction" in hist_df.columns:
            directions = hist_df["direction"].astype(str)
        else:
            directions = pd.Series(["long"] * len(hist_df), index=hist_df.index)
        if entry_col:
            entries = numcol(hist_df, entry_col, default=0.0)
        else:
            entries = pd.Series(0.0, index=hist_df.index)
        if exit_col:
            exits = numcol(hist_df, exit_col)
        else:
            exits = pd.Series(0.0, index=hist_df.index)
        raw_sizes = numcol(hist_df, "size", default=0.0).fillna(0)
        if SIZE_AS_NOTIONAL:
            entry_for_qty = numcol(hist_df, entry_col, default=np.nan) if entry_col else pd.Series(np.nan, index=hist_df.index)
            entry_for_qty = entry_for_qty.mask(entry_for_qty == 0)
            if "notional" in hist_df.columns:
                notional_for_qty = numcol(hist_df, "notional")
            else:
                notional_for_qty = raw_sizes
            sizes = (notional_for_qty / entry_for_qty).fillna(0)
        else:
            sizes = raw_sizes
        quantity_series = pd.to_numeric(sizes, errors="coerce").fillna(0.0)
        hist_df["Quantity"] = quantity_series
        notional_series = numcol(hist_df, "notional") if "notional" in hist_df.columns else pd.Series(0.0, index=hist_df.index)
        if notional_series.eq(0).all():
            if SIZE_AS_NOTIONAL:
                notional_series = pd.to_numeric(hist_df.get("size", 0.0), errors="coerce").fillna(0.0)
            else:
                notional_series = entries * quantity_series
        hist_df["Position Size (USDT)"] = pd.to_numeric(notional_series, errors="coerce").fillna(0.0)
        if np.isfinite(MAX_POSITION_USD):
            breaches_hist = hist_df["Position Size (USDT)"] > (MAX_POSITION_USD + 1e-6)
            if breaches_hist.any():
                st.error(
                    f"{breaches_hist.sum()} completed trade(s) exceeded the configured "
                    f"{MAX_POSITION_USD:.0f} USDT cap."
                )
        def _numeric_series(df: pd.DataFrame, name: str) -> pd.Series:
            """Return numeric column ``name`` aligned to ``df.index``.

            Falls back to a zero-filled series when the column is missing and
            gracefully handles DataFrame columns produced by duplicate headers.
            """

            if name in df:
                col = df[name]
                if isinstance(col, pd.DataFrame):  # duplicate columns
                    col = col.iloc[:, 0]
                series = pd.to_numeric(col, errors="coerce")
            else:
                series = pd.Series([0.0] * len(df), index=df.index, dtype=float)
            return series.reindex(df.index).fillna(0.0)

        entry_notional = entries.fillna(0.0).abs() * quantity_series
        exit_notional = exits.fillna(0.0).abs() * quantity_series
        estimated_fees = ((entry_notional + exit_notional) * BINANCE_FEE_RATE).fillna(0.0)
        existing_fees = _numeric_series(hist_df, "fees").fillna(0.0)
        fees = estimated_fees.where(existing_fees.abs() <= 1e-9, existing_fees)
        hist_df["Fees (USDT)"] = fees
        slippage = _numeric_series(hist_df, "slippage")
        # Determine direction multiplier and compute PnL safely
        def _to_float_series(s, index):
            if isinstance(s, (pd.Series, pd.Index)):
                s = pd.Series(s, index=getattr(s, "index", None))
            s = pd.to_numeric(pd.Series(s), errors="coerce")
            s.index = index
            return s.fillna(0.0)

        def _to_sign_series(direction, index):
            m = pd.Series(direction)
            m.index = index
            normalized = m.astype("string").str.lower().str.strip()
            sign = normalized.map({"long": 1, "short": -1}).fillna(0).astype(float)
            return sign

        idx = hist_df.index
        pnl_source = None
        pnl_source_name: str | None = None
        for col in ("net_pnl", "pnl"):
            if col in hist_df.columns:
                candidate = numcol(hist_df, col)
                if candidate.notna().any():
                    pnl_source = candidate
                    pnl_source_name = col
                    break
        if pnl_source is not None:
            if pnl_source_name == "net_pnl":
                pnl_net = pnl_source.fillna(0.0)
                pnl_gross = pnl_net + fees + slippage
            else:
                pnl_gross = pnl_source.fillna(0.0)
                pnl_net = pnl_gross - fees - slippage
        else:
            entries_aln = _to_float_series(entries, idx)
            exits_aln = _to_float_series(exits, idx)
            sizes_aln = _to_float_series(sizes, idx)
            sign_aln = _to_sign_series(directions, idx)
            pnl_vals = (
                (exits_aln.to_numpy() - entries_aln.to_numpy())
                * sizes_aln.to_numpy()
                * sign_aln.to_numpy()
            )
            pnl_gross = pd.Series(pnl_vals, index=idx)
            pnl_net = pnl_gross - fees - slippage
        pnl_pct_series: pd.Series | None = None
        if "pnl_pct" in hist_df.columns:
            candidate_pct = numcol(hist_df, "pnl_pct")
            if candidate_pct.notna().any():
                pnl_pct_series = candidate_pct
        if pnl_pct_series is not None:
            pnl_pct = pnl_pct_series.fillna(0.0)
        elif "notional" in hist_df.columns:
            notional_series = numcol(hist_df, "notional")
            safe_notional = notional_series.replace({0: np.nan})
            pnl_pct = pd.Series(0.0, index=idx, dtype=float)
            mask = safe_notional.notna()
            pnl_pct.loc[mask] = (
                pnl_net.loc[mask] / safe_notional.loc[mask]
            ) * 100
            pnl_pct = pnl_pct.fillna(0.0)
        else:
            entries_numeric = pd.to_numeric(entries, errors="coerce").fillna(0.0)
            sizes_numeric = pd.to_numeric(sizes, errors="coerce").fillna(0.0)
            pnl_numeric = pd.to_numeric(pnl_net, errors="coerce").fillna(0.0)
            if SIZE_AS_NOTIONAL:
                notional_vals = sizes_numeric
            else:
                notional_vals = entries_numeric * sizes_numeric
            pnl_pct = pd.Series(0.0, index=idx, dtype=float)
            mask = notional_vals != 0
            pnl_pct.loc[mask] = (pnl_numeric.loc[mask] / notional_vals.loc[mask]) * 100
        hist_df["PnL ($)"] = pnl_gross
        hist_df["PnL (net $)"] = pnl_net
        hist_df["PnL (%)"] = pnl_pct
        # Compute duration in minutes
        if {"entry_time", "exit_time"}.issubset(hist_df.columns):
            entry_times = pd.Series(
                pd.to_datetime(hist_df.get("entry_time"), errors="coerce", utc=True)
            )
            exit_times = pd.Series(
                pd.to_datetime(hist_df.get("exit_time"), errors="coerce", utc=True)
            )
            hist_df["Duration (min)"] = (
                exit_times - entry_times
            ).dt.total_seconds() / 60
        # Preview the most recent closed trades to reassure users the trades
        # are logged even after disappearing from the active table.
        sort_field = next(
            (col for col in ("exit_time", "timestamp") if col in hist_df.columns),
            None,
        )
        if sort_field:
            recent_closed = hist_df.sort_values(sort_field, ascending=False).head(10)
        else:
            recent_closed = hist_df.tail(10)
        if not recent_closed.empty:
            st.markdown("#### ✅ Recently Closed Trades")
            display_cols: list[str] = []
            rename_map: dict[str, str] = {}
            for src, dst in (
                ("symbol", "Symbol"),
                ("exit_time", "Exit Time"),
                ("PnL (net $)", "PnL (net $)"),
                ("PnL (%)", "PnL (%)"),
                ("outcome", "Outcome"),
            ):
                if src in recent_closed.columns:
                    display_cols.append(src)
                    rename_map[src] = dst
            if not display_cols:
                display_cols = list(recent_closed.columns)
            recent_display = recent_closed[display_cols].rename(columns=rename_map).copy()
            if "Exit Time" in recent_display.columns:
                exit_series = pd.to_datetime(
                    recent_display["Exit Time"], errors="coerce", utc=True
                )
                recent_display["Exit Time"] = exit_series.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            recent_safe = arrow_safe_dataframe(recent_display)
            recent_formatters = {}
            if "PnL (net $)" in recent_safe.columns:
                recent_formatters["PnL (net $)"] = _fmt_money
            if "PnL (%)" in recent_safe.columns:
                recent_formatters["PnL (%)"] = _fmt_percent
            st.dataframe(
                recent_safe.style.format(recent_formatters), use_container_width=True
            )
        # Summary metrics
        total_trades = len(hist_df)
        wins = (hist_df["PnL (net $)"] > 0).sum()
        losses = (hist_df["PnL (net $)"] < 0).sum()
        win_loss_ratio = wins / losses if losses > 0 else float("inf")
        total_gross = pnl_gross.sum()
        total_net = pnl_net.sum()
        win_rate = wins / total_trades * 100 if total_trades else 0.0
        avg_trade_pnl = total_net / total_trades if total_trades else 0.0
        largest_win = pnl_net.max() if not hist_df.empty else 0
        largest_loss = pnl_net.min() if not hist_df.empty else 0
        # Profit factor: gross profit / gross loss
        gross_profit = pnl_net[pnl_net > 0].sum()
        gross_loss = -pnl_net[pnl_net < 0].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        # Display summary metrics
        mcol1, mcol2, mcol3, mcol4, mcol5, mcol6, mcol7 = st.columns(7)
        mcol1.metric("Total Trades", total_trades)
        mcol2.metric("Profitable Trades", wins)
        mcol3.metric("Losing Trades", losses)
        mcol4.metric(
            "Win/Loss Ratio",
            f"{win_loss_ratio:.2f}" if np.isfinite(win_loss_ratio) else "∞",
        )
        mcol5.metric("Total Gross PnL", f"${total_gross:.2f}")
        mcol6.metric("Total Net PnL", f"${total_net:.2f}")
        mcol7.metric(
            "Profit Factor",
            f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞",
        )

        # Additional performance metrics
        wcol1, wcol2 = st.columns(2)
        wcol1.metric("Win Rate", f"{win_rate:.2f}%")
        wcol2.metric("Avg Trade PnL", f"${avg_trade_pnl:.2f}")

        # ------------------------------------------------------------------
        # Risk metrics
        # Compute per-trade returns using net PnL relative to notional value
        # If notional is zero or missing, fall back to size or entry*size
        # depending on ``SIZE_AS_NOTIONAL``
        returns: np.ndarray
        try:
            notional_series = numcol(hist_df, "notional")
            size_series = numcol(hist_df, "size", default=0.0)
            if entry_col:
                entry_series = numcol(hist_df, entry_col, default=0.0)
            else:
                entry_series = pd.Series([0.0] * len(hist_df), index=hist_df.index)
            if SIZE_AS_NOTIONAL:
                fallback_notional = size_series
            else:
                fallback_notional = entry_series * size_series
            notional_series = notional_series.where(notional_series > 0, fallback_notional)
            net_ret = hist_df["PnL (net $)"].astype(float)
            returns = np.where(notional_series > 0, net_ret / notional_series, 0.0)
            # Drop NaN values
            returns = np.array([r for r in returns if np.isfinite(r)])
        except Exception:
            returns = np.array([])

        # Calculate risk metrics only if we have at least two returns
        if len(returns) >= 2:
            sharpe_val = sharpe_ratio(returns)
            calmar_val = calmar_ratio(returns)
            # Build equity curve for max drawdown
            equity_curve = np.cumprod(1 + returns)
            mdd_val = max_drawdown(equity_curve)
            var_val = value_at_risk(returns, alpha=0.05)
            es_val = expected_shortfall(returns, alpha=0.05)
        else:
            sharpe_val = float("nan")
            calmar_val = float("nan")
            mdd_val = float("nan")
            var_val = float("nan")
            es_val = float("nan")

        # Display risk metrics in a separate row
        rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
        rcol1.metric(
            "Sharpe Ratio",
            f"{sharpe_val:.2f}" if np.isfinite(sharpe_val) else "N/A",
        )
        rcol2.metric(
            "Calmar Ratio",
            f"{calmar_val:.2f}" if np.isfinite(calmar_val) else "N/A",
        )
        rcol3.metric(
            "Max Drawdown",
            f"{mdd_val:.2%}" if np.isfinite(mdd_val) else "N/A",
        )
        rcol4.metric(
            "VaR (95%)",
            f"{var_val:.2%}" if np.isfinite(var_val) else "N/A",
        )
        rcol5.metric(
            "Expected Shortfall",
            f"{es_val:.2%}" if np.isfinite(es_val) else "N/A",
        )

        # ------------------------------------------------------------------
        # Aggregated PnL over different horizons
        # Compute PnL for trades closed today, this week, last 30 days and lifetime
        try:
            # Ensure exit_time is datetime with UTC timezone awareness
            exit_times = pd.to_datetime(
                hist_df.get("exit_time"), errors="coerce", utc=True
            )
            now_ts = pd.Timestamp.utcnow()
            # Start of day, week and month relative to current UTC time
            today_start = now_ts.normalize()
            week_start = now_ts - pd.Timedelta(days=7)
            month_start = now_ts - pd.Timedelta(days=30)
            pnl_net_series = hist_df["PnL (net $)"].astype(float)
            pnl_today = pnl_net_series[exit_times >= today_start].sum()
            pnl_week = pnl_net_series[exit_times >= week_start].sum()
            pnl_month = pnl_net_series[exit_times >= month_start].sum()
            pnl_lifetime = pnl_net_series.sum()
        except Exception:
            pnl_today = 0.0
            pnl_week = 0.0
            pnl_month = 0.0
            pnl_lifetime = 0.0

        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("PnL Today", f"${pnl_today:.2f}")
        pcol2.metric("PnL This Week", f"${pnl_week:.2f}")
        pcol3.metric("PnL 30 Days", f"${pnl_month:.2f}")
        pcol4.metric("PnL Lifetime", f"${pnl_lifetime:.2f}")
        # Equity curve chart
        if "exit_time" in hist_df.columns:
            curve_df = hist_df.sort_values("exit_time").copy()
            curve_df["Cumulative PnL"] = curve_df["PnL (net $)"].cumsum()
            st.line_chart(
                curve_df.set_index("exit_time")["Cumulative PnL"],
                use_container_width=True,
            )
        # Distribution of trade returns
        if "PnL (%)" in hist_df.columns:
            # Safely convert to numeric and drop missing/inf values before histogram
            pnl_series = (
                pd.to_numeric(hist_df["PnL (%)"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not pnl_series.empty:
                hist_vals, bins = np.histogram(pnl_series, bins=20)
                hist_chart_df = pd.DataFrame({"Return": bins[:-1], "Count": hist_vals})
                st.bar_chart(
                    hist_chart_df.set_index("Return"), use_container_width=True
                )
        # Performance analytics by symbol and strategy
        if "PnL (net $)" in hist_df.columns:
            if "symbol" in hist_df.columns:
                sym_group = hist_df.groupby("symbol")
                sym_stats = pd.DataFrame(
                    {
                        "Trades": sym_group.size(),
                        "Total PnL": sym_group["PnL (net $)"].sum(),
                        "Avg PnL": sym_group["PnL (net $)"].mean(),
                        "Win Rate (%)": sym_group["PnL (net $)"].apply(
                            lambda x: (x > 0).mean() * 100
                        ),
                    }
                ).sort_values("Total PnL", ascending=False)
                for col in ["Total PnL", "Avg PnL", "Win Rate (%)"]:
                    sym_stats[col] = sym_stats[col].round(2)
                sym_stats["Total PnL"] = sym_stats["Total PnL"].map(
                    lambda x: f"${x:,.2f}"
                )
                sym_stats["Avg PnL"] = sym_stats["Avg PnL"].map(
                    lambda x: f"${x:,.2f}"
                )
                st.subheader("Performance by Symbol")
                st.table(sym_stats)
            if "strategy" in hist_df.columns:
                strat_group = hist_df.groupby("strategy")
                strat_stats = pd.DataFrame(
                    {
                        "Trades": strat_group.size(),
                        "Total PnL": strat_group["PnL (net $)"].sum(),
                        "Avg PnL": strat_group["PnL (net $)"].mean(),
                        "Win Rate (%)": strat_group["PnL (net $)"].apply(
                            lambda x: (x > 0).mean() * 100
                        ),
                    }
                ).sort_values("Total PnL", ascending=False)
                for col in ["Total PnL", "Avg PnL", "Win Rate (%)"]:
                    strat_stats[col] = strat_stats[col].round(2)
                strat_stats["Total PnL"] = strat_stats["Total PnL"].map(
                    lambda x: f"${x:,.2f}"
                )
                strat_stats["Avg PnL"] = strat_stats["Avg PnL"].map(
                    lambda x: f"${x:,.2f}"
                )
                st.subheader("Performance by Strategy")
                st.table(strat_stats)
        # Session breakdown
        if "session" in hist_df.columns:
            session_perf = (
                hist_df.groupby("session")["PnL (net $)"].sum().reset_index()
            )
            st.bar_chart(
                session_perf.set_index("session"), use_container_width=True
            )
        # Map outcome codes to friendly descriptions
        outcome_descriptions = {
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
        }
        profit_codes = {
            "tp1",
            "tp1_partial",
            "tp2",
            "tp2_partial",
            "tp3",
            "tp4",
        }
        loss_codes = {"tp4_sl", "sl", "early_exit"}
        if "outcome" in hist_df.columns:
            hist_df["Outcome Description"] = hist_df["outcome"].map(
                lambda x: outcome_descriptions.get(str(x), str(x))
            )
        # Display historical trades table including per-stage metrics when available
        stage_map = {
            "pnl_tp1": "PnL TP1 ($)",
            "pnl_tp2": "PnL TP2 ($)",
            "pnl_tp3": "PnL TP3 ($)",
            "size_tp1": "Size TP1",
            "size_tp2": "Size TP2",
            "size_tp3": "Size TP3",
            "notional_tp1": "Notional TP1 ($)",
            "notional_tp2": "Notional TP2 ($)",
            "notional_tp3": "Notional TP3 ($)",
        }
        hist_display = pd.DataFrame(index=hist_df.index)
        base_columns = [
            ("trade_id", "Trade ID"),
            ("entry_time", "Entry Time"),
            ("exit_time", "Exit Time"),
            ("symbol", "Symbol"),
            ("direction", "Direction"),
            ("strategy", "Strategy"),
            ("session", "Session"),
        ]
        for raw, label in base_columns:
            if raw in hist_df.columns:
                hist_display[label] = hist_df[raw]
        if entry_col:
            hist_display["Entry Price"] = numcol(hist_df, entry_col, default=np.nan)
        if exit_col:
            hist_display["Exit Price"] = numcol(hist_df, exit_col, default=np.nan)
        hist_display["Quantity"] = hist_df.get("Quantity", 0.0)
        hist_display["Position Size (USDT)"] = hist_df["Position Size (USDT)"]
        hist_display["Gross PnL (USDT)"] = hist_df["PnL ($)"]
        hist_display["Net PnL (USDT)"] = hist_df["PnL (net $)"]
        hist_display["PnL (%)"] = hist_df["PnL (%)"]
        hist_display["Fees (USDT)"] = fees
        hist_display["Slippage (USDT)"] = slippage
        if "llm_decision" in hist_df.columns:
            hist_display["LLM Decision"] = hist_df["llm_decision"].fillna("")
        if "llm_approval" in hist_df.columns:
            approval_labels = hist_df["llm_approval"].map({True: "Approved", False: "Vetoed"})
            hist_display["LLM Approval"] = approval_labels.fillna("")
        if "llm_confidence" in hist_df.columns:
            hist_display["LLM Confidence Score"] = numcol(
                hist_df, "llm_confidence"
            )
        if "technical_indicator_score" in hist_df.columns:
            hist_display["Technical Indicators"] = numcol(
                hist_df, "technical_indicator_score"
            )
        if "exit_reason" in hist_df.columns:
            hist_display["Exit Reason"] = hist_df["exit_reason"].fillna("")
        if "Duration (min)" in hist_df.columns:
            hist_display["Duration (min)"] = hist_df["Duration (min)"]
        if "Outcome Description" in hist_df.columns:
            hist_display["Outcome Description"] = hist_df["Outcome Description"]
        stage_display_order: list[str] = []
        for key in (
            "pnl_tp1",
            "size_tp1",
            "notional_tp1",
            "pnl_tp2",
            "size_tp2",
            "notional_tp2",
            "pnl_tp3",
            "size_tp3",
            "notional_tp3",
        ):
            if key in hist_df.columns:
                label = stage_map[key]
                hist_display[label] = numcol(hist_df, key)
                stage_display_order.append(label)
        for dt_col in ("Entry Time", "Exit Time"):
            if dt_col in hist_display.columns:
                timestamps = pd.to_datetime(hist_display[dt_col], errors="coerce", utc=True)
                formatted = timestamps.dt.strftime("%Y-%m-%d %H:%M:%S")
                formatted = formatted.where(~timestamps.isna(), "")
                hist_display[dt_col] = formatted
        column_order = [
            "Trade ID",
            "Entry Time",
            "Exit Time",
            "Symbol",
            "Direction",
            "Strategy",
            "Session",
            "Entry Price",
            "Exit Price",
            "Quantity",
            "Position Size (USDT)",
            "Gross PnL (USDT)",
            "Net PnL (USDT)",
            "PnL (%)",
            "Fees (USDT)",
            "Slippage (USDT)",
            "LLM Decision",
            "LLM Approval",
            "LLM Confidence Score",
            "Technical Indicators",
            "Exit Reason",
            "Duration (min)",
            "Outcome Description",
            *stage_display_order,
        ]
        ordered = [col for col in column_order if col in hist_display.columns]
        remainder = [col for col in hist_display.columns if col not in ordered]
        hist_display = hist_display[ordered + remainder]
        hist_display = arrow_safe_dataframe(hist_display)
        positive_labels = {
            outcome_descriptions[c]
            for c in profit_codes
            if c in outcome_descriptions
        }
        negative_labels = {
            outcome_descriptions[c]
            for c in loss_codes
            if c in outcome_descriptions
        }

        def style_outcome(val: str) -> str:
            if val in positive_labels:
                return "color: green"
            if val in negative_labels:
                return "color: red"
            return ""

        formatters: dict[str, object] = {}
        for col in ("Entry Price", "Exit Price"):
            if col in hist_display.columns:
                formatters[col] = _fmt_price
        for col in ("Quantity", "Size TP1", "Size TP2"):
            if col in hist_display.columns:
                formatters[col] = _fmt_quantity
        money_cols = [
            "Position Size (USDT)",
            "Gross PnL (USDT)",
            "Net PnL (USDT)",
            "Fees (USDT)",
            "Slippage (USDT)",
            "PnL TP1 ($)",
            "PnL TP2 ($)",
            "Notional TP1 ($)",
            "Notional TP2 ($)",
        ]
        for col in money_cols:
            if col in hist_display.columns:
                formatters[col] = _fmt_money
        if "PnL (%)" in hist_display.columns:
            formatters["PnL (%)"] = _fmt_percent
        if "Duration (min)" in hist_display.columns:
            formatters["Duration (min)"] = _fmt_duration
        if "LLM Confidence Score" in hist_display.columns:
            formatters["LLM Confidence Score"] = _fmt_score
        if "Technical Indicators" in hist_display.columns:
            formatters["Technical Indicators"] = _fmt_score
        hist_display_style = hist_display.style.format(formatters)
        if "Outcome Description" in hist_display.columns:
            hist_display_style = hist_display_style.applymap(
                style_outcome, subset=["Outcome Description"]
            )
        if "Position Size (USDT)" in hist_display.columns:
            hist_display_style = hist_display_style.applymap(
                _highlight_position_limit, subset=["Position Size (USDT)"]
            )
        st.dataframe(hist_display_style, use_container_width=True)
        # CSV download button
        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Trade History",
            csv_data,
            "trade_history.csv",
            "text/csv",
        )
def render_backtest_tab() -> None:
    """Upload a CSV and visualise backtest or price-based results."""
    st.subheader("Backtest Trade Log")
    uploaded = st.file_uploader("Upload trade log or price CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded, encoding="utf-8")
        cols = {c.lower(): c for c in df.columns}

        # If neither 'pnl' nor 'close' columns are found, attempt to
        # interpret the file as a headerless Binance OHLCV export.  Such
        # files contain twelve columns in a fixed order but no header row.
        if "pnl" not in cols and "close" not in cols:
            uploaded.seek(0)
            binance_cols = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ]
            df = pd.read_csv(uploaded, names=binance_cols)
            cols = {c.lower(): c for c in df.columns}

        if "pnl" in cols:
            pnl_col = cols["pnl"]
            if "equity" not in cols:
                df["equity"] = (1 + df[pnl_col].astype(float)).cumprod()
            equity_col = "equity"
        elif "close" in cols:
            # Assume Binance OHLCV download; compute simple buy-and-hold PnL
            df = df.rename(columns={cols["close"]: "close"})
            df = compute_buy_and_hold_pnl(df)
            pnl_col = "pnl"
            equity_col = "equity"
            st.info("PnL computed from close prices using buy-and-hold assumption")
        else:
            st.error("CSV must contain either a 'pnl' or 'close' column")
            return
        st.line_chart(df[equity_col], use_container_width=True)
        st.bar_chart(df[pnl_col], use_container_width=True)
        # Display distribution of returns as a histogram
        hist, bins = np.histogram(df[pnl_col].astype(float), bins=20)
        hist_df = pd.DataFrame({"Return": bins[:-1], "Count": hist})
        st.bar_chart(hist_df.set_index("Return"), use_container_width=True)
        st.dataframe(arrow_safe_dataframe(df), use_container_width=True)

        # If OHLCV columns are present, generate synthetic trade labels and
        # train the ML model automatically.
        required_cols = {"open", "high", "low", "close"}
        if required_cols.issubset(cols):
            df_bt = df.rename(
                columns={
                    cols["open"]: "open",
                    cols["high"]: "high",
                    cols["low"]: "low",
                    cols["close"]: "close",
                }
            )
            if "open_time" in cols:
                df_bt["open_time"] = pd.to_datetime(
                    df[cols["open_time"]], unit="ms", errors="coerce"
                )
                df_bt = df_bt.set_index("open_time")
            symbol = df_bt["symbol"].iloc[0] if "symbol" in df_bt.columns else "UNKNOWN"
            trades = generate_trades_from_ohlcv(df_bt, symbol=str(symbol))
            for t in trades:
                trade_info = {
                    "symbol": t["symbol"],
                    "direction": "long",
                    "entry": t["entry"],
                    "entry_time": t["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "size": 1.0,
                    "strategy": "upload_backtest",
                    "session": "unknown",
                }
                log_trade_result(
                    trade_info,
                    t["outcome"],
                    t["exit"],
                    exit_time=t["exit_time"].strftime("%Y-%m-%d %H:%M:%S"),
                )
            st.success(
                f"Backtest generated {len(trades)} trades; results appended to {TRADE_HISTORY_FILE}"
            )
            train_model()
            st.info("Model training complete. Check logs for details.")
        else:
            st.warning(
                "CSV missing OHLCV columns; skipping trade generation and training."
            )


# Create tabs and render contents only when executed as a script
if __name__ == "__main__":
    tab_live, tab_backtest = st.tabs(["Dashboard", "Backtest"])
    with tab_live:
        render_live_tab()
    with tab_backtest:
        render_backtest_tab()
