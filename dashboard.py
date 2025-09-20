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
import os
from pathlib import Path
from datetime import datetime, timezone
from log_utils import setup_logger, LOG_FILE
from trade_schema import TRADE_HISTORY_COLUMNS, normalise_history_columns
from backtest import compute_buy_and_hold_pnl, generate_trades_from_ohlcv
from ml_model import train_model

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

@st.cache_data(ttl=30)
def _read_history_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(str(p), on_bad_lines="skip", engine="python")
    except Exception as e:
        # Show a hint in the UI and return empty rather than crashing the page
        st.warning(f"Could not read {p}: {e}")
        return pd.DataFrame()
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
    """Convert common string/int representations to a boolean."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in {"true", "1", "yes"}
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
    # Validate required fields
    if current_price is None or entry is None or direction is None:
        return None
    try:
        entry_price = float(entry)
    except Exception:
        return None
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

    def tp_flag(hit: bool | None, label: str) -> str:
        """Return an emoji tag indicating whether a TP level was hit."""
        return ("🟢" if hit else "🔵") + f" {label}"

    for key, label in [("tp1", "TP1"), ("tp2", "TP2"), ("tp3", "TP3")]:
        # Only show flags for targets that exist in the trade data
        if data.get(key) is not None:
            hit = to_bool(status.get(key))
            status_flags.append(tp_flag(hit, label))

    if to_bool(status.get("sl")):
        status_flags.append("🔴 SL")

    # Indicate TP4 profit-riding mode when enabled
    if profit_riding:
        status_flags.append("🚀 TP4 mode (Trailing)")

    status_str = " | ".join(status_flags) if status_flags else "Running"
    return {
        "Symbol": symbol,
        "Direction": direction_raw if direction_raw is not None else ("short" if is_short else "long"),
        "Entry": round(entry_price, 4),
        "Price": round(current_price, 4),
        "SL": round(sl, 4) if sl else None,
        "TP1": round(tp1, 4) if tp1 else None,
        "TP2": round(tp2, 4) if tp2 else None,
        "TP3": round(tp3, 4) if tp3 else None,
        "Size": size_val,
        "Notional ($)": round(notional, 2),
        "PnL ($)": round(pnl_dollars, 2),
        "PnL (%)": round(pnl_percent, 2),
        "Time in Trade (min)": round(time_delta_min, 1) if time_delta_min is not None else None,
        "Strategy": data.get("strategy", data.get("pattern", "")),
        "Session": data.get("session", ""),
        "Status": status_str,
    }


def compute_llm_decision_stats() -> tuple[int, int, int]:
    """Return counts of LLM-approved, vetoed and error trades."""
    approved = vetoed = errors = 0
    # Active trades
    for t in load_active_trades():
        if t.get("llm_error"):
            errors += 1
        elif t.get("llm_decision") is False:
            vetoed += 1
        else:
            approved += 1
    # Completed trades
    df = load_trade_history_df()
    if not df.empty:
        for _, row in df.iterrows():
            if str(row.get("llm_error")).lower() in {"true", "1"}:
                errors += 1
            elif str(row.get("llm_decision")).lower() in {"false", "0", "no"}:
                vetoed += 1
            else:
                approved += 1
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
    trades = load_active_trades()
    active_rows: list[dict] = []
    for data in trades:
        sym = data.get("symbol", "")
        row = format_active_row(sym, data)
        if row:
            active_rows.append(row)
    # LLM decision statistics
    approved, vetoed, errors = compute_llm_decision_stats()
    total_decisions = approved + vetoed + errors
    if total_decisions:
        st.subheader("🤖 LLM Decision Outcomes")
        c1, c2, c3 = st.columns(3)
        c1.metric("Approved", f"{approved} ({approved / total_decisions:.0%})")
        c2.metric("Vetoed", f"{vetoed} ({vetoed / total_decisions:.0%})")
        c3.metric("Errors", f"{errors} ({errors / total_decisions:.0%})")
    # Display live PnL section
    st.subheader("📈 Live PnL – Active Trades")
    if active_rows:
        df_active = pd.DataFrame(active_rows)
        # Summary metrics for active trades
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Active Trades", len(df_active))
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
        total_notional = df_active["Notional ($)"].sum() if not df_active.empty else 0.0
        col5.metric("Total Notional", f"${total_notional:,.2f}")
        # Display active trades table with formatted PnL columns
        df_display = df_active.copy()
        df_display["PnL (%)"] = df_display["PnL (%)"].apply(
            lambda x: f"{x:.2f}%"
        )
        df_display["PnL ($)"] = df_display["PnL ($)"].apply(
            lambda x: f"${x:,.2f}"
        )
        df_display["Notional ($)"] = df_display["Notional ($)"].apply(
            lambda x: f"${x:,.2f}"
        )
        st.dataframe(arrow_safe_dataframe(df_display), use_container_width=True)

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
    st.caption(
        f"Loaded {len(hist)} completed trades (partial exits collapsed) from "
        f"{Path(PRIMARY).as_posix()} and {sum(1 for p in LEGACY if p)} fallback file(s)."
    )

    if hist.empty:
        st.info("No completed trades logged yet.")
    else:
        # Only surface the raw dataframe on demand so we don't present two
        # different historical tables back-to-back in the UI.  The styled
        # view further below remains the primary presentation.
        latest_count = min(len(hist), 20)
        if latest_count:
            with st.expander(
                f"View latest {latest_count} trade(s) in raw format", expanded=False
            ):
                st.dataframe(
                    arrow_safe_dataframe(hist.tail(latest_count)),
                    use_container_width=True,
                )
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
                series = pd.Series([0] * len(df), index=df.index)
            return series.reindex(df.index).fillna(0)

        fees = _numeric_series(hist_df, "fees")
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
        for col in ("pnl", "net_pnl"):
            if col in hist_df.columns:
                candidate = numcol(hist_df, col)
                if candidate.notna().any():
                    pnl_source = candidate
                    break
        if pnl_source is not None:
            pnl_net = pnl_source.fillna(0.0)
            pnl_gross = pnl_net + fees + slippage
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
            "size_tp1": "Size TP1",
            "size_tp2": "Size TP2",
            "notional_tp1": "Notional TP1 ($)",
            "notional_tp2": "Notional TP2 ($)",
        }
        hist_display = hist_df.rename(columns=stage_map)
        base_cols = [
            "entry_time",
            "exit_time",
            "symbol",
            "direction",
            "strategy",
            "session",
            "entry",
            "entry_price",
            "exit",
            "exit_price",
            "size",
            "notional",
            "fees",
            "slippage",
            "Outcome Description",
            *stage_map.values(),
        ]
        display_cols = [col for col in base_cols if col in hist_display.columns] + [
            c
            for c in [
                "PnL (net $)",
                "PnL (%)",
                "Duration (min)",
                "Notional",
            ]
            if c in hist_display.columns
        ]
        hist_display = hist_display[display_cols].copy()
        # Ensure column labels are plain strings for Arrow/JSON serialisation
        hist_display.columns = [str(c) for c in hist_display.columns]
        # Format numeric columns safely by coercing to numbers before rounding
        numeric_cols = [
            "PnL (net $)",
            "PnL (%)",
            "Duration (min)",
            "entry",
            "entry_price",
            "exit",
            "exit_price",
            "size",
            "fees",
            "slippage",
            "notional",
            "Notional",
            *stage_map.values(),
        ]
        for col in numeric_cols:
            if col in hist_display.columns:
                converted = pd.to_numeric(hist_display[col], errors="coerce")
                if pd.api.types.is_bool_dtype(converted):
                    converted = converted.astype(float)
                hist_display[col] = converted.round(2)
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

        hist_display = arrow_safe_dataframe(hist_display)
        hist_display_style = hist_display.style.applymap(
            style_outcome, subset=["Outcome Description"]
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
