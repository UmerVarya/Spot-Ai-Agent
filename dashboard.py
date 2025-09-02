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
from datetime import datetime, timezone
from log_utils import setup_logger, LOG_FILE
from backtest import compute_buy_and_hold_pnl, generate_trades_from_ohlcv
from ml_model import train_model

try:
    import altair as alt  # type: ignore
except Exception:  # pragma: no cover
    alt = None

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
    COMPLETED_TRADES_FILE,
    ACTIVE_TRADES_FILE,
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
    "Paths: LOG_FILE=%s COMPLETED_TRADES=%s ACTIVE_TRADES=%s REJECTED_TRADES=%s LEARNING_LOG=%s",
    LOG_FILE,
    COMPLETED_TRADES_FILE,
    ACTIVE_TRADES_FILE,
    REJECTED_TRADES_FILE,
    TRADE_LEARNING_LOG_FILE,
)

st.title(" Spot AI Super Agent – Live Trade Dashboard")


def load_completed_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "side", "qty", "price", "pnl"]
        )
    try:
        return pd.read_csv(path, encoding="utf-8")
    except pd.errors.ParserError:
        # Some trade logs may contain malformed rows (e.g. mismatched commas).
        # Fall back to a more tolerant parser that skips bad lines so the
        # dashboard continues to function instead of crashing.
        return pd.read_csv(
            path,
            encoding="utf-8",
            on_bad_lines="skip",
            engine="python",
        )


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
    entry = data.get("entry")
    direction = data.get("direction")
    sl = data.get("sl")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")
    tp3 = data.get("tp3")
    size_field = data.get("size", data.get("position_size", 0))
    status = data.get("status", {})
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
        # ``size`` is the notional amount; derive quantity by dividing by entry price
        qty = size_val / entry_price if entry_price != 0 else 0.0
        notional = size_val
    else:
        # ``size`` is the quantity; derive notional by multiplying by entry price
        qty = size_val
        notional = entry_price * size_val
    # Compute unrealised PnL (absolute and percent)
    if direction == "long":
        pnl_abs = (current_price - entry_price) * qty
    else:
        pnl_abs = (entry_price - current_price) * qty
    # Avoid division by zero when computing percentage PnL
    if notional and notional != 0:
        pnl_percent = (pnl_abs / notional) * 100
    else:
        pnl_percent = 0.0
    # Time in trade (minutes)
    entry_time_str = data.get("entry_time") or data.get("timestamp")
    if entry_time_str:
        try:
            entry_dt = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            time_delta_min = (now - entry_dt).total_seconds() / 60
        except Exception:
            time_delta_min = None
    else:
        time_delta_min = None
    # Status flags
    status_flags: list[str] = []
    if status.get("tp1"):
        status_flags.append("TP1 hit")
    if status.get("tp2"):
        status_flags.append("TP2 hit")
    if status.get("tp3"):
        status_flags.append("TP3 hit")
    if status.get("sl"):
        status_flags.append("SL hit")
    status_str = " | ".join(status_flags) if status_flags else "Running"
    return {
        "Symbol": symbol,
        "Direction": direction,
        "Entry": round(entry_price, 4),
        "Price": round(current_price, 4),
        "SL": round(sl, 4) if sl else None,
        "TP1": round(tp1, 4) if tp1 else None,
        "TP2": round(tp2, 4) if tp2 else None,
        "TP3": round(tp3, 4) if tp3 else None,
        "Size": size_val,
        "Notional ($)": round(notional, 2),
        "PnL ($)": round(pnl_abs, 4),
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
    df = load_completed_df(COMPLETED_TRADES_FILE)
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
        st.dataframe(df_display, use_container_width=True)

        # Optional price chart for a selected trade
        selected_symbol = st.selectbox(
            "Select trade for price chart", df_active["Symbol"]
        )
        price_hist = get_price_history(selected_symbol)
        if price_hist is not None:
            trade_info = next(
                (t for t in trades if t.get("symbol") == selected_symbol), {}
            )
            entry_val = float(trade_info.get("entry", 0))
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
    hist_df = load_completed_df(COMPLETED_TRADES_FILE)
    st.subheader("📊 Historical Performance – Completed Trades")
    if not hist_df.empty:
        # Ensure date columns are parsed
        for col in ["entry_time", "exit_time"]:
            if col in hist_df.columns:
                hist_df[col] = pd.to_datetime(hist_df[col], errors="coerce")
        # Derive size column
        if "size" not in hist_df.columns and "position_size" in hist_df.columns:
            hist_df["size"] = hist_df["position_size"].astype(float)
        # Derive notional column if missing
        if "notional" not in hist_df.columns and {
            "size",
            "entry",
        }.issubset(hist_df.columns):
            hist_df["notional"] = pd.to_numeric(hist_df["size"], errors="coerce") * pd.to_numeric(
                hist_df["entry"], errors="coerce"
            )
        # Compute PnL absolute and percent
        if "direction" in hist_df.columns:
            directions = hist_df["direction"].astype(str)
        else:
            directions = pd.Series(["long"] * len(hist_df))
        if "entry" in hist_df.columns:
            entries = pd.to_numeric(hist_df["entry"], errors="coerce")
        else:
            entries = pd.Series([0] * len(hist_df), dtype=float)
        if "exit" in hist_df.columns:
            exits = pd.to_numeric(hist_df["exit"], errors="coerce")
        else:
            exits = pd.to_numeric(hist_df.get("exit_price", pd.Series([np.nan] * len(hist_df))), errors="coerce")
        if "size" in hist_df.columns:
            sizes = pd.to_numeric(hist_df["size"], errors="coerce").fillna(1)
        else:
            sizes = pd.Series([1] * len(hist_df), dtype=float)
        fees = pd.to_numeric(hist_df.get("fees", pd.Series([0] * len(hist_df))), errors="coerce").fillna(0)
        slippage = pd.to_numeric(hist_df.get("slippage", pd.Series([0] * len(hist_df))), errors="coerce").fillna(0)
        # Determine direction multiplier
        direction_sign = np.where(directions.str.lower().str.startswith("s"), -1, 1)
        # Use quantity times price difference
        pnl_abs = (exits - entries) * sizes * direction_sign
        pnl_net = pnl_abs - fees - slippage
        # Compute percentage based on notional when available
        if "notional" in hist_df.columns:
            notional_series = pd.to_numeric(
                hist_df["notional"], errors="coerce"
            ).replace(0, np.nan)
            pnl_pct = np.where(
                notional_series.notnull(),
                pnl_abs / notional_series * 100,
                0.0,
            )
        else:
            pnl_pct = np.where(
                entries > 0,
                pnl_abs / (entries * sizes) * 100,
                0,
            )
        hist_df["PnL ($)"] = pnl_abs
        hist_df["PnL (net $)"] = pnl_net
        hist_df["PnL (%)"] = pnl_pct
        # Compute duration in minutes
        if {"entry_time", "exit_time"}.issubset(hist_df.columns):
            hist_df["Duration (min)"] = (
                hist_df["exit_time"] - hist_df["entry_time"]
            ).dt.total_seconds() / 60
        # Summary metrics
        total_trades = len(hist_df)
        wins = (hist_df["PnL (net $)"] > 0).sum()
        losses = (hist_df["PnL (net $)"] < 0).sum()
        win_loss_ratio = wins / losses if losses > 0 else float("inf")
        total_gross = pnl_abs.sum()
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
        # If notional is zero or missing, fall back to entry * size
        returns: np.ndarray
        try:
            notional_series = pd.to_numeric(hist_df.get("notional"), errors="coerce")
            # Fallback: derive notional from entry and size if missing or zero
            fallback_notional = pd.to_numeric(hist_df.get("entry"), errors="coerce") * pd.to_numeric(
                hist_df.get("size"), errors="coerce"
            )
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
            # Ensure exit_time is datetime
            exit_times = pd.to_datetime(hist_df.get("exit_time"), errors="coerce")
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
            hist_vals, bins = np.histogram(hist_df["PnL (%)"].astype(float), bins=20)
            hist_chart_df = pd.DataFrame({"Return": bins[:-1], "Count": hist_vals})
            st.bar_chart(hist_chart_df.set_index("Return"), use_container_width=True)
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
        # Display historical trades table
        display_cols = [
            col
            for col in [
                "entry_time",
                "exit_time",
                "symbol",
                "direction",
                "strategy",
                "session",
                "entry",
                "exit",
                "size",
                "notional",
                "fees",
                "slippage",
                "outcome",
            ]
            if col in hist_df.columns
        ] + [
            c
            for c in [
                "PnL (net $)",
                "PnL (%)",
                "Duration (min)",
                "Notional",
            ]
            if c in hist_df.columns
        ]
        hist_display = hist_df[display_cols].copy()
        # Format numeric columns safely by coercing to numbers before rounding
        for col in [
            "PnL (net $)",
            "PnL (%)",
            "Duration (min)",
            "fees",
            "slippage",
            "notional",
            "Notional",
        ]:
            if col in hist_display.columns:
                hist_display[col] = (
                    pd.to_numeric(hist_display[col], errors="coerce").round(2)
                )
        st.dataframe(hist_display, use_container_width=True)
        # CSV download button
        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Trade History",
            csv_data,
            "trade_history.csv",
            "text/csv",
        )
    else:
        st.info("No completed trades logged yet.")


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
        st.dataframe(df, use_container_width=True)

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
                f"Backtest generated {len(trades)} trades; results appended to {COMPLETED_TRADES_FILE}"
            )
            train_model()
            st.info("Model training complete. Check logs for details.")
        else:
            st.warning(
                "CSV missing OHLCV columns; skipping trade generation and training."
            )


# Create tabs and render contents
tab_live, tab_backtest = st.tabs(["Dashboard", "Backtest"])
with tab_live:
    render_live_tab()
with tab_backtest:
    render_backtest_tab()
