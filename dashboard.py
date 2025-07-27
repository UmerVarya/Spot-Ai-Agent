"""
    Comprehensive Streamlit dashboard for the Spot AI Super Agent.

    This dashboard provides both real‑time monitoring of active trades and
    in‑depth analytics of historical performance.  It reads from the
    ``active_trades.json`` and ``trade_log.csv`` produced by the bot to
    compute metrics such as unrealised PnL, time in trade, net versus
    gross PnL, win/loss statistics, equity curves, session and strategy
    breakdowns.  Users can download their full trade history for offline
    analysis via a CSV export button.

    Note that many of these analytics rely on additional fields being
    recorded by the trading engine (e.g. ``entry_time``, ``exit_time``,
    ``size``, ``fees``, ``slippage``, ``strategy``, ``session``).  Ensure
    that your ``trade_storage`` and trade management code populates these
    fields accordingly.
    """

import streamlit as st
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

# Optional Binance client for live prices
try:
    from binance.client import Client  # type: ignore
except Exception:
    class Client:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("python-binance library not installed; cannot fetch live prices.")

# Optional dotenv support
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):  # type: ignore
        return None

from streamlit_autorefresh import st_autorefresh

load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
try:
    client = Client(api_key, api_secret)
except Exception:
    client = None

# Page configuration
st.set_page_config(page_title="Spot AI Super Agent Dashboard", page_icon="", layout="wide")
st.title(" Spot AI Super Agent – Live Trade Dashboard")

# Determine file paths for active and historical trades
import tempfile
ACTIVE_TRADES_FILE = os.environ.get(
    "ACTIVE_TRADES_FILE",
    os.path.join(tempfile.gettempdir(), "active_trades.json"),
)
TRADE_LOG_FILE = os.environ.get(
    "TRADE_LOG_FILE",
    os.path.join(os.path.dirname(__file__), "trade_log.csv"),
)


def load_active_trades() -> dict:
    """Return a dictionary of active trades from the configured JSON file."""
    try:
        with open(ACTIVE_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_trade_history() -> pd.DataFrame:
    """
    Load completed trade history from CSV.  This function first attempts to
    read the path specified by ``TRADE_LOG_FILE``.  If that file is empty
    or missing, it falls back to ``trade_log.csv`` and then
    ``trade_learning_log.csv`` in the same directory as this script.  If
    none are found, an empty DataFrame is returned.
    """
    candidates = [TRADE_LOG_FILE, os.path.join(os.path.dirname(__file__), "trade_log.csv"), os.path.join(os.path.dirname(__file__), "trade_learning_log.csv")]
    for candidate in candidates:
        try:
            if candidate and os.path.exists(candidate):
                df = pd.read_csv(candidate)
                if not df.empty:
                    return df
        except Exception:
            continue
    return pd.DataFrame()


def get_live_price(symbol: str) -> float:
    """Fetch the current price for a symbol from Binance."""
    try:
        res = client.get_symbol_ticker(symbol=symbol)
        return float(res["price"])
    except Exception:
        return None


def format_active_row(symbol: str, data: dict) -> dict:
    """Format a single active trade dictionary into a row for display."""
    entry = data.get("entry")
    direction = data.get("direction")
    sl = data.get("sl")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")
    tp3 = data.get("tp3")
    leverage = data.get("leverage", 1)
    size = data.get("size", data.get("position_size", 0))
    status = data.get("status", {})
    current_price = get_live_price(symbol)
    if current_price is None or entry is None:
        return None
    # Compute unrealised PnL (absolute and percent)
    if direction == "long":
        pnl_abs = (current_price - entry) * size
        pnl_percent = ((current_price - entry) / entry) * 100
    else:
        pnl_abs = (entry - current_price) * size
        pnl_percent = ((entry - current_price) / entry) * 100
    pnl_percent *= leverage
    # Time in trade (minutes)
    entry_time_str = data.get("entry_time") or data.get("timestamp")
    if entry_time_str:
        try:
            entry_dt = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            time_delta_min = (now - entry_dt).total_seconds() / 60
        except Exception:
            time_delta_min = None
    else:
        time_delta_min = None
    # Status flags
    status_flags = []
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
        "Entry": round(entry, 4),
        "Price": round(current_price, 4),
        "SL": round(sl, 4) if sl else None,
        "TP1": round(tp1, 4) if tp1 else None,
        "TP2": round(tp2, 4) if tp2 else None,
        "TP3": round(tp3, 4) if tp3 else None,
        "Leverage": leverage,
        "Size": size,
        "PnL ($)": round(pnl_abs, 4),
        "PnL (%)": round(pnl_percent, 2),
        "Time in Trade (min)": round(time_delta_min, 1) if time_delta_min is not None else None,
        "Strategy": data.get("strategy", data.get("pattern", "")),
        "Session": data.get("session", ""),
        "Status": status_str,
    }


# Sidebar controls
refresh_interval = st.sidebar.slider("⏱️ Refresh Interval (seconds)", 10, 60, 30)
st.sidebar.markdown("---")
st.sidebar.markdown("Built for  **Spot AI Super Agent**")

# Auto refresh
st_autorefresh(interval=refresh_interval * 1000, key="refresh")

# Load active trades and format into rows
trades = load_active_trades()
active_rows = []
for sym, data in trades.items():
    row = format_active_row(sym, data)
    if row:
        active_rows.append(row)

# Display live PnL section
st.subheader("📈 Live PnL – Active Trades")
if active_rows:
    df_active = pd.DataFrame(active_rows)
    # Summary metrics for active trades
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Trades", len(df_active))
    avg_pnl_pct = df_active["PnL (%)"].mean() if not df_active.empty else 0.0
    col2.metric("Average PnL (%)", f"{avg_pnl_pct:.2f}%", delta=f"{avg_pnl_pct:.2f}%")
    total_unrealised = df_active["PnL ($)"].sum() if not df_active.empty else 0.0
    col3.metric("Total Unrealised PnL", f"${total_unrealised:.2f}")
    wins_active = (df_active["PnL (%)"] > 0).sum()
    col4.metric("Winning Trades", wins_active)
    # Colour formatting for PnL columns
    df_display = df_active.copy()
    # Format PnL percent with colour indicator
    def colour_pnl(val):
        colour = "green" if val >= 0 else "red"
        return f"<span style='color:{colour}'>{val:.2f}%</span>"
    df_display["PnL (%)"] = df_display["PnL (%)"].apply(lambda x: f" {x:.2f}%")
    df_display["PnL ($)"] = df_display["PnL ($)"].apply(lambda x: f" ${x:.2f}")
    st.dataframe(df_display, use_container_width=True)
else:
    st.info("No active trades found.")


# Load trade history and compute summary statistics
hist_df = load_trade_history()
st.subheader("📊 Historical Performance – Completed Trades")
if not hist_df.empty:
    # Ensure date columns are parsed
    for col in ["entry_time", "exit_time"]:
        if col in hist_df.columns:
            hist_df[col] = pd.to_datetime(hist_df[col], errors="coerce")
    # Derive size column
    if "size" not in hist_df.columns and "position_size" in hist_df.columns:
        hist_df["size"] = hist_df["position_size"].astype(float)
    # Compute PnL absolute and percent
    directions = hist_df.get("direction", "long").astype(str)
    entries = pd.to_numeric(hist_df.get("entry", 0), errors="coerce")
    exits = pd.to_numeric(hist_df.get("exit", hist_df.get("exit_price", np.nan)), errors="coerce")
    sizes = pd.to_numeric(hist_df.get("size", 1), errors="coerce").fillna(1)
    fees = pd.to_numeric(hist_df.get("fees", 0), errors="coerce").fillna(0)
    slippage = pd.to_numeric(hist_df.get("slippage", 0), errors="coerce").fillna(0)
    # Determine direction multiplier
    direction_sign = np.where(directions.str.lower().str.startswith("s"), -1, 1)
    pnl_abs = (exits - entries) * sizes * direction_sign
    pnl_net = pnl_abs - fees - slippage
    pnl_pct = np.where(
        entries > 0,
        pnl_abs / (entries * sizes) * 100,
        0
    )
    hist_df["PnL ($)"] = pnl_abs
    hist_df["PnL (net $)"] = pnl_net
    hist_df["PnL (%)"] = pnl_pct
    # Compute duration in minutes
    if "entry_time" in hist_df.columns and "exit_time" in hist_df.columns:
        hist_df["Duration (min)"] = (hist_df["exit_time"] - hist_df["entry_time"]).dt.total_seconds() / 60
    # Summary metrics
    total_trades = len(hist_df)
    wins = (hist_df["PnL (net $)"] > 0).sum()
    losses = (hist_df["PnL (net $)"] < 0).sum()
    win_loss_ratio = wins / losses if losses > 0 else float('inf')
    total_gross = pnl_abs.sum()
    total_net = pnl_net.sum()
    largest_win = pnl_net.max() if not hist_df.empty else 0
    largest_loss = pnl_net.min() if not hist_df.empty else 0
    # Display summary metrics
    mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
    mcol1.metric("Total Trades", total_trades)
    mcol2.metric("Profitable Trades", wins)
    mcol3.metric("Losing Trades", losses)
    mcol4.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}" if np.isfinite(win_loss_ratio) else "∞")
    mcol5.metric("Total Gross PnL", f"${total_gross:.2f}")
    mcol6.metric("Total Net PnL", f"${total_net:.2f}")
    # Equity curve chart
    # Sort by exit_time and compute cumulative net PnL
    if "exit_time" in hist_df.columns:
        curve_df = hist_df.sort_values("exit_time").copy()
        curve_df["Cumulative PnL"] = curve_df["PnL (net $)"].cumsum()
        st.line_chart(curve_df.set_index("exit_time")["Cumulative PnL"], use_container_width=True)
    # Strategy breakdown
    if "strategy" in hist_df.columns:
        strat_perf = hist_df.groupby("strategy")["PnL (net $)"].sum().reset_index()
        st.bar_chart(strat_perf.set_index("strategy"), use_container_width=True)
    # Session breakdown
    if "session" in hist_df.columns:
        session_perf = hist_df.groupby("session")["PnL (net $)"].sum().reset_index()
        st.bar_chart(session_perf.set_index("session"), use_container_width=True)
    # Display historical trades table
    display_cols = [
        col for col in ["entry_time", "exit_time", "symbol", "direction", "strategy", "session", "entry", "exit", "size", "fees", "slippage", "outcome"] if col in hist_df.columns
    ] + ["PnL (net $)", "PnL (%)", "Duration (min)"]
    hist_display = hist_df[display_cols].copy()
    # Format numeric columns
    for col in ["PnL (net $)", "PnL (%)", "Duration (min)", "fees", "slippage"]:
        if col in hist_display.columns:
            hist_display[col] = hist_display[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    # Colour PnL column via HTML
    st.dataframe(hist_display, use_container_width=True)
    # CSV download button
    csv_data = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Trade History", csv_data, "trade_history.csv", "text/csv")
else:
    st.info("No completed trades logged yet.")
