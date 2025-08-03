"""
Comprehensive Streamlit dashboard for the Spot AI Super Agent (updated).

This dashboard provides both real‑time monitoring of active trades and
in‑depth analytics of historical performance.  It reads from the
``active_trades.json`` and ``trade_log.csv`` produced by the bot to
compute metrics such as unrealised PnL, time in trade, net versus
gross PnL, win/loss statistics, equity curves, session and strategy
breakdowns.  Users can download their full trade history for offline
analysis via a CSV export button.

Compared with the original version, this update imports the shared
file paths for the active trades JSON and trade log CSV from
``trade_storage``.  Centralising the locations ensures the dashboard
and trading engine operate on the same data regardless of where each
module is executed.  The ``load_trade_history`` function also falls
back to ``trades_log.csv`` for backward compatibility and filters out
rows where the outcome is recorded as ``open``.
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

# Load environment variables early so API keys are available
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

from trade_storage import ACTIVE_TRADES_FILE, TRADE_LOG_FILE  # shared paths

# Base directory for any fallback files.  By deriving this from
# ``TRADE_LOG_FILE`` we ensure all auxiliary paths reside alongside the
# primary trade log.
BASE_DIR = os.path.dirname(TRADE_LOG_FILE)


def load_active_trades() -> dict:
    """Return a dictionary of active trades from the configured JSON file."""
    try:
        with open(ACTIVE_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_trade_history() -> pd.DataFrame:
    """
    Load completed trade history from CSV.

    This function first attempts to read the path specified by
    ``TRADE_LOG_FILE``.  If that file is empty or missing, it falls
    back to ``trade_log.csv`` in the same directory, then to
    ``trades_log.csv`` (legacy signal log) and finally to
    ``trade_learning_log.csv`` in the same directory as this script.
    If none are found, an empty DataFrame is returned.
    Rows with ``outcome == 'open'`` are filtered out, since these are
    open trades mistakenly logged in earlier versions.
    """
    candidates = [
        TRADE_LOG_FILE,
        os.path.join(BASE_DIR, "trade_log.csv"),
        os.path.join(BASE_DIR, "trades_log.csv"),
        os.path.join(BASE_DIR, "trade_learning_log.csv"),
    ]
    for candidate in candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        try:
            df = pd.read_csv(candidate)
            if df.empty:
                # handle log files saved without a header row
                df = pd.read_csv(candidate, header=None)
            if not df.empty:
                # Filter out rows with outcome == 'open' (incomplete trades)
                if "outcome" in df.columns:
                    df = df[df["outcome"].astype(str).str.lower() != "open"]
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
    # Display active trades table with formatted PnL columns
    df_display = df_active.copy()
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
            "fees",
            "slippage",
            "outcome",
        ]
        if col in hist_df.columns
    ] + [c for c in ["PnL (net $)", "PnL (%)", "Duration (min)"] if c in hist_df.columns]
    hist_display = hist_df[display_cols].copy()
    # Format numeric columns
    for col in ["PnL (net $)", "PnL (%)", "Duration (min)", "fees", "slippage"]:
        if col in hist_display.columns:
            hist_display[col] = hist_display[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)
    st.dataframe(hist_display, use_container_width=True)
    # CSV download button
    csv_data = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Trade History", csv_data, "trade_history.csv", "text/csv")
else:
    st.info("No completed trades logged yet.")
