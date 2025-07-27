"""
Enhanced Streamlit dashboard for Spot AI Super Agent.

This updated dashboard extends the original ``dashboard.py`` by
displaying not only the active trades but also a running history of
completed trades.  It reads the ``trade_log.csv`` written by
``trade_storage.log_trade_result`` to compute high‑level statistics such
as the total number of trades taken, how many ended in profit or loss,
and the aggregate PnL across all closed trades.  The active trades
table remains unchanged, with live price updates via the Binance API.

To use this dashboard, ensure that ``ACTIVE_TRADES_FILE`` and
``TRADE_LOG_FILE`` are set via environment variables if you store
active or historical trades in custom locations.  When unset, the
defaults (``active_trades.json`` in a temporary directory and
``trade_log.csv`` alongside this script) will be used.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import os

# Attempt to import Binance client.  If unavailable, provide a stub
try:
    from binance.client import Client  # type: ignore
except Exception:
    class Client:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "python-binance library not installed; cannot fetch live prices."
            )

# Optional dotenv support
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):  # type: ignore
        return None

from streamlit_autorefresh import st_autorefresh

# Load environment variables
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

try:
    client = Client(api_key, api_secret)
except Exception:
    client = None

# Page configuration
st.set_page_config(
    page_title="Spot AI Super Agent Dashboard",
    page_icon="📈",
    layout="wide",
)
st.title("🤖 Spot AI Super Agent – Live Trade Dashboard")

import tempfile

# Paths for active and historical trades
ACTIVE_TRADES_FILE = os.environ.get(
    "ACTIVE_TRADES_FILE", os.path.join(tempfile.gettempdir(), "active_trades.json")
)
TRADE_LOG_FILE = os.environ.get(
    "TRADE_LOG_FILE", os.path.join(os.path.dirname(__file__), "trade_log.csv")
)


def load_active_trades() -> dict:
    """Return a dictionary of active trades from the configured JSON file."""
    try:
        with open(ACTIVE_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_trade_history() -> pd.DataFrame:
    """Load completed trade history from CSV.  If missing, return empty DataFrame."""
    try:
        return pd.read_csv(TRADE_LOG_FILE)
    except Exception:
        return pd.DataFrame()


def get_live_price(symbol: str) -> float:
    """Fetch the current price for a symbol from Binance."""
    try:
        res = client.get_symbol_ticker(symbol=symbol)
        return float(res["price"])
    except Exception:
        return None


def format_trade_row(symbol: str, data: dict) -> dict:
    """Format a single trade dictionary into a row for the active trades table."""
    entry = data.get("entry")
    direction = data.get("direction")
    sl = data.get("sl")
    tp1 = data.get("tp1")
    tp2 = data.get("tp2")
    tp3 = data.get("tp3")
    leverage = data.get("leverage", 1)
    size = data.get("position_size", 0)
    status = data.get("status", {})
    current_price = get_live_price(symbol)
    if current_price is None or entry is None:
        return None
    pnl_percent = (
        ((current_price - entry) / entry) * 100
        if direction == "long"
        else ((entry - current_price) / entry) * 100
    )
    pnl_percent *= leverage
    status_flags = []
    if status.get("tp1"):
        status_flags.append("✅ TP1")
    if status.get("tp2"):
        status_flags.append("✅ TP2")
    if status.get("tp3"):
        status_flags.append("🎯 TP3")
    if status.get("sl"):
        status_flags.append("🛑 SL Hit")
    status_str = " | ".join(status_flags) if status_flags else "⏳ In Progress"
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
        "Position ($)": size,
        "PnL%": round(pnl_percent, 2),
        "Status": status_str,
    }


# Sidebar controls
refresh_interval = st.sidebar.slider("⏱️ Refresh Interval (seconds)", 10, 60, 30)
st.sidebar.markdown("---")
st.sidebar.markdown("Built for 🔥 **Spot AI Super Agent**")

# Auto refresh
st_autorefresh(interval=refresh_interval * 1000, key="refresh")

# Load active trades and format into rows
trades = load_active_trades()
active_rows = []
for sym, data in trades.items():
    row = format_trade_row(sym, data)
    if row:
        active_rows.append(row)

# Display live PnL section
st.subheader("📊 Live PnL – Active Trades")
if active_rows:
    df_active = pd.DataFrame(active_rows)
    # Summary metrics for active trades
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Trades", len(df_active))
    avg_pnl = df_active["PnL%"].mean() if not df_active.empty else 0.0
    col2.metric("Average PnL (%)", f"{avg_pnl:.2f}%", delta=f"{avg_pnl:.2f}%")
    wins_active = (df_active["PnL%"] > 0).sum()
    col3.metric("Winning Trades", wins_active)
    # Colour PnL column
    df_display = df_active.copy()
    df_display["PnL"] = df_display["PnL%"].apply(
        lambda x: f"🟢 {x:.2f}%" if x >= 0 else f"🔴 {x:.2f}%"
    )
    df_display = df_display.drop(columns=["PnL%"])
    st.dataframe(df_display, use_container_width=True)
else:
    st.info("No active trades found.")

# Load trade history and compute summary statistics
hist_df = load_trade_history()
if not hist_df.empty:
    # Compute PnL percent per trade
    # Determine if direction field exists; fallback to long if missing
    if "direction" in hist_df.columns and "entry" in hist_df.columns and "exit" in hist_df.columns:
        hist_df["PnL%"] = np.where(
            hist_df["direction"].str.lower().str.startswith("s"),
            (hist_df["entry"] - hist_df["exit"]) / hist_df["entry"] * 100,
            (hist_df["exit"] - hist_df["entry"]) / hist_df["entry"] * 100,
        )
    else:
        hist_df["PnL%"] = 0.0
    total_trades = len(hist_df)
    # Derive win/loss counts.  Prefer the 'outcome' column if present;
    # fallback to PnL sign when 'outcome' is missing.
    if "outcome" in hist_df.columns:
        outcome_lower = hist_df["outcome"].astype(str).str.lower()
        wins_hist = outcome_lower.isin(["win", "tp", "profit", "tp1", "tp2", "tp3"]).sum()
        losses_hist = outcome_lower.isin(["loss", "sl", "stoploss", "stop loss"]).sum()
    else:
        wins_hist = (hist_df["PnL%"] > 0).sum()
        losses_hist = (hist_df["PnL%"] < 0).sum()
    total_pnl = hist_df["PnL%"].sum()
    # Section header
    st.subheader("📈 Historical Performance – Completed Trades")
    # Metrics row for history
    hcol1, hcol2, hcol3, hcol4 = st.columns(4)
    hcol1.metric("Total Trades", total_trades)
    hcol2.metric("Profitable Trades", wins_hist)
    hcol3.metric("Losing Trades", losses_hist)
    hcol4.metric(
        "Total PnL (%)",
        f"{total_pnl:.2f}%",
        delta=f"{total_pnl:.2f}%",
        help="Aggregate percentage profit/loss across all closed trades",
    )
    # Display historical trades table with coloured PnL
    hist_display = hist_df.copy()
    hist_display["PnL"] = hist_display["PnL%"].apply(
        lambda x: f"🟢 {x:.2f}%" if x >= 0 else f"🔴 {x:.2f}%"
    )
    # Keep key columns and formatted columns for display
    keep_cols = [
        col
        for col in ["timestamp", "symbol", "direction", "entry", "exit", "outcome"]
        if col in hist_display.columns
    ] + ["PnL"]
    hist_display = hist_display[keep_cols]
    st.dataframe(hist_display, use_container_width=True)
else:
    st.info("No completed trades logged yet.")
