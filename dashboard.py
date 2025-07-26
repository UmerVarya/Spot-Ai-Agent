"""
Streamlit dashboard for visualising active Spot AI trades.

This dashboard reads the active trades file created by the agent and
calculates live mark‑to‑market P&L using real‑time prices from Binance.
It refreshes automatically at an interval chosen by the user and
displays a table of open positions with their stop‑loss and take‑profit
levels and a status indicator.

Modifications: The dashboard now loads the ``active_trades.json`` file
using a constant path relative to this script.  This ensures that
regardless of the working directory, the file is found correctly.
"""

import streamlit as st
import json
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import os
from streamlit_autorefresh import st_autorefresh

# === Load API Keys ===
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

# === Streamlit Page Settings ===
st.set_page_config(page_title="📈 Spot AI Super Agent Dashboard", layout="wide")
st.title("🤖 Spot AI Super Agent – Live Trade Dashboard")

# === Paths ===
import tempfile
ACTIVE_TRADES_FILE = os.environ.get(
    "ACTIVE_TRADES_FILE",
    os.path.join(tempfile.gettempdir(), "active_trades.json")
)


def load_active_trades() -> dict:
    """Load open trades from the JSON file.

    Returns
    -------
    dict
        Mapping of symbol to trade information.  An empty dict is returned
        if the file is missing or malformed.
    """
    try:
        with open(ACTIVE_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def get_live_price(symbol: str) -> float:
    """Fetch the latest price for a symbol from Binance.

    If the API call fails, ``None`` is returned.
    """
    try:
        res = client.get_symbol_ticker(symbol=symbol)
        return float(res['price'])
    except Exception:
        return None


def format_trade_row(symbol: str, data: dict) -> dict:
    """Prepare a trade dict for display in a DataFrame.

    This function computes the percentage P&L for the trade and formats
    stop‑loss and take‑profit values.  It also concatenates status
    flags to give a quick at‑a‑glance view of progress through the
    profit ladder.
    """
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

    if direction == "long":
        pnl_percent = ((current_price - entry) / entry) * 100
    else:
        pnl_percent = ((entry - current_price) / entry) * 100
    pnl_percent *= leverage

    # Status string
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
        "PnL %": round(pnl_percent, 2),
        "Status": status_str,
    }

# === Sidebar Refresh Control ===
refresh_interval = st.sidebar.slider("⏱️ Refresh Interval (seconds)", 10, 60, 30)
st.sidebar.markdown("---")
st.sidebar.markdown("Built for 🔥 **Spot AI Super Agent**")

# Auto-refresh every N seconds
st_autorefresh(interval=refresh_interval * 1000, key="refresh")

# === Display Live Trades ===
trades = load_active_trades()
rows = []
for sym, data in trades.items():
    row = format_trade_row(sym, data)
    if row:
        rows.append(row)

st.subheader("📊 Live PnL – Active Trades")

if rows:
    df = pd.DataFrame(rows)
    # Colour code PnL in a separate column for clarity
    df["🟩 PnL %"] = df["PnL %"].apply(lambda x: f"🟢 {x:.2f}%" if x >= 0 else f"🔴 {x:.2f}%")
    df = df.drop(columns=["PnL %"])  # remove the original column
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No active trades found.")
    
