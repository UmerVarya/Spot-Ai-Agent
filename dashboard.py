"""
Professional Streamlit dashboard for Spot AI Super Agent.

This updated dashboard refines the original design with a cleaner layout,
summary metrics and conditional formatting.  It preserves the refresh
slider and active trades table while adding high‑level statistics (number
of trades and average PnL).  Colours and icons provide an at‑a‑glance
view of performance.  The dashboard automatically refreshes according
to the interval selected.
"""

import streamlit as st
import json
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import os
from streamlit_autorefresh import st_autorefresh

# Load API keys
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")
client = Client(api_key, api_secret)

# Page configuration
st.set_page_config(
    page_title="Spot AI Super Agent Dashboard",
    page_icon="📈",
    layout="wide"
)
st.title("🤖 Spot AI Super Agent – Live Trade Dashboard")

# Path to active trades file
import tempfile
ACTIVE_TRADES_FILE = os.environ.get(
    "ACTIVE_TRADES_FILE", os.path.join(tempfile.gettempdir(), "active_trades.json")
)


def load_active_trades() -> dict:
    try:
        with open(ACTIVE_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def get_live_price(symbol: str) -> float:
    try:
        res = client.get_symbol_ticker(symbol=symbol)
        return float(res['price'])
    except Exception:
        return None


def format_trade_row(symbol: str, data: dict) -> dict:
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
    pnl_percent = ((current_price - entry) / entry) * 100 if direction == "long" else ((entry - current_price) / entry) * 100
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

# Main content
trades = load_active_trades()
rows = []
for sym, data in trades.items():
    row = format_trade_row(sym, data)
    if row:
        rows.append(row)

st.subheader("📊 Live PnL – Active Trades")
if rows:
    df = pd.DataFrame(rows)
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Trades", len(df))
    avg_pnl = df['PnL%'].mean() if not df.empty else 0.0
    col2.metric("Average PnL (%)", f"{avg_pnl:.2f}%", delta=f"{avg_pnl:.2f}%")
    wins = (df['PnL%'] > 0).sum()
    col3.metric("Winning Trades", wins)
    # Colour PnL column
    df_display = df.copy()
    df_display["PnL"] = df_display['PnL%'].apply(lambda x: f"🟢 {x:.2f}%" if x >= 0 else f"🔴 {x:.2f}%")
    df_display = df_display.drop(columns=["PnL%"])  # remove raw PnL%
    st.dataframe(df_display, use_container_width=True)
else:
    st.info("No active trades found.")
