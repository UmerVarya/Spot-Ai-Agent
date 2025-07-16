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

# === Load Trades ===
def load_active_trades():
    try:
        with open("active_trades.json", "r") as f:
            return json.load(f)
    except:
        return {}

# === Fetch Live Price ===
def get_live_price(symbol):
    try:
        res = client.get_symbol_ticker(symbol=symbol)
        return float(res['price'])
    except:
        return None

# === Format Trade Data ===
def format_trade_row(symbol, data):
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
    if status.get("tp1"): status_flags.append("✅ TP1")
    if status.get("tp2"): status_flags.append("✅ TP2")
    if status.get("tp3"): status_flags.append("🎯 TP3")
    if status.get("sl"): status_flags.append("🛑 SL Hit")
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
        "Status": status_str
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
for symbol, data in trades.items():
    row = format_trade_row(symbol, data)
    if row:
        rows.append(row)

st.subheader("📊 Live PnL – Active Trades")

if rows:
    df = pd.DataFrame(rows)
    df["🟩 PnL %"] = df["PnL %"].apply(lambda x: f"🟢 {x:.2f}%" if x >= 0 else f"🔴 {x:.2f}%")
    df = df.drop(columns=["PnL %"])  # Clean old column
    st.dataframe(df, use_container_width=True)
else:
    st.warning("No active trades found.")
