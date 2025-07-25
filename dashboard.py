"""
Enhanced Streamlit dashboard for the Spot AI Super Agent.

This dashboard reads the current set of active trades from a JSON file
whose location is determined by the ``ACTIVE_TRADES_FILE`` environment
variable (falling back to a temporary file if unset).  It also reads
historical trade performance from ``trade_learning_log.csv`` and
``trades_log.csv`` to display statistics such as win rate and daily
profit & loss.  A professional‑looking set of summary cards presents
high‑level metrics at a glance, and an expanded table lists each
active trade with detailed fields including pattern, technical score,
LLM confidence and machine‑learning probability.

To run this dashboard locally:

    streamlit run dashboard.py

Ensure that the agent process is writing to the same
``ACTIVE_TRADES_FILE`` so the dashboard can display live trades.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import streamlit as st
from binance.client import Client

from trade_utils import get_market_session

# Attempt to import our ML model; if unavailable, treat probabilities as 0.5
try:
    from ml_model import predict_success_probability
except Exception:
    predict_success_probability = None


def get_active_trades_path() -> str:
    """Return the path to the active trades JSON file.

    The location is controlled by the ``ACTIVE_TRADES_FILE`` environment
    variable.  If this variable is not set, default to a file in
    ``/tmp`` so that both the agent and dashboard can write and read it.
    """
    return os.getenv("ACTIVE_TRADES_FILE", "/tmp/active_trades.json")


def load_active_trades() -> Dict[str, Any]:
    """Load the dictionary of active trades from the configured file.

    Returns an empty dict if the file is missing or unreadable.
    """
    path = get_active_trades_path()
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def get_live_price(symbol: str) -> float:
    """Fetch the latest price for a symbol from Binance.

    If the API call fails (e.g. network error or invalid symbol),
    return ``None`` so that callers can handle missing data gracefully.
    """
    try:
        client = Client()
        mapped = symbol  # The agent uses unmapped symbols for display
        ticker = client.get_symbol_ticker(symbol=mapped)
        return float(ticker["price"])
    except Exception:
        return None


def load_log(csv_path: str) -> pd.DataFrame:
    """Load a CSV log into a DataFrame with tolerant parsing.

    The function attempts to read with ``on_bad_lines='skip'`` so
    partially corrupted rows are ignored rather than raising an exception.
    Returns an empty DataFrame if the file is missing.
    """
    try:
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def compute_summary_metrics(trades: Dict[str, Any]) -> Dict[str, Any]:
    """Compute high‑level summary statistics from the active trades dict."""
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "count": 0,
            "avg_conf": 0.0,
            "avg_ml_prob": 0.0,
            "exposure": 0.0,
        }
    confidences = []
    ml_probs = []
    exposure = 0.0
    for trade in trades.values():
        conf = float(trade.get("confidence", 0))
        confidences.append(conf)
        ml = float(trade.get("ml_prob", 0))
        ml_probs.append(ml)
        price = float(trade.get("entry", 0))
        size = int(trade.get("position_size", 0))
        exposure += price * size
    avg_conf = sum(confidences) / total_trades
    avg_ml = sum(ml_probs) / total_trades if ml_probs else 0.0
    return {
        "count": total_trades,
        "avg_conf": round(avg_conf, 2),
        "avg_ml_prob": round(avg_ml, 2),
        "exposure": round(exposure, 2),
    }


def build_trade_rows(trades: Dict[str, Any]) -> pd.DataFrame:
    """Build a DataFrame with detailed fields for each active trade."""
    rows = []
    for sym, trade in trades.items():
        entry = float(trade.get("entry", 0))
        size = int(trade.get("position_size", 0))
        live_price = get_live_price(sym)
        pnl = None
        if live_price is not None and entry != 0:
            pnl = round((live_price - entry) * size, 4)
        row = {
            "Symbol": sym,
            "Direction": trade.get("direction"),
            "Entry": entry,
            "Size": size,
            "Current Price": live_price,
            "PnL": pnl,
            "Score": float(trade.get("score", 0)),
            "Confidence": float(trade.get("confidence", 0)),
            "ML Prob": float(trade.get("ml_prob", 0)),
            "Pattern": trade.get("pattern"),
            "Session": trade.get("session"),
            "Opened": trade.get("timestamp"),
        }
        rows.append(row)
    if rows:
        return pd.DataFrame(rows).sort_values(by=["Symbol"])
    return pd.DataFrame()


def display_performance_summary(log_df: pd.DataFrame) -> None:
    """Display historical performance metrics and charts."""
    if log_df.empty:
        st.info("No historical trades to display yet.")
        return
    # Ensure required columns exist
    if not {"timestamp", "outcome", "confidence"}.issubset(log_df.columns):
        st.info("Historical log missing columns for performance summary.")
        return
    # Convert timestamp to datetime and derive date
    log_df["date"] = pd.to_datetime(log_df["timestamp"]).dt.date
    # Compute win rate
    total = len(log_df)
    wins = len(log_df[log_df["outcome"] == "win"])
    win_rate = wins / total if total > 0 else 0.0
    st.subheader("Historical Performance")
    st.write(f"Total trades: {total} | Wins: {wins} | Win rate: {win_rate:.2%}")
    # Aggregate PnL by date if available
    if {"exit_price", "entry", "position_size"}.issubset(log_df.columns):
        def calc_pnl(row):
            try:
                return (row["exit_price"] - row["entry"]) * row["position_size"]
            except Exception:
                return 0
        log_df["pnl"] = log_df.apply(calc_pnl, axis=1)
        daily_pnl = log_df.groupby("date")["pnl"].sum()
        st.bar_chart(daily_pnl)


def main() -> None:
    st.set_page_config(page_title="Spot AI Super Agent", layout="wide")
    st.title("🤖 Spot AI Super Agent – Live Trade Dashboard")
    # Sidebar refresh interval
    refresh = st.sidebar.slider("Refresh Interval (seconds)", 10, 60, 30)
    # Load active trades
    trades = load_active_trades()
    # Summary cards
    metrics = compute_summary_metrics(trades)
    cols = st.columns(4)
    cols[0].metric("Active Trades", metrics["count"])
    cols[1].metric("Avg Confidence", metrics["avg_conf"])
    cols[2].metric("Avg ML Prob", metrics["avg_ml_prob"])
    cols[3].metric("Total Exposure", f"${metrics['exposure']:.2f}")
    st.markdown("---")
    # Detailed trade table
    if trades:
        df_trades = build_trade_rows(trades)
        st.subheader("Active Trades – Details")
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("No active trades found.")
    st.markdown("---")
    # Load and display historical logs
    learning_log_path = os.getenv("TRADE_LEARNING_FILE", "trade_learning_log.csv")
    df_learning = load_log(learning_log_path)
    if not df_learning.empty:
        display_performance_summary(df_learning)
    st.markdown("---")
    st.caption("Built for Spot AI Super Agent")


if __name__ == "__main__":
    main()
