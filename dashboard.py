"""
Professional Streamlit dashboard for the Spot AI Super Agent.

This dashboard presents an organised view of current and historical
trading activity.  It is designed with a modern aesthetic and provides
additional risk metrics beyond the original implementation.  Where
available, Plotly charts are used for interactivity; otherwise, the
dashboard falls back to Streamlit's built‑in charting functions.  The
dashboard reads data from shared CSV and JSON files produced by the
agent and trade manager.

Usage
-----
Run ``streamlit run dashboard.py`` from the project root.  The
refresh interval can be adjusted via the sidebar.  Ensure that the
agent process writes to the same ``ACTIVE_TRADES_FILE`` path so that
live trades are visible.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

# Try to import Plotly for interactive charts
try:
    import plotly.express as px  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:
    px = None  # type: ignore
    PLOTLY_AVAILABLE = False

from trade_utils import get_market_session

# Import ML model if available for predictive probabilities
try:
    from ml_model import predict_success_probability
except Exception:
    predict_success_probability = None  # type: ignore


def get_active_trades_path() -> str:
    """Return the path to the active trades JSON file."""
    return os.getenv("ACTIVE_TRADES_FILE", "/tmp/active_trades.json")


def load_active_trades() -> Dict[str, Any]:
    """Load the dictionary of active trades from the configured file."""
    path = get_active_trades_path()
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def get_live_price(symbol: str) -> Optional[float]:
    """Fetch the latest price for a symbol from Binance.  Returns None on failure."""
    try:
        from binance.client import Client  # local import to avoid heavy import on dashboard load
        client = Client()
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker.get("price"))
    except Exception:
        return None


def load_log(csv_path: str) -> pd.DataFrame:
    """Load a CSV log into a DataFrame with tolerant parsing."""
    try:
        if not os.path.exists(csv_path):
            return pd.DataFrame()
        return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def compute_summary_metrics(trades: Dict[str, Any]) -> Dict[str, float]:
    """Compute high‑level summary statistics from the active trades dict."""
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "count": 0,
            "avg_conf": 0.0,
            "avg_ml_prob": 0.0,
            "exposure": 0.0,
            "unrealised_pnl": 0.0,
        }
    confidences = []
    ml_probs = []
    exposure = 0.0
    unreal_pnl = 0.0
    for trade in trades.values():
        try:
            conf = float(trade.get("confidence", 0))
            confidences.append(conf)
            ml = float(trade.get("ml_prob", 0))
            ml_probs.append(ml)
            entry = float(trade.get("entry", 0))
            size = float(trade.get("position_size", 0))
            live = get_live_price(trade.get("symbol"))
            if live is not None and entry > 0:
                unreal_pnl += (live - entry) * size
            exposure += entry * size
        except Exception:
            continue
    avg_conf = sum(confidences) / total_trades if confidences else 0.0
    avg_ml = sum(ml_probs) / total_trades if ml_probs else 0.0
    return {
        "count": total_trades,
        "avg_conf": round(avg_conf, 2),
        "avg_ml_prob": round(avg_ml, 2),
        "exposure": round(exposure, 2),
        "unrealised_pnl": round(unreal_pnl, 2),
    }


def build_trade_rows(trades: Dict[str, Any]) -> pd.DataFrame:
    """Build a DataFrame with detailed fields for each active trade."""
    rows: list[dict] = []
    for sym, trade in trades.items():
        entry = float(trade.get("entry", 0))
        size = float(trade.get("position_size", 0))
        live_price = get_live_price(sym)
        pnl = None
        if live_price is not None and entry != 0:
            pnl = round((live_price - entry) * size, 4)
        rows.append({
            "Symbol": sym,
            "Direction": trade.get("direction"),
            "Entry": entry,
            "Size": size,
            "Current Price": live_price,
            "Unrealised PnL": pnl,
            "Score": float(trade.get("score", 0)),
            "Confidence": float(trade.get("confidence", 0)),
            "ML Prob": float(trade.get("ml_prob", 0)),
            "Pattern": trade.get("pattern"),
            "Session": trade.get("session"),
            "Opened": trade.get("timestamp"),
        })
    if rows:
        return pd.DataFrame(rows).sort_values(by=["Symbol"])
    return pd.DataFrame()


def display_performance_summary(log_df: pd.DataFrame) -> None:
    """Display historical performance metrics and charts."""
    if log_df.empty:
        st.info("No historical trades to display yet.")
        return
    required_cols = {"timestamp", "outcome"}
    if not required_cols.issubset(log_df.columns):
        st.info("Historical log missing columns for performance summary.")
        return
    log_df = log_df.copy()
    log_df["date"] = pd.to_datetime(log_df["timestamp"]).dt.date
    total = len(log_df)
    wins = len(log_df[log_df["outcome"] == "win"])
    win_rate = wins / total if total > 0 else 0.0
    st.subheader("Historical Performance")
    st.write(f"Total trades: {total} | Wins: {wins} | Win rate: {win_rate:.2%}")
    # Compute PnL if columns exist
    if {"exit_price", "entry", "position_size"}.issubset(log_df.columns):
        def calc_pnl(row: pd.Series) -> float:
            try:
                return (float(row["exit_price"]) - float(row["entry"])) * float(row["position_size"])
            except Exception:
                return 0.0
        log_df["pnl"] = log_df.apply(calc_pnl, axis=1)
        daily_pnl = log_df.groupby("date")["pnl"].sum().reset_index()
        if PLOTLY_AVAILABLE:
            fig = px.bar(daily_pnl, x="date", y="pnl", title="Daily PnL")
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(daily_pnl.set_index("date"))


def main() -> None:
    st.set_page_config(page_title="Spot AI Super Agent Dashboard", layout="wide")
    # Custom CSS for professional look
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #f7f9fa;
            padding: 2rem;
        }
        .sidebar .sidebar-content {
            background-color: #f2f4f5;
        }
        .stMetric > div > div {
            font-size: 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🤖 Spot AI Super Agent – Live Dashboard")
    # Sidebar controls
    st.sidebar.title("Settings")
    refresh = st.sidebar.slider("Refresh Interval (seconds)", 10, 60, 30)
    # Auto refresh using Streamlit's experimental function
    st_autorefresh = getattr(st, "experimental_rerun", None)
    # Display session information
    st.sidebar.write(f"**Market Session:** {get_market_session()}")
    # Load data
    trades = load_active_trades()
    metrics = compute_summary_metrics(trades)
    # Summary cards
    cols = st.columns(5)
    cols[0].metric("Active Trades", metrics["count"])
    cols[1].metric("Avg Confidence", metrics["avg_conf"])
    cols[2].metric("Avg ML Prob", metrics["avg_ml_prob"])
    cols[3].metric("Total Exposure (USDT)", f"${metrics['exposure']:.2f}")
    cols[4].metric("Unrealised PnL", f"${metrics['unrealised_pnl']:.2f}")
    st.markdown("---")
    # Active trades table
    if trades:
        df_trades = build_trade_rows(trades)
        st.subheader("Active Trades – Details")
        st.dataframe(df_trades, use_container_width=True)
    else:
        st.info("No active trades found.")
    st.markdown("---")
    # Historical performance
    learning_log_path = os.getenv("TRADE_LEARNING_FILE", os.path.join(os.path.dirname(__file__), "trade_learning_log.csv"))
    df_learning = load_log(learning_log_path)
    if not df_learning.empty:
        display_performance_summary(df_learning)
    st.markdown("---")
    st.caption("Built with ❤️ for Spot AI Super Agent")


if __name__ == "__main__":
    main()
