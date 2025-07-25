"""
Utilities for retrieving past trade memories.

This module provides helper functions to summarise recent trades from the
trade learning log.  The summary can be used to give the LLM context
about similar past trades, enabling it to reason based on prior
experience (retrieval‑augmented generation).
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

# Path to the learning log file (same directory as this module)
LOG_FILE = os.path.join(os.path.dirname(__file__), "trade_learning_log.csv")


def get_recent_trade_summary(symbol: str, pattern: str, max_entries: int = 3) -> str:
    """Summarise recent trades for a given symbol or pattern.

    Parameters
    ----------
    symbol : str
        The symbol to match (e.g., "BTCUSDT").
    pattern : str
        The pattern name used to filter trades (case‑insensitive).  If
        empty or "None", the pattern filter is ignored.
    max_entries : int, optional
        The maximum number of recent trade entries to include.  Defaults
        to 3.

    Returns
    -------
    str
        A concatenated summary of the most recent matching trades.  Each
        trade is summarised by its timestamp, outcome and pattern.  If
        no matching trades exist, a default string is returned.
    """
    if not os.path.exists(LOG_FILE):
        return "No prior trades on record."
    try:
        df = pd.read_csv(LOG_FILE, engine="python", on_bad_lines="skip")
    except Exception:
        return "(Unable to read trade log.)"
    # Filter by symbol and (optionally) pattern
    df = df[df.get("symbol") == symbol]
    if pattern and pattern.lower() != "none":
        df = df[df.get("pattern", "").str.lower() == pattern.lower()]
    if df.empty:
        return "No prior trades on record."
    # Sort by timestamp (assuming timestamp column exists)
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp", ascending=False)
    # Take the most recent entries
    df = df.head(max_entries)
    summaries = []
    for _, row in df.iterrows():
        ts = row.get("timestamp", "?")
        outcome = row.get("outcome", "open")
        pat = row.get("pattern", "?")
        conf = row.get("confidence", "?")
        summaries.append(f"[{ts}] {pat} → {outcome} (conf {conf})")
    return "; ".join(summaries)
