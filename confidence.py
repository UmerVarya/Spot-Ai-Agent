"""
Historical confidence calculator for trade outcomes.

This module provides a function to compute a confidence estimate for a
potential trade based on past performance recorded in ``trade_learning_log.csv``.
It reads the log with tolerant parsing (skipping bad lines) and filters
similar trades by symbol, direction, score range and session.  The
confidence is scaled between 0 and 100 and includes a small boost if
recent trades show better performance.
"""

import pandas as pd
import os

import os

# Path to the learning log CSV.  Use a fixed path relative to this module
# so it is consistent regardless of the current working directory.
LOG_FILE = os.path.join(os.path.dirname(__file__), "trade_learning_log.csv")


def calculate_historical_confidence(symbol, score, direction, session="Unknown", pattern_name=None):
    """
    Calculate a historical confidence estimate based on prior trades.

    Parameters
    ----------
    symbol : str
        The trading symbol (e.g., BTCUSDT).
    score : float
        The technical score for the current setup.
    direction : str
        The trade direction ("long" or "short").
    session : str, optional
        The trading session (e.g., Asia, Europe, US).  Default is "Unknown".
    pattern_name : str, optional
        Name of the pattern detected for confluence.

    Returns
    -------
    dict
        A dictionary with keys ``confidence`` (0–100) and ``reasoning``.
    """
    try:
        if not os.path.exists(LOG_FILE):
            return {"confidence": 50, "reasoning": "No historical data yet."}

        # Use python engine and skip bad lines to handle inconsistent logs
        df = pd.read_csv(LOG_FILE, engine="python", on_bad_lines="skip", encoding="utf-8")

        if df.empty or 'score' not in df.columns or 'direction' not in df.columns or 'outcome' not in df.columns:
            return {"confidence": 50, "reasoning": "Incomplete or invalid learning log."}

        # Filter recent trades for the same symbol and direction
        symbol_trades = df[
            (df['symbol'] == symbol) &
            (df['direction'] == direction)
        ].tail(50)

        # Pattern-specific reinforcement if available
        if pattern_name:
            pattern_trades = df[
                (df['symbol'] == symbol) &
                (df['direction'] == direction) &
                (df['score'].between(score - 1.5, score + 1.5)) &
                (df.get('session') == session)
            ]
        else:
            pattern_trades = symbol_trades

        total_trades = len(pattern_trades)
        wins = pattern_trades[pattern_trades["outcome"] == "win"]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.5

        # Confidence scaling: base confidence proportional to win rate
        base_confidence = win_rate * 100

        # Boost confidence if many recent wins
        recent = symbol_trades.tail(10)
        if len(recent) > 0:
            recent_win_rate = len(recent[recent["outcome"] == "win"]) / len(recent)
        else:
            recent_win_rate = 0.5
        boost = (recent_win_rate - 0.5) * 30  # ±15 max boost

        final_conf = min(max(base_confidence + boost, 0), 100)

        return {
            "confidence": round(final_conf, 2),
            "reasoning": f"{symbol}: {len(wins)} wins out of {total_trades} similar trades. Boost from recent: {round(boost, 2)}"
        }

    except Exception as e:
        return {
            "confidence": 50,
            "reasoning": f"Confidence calculation error: {e}"
        }
