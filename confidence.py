# confidence.py

import pandas as pd
import os

LOG_FILE = "trade_learning_log.csv"

def calculate_historical_confidence(symbol, score, direction, session="Unknown", pattern_name=None):
    try:
        if not os.path.exists(LOG_FILE):
            return {"confidence": 50, "reasoning": "No historical data yet."}

        df = pd.read_csv(LOG_FILE)

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
                (df['session'] == session)
            ]
        else:
            pattern_trades = symbol_trades

        total_trades = len(pattern_trades)
        wins = pattern_trades[pattern_trades["outcome"] == "win"]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.5

        # Confidence scaling
        base_confidence = win_rate * 100

        # Boost confidence if many recent wins
        recent = symbol_trades.tail(10)
        recent_win_rate = len(recent[recent["outcome"] == "win"]) / len(recent) if len(recent) > 0 else 0.5
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
