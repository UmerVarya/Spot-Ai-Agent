# trade_logger.py

import csv
import os
from datetime import datetime

# === Log Final Trade Outcome ===
def log_trade_result(trade, outcome):
    log_file = "trade_learning_log.csv"
    fields = [
        "timestamp", "symbol", "session", "score", "direction",
        "outcome", "btc_dominance", "fear_greed", "sentiment",
        "pattern", "support_zone", "resistance_zone", "volume", "confidence"
    ]
    
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": trade["symbol"],
        "session": trade.get("session", "unknown"),
        "score": trade.get("confidence", 0),
        "direction": trade.get("direction", "long"),
        "outcome": outcome,
        "btc_dominance": trade.get("btc_dominance", 0),
        "fear_greed": trade.get("fear_greed", 0),
        "sentiment": trade.get("sentiment_bias", "unknown"),
        "pattern": trade.get("chart_pattern", "none"),
        "support_zone": trade.get("support_zone", False),
        "resistance_zone": trade.get("resistance_zone", False),
        "volume": trade.get("volume", 0),
        "confidence": trade.get("sentiment_confidence", 0)
    }

    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
