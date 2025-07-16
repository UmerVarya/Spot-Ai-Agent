# trade_storage.py

import json
import os
import csv
from datetime import datetime

ACTIVE_FILE = "active_trades.json"
LOG_FILE = "trade_log.csv"

def load_active_trades():
    if os.path.exists(ACTIVE_FILE):
        with open(ACTIVE_FILE, "r") as f:
            return json.load(f)
    return []

def save_active_trades(trades):
    with open(ACTIVE_FILE, "w") as f:
        json.dump(trades, f, indent=4)

def is_trade_active(symbol):
    trades = load_active_trades()
    for trade in trades:
        if trade["symbol"] == symbol:
            return True
    return False

def store_trade(trade):
    trades = load_active_trades()
    trades.append(trade)
    save_active_trades(trades)

def remove_trade(symbol):
    trades = load_active_trades()
    updated = [t for t in trades if t["symbol"] != symbol]
    save_active_trades(updated)

def log_trade_result(trade, outcome, exit_price):
    headers = [
        "timestamp", "symbol", "direction", "entry", "exit", "outcome",
        "confidence", "btc_dominance", "fear_greed", "macro_filtered",
        "score", "pattern", "narrative"
    ]

    log_data = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": trade["symbol"],
        "direction": trade["direction"],
        "entry": trade["entry"],
        "exit": exit_price,
        "outcome": outcome,
        "confidence": trade.get("confidence", 0),
        "btc_dominance": trade.get("btc_dominance", 0),
        "fear_greed": trade.get("fear_greed", 0),
        "macro_filtered": trade.get("sentiment_bias") or trade.get("macro_filtered", "neutral"),
        "score": trade.get("strength", 0),
        "pattern": trade.get("pattern", "None"),
        "narrative": trade.get("narrative", "No explanation")
    }

    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
