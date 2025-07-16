# drawdown_guard.py

import os
import json
from datetime import datetime

GUARD_FILE = "drawdown_guard.json"
SL_LIMIT = 2
PNL_LIMIT = -3  # in percent

def load_guard():
    if os.path.exists(GUARD_FILE):
        with open(GUARD_FILE, "r") as f:
            return json.load(f)
    return {}

def save_guard(data):
    with open(GUARD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_guard(outcome, pnl_percent):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    guard = load_guard()
    if today not in guard:
        guard[today] = {"sl_count": 0, "pnl": 0.0, "blocked": False}

    if outcome == "sl":
        guard[today]["sl_count"] += 1
    guard[today]["pnl"] += pnl_percent

    # Trigger block
    if guard[today]["sl_count"] >= SL_LIMIT or guard[today]["pnl"] <= PNL_LIMIT:
        guard[today]["blocked"] = True

    save_guard(guard)

def is_trading_blocked():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    guard = load_guard()
    return guard.get(today, {}).get("blocked", False)
