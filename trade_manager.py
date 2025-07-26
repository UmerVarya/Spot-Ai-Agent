"""
Enhanced trade management utilities for the Spot AI Super Agent.

This module refines the lifecycle management of open trades.  It adds
configurable ATR‑based exit criteria, dynamic trailing stop rules, and
robust error handling so that the agent can operate reliably over long
periods.  All active trades are stored in a shared JSON file whose
location may be customised via the ``ACTIVE_TRADES_FILE`` environment
variable.  Only spot (long) trades are supported.

Functions
---------
load_active_trades()
    Load the current open trades from disk.
save_active_trades(trades)
    Persist the open trades to disk.
create_new_trade(trade)
    Insert a newly opened trade into the active trades file.
should_exit_early(trade, current_price, price_data)
    Determine if an open trade should exit early before TP or SL.
manage_trades()
    Iterate through active trades, update trailing stops and close trades
    based on take‑profit hits, stop‑loss triggers or weakening momentum.

Configuration
-------------
The following environment variables control risk parameters:

* ``EARLY_EXIT_ATR_MULTIPLIER`` – multiplier of the ATR to determine
  early exit threshold.  Default 1.5.  A trade exits early if the
  price drops by more than this multiplier times the ATR from the
  entry price.
* ``TRAILING_ATR_MULTIPLIER`` – multiplier of the ATR used when
  tightening or trailing stop losses after TP1 and during TP4 profit
  riding.  Default 1.0.
* ``MACRO_CONFIDENCE_EXIT_THRESHOLD`` – macro sentiment confidence
  threshold below which trades exit on bearish bias.  Default 4.

By tuning these parameters, you can adjust the bot's risk tolerance
without modifying the code.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np

from trade_utils import get_price_data, calculate_indicators
from macro_sentiment import analyze_macro_sentiment
from notifier import send_email
from trade_logger import log_trade_result

# Path to the JSON file storing active trades
import tempfile
ACTIVE_TRADES_FILE = os.environ.get(
    "ACTIVE_TRADES_FILE",
    os.path.join(tempfile.gettempdir(), "active_trades.json"),
)

# Risk parameters configurable via environment variables
EARLY_EXIT_ATR_MULTIPLIER = float(os.getenv("EARLY_EXIT_ATR_MULTIPLIER", 1.5))
TRAILING_ATR_MULTIPLIER = float(os.getenv("TRAILING_ATR_MULTIPLIER", 1.0))
MACRO_CONFIDENCE_EXIT_THRESHOLD = float(os.getenv("MACRO_CONFIDENCE_EXIT_THRESHOLD", 4))


def load_active_trades() -> Dict[str, dict]:
    """Load all currently active trades from disk.  Returns empty dict on failure."""
    try:
        with open(ACTIVE_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_active_trades(trades: Dict[str, dict]) -> None:
    """Persist the current active trades to disk."""
    try:
        with open(ACTIVE_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=4)
    except Exception as e:
        print(f"⚠️ Unable to save active trades: {e}")


def create_new_trade(trade: dict) -> None:
    """Add a new trade to the active trades file."""
    active_trades = load_active_trades()
    active_trades[trade["symbol"]] = trade
    save_active_trades(active_trades)


def should_exit_early(trade: dict, current_price: float, price_data) -> Tuple[bool, Optional[str]]:
    """Evaluate if a trade should exit early before reaching TP or SL.

    Early exit triggers if the price moves against the entry by more
    than ``EARLY_EXIT_ATR_MULTIPLIER`` times the ATR, if RSI and MACD
    show weakening momentum, or if macro sentiment flips bearish with
    low confidence.  Only long trades are considered.
    """
    entry = trade.get('entry')
    direction = trade.get('direction')
    if entry is None or direction != "long":
        return False, None
    # Compute latest indicators to obtain ATR, RSI and MACD histogram
    try:
        indicators = calculate_indicators(price_data)
        atr_series = indicators.get("atr", 0)
        rsi_series = indicators.get("rsi", 50)
        macd_series = indicators.get("macd", 0)
        atr = float(atr_series.iloc[-1]) if hasattr(atr_series, 'iloc') else float(atr_series)
        rsi = float(rsi_series.iloc[-1]) if hasattr(rsi_series, 'iloc') else float(rsi_series)
        macd_hist = float(macd_series.iloc[-1]) if hasattr(macd_series, 'iloc') else float(macd_series)
    except Exception:
        # If indicator computation fails, skip early exit
        return False, None
    # 1. Price moved adversely relative to ATR
    if current_price < entry - atr * EARLY_EXIT_ATR_MULTIPLIER:
        return True, f"Price dropped {EARLY_EXIT_ATR_MULTIPLIER}× ATR below entry"
    # 2. Weakening momentum based on RSI and MACD histogram
    if rsi < 45:
        return True, f"Weak RSI: {rsi:.2f}"
    if macd_hist < 0:
        return True, f"MACD histogram reversed: {macd_hist:.4f}"
    # 3. Macro sentiment turned bearish with low confidence
    macro = analyze_macro_sentiment()
    if macro.get('bias') == 'bearish' and float(macro.get('confidence', 0)) < MACRO_CONFIDENCE_EXIT_THRESHOLD:
        return True, f"Macro sentiment bearish (Confidence: {macro.get('confidence')})"
    return False, None


def manage_trades() -> None:
    """Iterate over active trades, update SL/TP and close trades when needed."""
    active_trades = load_active_trades()
    updated_trades: Dict[str, dict] = {}
    for symbol, trade in active_trades.items():
        try:
            price_data = get_price_data(symbol)
            if price_data is None or price_data.empty:
                continue
            current_price = float(price_data['close'].iloc[-1])
        except Exception:
            continue
        direction = trade.get('direction')
        entry = trade.get('entry')
        sl = trade.get('sl')
        tp1 = trade.get('tp1')
        tp2 = trade.get('tp2')
        tp3 = trade.get('tp3')
        # Early exit evaluation
        exit_now, reason = should_exit_early(trade, current_price, price_data)
        if exit_now:
            print(f"🚨 Early exit triggered for {symbol}: {reason}")
            trade['exit_price'] = current_price
            trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            trade['outcome'] = "early_exit"
            trade['exit_reason'] = reason
            log_trade_result(trade, outcome="early_exit", exit_price=current_price)
            send_email(f"🚨 Early Exit: {symbol}", f"{trade}\n\n🧠 Narrative:\n{trade.get('narrative', 'N/A')}")
            continue
        # Compute indicators for trailing logic
        indicators = calculate_indicators(price_data)
        adx_series = indicators.get("adx", 20)
        macd_series = indicators.get("macd", 0)
        atr_series = indicators.get("atr", 0.005)
        adx = float(adx_series.iloc[-1]) if hasattr(adx_series, 'iloc') else float(adx_series)
        macd_hist = float(macd_series.iloc[-1]) if hasattr(macd_series, 'iloc') else float(macd_series)
        atr = float(atr_series.iloc[-1]) if hasattr(atr_series, 'iloc') else float(atr_series)
        # Only long trades supported
        if direction == "long":
            # Take profit hits update status and stop loss
            status = trade.setdefault('status', {})
            # Hit TP1
            if not status.get('tp1') and current_price >= tp1:
                status['tp1'] = True
                trade['sl'] = entry  # move stop to entry price
                print(f"🎯 {symbol} hit TP1 — SL moved to entry")
            # Hit TP2
            elif status.get('tp1') and not status.get('tp2') and current_price >= tp2:
                status['tp2'] = True
                trade['sl'] = tp1
                print(f"🎯 {symbol} hit TP2 — SL moved to TP1")
            # Hit TP3
            elif status.get('tp2') and not status.get('tp3') and current_price >= tp3:
                status['tp3'] = True
                trade['profit_riding'] = True
                trade['sl'] = tp2
                print(f"🚀 {symbol} hit TP3 — entering TP4 profit riding mode")
                updated_trades[symbol] = trade
                continue
            # Stop loss triggered
            elif current_price <= sl:
                print(f"🛑 {symbol} hit stop loss!")
                trade['exit_price'] = sl
                trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trade['outcome'] = "tp4_sl" if trade.get('profit_riding') else "sl"
                log_trade_result(trade, outcome=trade['outcome'], exit_price=sl)
                send_email(f"🛑 Stop Loss Hit: {symbol}", f"{trade}\n\n🧠 Narrative:\n{trade.get('narrative', 'N/A')}")
                continue
            # Tighten SL after TP1 before TP3
            if status.get('tp1') and not trade.get('profit_riding'):
                # If ADX is low or MACD histogram negative, tighten stop by ATR * TRAILING_ATR_MULTIPLIER
                if adx < 15 or macd_hist < 0:
                    tightened_sl = round(current_price - atr * TRAILING_ATR_MULTIPLIER, 6)
                    if tightened_sl > trade['sl']:
                        trade['sl'] = tightened_sl
                        print(f"🔒 SL tightened for {symbol} to {trade['sl']}")
            # Profit riding trailing logic (TP4)
            if trade.get('profit_riding'):
                if adx > 25 and macd_hist > 0:
                    trail_sl = round(current_price - atr * TRAILING_ATR_MULTIPLIER, 6)
                    if trail_sl > trade['sl']:
                        trade['sl'] = trail_sl
                        print(f"🏇 {symbol} TP4 ride: SL trailed to {trail_sl}")
                else:
                    print(f"⚠️ {symbol} momentum weakening — exiting TP4")
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    trade['outcome'] = "tp4"
                    log_trade_result(trade, outcome="tp4", exit_price=current_price)
                    send_email(f"✅ TP4 Exit: {symbol}", f"{trade}\n\n🧠 Narrative:\n{trade.get('narrative', 'N/A')}")
                    continue
        # Add still active trade back to updated list
        updated_trades[symbol] = trade
    # Save updated trades
    save_active_trades(updated_trades)
