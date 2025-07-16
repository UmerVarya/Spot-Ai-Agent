import json
import time
from datetime import datetime
from trade_utils import get_price_data, calculate_indicators
from macro_sentiment import analyze_macro_sentiment
from notifier import send_email
from trade_logger import log_trade_result

EARLY_EXIT_THRESHOLD = 0.015  # 1.5%
MACRO_CONFIDENCE_EXIT_THRESHOLD = 4

def load_active_trades():
    try:
        with open("active_trades.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_active_trades(trades):
    with open("active_trades.json", "w") as f:
        json.dump(trades, f, indent=4)

def create_new_trade(trade):
    active_trades = load_active_trades()
    active_trades[trade["symbol"]] = trade
    save_active_trades(active_trades)

def should_exit_early(trade, current_price, price_data):
    entry = trade['entry']
    direction = trade['direction']

    # 1. Price reversed significantly
    if direction == "long" and current_price < entry * (1 - EARLY_EXIT_THRESHOLD):
        return True, "Price dropped beyond early exit threshold"
    if direction == "short" and current_price > entry * (1 + EARLY_EXIT_THRESHOLD):
        return True, "Price rose beyond early exit threshold"

    # 2. Indicator Weakness
    indicators = calculate_indicators(price_data)
    rsi = indicators.get("rsi", 50)
    macd_hist = indicators.get("macd", 0)

    if hasattr(rsi, 'iloc'):
        rsi = rsi.iloc[-1]
    if hasattr(macd_hist, 'iloc'):
        macd_hist = macd_hist.iloc[-1]

    if direction == "long" and rsi < 45:
        return True, f"Weak RSI: {rsi:.2f}"
    if direction == "short" and rsi > 55:
        return True, f"Weak RSI: {rsi:.2f}"

    if direction == "long" and macd_hist < 0:
        return True, f"MACD histogram reversed: {macd_hist:.4f}"
    if direction == "short" and macd_hist > 0:
        return True, f"MACD histogram reversed: {macd_hist:.4f}"

    # 3. Macro Shift
    macro = analyze_macro_sentiment()
    if macro['bias'] == "bearish" and macro['confidence'] < MACRO_CONFIDENCE_EXIT_THRESHOLD:
        return True, f"Macro sentiment shifted to bearish (Confidence: {macro['confidence']})"

    return False, None

def manage_trades():
    active_trades = load_active_trades()
    updated_trades = {}

    for symbol, trade in active_trades.items():
        price_data = get_price_data(symbol)
        if price_data is None or price_data.empty:
            continue

        current_price = price_data['close'].iloc[-1]
        direction = trade['direction']
        entry = trade['entry']
        sl = trade['sl']
        tp1 = trade['tp1']
        tp2 = trade['tp2']
        tp3 = trade['tp3']

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

        indicators = calculate_indicators(price_data)
        adx = indicators.get("adx", 20)
        macd_hist = indicators.get("macd", 0)
        atr = indicators.get("atr", 0.005)

        if hasattr(macd_hist, 'iloc'):
            macd_hist = macd_hist.iloc[-1]
        if hasattr(adx, 'iloc'):
            adx = adx.iloc[-1]

        if direction == "long":
            if not trade['status']['tp1'] and current_price >= tp1:
                trade['status']['tp1'] = True
                trade['sl'] = entry
                print(f"🎯 {symbol} hit TP1 — SL moved to Entry")

            elif trade['status']['tp1'] and not trade['status']['tp2'] and current_price >= tp2:
                trade['status']['tp2'] = True
                trade['sl'] = tp1
                print(f"🎯 {symbol} hit TP2 — SL moved to TP1")

            elif trade['status']['tp2'] and not trade['status']['tp3'] and current_price >= tp3:
                trade['status']['tp3'] = True
                trade['profit_riding'] = True  # ✅ Enable TP4 mode
                trade['sl'] = tp2  # Set SL just below
                print(f"🚀 {symbol} hit TP3 — Entering TP4 Profit Riding Mode")
                continue

            elif current_price <= sl:
                print(f"🛑 {symbol} hit Stop Loss!")
                trade['exit_price'] = sl
                trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trade['outcome'] = "tp4_sl" if trade.get("profit_riding") else "sl"
                log_trade_result(trade, outcome=trade['outcome'], exit_price=sl)
                send_email(f"🛑 Stop Loss Hit: {symbol}", f"{trade}\n\n🧠 Narrative:\n{trade.get('narrative', 'N/A')}")
                continue

            if trade['status']['tp1'] and not trade.get('profit_riding'):
                if adx < 15 or macd_hist < 0:
                    tightened_sl = round(current_price - atr, 6)
                    if tightened_sl > trade['sl']:
                        trade['sl'] = tightened_sl
                        print(f"🔒 SL tightened for {symbol} to {trade['sl']}")

            # ✅ TP4 Profit Riding
            if trade.get("profit_riding"):
                if adx > 25 and macd_hist > 0:
                    trail_sl = round(current_price - atr, 6)
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

        # === SHORT logic remains unchanged
        elif direction == "short":
            if not trade['status']['tp1'] and current_price <= tp1:
                trade['status']['tp1'] = True
                trade['sl'] = entry
                print(f"🎯 {symbol} hit TP1 — SL moved to Entry")

            elif trade['status']['tp1'] and not trade['status']['tp2'] and current_price <= tp2:
                trade['status']['tp2'] = True
                trade['sl'] = tp1
                print(f"🎯 {symbol} hit TP2 — SL moved to TP1")

            elif trade['status']['tp2'] and not trade['status']['tp3'] and current_price <= tp3:
                trade['status']['tp3'] = True
                trade['exit_price'] = tp3
                trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trade['outcome'] = "tp3"
                log_trade_result(trade, outcome="tp3", exit_price=tp3)
                send_email(f"✅ TP3 Hit: {symbol}", f"{trade}\n\n🧠 Narrative:\n{trade.get('narrative', 'N/A')}")
                continue

            elif current_price >= sl:
                print(f"🛑 {symbol} hit Stop Loss!")
                trade['exit_price'] = sl
                trade['exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                trade['outcome'] = "sl"
                log_trade_result(trade, outcome="sl", exit_price=sl)
                send_email(f"🛑 Stop Loss Hit: {symbol}", f"{trade}\n\n🧠 Narrative:\n{trade.get('narrative', 'N/A')}")
                continue

            if trade['status']['tp1'] and not trade['status']['tp3']:
                if adx < 15 or macd_hist > 0:
                    tightened_sl = round(current_price + atr, 6)
                    if tightened_sl < trade['sl']:
                        trade['sl'] = tightened_sl
                        print(f"🔒 SL tightened for {symbol} to {trade['sl']}")

        updated_trades[symbol] = trade

    save_active_trades(updated_trades)
