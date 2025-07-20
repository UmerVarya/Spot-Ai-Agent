import logging
logging.basicConfig(level=logging.INFO)
logging.info("agent.py starting...")

try:
    # Your existing code here
    pass
except Exception as e:
    logging.error(f"Unhandled exception: {e}", exc_info=True)
import time
import os
from dotenv import load_dotenv
load_dotenv()
import json
import threading
from trade_utils import get_top_symbols, get_price_data, evaluate_signal
from trade_manager import manage_trades, create_new_trade, save_active_trades
from notifier import send_email
from brain import should_trade
from sentiment import get_macro_sentiment
from btc_dominance import get_btc_dominance
from fear_greed import get_fear_greed_index
from fetch_news import run_news_fetcher
from orderflow import detect_aggression
from drawdown_guard import is_trading_blocked  # ✅ Drawdown Guard

MAX_ACTIVE_TRADES = 2
SCAN_INTERVAL = 15  # reduced from 60s to 15s for faster scans
NEWS_INTERVAL = 3600

def load_active_trades():
    try:
        with open("active_trades.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_active_trades(trades):
    with open("active_trades.json", "w") as f:
        json.dump(trades, f, indent=4)

def auto_run_news():
    while True:
        print("🗞️ Running scheduled news fetcher...")
        run_news_fetcher()
        time.sleep(NEWS_INTERVAL)

def run_streamlit():
    port = os.environ.get("PORT", "10000")
    os.system(f"streamlit run dashboard.py --server.port {port} --server.headless true")

def run_agent_loop():
    print("\n🤖 Spot AI Super Agent running in paper trading mode...\n")
    threading.Thread(target=auto_run_news, daemon=True).start()

    while True:
        try:
            print(f"=== Scan @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

            # Check if trading is temporarily blocked (e.g., due to drawdown limits)
            if is_trading_blocked():
                print("⛔ Drawdown limit reached. Skipping trading for today.\n")
                time.sleep(SCAN_INTERVAL)
                continue

            # Get macro context
            btc_d = get_btc_dominance()
            fg = get_fear_greed_index()
            sentiment = get_macro_sentiment()
            try:
                sentiment_confidence = float(sentiment.get("confidence", 5.0))
            except:
                sentiment_confidence = 5.0
            sentiment_bias = str(sentiment.get("bias", "neutral"))
            try:
                btc_d = float(btc_d)
            except:
                btc_d = 0.0
            try:
                fg = int(fg)
            except:
                fg = 0

            print(f"🌐 BTC Dominance: {btc_d:.2f}% | Fear & Greed: {fg} | Sentiment: {sentiment_bias} (Confidence: {sentiment_confidence})")

            # Market sentiment gating – only block if conditions indicate strongly bearish environment
            if (sentiment_bias == "bearish" and sentiment_confidence >= 7) or fg < 20 or btc_d > 60:
                reason = []
                if sentiment_bias == "bearish" and sentiment_confidence >= 7:
                    reason.append("strong bearish sentiment")
                if fg < 20:
                    reason.append("extreme Fear & Greed index")
                if btc_d > 60:
                    reason.append("very high BTC dominance")
                reason_text = " + ".join(reason) if reason else "unfavorable conditions"
                print(f"🛑 Market unfavorable ({reason_text}). Skipping scan.\n")
                time.sleep(SCAN_INTERVAL)
                continue

            active_trades = load_active_trades()
            top_symbols = get_top_symbols(limit=30)
            potential_trades = []  # collect potential trade signals

            for symbol in top_symbols:
                # Enforce max concurrency: still scan all symbols for logging, but do not open beyond cap
                # Skip symbols that already have an active trade open
                if symbol in active_trades:
                    print(f"⚠️ Skipping {symbol}: already in an active trade.")
                    continue

                try:
                    price_data = get_price_data(symbol)
                    if price_data is None or price_data.empty or len(price_data) < 20:
                        print(f"⚠️ Skipping {symbol} due to insufficient data.\n")
                        continue

                    score, direction, position_size, pattern_name = evaluate_signal(price_data, symbol)
                    # ✅ Fallback: if no direction but score meets threshold, assume long (for neutral sentiment scenarios)
                    if direction is None and score >= 5.5:
                        print(f"⚠️ No clear direction for {symbol} despite score={score:.2f}. Forcing fallback 'long'.")
                        direction = "long"

                    # Skip if signal does not qualify (no long direction or position size 0)
                    if direction != "long" or position_size <= 0:
                        # Log skip reasons were already printed during evaluate_signal (volume, VWMA, sentiment, zone filters, etc.)
                        continue

                    # Order flow check (must have bullish aggression)
                    flow_status = detect_aggression(price_data)
                    if flow_status != "buyers in control":
                        if flow_status == "sellers in control":
                            print(f"🚫 Bearish order flow detected in {symbol}. Skipping trade.")
                        else:
                            print(f"🚫 No buy-side aggression detected in {symbol}. Skipping trade.")
                        continue

                    # If we reach here, we have a valid potential long trade signal
                    potential_trades.append({
                        "symbol": symbol,
                        "score": score,
                        "direction": "long",
                        "position_size": position_size,
                        "pattern": pattern_name,
                        "price_data": price_data
                    })

                except Exception as e:
                    print(f"❌ Error evaluating {symbol}: {e}")
                    continue

            # Rank all potential trades by score (descending) to prioritize top signals
            potential_trades.sort(key=lambda x: x['score'], reverse=True)

            # Determine how many new trades we can open this cycle
            allowed_new = MAX_ACTIVE_TRADES - len(active_trades)
            opened_count = 0

            for signal in potential_trades:
                symbol = signal["symbol"]
                score = signal["score"]
                direction = signal["direction"]
                position_size = signal["position_size"]
                pattern_name = signal["pattern"]
                price_data = signal["price_data"]

                if opened_count >= allowed_new:
                    # Concurrency limit reached for this cycle - highlight skipped trade
                    print(f"⚠️ Skipping {symbol} due to concurrency cap (max {MAX_ACTIVE_TRADES} trades).")
                    continue

                # Prepare indicators for brain decision and call should_trade
                indicators = {
                    "rsi": price_data["rsi"].iloc[-1] if "rsi" in price_data else 50,
                    "macd": price_data["macd"].iloc[-1] if "macd" in price_data else 0,
                    "adx": price_data["adx"].iloc[-1] if "adx" in price_data else 20,
                    "volume": price_data["volume"].iloc[-1] if "volume" in price_data else 0,
                }
                orderflow_tag = "buy-side aggression"  # we've ensured buy-side flow is present
                decision_obj = should_trade(
                    symbol=symbol,
                    score=score,
                    direction=direction,
                    indicators=indicators,
                    session="default",
                    pattern_name=pattern_name,
                    orderflow=orderflow_tag,
                    sentiment=sentiment,
                    macro_news=sentiment
                )
                decision = decision_obj.get("decision", False)
                final_conf = decision_obj.get("confidence", 0.0)
                narrative = decision_obj.get("narrative", "")

                # Log brain decision outcome with confidence and reason/narrative
                if decision:
                    print(f"🤖 Brain Decision for {symbol}: True | Confidence: {final_conf:.2f}\n{narrative}\n")
                else:
                    reason = decision_obj.get("reason", "Unknown reason")
                    print(f"🤖 Brain Decision for {symbol}: False | Confidence: {final_conf:.2f}\nReason: {reason}\n")

                # If brain approves the trade, execute in paper mode
                if decision and direction == "long" and position_size > 0:
                    entry_price = round(price_data['close'].iloc[-1], 6)
                    sl = round(entry_price - entry_price * 0.01, 6)
                    tp1 = round(entry_price + entry_price * 0.01, 6)
                    tp2 = round(entry_price + entry_price * 0.015, 6)
                    tp3 = round(entry_price + entry_price * 0.025, 6)

                    new_trade = {
                        "symbol": symbol,
                        "direction": "long",
                        "entry": entry_price,
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "tp3": tp3,
                        "position_size": position_size,
                        "confidence": final_conf,
                        "btc_dominance": btc_d,
                        "fear_greed": fg,
                        "sentiment_bias": sentiment_bias,
                        "sentiment_confidence": sentiment_confidence,
                        "sentiment_summary": sentiment.get("summary", ""),
                        "pattern": pattern_name,
                        "narrative": narrative,
                        "status": {"tp1": False, "tp2": False, "tp3": False},
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    print(f"📖 Narrative:\n{narrative}\n")
                    print(f"✅ Trade: {symbol} | Score: {score:.2f} | Size: ${position_size}")
                    active_trades[symbol] = new_trade
                    save_active_trades(active_trades)  # update persistent active trades
                    send_email(f"New Trade Opened: {symbol}", str(new_trade))
                    opened_count += 1
                # If decision was False or position_size 0, we do not open trade (already logged reason above).

            # Manage existing trades (update SL/TP, handle exits) and then pause until next scan
            manage_trades()
            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            print(f"❌ Main Loop Error: {e}")
            time.sleep(10)
