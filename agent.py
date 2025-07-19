import time
import os
from dotenv import load_dotenv
load_dotenv()
import json
import threading
from trade_utils import get_top_symbols, get_price_data, evaluate_signal
from trade_manager import manage_trades, create_new_trade
from notifier import send_email
from brain import should_trade
from sentiment import get_macro_sentiment
from btc_dominance import get_btc_dominance
from fear_greed import get_fear_greed_index
from fetch_news import run_news_fetcher
from orderflow import detect_aggression
from drawdown_guard import is_trading_blocked  # ✅ Drawdown Guard

MAX_ACTIVE_TRADES = 2
SCAN_INTERVAL = 60
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

            if is_trading_blocked():
                print("⛔ Drawdown limit reached. Skipping trading for today.\n")
                time.sleep(SCAN_INTERVAL)
                continue

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

            if sentiment_bias == "bearish" or sentiment_confidence < 5 or fg < 30 or btc_d > 55:
                print("🛑 Market unfavorable. Skipping scan.\n")
                time.sleep(SCAN_INTERVAL)
                continue

            active_trades = load_active_trades()
            top_symbols = get_top_symbols()

            for symbol in top_symbols:
                if len(active_trades) >= MAX_ACTIVE_TRADES:
                    print("⚠️ Max active trades reached. Waiting...\n")
                    break

                try:
                    price_data = get_price_data(symbol)
                    if price_data is None or price_data.empty or len(price_data) < 20:
                        print(f"⚠️ Skipping {symbol} due to insufficient data.\n")
                        continue

                    score, direction, position_size, pattern_name = evaluate_signal(price_data, symbol)

                    # ✅ Fallback direction logic
                    if direction is None and score >= 6.5:
                        print(f"⚠️ No direction for {symbol} despite score={score}. Forcing fallback 'long'.")
                        direction = "long"

                    if not detect_aggression(price_data):
                        print(f"🚫 No buy-side aggression detected in {symbol}. Skipping.")
                        continue

                    indicators = {
                        "rsi": price_data["rsi"].iloc[-1] if "rsi" in price_data else 50,
                        "macd": price_data["macd"].iloc[-1] if "macd" in price_data else 0,
                        "adx": price_data["adx"].iloc[-1] if "adx" in price_data else 20,
                        "volume": price_data["volume"].iloc[-1] if "volume" in price_data else 0,
                    }

                    decision_obj = should_trade(
                        symbol=symbol,
                        score=score,
                        direction=direction,
                        indicators=indicators,
                        session="default",
                        pattern_name=pattern_name,
                        orderflow="buy-side aggression",
                        sentiment=sentiment,
                        macro_news=sentiment
                    )

                    decision = decision_obj.get("decision", False)
                    final_conf = decision_obj.get("confidence", 0.0)
                    narrative = decision_obj.get("narrative", "No narrative available.")

                    print(f"🤖 Brain Decision for {symbol}: {decision} | Confidence: {final_conf:.2f}\n{narrative}\n")

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
                        create_new_trade(new_trade)
                        send_email(f"New Trade Opened: {symbol}", str(new_trade))
                        active_trades[symbol] = new_trade
                        save_active_trades(active_trades)

                    else:
                        print(f"⛔ Skipped {symbol}: Score={score}, Dir={direction}, PosSize={position_size}")

                except Exception as e:
                    print(f"❌ Error evaluating {symbol}: {e}")
                    continue

            manage_trades()
            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            print(f"❌ Main Loop Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    threading.Thread(target=run_streamlit).start()
    threading.Thread(target=run_agent_loop).start()

    # ✅ Hardcoded Top 30 (for other tasks if needed)
    from sentiment import get_macro_sentiment
    from trade_utils import get_price_data, evaluate_signal

    top_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT",
        "DOTUSDT", "MATICUSDT", "TRXUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT", "BCHUSDT", "ETCUSDT",
        "FILUSDT", "ICPUSDT", "APTUSDT", "SUIUSDT", "INJUSDT", "OPUSDT", "ARBUSDT", "XLMUSDT",
        "HBARUSDT", "EGLDUSDT", "VETUSDT", "RUNEUSDT", "FTMUSDT", "AAVEUSDT"
    ]

    sentiment = get_macro_sentiment()
    sentiment_bias = sentiment.get("bias", "neutral")

    symbol_scores = {}

    for symbol in top_symbols:
        df = get_price_data(symbol)
        score, direction, position_size, pattern = evaluate_signal(df, symbol, sentiment_bias)
        symbol_scores[symbol] = {
            "score": score,
            "direction": direction
        }

    with open("symbol_scores.json", "w") as f:
        json.dump(symbol_scores, f)
