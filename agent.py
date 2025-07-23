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
from fetch_news import fetch_news
from trade_utils import simulate_slippage, estimate_commission
from trade_utils import get_top_symbols, get_price_data, evaluate_signal
from trade_manager import manage_trades, load_active_trades, save_active_trades
from notifier import send_email
from brain import should_trade
from sentiment import get_macro_sentiment
from btc_dominance import get_btc_dominance
from fear_greed import get_fear_greed_index
from orderflow import detect_aggression
from drawdown_guard import is_trading_blocked  # ✅ Drawdown Guard

class Agent:
    def __init__(self, symbol, starting_balance=10000.0, trade_quantity=1):
        """
        Initialize the trading agent.
        Args:
            symbol (str): The trading symbol (e.g., stock ticker or crypto pair).
            starting_balance (float): Starting account balance for the agent.
            trade_quantity (int): Default quantity of shares/units to trade.
        """
        self.symbol = symbol
        self.balance = starting_balance
        self.trade_quantity = trade_quantity
        self.in_position = False      # Whether currently holding an open position
        self.entry_price = None       # Price at which current position was entered
        self.total_profit = 0.0       # Cumulative profit/loss

MAX_ACTIVE_TRADES = 2
SCAN_INTERVAL = 15   # scan more frequently (15s)
NEWS_INTERVAL = 3600

def auto_run_news():
    while True:
        print("🗞️ Running scheduled news fetcher...")
        run_news_fetcher()
        time.sleep(NEWS_INTERVAL)

def run_streamlit():
    port = os.environ.get("PORT", "10000")  # Render provides a dynamic port
    os.system(f"streamlit run dashboard.py --server.port {port} --server.headless true")

def run_agent_loop():
    print("\n🤖 Spot AI Super Agent running in paper trading mode...\n")
    threading.Thread(target=auto_run_news, daemon=True).start()
    threading.Thread(target=run_streamlit, daemon=True).start()

    while True:
        try:
            print(f"=== Scan @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

            # Check if trading is temporarily blocked (e.g., drawdown limit hit)
            if is_trading_blocked():
                print("⛔ Drawdown limit reached. Skipping trading for today.\n")
                time.sleep(SCAN_INTERVAL)
                continue

            # Get macro context indicators
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

            # Market sentiment gating – skip trading if conditions indicate a strongly bearish environment
            if (sentiment_bias == "bearish" and sentiment_confidence >= 7) or fg < 20 or btc_d > 60:
                reasons = []
                if sentiment_bias == "bearish" and sentiment_confidence >= 7:
                    reasons.append("strong bearish sentiment")
                if fg < 20:
                    reasons.append("extreme Fear & Greed index")
                if btc_d > 60:
                    reasons.append("very high BTC dominance")
                reason_text = " + ".join(reasons) if reasons else "unfavorable conditions"
                print(f"🛑 Market unfavorable ({reason_text}). Skipping scan.\n")
                time.sleep(SCAN_INTERVAL)
                continue

            active_trades = load_active_trades()
            # Purge any remaining short trades from active trades (spot-only mode)
            for sym, trade in list(active_trades.items()):
                if trade.get("direction") == "short":
                    print(f"⛔ Removing non-long trade {sym} from active trades (spot-only mode).")
                    del active_trades[sym]
            save_active_trades(active_trades)

            top_symbols = get_top_symbols(limit=30)
            potential_trades = []  # collect potential trade signals
            symbol_scores = {}  # save scores for context boosting

            for symbol in top_symbols:
                # Enforce max concurrency but still evaluate all symbols for logging
                if symbol in active_trades:
                    print(f"⚠️ Skipping {symbol}: already in an active trade.")
                    continue

                try:
                    price_data = get_price_data(symbol)
                    if price_data is None or price_data.empty or len(price_data) < 20:
                        print(f"⚠️ Skipping {symbol} due to insufficient data.\n")
                        continue

                    score, direction, position_size, pattern_name = evaluate_signal(price_data, symbol)
                    symbol_scores[symbol] = {"score": score, "direction": direction}
                    # ✅ If no clear direction but score meets threshold, assume long (neutral sentiment fallback)
                    if direction is None and score >= 5.5:
                        print(f"⚠️ No clear direction for {symbol} despite score={score:.2f}. Forcing 'long' direction.")
                        direction = "long"

                    # Skip if signal does not qualify (we only trade long spot positions)
                    if direction != "long" or position_size <= 0:
                        # Skip reason already logged in evaluate_signal
                        continue

                    # Order flow check – require bullish aggression
                    flow_status = detect_aggression(price_data)
                    if flow_status != "buyers in control":
                        if flow_status == "sellers in control":
                            print(f"🚫 Bearish order flow detected in {symbol}. Skipping trade.")
                        else:
                            print(f"🚫 No buy-side aggression detected in {symbol}. Skipping trade.")
                        continue

                    # If we reach here, we have a valid potential **long** trade signal
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

            # Rank potential trades by score (highest first)
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
                    print(f"⚠️ Skipping {symbol} due to concurrency cap (max {MAX_ACTIVE_TRADES} trades).")
                    continue

                # Prepare indicators for decision and consult the brain module
                indicators = {
                    "rsi": price_data["rsi"].iloc[-1] if "rsi" in price_data else 50,
                    "macd": price_data["macd"].iloc[-1] if "macd" in price_data else 0,
                    "adx": price_data["adx"].iloc[-1] if "adx" in price_data else 20,
                    "volume": price_data["volume"].iloc[-1] if "volume" in price_data else 0,
                }
                orderflow_tag = "buy-side aggression"  # we ensured buy-side flow above

                decision_obj = should_trade(
                    symbol=symbol,
                    score=score,
                    direction=direction,
                    indicators=indicators,
                    session="default",
                    pattern_name=pattern_name,
                    orderflow=orderflow_tag,
                    sentiment=sentiment,
                    macro_news=sentiment  # using macro sentiment as macro_news context
                )
                decision = decision_obj.get("decision", False)
                final_conf = decision_obj.get("confidence", 0.0)
                narrative = decision_obj.get("narrative", "")

                # Log the brain's decision and reasoning
                if decision:
                    print(f"🤖 Brain Decision for {symbol}: True | Confidence: {final_conf:.2f}\n{narrative}\n")
                else:
                    reason = decision_obj.get("reason", "Unknown reason")
                    print(f"🤖 Brain Decision for {symbol}: False | Confidence: {final_conf:.2f}\nReason: {reason}\n")

                # If brain approves the trade, simulate executing it (paper trade)
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
                    print(f"✅ Trade: {symbol} | Score: {score:.2f} | Position Size: ${position_size}")
                    active_trades[symbol] = new_trade
                    save_active_trades(active_trades)
                    send_email(f"New Trade Opened: {symbol}", str(new_trade))
                    opened_count += 1

            # Update and manage any open trades, then pause
            manage_trades()
            # Save symbol_scores to JSON file
            with open("symbol_scores.json", "w") as f:
                json.dump(symbol_scores, f, indent=4)
            print("💾 Saved symbol scores.")
            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            print(f"❌ Main Loop Error: {e}")
            time.sleep(10)
def run(self):
        """
        Main loop to run the trading agent continuously.
        Continuously checks for trade signals and executes trades with risk management.
        """
        print(f"Starting agent for {self.symbol}...")
        while True:
            # 1. Fetch latest market data and generate trade signal (buy/sell/hold)
            signal = self.generate_trade_signal()  # (Assume this method exists and is unchanged)
            # ... (any other pre-trade logic remains unchanged) ...

            # 2. Incorporate news-based filter before executing any trade
            if signal is not None and signal in ["BUY", "SELL"]:
                news_articles = fetch_news(self.symbol)
                negative_news = False
                positive_news = False
                # Analyze recent news headlines for sentiment keywords
                for article in news_articles:
                    title = article.get('title', '').lower()
                    if any(word in title for word in ["plunge", "tumble", "crash", "fraud", "scandal", "down", "falls", "drops", "misses"]):
                        negative_news = True
                    if any(word in title for word in ["soar", "surge", "jump", "rise", "record high", "beats", "profit", "upward"]):
                        positive_news = True
                # If buying but news is negative, skip the trade; if selling (or shorting) but news is positive, skip.
                if signal == "BUY" and negative_news:
                    print(f"Negative news detected for {self.symbol}. Skipping BUY signal.")
                    signal = None
                elif signal == "SELL" and positive_news:
                    print(f"Positive news detected for {self.symbol}. Skipping SELL signal.")
                    signal = None

            # 3. Execute trade based on signal (if not filtered out)
            if signal == "BUY" and not self.in_position:
                current_price = self.get_current_price()  # (Assume this method provides latest market price)
                quantity = self.trade_quantity
                # Place buy order (simulate immediate execution at current_price)
                executed_price = current_price
                self.in_position = True
                self.entry_price = executed_price
                print(f"Entered BUY position at {executed_price:.2f} for {quantity} units of {self.symbol}")
                # Calculate slippage (if any) compared to intended price (current_price here)
                slippage_pct = simulate_slippage(executed_price, current_price)
                # Calculate commission for the trade
                commission_cost = estimate_commission(self.symbol, quantity, executed_price)
                # Update balance for commission cost
                self.balance -= commission_cost
                self.total_profit -= commission_cost  # account for commission as an immediate cost
                # Log slippage and commission
                if slippage_pct != 0:
                    print(f"Trade slippage: {slippage_pct:+.4f}% on BUY order.")
                print(f"Commission paid: ${commission_cost:.2f}")

            elif signal == "SELL" and self.in_position:
                current_price = self.get_current_price()  # current market price at which we'll sell
                quantity = self.trade_quantity
                # Place sell order (simulate execution at current_price)
                executed_price = current_price
                print(f"Executing SELL at {executed_price:.2f} for {quantity} units of {self.symbol}")
                # Calculate slippage (if any) for the sell relative to intended price
                slippage_pct = simulate_slippage(executed_price, current_price)
                # Calculate commission for the trade
                commission_cost = estimate_commission(self.symbol, quantity, executed_price)
                # Compute profit from the trade (difference between sell and buy prices)
                trade_profit = 0.0
                if self.entry_price is not None:
                    # Profit (or loss) = (sell_price - entry_price) * quantity
                    trade_profit = (executed_price - self.entry_price) * quantity
                # Subtract commission cost from profit and update balance and total_profit
                trade_profit -= commission_cost
                self.balance += executed_price * quantity  # add proceeds from selling back to balance
                self.balance -= commission_cost           # subtract commission from balance
                self.total_profit += trade_profit
                # Reset position
                self.in_position = False
                self.entry_price = None
                # Log results of the trade
                if slippage_pct != 0:
                    print(f"Trade slippage: {slippage_pct:+.4f}% on SELL order.")
                print(f"Commission paid: ${commission_cost:.2f}")
                if trade_profit >= 0:
                    print(f"Trade closed with profit: ${trade_profit:.2f}")
                else:
                    print(f"Trade closed with loss: ${-trade_profit:.2f}")
                print(f"Total cumulative profit: ${self.total_profit:.2f}")

            # 4. (Optional) Implement stop-loss or take-profit checks if needed
            # ... (any existing risk management logic remains unchanged) ...

            # 5. Wait for a short interval before next iteration (to avoid high-frequency churn)
            time.sleep(1)  # Adjust sleep as needed for desired trading frequency
            
if __name__ == "__main__":
    logging.info("🚀 Starting Spot AI Super Agent loop...")
    run_agent_loop()
