import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from binance.client import Client
from symbol_mapper import map_symbol_for_binance
from datetime import datetime as dt
from ta.volume import VolumeWeightedAveragePrice  # ✅ VWMA fix
import os

from price_action import detect_support_resistance_zones, is_price_near_zone
from orderflow import detect_aggression
from pattern_memory import recall_pattern_confidence  # ✅ Pattern memory
from datetime import datetime

def get_current_session():
    now = datetime.utcnow().hour
    if 0 <= now < 8:
        return "Asia"
    elif 8 <= now < 16:
        return "Europe"
    else:
        return "New York"

client = Client()

# === Indicator Calculation ===
def calculate_indicators(df):
    df = df.copy()
    df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    adx = ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    return df

# === Binance Price Fetch ===
def get_top_symbols(limit=30):
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
    symbols = [x['symbol'] for x in sorted_tickers if x['symbol'].endswith("USDT") and not x['symbol'].endswith("BUSD")]
    return symbols[:limit]

def get_price_data(symbol):
    try:
        mapped_symbol = map_symbol_for_binance(symbol)
        klines = client.get_klines(symbol=mapped_symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=100)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"⚠️ Failed to fetch data for {symbol}: {e}")
        return None

# === Pattern Detection ===
def detect_candlestick_patterns(df):
    if len(df) < 2:
        return []
    latest = df.iloc[-2:]
    patterns = []
    body = abs(latest['close'].iloc[-1] - latest['open'].iloc[-1])
    lower_shadow = latest['open'].iloc[-1] - latest['low'].iloc[-1] if latest['open'].iloc[-1] > latest['close'].iloc[-1] else latest['close'].iloc[-1] - latest['low'].iloc[-1]
    upper_shadow = latest['high'].iloc[-1] - max(latest['open'].iloc[-1], latest['close'].iloc[-1])
    if lower_shadow > 2 * body:
        patterns.append("Hammer")
    if upper_shadow > 2 * body:
        patterns.append("ShootingStar")
    if latest['close'].iloc[-2] < latest['open'].iloc[-2] and latest['close'].iloc[-1] > latest['open'].iloc[-1] and latest['close'].iloc[-1] > latest['open'].iloc[-2] and latest['open'].iloc[-1] < latest['close'].iloc[-2]:
        patterns.append("BullishEngulfing")
    return patterns

def detect_flag_pattern(df):
    if len(df) < 30:
        return False

    recent = df.tail(20)
    prev = df.tail(40).head(20)

    # 1. Check prior uptrend strength
    prior_close_change = (prev['close'].iloc[-1] - prev['close'].iloc[0]) / prev['close'].iloc[0]
    if prior_close_change < 0.04:  # At least 4% up move
        return False

    # 2. Check consolidation
    high_range = recent['high'].max()
    low_range = recent['low'].min()
    price_range = high_range - low_range
    avg_range = recent['high'].mean() - recent['low'].mean()

    if price_range / recent['close'].iloc[-1] > 0.03:  # Too wide for a flag
        return False
    if avg_range / recent['close'].iloc[-1] > 0.015:  # Too volatile
        return False

    # 3. Volume drop during flag (optional but useful)
    if recent['volume'].mean() > prev['volume'].mean():
        return False

    return True


def detect_triangle_wedge(df):
    if len(df) < 40:
        return None

    recent = df.tail(30).reset_index(drop=True)
    highs = recent['high']
    lows = recent['low']
    closes = recent['close']

    # Calculate highs/lows slope
    high_slope = highs.iloc[-1] - highs.iloc[0]
    low_slope = lows.iloc[-1] - lows.iloc[0]

    slope_diff = abs(high_slope - low_slope)
    squeeze_range = highs.max() - lows.min()

    # If range is narrowing and slopes converge
    if slope_diff < squeeze_range * 0.3:
        if high_slope < 0 and low_slope > 0:
            return "SymTriangle"
        elif high_slope < 0 and low_slope < 0:
            return "FallingWedge"
        elif high_slope > 0 and low_slope > 0:
            return "RisingWedge"

    return None


def get_market_session():
    utc_hour = dt.utcnow().hour
    if 0 <= utc_hour < 8:
        return "Asia"
    elif 8 <= utc_hour < 16:
        return "Europe"
    else:
        return "US"

def log_signal(symbol, session, score, direction, weights, candle_patterns, chart_pattern):
    log_entry = {
        "timestamp": dt.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "session": session,
        "score": score,
        "direction": direction,
        "ema_weight": weights.get("ema", 0),
        "macd_weight": weights.get("macd", 0),
        "rsi_weight": weights.get("rsi", 0),
        "adx_weight": weights.get("adx", 0),
        "vwma_weight": weights.get("vwma", 0),
        "bb_weight": weights.get("bb", 0),
        "candle_patterns": ", ".join(candle_patterns),
        "chart_pattern": chart_pattern if chart_pattern else "None"
    }
    df_entry = pd.DataFrame([log_entry])
    path = "trades_log.csv"
    if os.path.exists(path):
        df_entry.to_csv(path, mode='a', header=False, index=False)
    else:
        df_entry.to_csv(path, index=False)

def get_reinforcement_bonus(symbol, session):
    try:
        df = pd.read_csv("trade_learning_log.csv")
        df = df[(df['symbol'] == symbol) & (df['session'] == session)]
        if df.empty:
            return 1.0
        win_rate = df[df['outcome'] == 'win'].shape[0] / df.shape[0]
        return round(1.0 + (win_rate - 0.5), 2)
    except:
        return 1.0

def get_position_size(confidence):
    if confidence >= 8.5:
        return 100
    elif confidence >= 6.5:
        return 80
    else:
        return 0

# === Main Signal Function ===
# === Main Signal Function ===
def evaluate_signal(price_data, symbol="", sentiment_bias="neutral"):
    try:
        if price_data is None or price_data.empty or len(price_data) < 20:
            print(f"[DEBUG] Skipping {symbol}: insufficient price data.")
            return 0, None, 0, None

        open_ = price_data['open']
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        volume = price_data['volume']

        ema_short = EMAIndicator(close, window=20).ema_indicator()
        ema_long = EMAIndicator(close, window=50).ema_indicator()
        macd_line = MACD(close).macd_diff()
        rsi = RSIIndicator(close, window=14).rsi()
        adx = ADXIndicator(high=high, low=low, close=close).adx()
        bb = BollingerBands(close=close, window=20, window_dev=2)
        vwma_calc = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=20)
        vwma = vwma_calc.volume_weighted_average_price()
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()

        print(f"🔍 [{symbol}] Volume: {volume.iloc[-1]:,.0f} | VWMA: {vwma.iloc[-1]:.2f} | Sentiment: {sentiment_bias}")

        if volume.iloc[-1] < 250000:
            print(f"⛔ Skipping due to low volume: {volume.iloc[-1]}")
            return 0, None, 0, None
        
        vwma_value = vwma.iloc[-1]
        price_now = close.iloc[-1]

        # Check if price is not significantly under VWMA (within ±3%)
        if np.isnan(vwma_value) or abs(price_now - vwma_value) / price_now > 0.03:
            print(f"⛔ Skipping due to VWMA mismatch | VWMA: {vwma_value}, Price: {price_now}")
            return 0, None, 0, None
        
    except Exception as e:
        print(f"⚠️ Error calculating indicators for {symbol}: {e}")
        return 0, None, 0, None

        session = get_market_session()
        weights = {
            "Asia":   {"ema": 1.0, "macd": 1.0, "rsi": 1.0, "adx": 1.2, "vwma": 1.3, "bb": 1.2, "candle": 1.0, "chart": 1.0, "flag": 1.0, "flow": 1.5},
            "Europe": {"ema": 1.2, "macd": 1.0, "rsi": 1.0, "adx": 1.3, "vwma": 1.0, "bb": 1.0, "candle": 1.0, "chart": 1.0, "flag": 1.0, "flow": 1.5},
            "US":     {"ema": 1.0, "macd": 1.2, "rsi": 1.2, "adx": 1.2, "vwma": 1.0, "bb": 1.0, "candle": 1.0, "chart": 1.0, "flag": 1.0, "flow": 1.5},
        }
        w = weights[session]
        reinforcement = get_reinforcement_bonus(symbol, session)
        score = 0

        if ema_short.iloc[-1] > ema_long.iloc[-1]: 
            score += w["ema"]
            print(f"[DEBUG] EMA condition passed: {ema_short.iloc[-1]:.2f} > {ema_long.iloc[-1]:.2f}")
        if macd_line.iloc[-1] > 0: 
            score += w["macd"]
            print(f"[DEBUG] MACD condition passed: {macd_line.iloc[-1]:.2f}")
        if rsi.iloc[-1] > 50: 
            score += w["rsi"]
            print(f"[DEBUG] RSI condition passed: {rsi.iloc[-1]:.2f}")
        if adx.iloc[-1] > 20: 
            score += w["adx"]
            print(f"[DEBUG] ADX condition passed: {adx.iloc[-1]:.2f}")
        if close.iloc[-1] > vwma.iloc[-1]: 
            score += w["vwma"]
            print(f"[DEBUG] VWMA condition passed: {close.iloc[-1]:.2f} > {vwma.iloc[-1]:.2f}")
        if bb_lower.iloc[-1] < close.iloc[-1] < bb_upper.iloc[-1]: 
            score += w["bb"]
            print(f"[DEBUG] BB range condition passed: {bb_lower.iloc[-1]:.2f} < {close.iloc[-1]:.2f} < {bb_upper.iloc[-1]:.2f}")

        candle_patterns = detect_candlestick_patterns(price_data)
        chart_pattern = detect_triangle_wedge(price_data)
        flag = detect_flag_pattern(price_data)

        if candle_patterns: 
            score += w["candle"]
            print(f"[DEBUG] Candlestick pattern(s) found: {candle_patterns}")
        if chart_pattern: 
            score += w["chart"]
            print(f"[DEBUG] Chart pattern found: {chart_pattern}")
        if flag: 
            score += w["flag"]
            print(f"[DEBUG] Flag pattern confirmed")

        aggression = detect_aggression(price_data)
        if aggression == "buyers in control":
            score += w["flow"]
            print(f"[DEBUG] Buy-side aggression detected")
        elif aggression == "sellers in control":
            print(f"[DEBUG] Seller aggression detected — skipping.")
            return 0, None, 0, None
        
        # 📊 Multi-Symbol Context Boost
        try:
            with open("symbol_scores.json", "r") as f:
                scores_data = json.load(f)
                bullish_allies = 0
                for sym, data in scores_data.items():
                    if sym != symbol and data["direction"] == "long" and data["score"] >= 7.0:
                        bullish_allies += 1

                if bullish_allies >= 2:
                    score += 0.4
                    print(f"[CONTEXT BOOST] {bullish_allies} correlated bullish symbols detected — boosting score.")
        except Exception as e:
            print(f"[CONTEXT ERROR] Failed to apply context boost: {e}")        

        max_possible = sum(w.values())
        normalized_score = round((score / max_possible) * 10 * reinforcement, 2)

        # 🌤️ Sentiment-based boost
        if sentiment_bias == "bullish" and normalized_score < 6.5:
            normalized_score += 0.8
        elif sentiment_bias == "bearish" and normalized_score > 7.5:
            normalized_score -= 0.8
        normalized_score = round(normalized_score, 2)
        # ✨ Chart + Candle Confluence Bonus
        if candle_patterns and chart_pattern:
            score += 0.5
            print(f"[CONFLUENCE] Chart + Candle pattern confluence detected")
        score_long = normalized_score
        score_short = 0
        direction = "long" if score_long > score_short and score_long >= 6.5 else None

        # 🧠 Fallback direction
        if direction is None and score_long >= 5.0 and sentiment_bias == "bullish":
            direction = "long"
            print(f"[DEBUG] Fallback direction applied due to bullish sentiment and score: {score_long}")

        position_size = get_position_size(score_long)

        zones = detect_support_resistance_zones(price_data)
        zones = zones.to_dict() if isinstance(zones, pd.Series) else zones or {"support": [], "resistance": []}
        current_price = float(close.iloc[-1])

        if direction == "long":
            if is_price_near_zone(current_price, zones, 'resistance', 0.005):
                print(f"[ZONE FILTER] Skipping long trade near resistance at {current_price}")
                return 0, None, 0, None
            if not is_price_near_zone(current_price, zones, 'support', 0.015 if sentiment_bias == "bullish" else 0.01):
                print(f"[ZONE FILTER] Skipping long trade with no nearby support at {current_price}")
                return 0, None, 0, None

        print(f"✅ [{symbol}] Final Score: {normalized_score} | Dir: {direction} | PosSize: {position_size}")

        pattern_name = candle_patterns[0] if candle_patterns else chart_pattern if chart_pattern else "None"

        # 📚 Pattern Memory Boost
        pattern_boost = recall_pattern_confidence(symbol, pattern_name)
        if pattern_boost >= 0.6:
            score += 0.4
            print(f"[MEMORY] Pattern memory boost: {pattern_name} | Conf: {pattern_boost:.2f}")
        log_signal(symbol, session, normalized_score, direction, w, candle_patterns, chart_pattern)

        return normalized_score, direction, position_size, pattern_name

    except Exception as e:
        print(f"⚠️ Signal evaluation error in {symbol}: {e}")
        return 0, None, 0, None


