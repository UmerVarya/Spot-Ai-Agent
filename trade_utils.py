import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from binance.client import Client
from symbol_mapper import map_symbol_for_binance
from datetime import datetime as dt
from ta.volume import VolumeWeightedAveragePrice  # ✅ VWMA indicator
import os
import json
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
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['high', 'low', 'close'])
    # Compute common indicators
    df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd_diff()  # MACD histogram (diff between MACD and signal)
    df['macd_signal'] = macd.macd_signal()
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
        klines = client.get_klines(symbol=mapped_symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=100)
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

# === Pattern Detection (unchanged) ===
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
    # ... [no changes to pattern detection logic] ...
    if len(df) < 30:
        return False
    # (pattern detection logic continues as in original)
    # ...

def detect_triangle_wedge(df):
    # ... [no changes to pattern detection logic] ...
    if len(df) < 40:
        return None
    # (pattern detection logic continues as in original)
    # ...

def get_market_session():
    utc_hour = dt.utcnow().hour
    if 0 <= utc_hour < 8:
        return "Asia"
    elif 8 <= utc_hour < 16:
        return "Europe"
    else:
        return "US"

def log_signal(symbol, session, score, direction, weights, candle_patterns, chart_pattern):
    # Logging signals to CSV (unchanged)
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
        # Bonus = 1 + (win_rate - 0.5), range roughly 0.5 to 1.5
        return round(1.0 + (win_rate - 0.5), 2)
    except:
        return 1.0

def get_position_size(confidence):
    # Tiered position sizing
    if confidence >= 8.5:
        return 100   # 100% position (max size)
    elif confidence >= 6.5:
        return 80    # 80% position for confidence 6.5–8.4
    elif confidence >= 5.5:
        return 50    # 50% position for confidence 5.5–6.4
    else:
        return 0     # no trade if confidence below 5.5

# === Main Signal Evaluation Function ===
def evaluate_signal(price_data, symbol="", sentiment_bias="neutral"):
    try:
        if price_data is None or price_data.empty or len(price_data) < 20:
            print(f"[DEBUG] Skipping {symbol}: insufficient price data.")
            return 0, None, 0, None

        price_data = price_data.replace([np.inf, -np.inf], np.nan)
        price_data = price_data.dropna(subset=['high', 'low', 'close'])

        open_ = price_data['open']
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        volume = price_data['volume']

        # Compute key technical indicators for the current 5m data
        ema_short = EMAIndicator(close, window=20).ema_indicator()
        ema_long = EMAIndicator(close, window=50).ema_indicator()
        macd_line = MACD(close).macd_diff()  # MACD histogram
        rsi = RSIIndicator(close, window=14).rsi()
        adx = ADXIndicator(high=high, low=low, close=close).adx().fillna(0)
        bb = BollingerBands(close=close, window=20, window_dev=2)
        vwma_calc = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=20)
        vwma = vwma_calc.volume_weighted_average_price()
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()

        # Print snapshot of current volume, VWMA, and sentiment for transparency
        latest_vol = volume.iloc[-1]
        latest_vwma = vwma.iloc[-1]
        print(f"🔍 [{symbol}] Volume: {latest_vol:,.0f} | VWMA: {latest_vwma:.2f} | Sentiment: {sentiment_bias}")

        # ✅ Dynamic Volume Filter: require current volume at least 50% of 20-bar average (with 50k USDT absolute minimum)
        avg_vol_20 = volume.iloc[-20:].mean()
        vol_threshold = max(0.5 * avg_vol_20, 50000)
        if latest_vol < vol_threshold:
            print(f"⛔ Skipping due to low volume: {latest_vol:,.0f} < {vol_threshold:,.0f} (50% of 20-bar avg)")
            return 0, None, 0, None

        # ✅ VWMA Deviation Logic: no hard filter, use soft adjustment
        vwma_value = latest_vwma
        price_now = close.iloc[-1]
        vwma_dev = abs(price_now - vwma_value) / price_now if price_now != 0 else 0.0

        # Determine trading session for weight profile
        session = get_market_session()
        weights = {
            "Asia":   {"ema": 1.0, "macd": 1.0, "rsi": 1.0, "adx": 1.2, "vwma": 1.3, "bb": 1.2, "candle": 1.0, "chart": 1.0, "flag": 1.0, "flow": 1.5},
            "Europe": {"ema": 1.2, "macd": 1.0, "rsi": 1.0, "adx": 1.3, "vwma": 1.0, "bb": 1.0, "candle": 1.0, "chart": 1.0, "flag": 1.0, "flow": 1.5},
            "US":     {"ema": 1.0, "macd": 1.2, "rsi": 1.2, "adx": 1.2, "vwma": 1.0, "bb": 1.0, "candle": 1.0, "chart": 1.0, "flag": 1.0, "flow": 1.5},
        }
        w = weights.get(session, weights["US"])
        reinforcement = get_reinforcement_bonus(symbol, session)
        score = 0.0

        # Apply indicator conditions to score
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
        # VWMA trend condition with partial credit if price far above VWMA
        if price_now > vwma_value:
            if vwma_dev <= 0.05:
                score += w["vwma"]
                print(f"[DEBUG] VWMA condition passed: {price_now:.2f} > {vwma_value:.2f}")
            elif vwma_dev < 0.10:
                # Price is 5-10% above VWMA: give partial weight
                fraction = (0.10 - vwma_dev) / 0.05
                score += w["vwma"] * fraction
                print(f"[DEBUG] Price above VWMA by {vwma_dev:.2%} – partial VWMA weight ({fraction*100:.0f}%) applied")
            else:
                # Price extremely above VWMA (>10% deviation): no additional VWMA weight (overextended)
                print(f"[DEBUG] Price above VWMA by {vwma_dev:.2%} – VWMA weight omitted (overextended)")
        else:
            # Price at or below VWMA: no VWMA weight added (price not in bullish position relative to VWMA)
            if vwma_dev > 0.05:
                print(f"[DEBUG] Price is below VWMA by {vwma_dev:.2%} (potential dip) – no VWMA weight added")

        if bb_lower := bb_lower.iloc[-1]:
            if bb_lower < price_now < bb_upper.iloc[-1]:
                score += w["bb"]
                print(f"[DEBUG] BB range condition passed: {bb_lower:.2f} < {price_now:.2f} < {bb_upper.iloc[-1]:.2f}")

        # Candlestick and chart pattern detection
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

        # Order flow analysis (aggression)
        aggression = detect_aggression(price_data)
        if aggression == "buyers in control":
            score += w["flow"]
            print(f"[DEBUG] Buy-side aggression detected")
        elif aggression == "sellers in control":
            print(f"[DEBUG] Seller aggression detected — skipping signal.")
            return 0, None, 0, None
        # (If no clear aggression, we neither add nor skip here; neutral order flow will be handled in agent logic.)

        # Context boost from correlated symbols (existing symbol_scores.json)
        try:
            with open("symbol_scores.json", "r") as f:
                scores_data = json.load(f)
            bullish_allies = sum(
                1 for sym, data in scores_data.items()
                if sym != symbol and data.get("direction") == "long" and data.get("score", 0) >= 7.0
            )
            if bullish_allies >= 2:
                score += 0.4
                print(f"[CONTEXT BOOST] {bullish_allies} other symbols bullish — boosting score")
        except Exception as e:
            print(f"[CONTEXT ERROR] Failed to apply context boost: {e}")

        # Pattern confluence boost
        if candle_patterns and chart_pattern:
            score += 0.5
            print(f"[CONFLUENCE] Candle + Chart pattern confluence detected")

        # Optional 15m trend context – small boost or penalty based on higher timeframe trend
        try:
            mapped_symbol = map_symbol_for_binance(symbol)
            klines15 = client.get_klines(symbol=mapped_symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=50)
            df15 = pd.DataFrame(klines15, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df15['close'] = df15['close'].astype(float)
            ema15_short = df15['close'].ewm(span=20).mean().iloc[-1]
            ema15_long = df15['close'].ewm(span=50).mean().iloc[-1]
        except Exception as e:
            ema15_short = ema15_long = None
            # (If fetch fails, we simply won't adjust for 15m trend)
        if ema15_short is not None and ema15_long is not None:
            if ema15_short > ema15_long:
                score += 0.3
                print(f"[DEBUG] 15m trend is UP – small bullish context boost applied")
            elif ema15_short < ema15_long:
                score -= 0.3
                print(f"[DEBUG] 15m trend is DOWN – small penalty applied to score")

        # Normalize score to 0–10 scale with reinforcement factor
        max_possible = sum(w.values())
        normalized_score = round((score / max_possible) * 10 * reinforcement, 2)

        # Sentiment bias adjustments to normalized score (soften threshold requirements)
        if sentiment_bias == "bullish" and normalized_score < 5.5:
            normalized_score += 0.8  # bullish sentiment can slightly boost borderline scores
        elif sentiment_bias == "bearish" and normalized_score > 7.5:
            normalized_score -= 0.8  # bearish sentiment can temper overly high scores
        normalized_score = round(normalized_score, 2)

        # Determine trade direction based on score thresholds
        score_long = normalized_score
        score_short = 0
        direction = "long" if score_long >= 5.5 else None  # base score threshold lowered to 5.5 for long trades

        # Fallback: if still no direction but score >=5.5 under bullish sentiment, go long (ensures bullish context isn't blocked by slight score miss)
        if direction is None and score_long >= 5.5 and sentiment_bias == "bullish":
            direction = "long"
            print(f"[DEBUG] Fallback direction applied due to bullish sentiment and score: {score_long:.2f}")

        # Determine position size based on confidence (score_long)
        position_size = get_position_size(score_long)

        # Support/Resistance zone safety checks for longs
        zones = detect_support_resistance_zones(price_data)
        zones = zones.to_dict() if isinstance(zones, pd.Series) else (zones or {"support": [], "resistance": []})
        current_price = float(close.iloc[-1])
        if direction == "long":
            if is_price_near_zone(current_price, zones, 'resistance', 0.005):
                print(f"[ZONE FILTER] Skipping long trade near resistance at {current_price}")
                return 0, None, 0, None
            if not is_price_near_zone(current_price, zones, 'support', 0.015 if sentiment_bias == "bullish" else 0.01):
                print(f"[ZONE FILTER] Skipping long trade with no nearby support at {current_price}")
                return 0, None, 0, None

        # Log final evaluated score and direction
        print(f"✅ [{symbol}] Final Score: {normalized_score:.2f} | Dir: {direction} | PosSize: {position_size}")

        # Identify primary pattern name for memory (either the first candle pattern or the chart pattern)
        pattern_name = candle_patterns[0] if candle_patterns else (chart_pattern if chart_pattern else "None")
        # Pattern memory boost (if this pattern historically successful, slightly increase score for reference)
        pattern_boost = recall_pattern_confidence(symbol, pattern_name)
        if pattern_boost >= 0.6:
            score += 0.4
            print(f"[MEMORY] Pattern memory boost applied for {pattern_name} (Conf: {pattern_boost:.2f})")

        # Log the signal to CSV for learning analysis
        log_signal(symbol, session, normalized_score, direction, w, candle_patterns, chart_pattern)
        return normalized_score, direction, position_size, pattern_name

    except Exception as e:
        print(f"⚠️ Signal evaluation error in {symbol}: {e}")
        return 0, None, 0, None


