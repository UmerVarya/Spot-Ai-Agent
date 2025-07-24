import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from binance.client import Client
from symbol_mapper import map_symbol_for_binance
from datetime import datetime as dt
from ta.volume import VolumeWeightedAveragePrice
import os
import json
from price_action import detect_support_resistance_zones, is_price_near_zone
from orderflow import detect_aggression
from pattern_memory import recall_pattern_confidence
from datetime import datetime


def simulate_slippage(executed_price: float, expected_price: float) -> float:
    """
    Calculate slippage as a percentage difference between the executed price and expected price.
    Positive slippage means the executed price was worse for the trader (e.g., higher buy price or lower sell price),
    negative slippage means an executed price better than expected.
    """
    if expected_price == 0 or expected_price is None:
        return 0.0  # Avoid division by zero or undefined expected price
    slippage_pct = (executed_price - expected_price) / expected_price * 100
    return round(slippage_pct, 4)


def estimate_commission(symbol: str, quantity: int, price: float, broker: str = "generic") -> float:
    """
    Estimate the commission cost for a trade based on the specified broker model.
    - "generic": A generic brokerage with $0.005 per share and $1 minimum commission.
    - "free": Commission-free trading (e.g., many crypto exchanges or zero-commission brokers).
    - "percent": Commission as 0.1% of trade value with $1 minimum.
    """
    # Calculate commission based on the chosen broker model
    if broker == "generic":
        cost_per_unit = 0.005  # $0.005 per share/unit
        commission = quantity * cost_per_unit
        return round(max(commission, 1.0), 4)  # enforce a minimum of $1.00
    elif broker == "free":
        return 0.0  # no commission
    elif broker == "percent":
        commission = quantity * price * 0.001  # 0.1% of trade value
        return round(max(commission, 1.0), 4)  # minimum $1.00
    else:
        # If an unknown broker model is passed, default to no commission
        return 0.0


def get_current_session():
    now = datetime.utcnow().hour
    if 0 <= now < 8:
        return "Asia"
    elif 8 <= now < 16:
        return "Europe"
    else:
        return "New York"


# initialise a Binance client for price data
client = Client()


def calculate_indicators(df: pd.DataFrame):
    """Compute a suite of technical indicators on a price DataFrame."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['high', 'low', 'close'])
    df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd_diff()  # MACD histogram (diff between MACD and signal)
    df['macd_signal'] = macd.macd_signal()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    # Suppress RuntimeWarnings for ADX calculation
    with np.errstate(invalid='ignore', divide='ignore'):
        adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_indicator.adx()
    df['adx'] = df['adx'].fillna(0)
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    return df


def get_top_symbols(limit=30):
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
    symbols = [x['symbol'] for x in sorted_tickers if x['symbol'].endswith("USDT") and not x['symbol'].endswith("BUSD")]
    return symbols[:limit]


def get_price_data(symbol: str):
    try:
        mapped_symbol = map_symbol_for_binance(symbol)
        klines = client.get_klines(symbol=mapped_symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=100)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']] = df[['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']].astype(float)
        df['quote_volume'] = df['quote_asset_volume']
        return df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']]
    except Exception as e:
        print(f"⚠️ Failed to fetch data for {symbol}: {e}")
        return None


def detect_candlestick_patterns(df: pd.DataFrame):
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


def detect_flag_pattern(df: pd.DataFrame):
    # Placeholder for flag pattern detection logic.
    if len(df) < 30:
        return False
    return False


def detect_triangle_wedge(df: pd.DataFrame):
    # Placeholder for triangle/wedge pattern detection logic.
    if len(df) < 40:
        return None
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


def get_reinforcement_bonus(symbol: str, session: str):
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


def get_position_size(confidence: float) -> int:
    # Tiered position sizing based on confidence score
    if confidence >= 8.5:
        return 100
    elif confidence >= 6.5:
        return 80
    elif confidence >= 5.5:
        return 50
    else:
        return 0


def evaluate_signal(price_data: pd.DataFrame, symbol: str = "", sentiment_bias: str = "neutral"):
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
        macd_line = MACD(close).macd_diff()
        rsi = RSIIndicator(close, window=14).rsi()
        with np.errstate(invalid='ignore', divide='ignore'):
            adx_series = ADXIndicator(high=high, low=low, close=close, window=14).adx()
        adx = adx_series.fillna(0)
        bb = BollingerBands(close=close, window=20, window_dev=2)
        vwma_calc = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=20)
        vwma = vwma_calc.volume_weighted_average_price()
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()

        # Print snapshot of current volume, VWMA, and sentiment for transparency
        latest_vol = volume.iloc[-1]
        latest_vwma = vwma.iloc[-1]
        price_now = close.iloc[-1]
        print(f"🔍 [{symbol}] Volume: {latest_vol:,.0f} | VWMA: {latest_vwma:.2f} | Sentiment: {sentiment_bias}")

        # Dynamic Volume Filter: require current quote volume at least X% of 20-bar average or 50k USDT minimum
        avg_quote_vol_20 = price_data['quote_volume'].iloc[-20:].mean() if 'quote_volume' in price_data else None
        latest_quote_vol = price_data['quote_volume'].iloc[-1] if 'quote_volume' in price_data else None
        if avg_quote_vol_20 is None:
            # fallback to base volume times price
            avg_quote_vol_20 = volume.iloc[-20:].mean() * price_now
            latest_quote_vol = latest_vol * price_now

        session_name = get_market_session()
        session_factor = {"Asia": 0.3, "Europe": 0.3, "US": 0.4}
        vol_factor = session_factor.get(session_name, 0.4)
        vol_threshold = max(vol_factor * avg_quote_vol_20, 50_000)
        if latest_quote_vol < vol_threshold:
            print(f"⛔ Skipping due to low volume: {latest_quote_vol:,.0f} < {vol_threshold:,.0f} ({vol_factor*100:.0f}% of 20-bar avg)")
            return 0, None, 0, None

        # Determine trading session for weight profile and reinforcement factor
        session = get_market_session()
        weights_map = {
            "Asia":   {"ema": 1.5, "macd": 1.3, "rsi": 1.3, "adx": 1.5, "vwma": 1.5, "bb": 1.3, "candle": 1.2, "chart": 1.2, "flag": 1.0, "flow": 1.5},
            "Europe": {"ema": 1.5, "macd": 1.3, "rsi": 1.3, "adx": 1.6, "vwma": 1.3, "bb": 1.3, "candle": 1.2, "chart": 1.2, "flag": 1.0, "flow": 1.5},
            "US":     {"ema": 1.3, "macd": 1.5, "rsi": 1.5, "adx": 1.5, "vwma": 1.3, "bb": 1.3, "candle": 1.2, "chart": 1.2, "flag": 1.0, "flow": 1.5},
        }
        w = weights_map.get(session, weights_map["US"])
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
        vwma_value = latest_vwma
        vwma_dev = abs(price_now - vwma_value) / price_now if price_now != 0 else 0.0
        if price_now > vwma_value:
            if vwma_dev <= 0.05:
                score += w["vwma"]
                print(f"[DEBUG] VWMA condition passed: {price_now:.2f} > {vwma_value:.2f}")
            elif vwma_dev < 0.10:
                fraction = (0.10 - vwma_dev) / 0.05
                score += w["vwma"] * fraction
                print(f"[DEBUG] Price above VWMA by {vwma_dev:.2%} – partial VWMA weight ({fraction*100:.0f}%) applied")
            else:
                print(f"[DEBUG] Price above VWMA by {vwma_dev:.2%} – VWMA weight omitted (overextended)")
        else:
            if vwma_dev > 0.05:
                print(f"[DEBUG] Price is below VWMA by {vwma_dev:.2%} (potential dip) – no VWMA weight added")

        if bb_lower.iloc[-1] and bb_lower.iloc[-1] < price_now < bb_upper.iloc[-1]:
            score += w["bb"]
            print(f"[DEBUG] BB range condition passed: {bb_lower.iloc[-1]:.2f} < {price_now:.2f} < {bb_upper.iloc[-1]:.2f}")

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

        # Order flow analysis – apply boost/penalty instead of skipping
        aggression = detect_aggression(price_data)
        if aggression == "buyers in control":
            score += w["flow"]
            print(f"[DEBUG] Buy-side aggression detected")
        elif aggression == "sellers in control":
            score -= w["flow"]
            print(f"[DEBUG] Seller aggression detected — penalizing score.")

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

        # Optional 15m trend context – smaller penalty/boost
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
        except Exception:
            ema15_short = ema15_long = None
        if ema15_short is not None and ema15_long is not None:
            if ema15_short > ema15_long:
                score += 0.4  # stronger positive context
                print(f"[DEBUG] 15m trend is UP – bullish context boost applied")
            elif ema15_short < ema15_long:
                score -= 0.1  # milder penalty
                print(f"[DEBUG] 15m trend is DOWN – small penalty applied to score")

        # Normalize score to 0–10 scale with reinforcement factor
        max_possible = sum(w.values())
        normalized_score = round((score / max_possible) * 10 * reinforcement, 2)

        # Sentiment bias adjustments to normalized score (soften threshold requirements)
        if sentiment_bias == "bullish" and normalized_score < 5.0:
            normalized_score += 0.8
        elif sentiment_bias == "bearish" and normalized_score > 7.5:
            normalized_score -= 0.8
        normalized_score = round(normalized_score, 2)

        # Determine trade direction based on score thresholds
        direction = "long" if normalized_score >= 4.5 else None  # lowered base threshold

        # Determine position size based on confidence
        position_size = get_position_size(normalized_score)

        # Support/Resistance zone safety checks for longs (more permissive)
        zones = detect_support_resistance_zones(price_data)
        zones = zones.to_dict() if isinstance(zones, pd.Series) else (zones or {"support": [], "resistance": []})
        current_price = float(close.iloc[-1])
        if direction == "long":
            near_resistance = is_price_near_zone(current_price, zones, 'resistance', 0.005)
            near_support = is_price_near_zone(current_price, zones, 'support', 0.015 if sentiment_bias == "bullish" else 0.01)
            if near_resistance and normalized_score < 7.0:
                print(f"[ZONE FILTER] Skipping long trade near resistance at {current_price}")
                return 0, None, 0, None
            if not near_support and normalized_score < 6.5:
                print(f"[ZONE FILTER] Skipping long trade with no nearby support at {current_price}")
                return 0, None, 0, None

        # Log final evaluated score and direction
        print(f"✅ [{symbol}] Final Score: {normalized_score:.2f} | Dir: {direction} | PosSize: {position_size}")

        # Identify primary pattern name for memory
        pattern_name = candle_patterns[0] if candle_patterns else (chart_pattern if chart_pattern else "None")

        # Apply pattern memory confidence boost (if pattern historically successful)
        pattern_boost = recall_pattern_confidence(symbol, pattern_name)
        if pattern_boost >= 0.6:
            score += 0.4
            print(f"[MEMORY] Pattern memory boost applied for {pattern_name} (Conf: {pattern_boost:.2f})")

        # Log the signal for learning analysis
        log_signal(symbol, session, normalized_score, direction, w, candle_patterns, chart_pattern)
        return normalized_score, direction, position_size, pattern_name

    except Exception as e:
        print(f"⚠️ Signal evaluation error in {symbol}: {e}")
        return 0, None, 0, None
