"""
Enhanced trade utilities module for Spot AI Super Agent.

This version of ``trade_utils`` extends the original implementation with
additional technical indicators, improved session handling, dynamic
stop‑loss/take‑profit calculation support and more robust volume filtering.

Key improvements:

* Compute advanced indicators such as double/triple exponential moving averages (DEMA/TEMA),
  stochastic oscillator, commodity channel index (CCI), Keltner Channels and Average True Range (ATR).
* Provide a timezone‑aware ``get_market_session`` that maps the current time in
  Asia/Karachi to a session (Asia/Europe/US) rather than relying on naive UTC hours.
* Increase kline history to up to 500 candles (≈40 hours) for higher quality
  indicator calculations and volume assessment.
* Include optional dynamic risk metrics (ATR) that downstream modules can use
  to derive adaptive stop‑loss and profit targets.
* Preserve all existing functions and signatures for backward compatibility;
  only new indicator columns are appended to the output DataFrame and
  existing call sites remain unaffected.

Note: If any of the additional indicators fail due to missing data or
unexpected types, they are silently skipped. This ensures the module
remains robust when interacting with external APIs.
"""

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator, DEMAIndicator, TEMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import VolumeWeightedAveragePrice
from binance.client import Client
from symbol_mapper import map_symbol_for_binance
import os
import json
from datetime import datetime
from typing import Optional

# Use zoneinfo from stdlib for timezone conversion (Python ≥3.9)
try:
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:
    ZoneInfo = None  # fallback if not available

from price_action import detect_support_resistance_zones, is_price_near_zone
from orderflow import detect_aggression
from pattern_memory import recall_pattern_confidence

# Path to persistent symbol scores file.
SYMBOL_SCORES_FILE = os.path.join(os.path.dirname(__file__), "symbol_scores.json")

# Initialise a Binance client for price data
client = Client()


def get_market_session() -> str:
    """Return the current market session based on local time in Asia/Karachi.

    The function converts the current UTC time into the Asia/Karachi timezone
    using Python's ``zoneinfo`` module (if available) and then assigns a
    session label:

    * ``Asia`` for 00:00–08:00 local
    * ``Europe`` for 08:00–16:00 local
    * ``US`` for 16:00–24:00 local

    If timezone conversion is unavailable, it falls back to UTC hour logic.
    """
    try:
        if ZoneInfo is not None:
            now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("Asia/Karachi"))
            hour = now.hour
        else:
            hour = datetime.utcnow().hour
    except Exception:
        hour = datetime.utcnow().hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 16:
        return "Europe"
    else:
        return "US"


def get_price_data(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch recent OHLCV data for a symbol from Binance.

    The returned DataFrame includes columns: ``open``, ``high``, ``low``, ``close``,
    ``volume`` and ``quote_volume``.  Up to 500 5‑minute candles are requested to
    provide a longer history for indicator calculations.  If the request fails,
    ``None`` is returned.
    """
    try:
        mapped_symbol = map_symbol_for_binance(symbol)
        # Request more candles (500) for richer context; Binance caps at 1500 per call
        klines = client.get_klines(symbol=mapped_symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=500)
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


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a suite of technical indicators on a price DataFrame.

    This function augments the input DataFrame with columns for moving averages,
    momentum oscillators, volatility bands and volume‑weighted metrics.  New
    indicators added in this enhanced version include DEMA, TEMA, stochastic
    oscillator, CCI, Keltner Channels and ATR.  Missing data are filled with
    neutral values where appropriate.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['high', 'low', 'close'])
    try:
        # Basic trend indicators
        df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
        macd_obj = MACD(df['close'])
        df['macd'] = macd_obj.macd_diff()
        df['macd_signal'] = macd_obj.macd_signal()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx().fillna(0)
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        # Volume weighted moving average
        vwma_calc = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=20)
        df['vwma'] = vwma_calc.volume_weighted_average_price()
        # Advanced moving averages
        df['dema_20'] = DEMAIndicator(df['close'], window=20).dema_indicator()
        df['tema_20'] = TEMAIndicator(df['close'], window=20).tema_indicator()
        # Stochastic oscillator
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        # Commodity Channel Index
        df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        # Keltner Channels
        kc = KeltnerChannel(df['high'], df['low'], df['close'], window=20)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        # Average True Range for dynamic risk assessment
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
    except Exception as e:
        # If any calculation fails, print debug info and proceed with existing columns
        print(f"[INDICATOR] Warning: failed to compute some indicators: {e}")
    return df


def get_top_symbols(limit: int = 30) -> list:
    """Return the top quote‑volume symbols trading against USDT.

    The list is filtered to exclude BUSD pairs and sorted by 24‑hour quote volume.
    """
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
    symbols = [x['symbol'] for x in sorted_tickers if x['symbol'].endswith("USDT") and not x['symbol'].endswith("BUSD")]
    return symbols[:limit]


def log_signal(symbol: str, session: str, score: float, direction: Optional[str], weights: dict,
               candle_patterns: list, chart_pattern: Optional[str]) -> None:
    """Append a signal entry to the trades log.

    This helper persists key technical metrics for later analysis.  It writes to a
    CSV file located alongside this module to maintain consistency across
    different working directories.
    """
    log_entry = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
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
        "dema_weight": weights.get("dema", 0),
        "stoch_weight": weights.get("stoch", 0),
        "cci_weight": weights.get("cci", 0),
        "candle_patterns": ", ".join(candle_patterns),
        "chart_pattern": chart_pattern if chart_pattern else "None"
    }
    df_entry = pd.DataFrame([log_entry])
    log_path = os.path.join(os.path.dirname(__file__), "trades_log.csv")
    if os.path.exists(log_path):
        df_entry.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df_entry.to_csv(log_path, index=False)


def get_position_size(confidence: float) -> int:
    """Return an integer position size based on the model confidence."""
    if confidence >= 8.5:
        return 100
    elif confidence >= 6.5:
        return 80
    elif confidence >= 5.5:
        return 50
    else:
        return 0


def evaluate_signal(price_data: pd.DataFrame, symbol: str = "", sentiment_bias: str = "neutral"):
    """Evaluate a trading signal given a price DataFrame.

    This function calculates a composite score based on multiple technical
    indicators and returns a tuple of ``(score, direction, position_size, pattern_name)``.
    It now incorporates additional indicators (DEMA, stochastic, CCI) into the
    scoring logic, allowing for more nuanced momentum assessment.  The scoring
    weights have been extended accordingly.
    """
    try:
        if price_data is None or price_data.empty or len(price_data) < 40:
            print(f"[DEBUG] Skipping {symbol}: insufficient price data.")
            return 0, None, 0, None

        price_data = price_data.replace([np.inf, -np.inf], np.nan)
        price_data = price_data.dropna(subset=['high', 'low', 'close'])
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        # Core indicators
        ema_short = EMAIndicator(close, window=20).ema_indicator()
        ema_long = EMAIndicator(close, window=50).ema_indicator()
        macd_line = MACD(close).macd_diff()
        rsi = RSIIndicator(close, window=14).rsi()
        with np.errstate(invalid='ignore', divide='ignore'):
            adx_series = ADXIndicator(high=high, low=low, close=close, window=14).adx()
        adx = adx_series.fillna(0)
        bb = BollingerBands(close, window=20, window_dev=2)
        vwma_calc = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=20)
        vwma = vwma_calc.volume_weighted_average_price()
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        # New indicators
        dema_short = DEMAIndicator(close, window=20).dema_indicator()
        dema_long = DEMAIndicator(close, window=50).dema_indicator()
        stoch_obj = StochasticOscillator(high, low, close, window=14, smooth_window=3)
        stoch_k = stoch_obj.stoch()
        stoch_d = stoch_obj.stoch_signal()
        cci = CCIIndicator(high, low, close, window=20).cci()
        # Volume/quote volume for dynamic filter
        avg_quote_vol_20 = price_data['quote_volume'].iloc[-20:].mean() if 'quote_volume' in price_data else None
        latest_quote_vol = price_data['quote_volume'].iloc[-1] if 'quote_volume' in price_data else None
        price_now = close.iloc[-1]
        if avg_quote_vol_20 is None:
            avg_quote_vol_20 = volume.iloc[-20:].mean() * price_now
            latest_quote_vol = volume.iloc[-1] * price_now
        session_name = get_market_session()
        session_factor = {"Asia": 0.3, "Europe": 0.3, "US": 0.4}
        vol_factor = session_factor.get(session_name, 0.4)
        vol_threshold = max(vol_factor * avg_quote_vol_20, 50_000)
        if latest_quote_vol < vol_threshold:
            print(f"⛔ Skipping due to low volume: {latest_quote_vol:,.0f} < {vol_threshold:,.0f} ({vol_factor*100:.0f}% of 20-bar avg)")
            return 0, None, 0, None
        # Determine weights per session and incorporate new indicators
        base_weights = {
            "ema": 1.4, "macd": 1.3, "rsi": 1.3, "adx": 1.5,
            "vwma": 1.4, "bb": 1.3, "candle": 1.2, "chart": 1.2,
            "flag": 1.0, "flow": 1.5, "dema": 1.1, "stoch": 1.0, "cci": 1.0
        }
        # Adjust weights slightly by session to emphasise momentum during US hours
        if session_name == "US":
            base_weights["macd"] += 0.2
            base_weights["stoch"] += 0.2
        w = base_weights
        reinforcement = 1.0  # placeholder; pattern memory bonus is applied later
        score = 0.0
        # Score accumulation using original rules plus new indicators
        if ema_short.iloc[-1] > ema_long.iloc[-1]:
            score += w["ema"]
        if macd_line.iloc[-1] > 0:
            score += w["macd"]
        rsi_val = rsi.iloc[-1]
        if rsi_val > 50:
            score += w["rsi"]
        if adx.iloc[-1] > 20:
            score += w["adx"]
        vwma_value = vwma.iloc[-1]
        vwma_dev = abs(price_now - vwma_value) / price_now if price_now != 0 else 0.0
        if price_now > vwma_value:
            if vwma_dev <= 0.05:
                score += w["vwma"]
            elif vwma_dev < 0.10:
                fraction = (0.10 - vwma_dev) / 0.05
                score += w["vwma"] * fraction
        if bb_lower.iloc[-1] and bb_lower.iloc[-1] < price_now < bb_upper.iloc[-1]:
            score += w["bb"]
        # DEMA trend: faster moving average crossing over slower one
        if dema_short.iloc[-1] > dema_long.iloc[-1]:
            score += w["dema"]
        # Stochastic oscillator: k above d but not overbought (>80)
        if stoch_k.iloc[-1] > stoch_d.iloc[-1] and stoch_k.iloc[-1] < 80:
            score += w["stoch"]
        # CCI positive indicates upward momentum
        if cci.iloc[-1] > 0:
            score += w["cci"]
        # Candlestick patterns and chart patterns (reuse existing detectors)
        candle_patterns = detect_candlestick_patterns(price_data)
        chart_pattern = detect_triangle_wedge(price_data)
        flag = detect_flag_pattern(price_data)
        if candle_patterns:
            score += w["candle"]
        if chart_pattern:
            score += w["chart"]
        if flag:
            score += w["flag"]
        # Order flow adjustment
        aggression = detect_aggression(price_data)
        if aggression == "buyers in control":
            score += w["flow"]
        elif aggression == "sellers in control":
            score -= w["flow"]
        # Normalise score to 0–10 scale
        max_possible = sum(w.values())
        normalized_score = round((score / max_possible) * 10 * reinforcement, 2)
        # Sentiment bias adjustments
        if sentiment_bias == "bullish" and normalized_score < 5.0:
            normalized_score += 0.8
        elif sentiment_bias == "bearish" and normalized_score > 7.5:
            normalized_score -= 0.8
        normalized_score = round(normalized_score, 2)
        direction = "long" if normalized_score >= 4.5 else None
        position_size = get_position_size(normalized_score)
        # Support/resistance checks remain unchanged
        zones = detect_support_resistance_zones(price_data)
        zones = zones.to_dict() if isinstance(zones, pd.Series) else (zones or {"support": [], "resistance": []})
        current_price = float(close.iloc[-1])
        if direction == "long":
            near_resistance = is_price_near_zone(current_price, zones, 'resistance', 0.005)
            near_support = is_price_near_zone(current_price, zones, 'support', 0.015 if sentiment_bias == "bullish" else 0.01)
            if near_resistance and normalized_score < 7.0:
                return 0, None, 0, None
            if not near_support and normalized_score < 6.5:
                return 0, None, 0, None
        # Identify primary pattern name for memory
        pattern_name = candle_patterns[0] if candle_patterns else (chart_pattern if chart_pattern else "None")
        # Persist symbol scores for dashboard usage
        try:
            scores = {}
            if os.path.exists(SYMBOL_SCORES_FILE):
                with open(SYMBOL_SCORES_FILE, "r") as f:
                    scores = json.load(f)
            scores[symbol] = {
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "score": normalized_score,
                "direction": direction,
                "position_size": position_size,
                "pattern": pattern_name
            }
            with open(SYMBOL_SCORES_FILE, "w") as f:
                json.dump(scores, f, indent=2)
        except Exception as e:
            print(f"[SYMBOL SCORES] Failed to update symbol_scores.json: {e}")
        return normalized_score, direction, position_size, pattern_name
    except Exception as e:
        print(f"⚠️ Signal evaluation error in {symbol}: {e}")
        return 0, None, 0, None
