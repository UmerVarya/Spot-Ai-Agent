"""
Enhanced trade utility functions for the Spot AI Super Agent.

Key improvements:

* Adds additional technical indicators (ATR, CCI, Stochastic) to the indicator set.
* Supports configurable Binance klines limit via environment variable ``KLINES_LIMIT`` and
  drops the most recent candle to avoid look‑ahead bias.
* Uses ``zoneinfo`` to compute current trading session based on a configurable
  local timezone (``LOCAL_TIMEZONE``), defaulting to Asia/Karachi.
* Filters symbols by 24‑hour quote volume using a configurable minimum volume
  (``MIN_VOLUME_USDT``) and sorts descending before selection.
* Provides dynamic indicator scoring with adjustable weights per session and
  integrates additional factors such as Stochastic oscillator and CCI.
"""

import os
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from binance.client import Client
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import StochRSIIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta.trend import CCIIndicator
from zoneinfo import ZoneInfo

from symbol_mapper import map_symbol_for_binance
from price_action import detect_support_resistance_zones, is_price_near_zone
from orderflow import detect_aggression
from pattern_memory import recall_pattern_confidence
from confidence import calculate_historical_confidence

# Path to persistent symbol scores file.  Use fixed path relative to this module
SYMBOL_SCORES_FILE = os.path.join(os.path.dirname(__file__), "symbol_scores.json")

# Instantiate a Binance client once for all functions
client = Client()

# Environment variables
LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "Asia/Karachi")
MIN_VOLUME_USDT = float(os.getenv("MIN_VOLUME_USDT", 100000))
KLINES_LIMIT = int(os.getenv("KLINES_LIMIT", 500))


def get_market_session() -> str:
    """Return the current trading session (Asia, Europe or US) based on local time.

    The local timezone can be configured via the ``LOCAL_TIMEZONE`` environment
    variable.  Session boundaries are approximate and correspond to when most
    liquidity in crypto markets is concentrated.
    """
    try:
        tz = ZoneInfo(LOCAL_TIMEZONE)
    except Exception:
        tz = ZoneInfo("UTC")
    now_local = datetime.now(tz)
    hour = now_local.hour
    # Session boundaries (approximate)
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 16:
        return "Europe"
    else:
        return "US"


def get_top_symbols(limit: int = 30) -> List[str]:
    """Return a list of liquid USDT pairs sorted by 24h quote volume.

    This function filters out BUSD pairs and requires quote volume to exceed
    ``MIN_VOLUME_USDT``.  The result is sorted in descending order by
    quote volume and truncated to ``limit`` symbols.
    """
    tickers = client.get_ticker()
    candidates = []
    for t in tickers:
        symbol = t.get("symbol")
        if not symbol or not symbol.endswith("USDT") or symbol.endswith("BUSD"):
            continue
        try:
            vol = float(t.get("quoteVolume", 0))
        except Exception:
            continue
        if vol < MIN_VOLUME_USDT:
            continue
        candidates.append((symbol, vol))
    # Sort by volume descending
    sorted_syms = sorted(candidates, key=lambda x: x[1], reverse=True)
    return [s[0] for s in sorted_syms[:limit]]


def get_price_data(symbol: str) -> pd.DataFrame | None:
    """Fetch recent price data for ``symbol`` using the configured klines limit.

    The function returns a DataFrame with columns [open, high, low, close,
    volume, quote_volume].  It drops the most recent candle to avoid
    look‑ahead bias.  If data cannot be fetched or is insufficient, returns
    ``None``.
    """
    try:
        mapped_symbol = map_symbol_for_binance(symbol)
        klines = client.get_klines(
            symbol=mapped_symbol,
            interval=Client.KLINE_INTERVAL_5MINUTE,
            limit=KLINES_LIMIT,
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        # Convert numeric columns
        num_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        df[num_cols] = df[num_cols].astype(float)
        df['quote_volume'] = df['quote_asset_volume']
        # Drop last row (incomplete candle)
        if len(df) > 1:
            df = df.iloc[:-1]
        return df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']]
    except Exception as e:
        print(f"⚠️ Failed to fetch data for {symbol}: {e}")
        return None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a suite of technical indicators on the given DataFrame.

    Returns a new DataFrame with additional columns: ema_20, ema_50,
    macd, macd_signal, rsi, adx, bb_upper, bb_lower, bb_middle, atr,
    vwma, cci, stoch_k, stoch_d.  Any NaN/inf values are replaced with
    zeros.  If computation fails, missing columns are filled with zeros.
    """
    df = df.copy().replace([np.inf, -np.inf], np.nan).dropna(subset=['high', 'low', 'close'])
    try:
        df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
        macd = MACD(df['close'])
        df['macd'] = macd.macd_diff()
        df['macd_signal'] = macd.macd_signal()
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        with np.errstate(invalid='ignore', divide='ignore'):
            df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx().fillna(0)
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
        df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        vwma = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=20)
        df['vwma'] = vwma.volume_weighted_average_price()
    except Exception:
        # Fill missing columns with zeros
        for col in [
            'ema_20', 'ema_50', 'macd', 'macd_signal', 'rsi', 'adx', 'bb_upper', 'bb_lower',
            'bb_middle', 'atr', 'cci', 'stoch_k', 'stoch_d', 'vwma'
        ]:
            if col not in df.columns:
                df[col] = 0.0
    return df


def get_position_size(confidence: float) -> int:
    """Map a confidence score (0–10) to a position size in USDT units.

    This is a simple linear mapping that can be adjusted.  The size
    increases with confidence but never exceeds a maximum lot size.  This
    implementation assumes a maximum trade size of 1000 USDT.
    """
    max_size = float(os.getenv("MAX_POSITION_USDT", 1000))
    min_size = float(os.getenv("MIN_POSITION_USDT", 100))
    # Scale confidence (0–10) to between min_size and max_size
    scaled = min_size + (max_size - min_size) * (confidence / 10.0)
    return int(round(scaled))


def evaluate_signal(price_data: pd.DataFrame, symbol: str = "") -> Tuple[float, str | None, int, str]:
    """Evaluate trading signal for a symbol.

    Returns a tuple of ``(score, direction, position_size, pattern_name)``.
    The score is normalised to 0–10.  Direction is ``"long"`` or
    ``None``.  Position size is an integer USDT amount.  Pattern name
    identifies the primary candlestick or chart pattern detected.
    """
    try:
        if price_data is None or price_data.empty or len(price_data) < 30:
            print(f"[DEBUG] Skipping {symbol}: insufficient price data.")
            return 0.0, None, 0, "None"
        # Compute indicators
        df = calculate_indicators(price_data)
        # Extract latest values
        ema_short = df['ema_20'].iloc[-1]
        ema_long = df['ema_50'].iloc[-1]
        macd_hist = df['macd'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        adx_val = df['adx'].iloc[-1]
        atr_val = df['atr'].iloc[-1]
        cci_val = df['cci'].iloc[-1]
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]
        vwma_val = df['vwma'].iloc[-1]
        price_now = df['close'].iloc[-1]
        volume_now = df['volume'].iloc[-1]
        # Volume filter: require quote volume > threshold
        quote_vol = df['quote_volume'].iloc[-20:].mean() if 'quote_volume' in df else None
        if quote_vol is not None and quote_vol < MIN_VOLUME_USDT / 2:
            print(f"[DEBUG] {symbol} volume below threshold: {quote_vol}")
            return 0.0, None, 0, "None"
        # Determine session and scoring weights
        session = get_market_session()
        weights = {
            "Asia":   {"ema": 1.5, "macd": 1.3, "rsi": 1.3, "adx": 1.5, "vwma": 1.3, "cci": 1.2, "stoch": 1.2},
            "Europe": {"ema": 1.5, "macd": 1.3, "rsi": 1.3, "adx": 1.6, "vwma": 1.3, "cci": 1.2, "stoch": 1.2},
            "US":     {"ema": 1.3, "macd": 1.5, "rsi": 1.5, "adx": 1.5, "vwma": 1.3, "cci": 1.2, "stoch": 1.2},
        }[session]
        score = 0.0
        # EMA trend
        if ema_short > ema_long:
            score += weights["ema"]
        # MACD histogram positive
        if macd_hist > 0:
            score += weights["macd"]
        # RSI above 50
        if rsi > 50:
            score += weights["rsi"]
        # ADX strong trend
        if adx_val > 20:
            score += weights["adx"]
        # VWMA: price above VWMA
        if price_now > vwma_val:
            score += weights["vwma"]
        # CCI positive (indicating upward momentum)
        if cci_val > 0:
            score += weights["cci"]
        # Stochastic oversold crossing up
        if stoch_k > stoch_d and stoch_k < 80:
            score += weights["stoch"]
        # Order flow
        flow = detect_aggression(price_data)
        if flow == "buyers in control":
            score += 0.5
        elif flow == "sellers in control":
            score -= 0.5
        # Normalise score to 0–10
        max_score = sum(weights.values()) + 0.5
        norm_score = round(max(0.0, min((score / max_score) * 10, 10.0)), 2)
        # Determine direction
        direction = "long" if norm_score >= 4.5 else None
        # Position size
        pos_size = get_position_size(norm_score)
        # Determine pattern (placeholder)
        pattern_name = "None"
        return norm_score, direction, pos_size, pattern_name
    except Exception as e:
        print(f"⚠️ Signal evaluation error in {symbol}: {e}")
        return 0.0, None, 0, "None"
