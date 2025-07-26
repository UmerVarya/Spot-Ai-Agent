"""
Enhanced trade utility functions for the Spot AI Super Agent.

This module consolidates common operations such as fetching price data,
computing technical indicators, evaluating trading signals and mapping
confidence scores to position sizes.  It introduces additional
indicators (ATR, CCI, Stochastic oscillator, VWMA), avoids look‑ahead
bias by dropping the most recent candle, and supports configurable
timezone handling and symbol filtering based on 24h volume.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from binance.client import Client
from ta.trend import EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from zoneinfo import ZoneInfo

from symbol_mapper import map_symbol_for_binance
from price_action import detect_support_resistance_zones, is_price_near_zone
from orderflow import detect_aggression
from pattern_memory import recall_pattern_confidence
from confidence import calculate_historical_confidence

# Persistent symbol scores file path
SYMBOL_SCORES_FILE = os.path.join(os.path.dirname(__file__), "symbol_scores.json")

# Instantiate Binance client once
client = Client()

# Environment variables
LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "Asia/Karachi")
# Default minimum 24h quote volume threshold for selecting symbols.  A lower
# default (30000 USDT) yields a broader universe; adjust via the
# ``MIN_VOLUME_USDT`` environment variable if needed.
MIN_VOLUME_USDT = float(os.getenv("MIN_VOLUME_USDT", 30000))
KLINES_LIMIT = int(os.getenv("KLINES_LIMIT", 500))


def get_market_session() -> str:
    """Return the current trading session (Asia, Europe or US) based on local time."""
    try:
        tz = ZoneInfo(LOCAL_TIMEZONE)
    except Exception:
        tz = ZoneInfo("UTC")
    now_local = datetime.now(tz)
    hour = now_local.hour
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 16:
        return "Europe"
    else:
        return "US"


def get_top_symbols(limit: int = 30) -> List[str]:
    """Return a list of liquid USDT pairs sorted by 24h quote volume.

    This function fetches the 24h ticker statistics from Binance,
    filters out pairs with insufficient quote volume or unsupported
    settlement assets, and validates that the symbol exists on Binance
    before returning it.  If ``MIN_VOLUME_USDT`` is set very high you
    may see few or no candidates; adjust the environment variable to
    widen the selection.
    """
    try:
        tickers = client.get_ticker()
    except Exception:
        return []
    candidates: list[tuple[str, float]] = []
    for t in tickers:
        symbol = t.get("symbol")
        # Only consider USDT pairs (exclude BUSD pairs).  We no longer call
        # ``client.get_symbol_info`` here to avoid rate limiting; invalid
        # symbols will be filtered out later by ``get_price_data``.
        if not symbol or not symbol.endswith("USDT") or symbol.endswith("BUSD"):
            continue
        try:
            vol = float(t.get("quoteVolume", 0))
        except Exception:
            continue
        # Skip low volume pairs
        if vol < MIN_VOLUME_USDT:
            continue
        candidates.append((symbol, vol))
    if not candidates:
        return []
    sorted_syms = sorted(candidates, key=lambda x: x[1], reverse=True)
    # Return only the symbols; invalid pairs will be filtered later in ``get_price_data``
    return [s for s, _ in sorted_syms[:limit]]


def get_price_data(symbol: str) -> pd.DataFrame | None:
    """Fetch recent price data for ``symbol`` using the configured klines limit.

    Returns a DataFrame with columns [open, high, low, close, volume, quote_volume].
    Drops the most recent candle to avoid look‑ahead bias.  Returns None on error.
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
        num_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
        df[num_cols] = df[num_cols].astype(float)
        df['quote_volume'] = df['quote_asset_volume']
        # Drop incomplete last candle
        if len(df) > 1:
            df = df.iloc[:-1]
        return df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']]
    except Exception as e:
        print(f"⚠️ Failed to fetch data for {symbol}: {e}")
        return None


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a suite of technical indicators on the given DataFrame."""
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
        for col in [
            'ema_20', 'ema_50', 'macd', 'macd_signal', 'rsi', 'adx',
            'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'cci', 'stoch_k',
            'stoch_d', 'vwma'
        ]:
            if col not in df.columns:
                df[col] = 0.0
    return df


def get_position_size(confidence: float) -> int:
    """Map a confidence score (0–10) to a position size in USDT units."""
    max_size = float(os.getenv("MAX_POSITION_USDT", 1000))
    min_size = float(os.getenv("MIN_POSITION_USDT", 100))
    scaled = min_size + (max_size - min_size) * (confidence / 10.0)
    return int(round(scaled))


def evaluate_signal(price_data: pd.DataFrame, symbol: str = "") -> Tuple[float, str | None, int, str]:
    """Evaluate trading signal for a symbol.

    Returns a tuple of (score, direction, position_size, pattern_name).
    Score is normalised to 0–10.  Direction is 'long' or None.  Position size
    is an integer USDT amount.  Pattern name identifies the primary
    candlestick or chart pattern detected.
    """
    try:
        if price_data is None or price_data.empty or len(price_data) < 30:
            print(f"[DEBUG] Skipping {symbol}: insufficient price data.")
            return 0.0, None, 0, "None"
        df = calculate_indicators(price_data)
        ema_short = df['ema_20'].iloc[-1]
        ema_long = df['ema_50'].iloc[-1]
        macd_hist = df['macd'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        adx = df['adx'].iloc[-1]
        stoch_k = df['stoch_k'].iloc[-1]
        stoch_d = df['stoch_d'].iloc[-1]
        cci = df['cci'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        close = df['close'].iloc[-1]
        # Signal components
        trend_score = 0
        if ema_short > ema_long:
            trend_score += 1
        if macd_hist > 0:
            trend_score += 1
        if rsi > 55:
            trend_score += 1
        if adx > 20:
            trend_score += 1
        momentum_score = 0
        if stoch_k > stoch_d and stoch_k < 80:
            momentum_score += 1
        if cci > 100:
            momentum_score += 1
        if close > bb_upper:
            momentum_score -= 1
        elif close < bb_lower:
            momentum_score += 1
        # Range score – price near support/resistance
        support_zone, resistance_zone = detect_support_resistance_zones(df)
        is_support = is_price_near_zone(close, support_zone)
        is_resist = is_price_near_zone(close, resistance_zone)
        range_score = 0
        if is_support:
            range_score += 1
        if is_resist:
            range_score -= 1
        # Order flow score
        flow = detect_aggression(df)
        flow_score = 0
        if flow == "buyers in control":
            flow_score += 1
        elif flow == "sellers in control":
            flow_score -= 1
        # Pattern confidence
        pattern_name = "None"
        pattern_confidence = 0
        # Try to recall pattern confidence from memory
        try:
            pattern_confidence = recall_pattern_confidence(pattern_name)
        except Exception:
            pattern_confidence = 0
        # Historical confidence adjustment
        historical_conf = calculate_historical_confidence(symbol)
        # Weighted sum
        score_raw = (trend_score * 0.4 + momentum_score * 0.3 + range_score * 0.1 + flow_score * 0.1 + pattern_confidence * 0.1)
        # Normalise to 0–10
        score = max(0.0, min(10.0, (score_raw + 3) * 2))  # shift and scale
        # Determine direction and position size
        direction = "long" if score >= 5 else None
        position_size = get_position_size(score / 10.0 * 10)
        return score, direction, position_size, pattern_name
    except Exception as e:
        print(f"⚠️ evaluate_signal error for {symbol}: {e}")
        return 0.0, None, 0, "None"
