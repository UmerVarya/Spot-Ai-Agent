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

# ---------------------------------------------------------------------------
# Optional TA-Lib imports
#
# The Spot AI bot originally relies on the ``ta`` library for computing
# technical indicators (EMA, MACD, RSI, ADX, Bollinger Bands, etc.).
# Unfortunately, the production environment may not always have the
# ``ta`` package installed.  To ensure this module still loads and
# operates without crashing, we attempt to import these indicators from
# ``ta``.  If that fails, we define lightweight fallback classes that
# compute approximate versions of these indicators using pandas.  These
# fallbacks may not be as precise or feature‑rich as the ``ta``
# implementation, but they maintain functional parity and prevent
# deployment failures due to missing dependencies.

try:
    from ta.trend import EMAIndicator as _TA_EMAIndicator, MACD as _TA_MACD, ADXIndicator as _TA_ADXIndicator, DEMAIndicator as _TA_DEMAIndicator, TEMAIndicator as _TA_TEMAIndicator
    from ta.momentum import RSIIndicator as _TA_RSIIndicator, StochasticOscillator as _TA_StochasticOscillator, CCIIndicator as _TA_CCIIndicator
    from ta.volatility import BollingerBands as _TA_BollingerBands, AverageTrueRange as _TA_AverageTrueRange, KeltnerChannel as _TA_KeltnerChannel
    from ta.volume import VolumeWeightedAveragePrice as _TA_VolumeWeightedAveragePrice
    # Alias the imported classes so the rest of the code can use generic names
    EMAIndicator = _TA_EMAIndicator
    MACD = _TA_MACD
    ADXIndicator = _TA_ADXIndicator
    DEMAIndicator = _TA_DEMAIndicator
    TEMAIndicator = _TA_TEMAIndicator
    RSIIndicator = _TA_RSIIndicator
    StochasticOscillator = _TA_StochasticOscillator
    CCIIndicator = _TA_CCIIndicator
    BollingerBands = _TA_BollingerBands
    AverageTrueRange = _TA_AverageTrueRange
    KeltnerChannel = _TA_KeltnerChannel
    VolumeWeightedAveragePrice = _TA_VolumeWeightedAveragePrice
except Exception:
    # Provide simple fallback implementations for core indicators.
    # These implementations use pandas operations to approximate the
    # behaviour of the indicators.  They are not perfect substitutes but
    # prevent the absence of the ``ta`` library from crashing the agent.

    def _ema(series: pd.Series, span: int) -> pd.Series:
        """Return the exponential moving average of a series."""
        return series.ewm(span=span, adjust=False).mean()

    class EMAIndicator:
        def __init__(self, series: pd.Series, window: int = 14):
            self.series = series
            self.window = window
        def ema_indicator(self) -> pd.Series:
            return _ema(self.series, self.window)

    class MACD:
        def __init__(self, series: pd.Series, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9):
            self.series = series
            self.window_slow = window_slow
            self.window_fast = window_fast
            self.window_sign = window_sign
        def macd_diff(self) -> pd.Series:
            ema_fast = _ema(self.series, self.window_fast)
            ema_slow = _ema(self.series, self.window_slow)
            return ema_fast - ema_slow
        def macd_signal(self) -> pd.Series:
            diff = self.macd_diff()
            return diff.ewm(span=self.window_sign, adjust=False).mean()

    class RSIIndicator:
        def __init__(self, series: pd.Series, window: int = 14):
            self.series = series
            self.window = window
        def rsi(self) -> pd.Series:
            delta = self.series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(self.window).mean()
            avg_loss = loss.rolling(self.window).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            return 100 - (100 / (1 + rs))

    class ADXIndicator:
        def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
            self.high = high
            self.low = low
            self.close = close
            self.window = window
        def adx(self) -> pd.Series:
            # Compute directional movement
            up_move = self.high.diff()
            down_move = self.low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            # True range
            tr1 = self.high - self.low
            tr2 = (self.high - self.close.shift()).abs()
            tr3 = (self.low - self.close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(self.window).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(self.window).mean() / (atr + 1e-9))
            minus_di = 100 * (pd.Series(minus_dm).rolling(self.window).mean() / (atr + 1e-9))
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
            adx = dx.rolling(self.window).mean()
            return adx

    class BollingerBands:
        def __init__(self, series: pd.Series, window: int = 20, window_dev: int = 2):
            self.series = series
            self.window = window
            self.window_dev = window_dev
        def bollinger_hband(self) -> pd.Series:
            sma = self.series.rolling(self.window).mean()
            std = self.series.rolling(self.window).std()
            return sma + self.window_dev * std
        def bollinger_lband(self) -> pd.Series:
            sma = self.series.rolling(self.window).mean()
            std = self.series.rolling(self.window).std()
            return sma - self.window_dev * std
        def bollinger_mavg(self) -> pd.Series:
            return self.series.rolling(self.window).mean()

    class VolumeWeightedAveragePrice:
        def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20):
            self.price = (high + low + close) / 3
            self.volume = volume
            self.window = window
        def volume_weighted_average_price(self) -> pd.Series:
            pv = self.price * self.volume
            cum_pv = pv.rolling(self.window).sum()
            cum_vol = self.volume.rolling(self.window).sum()
            return cum_pv / (cum_vol + 1e-9)

    class DEMAIndicator:
        def __init__(self, series: pd.Series, window: int = 20):
            self.series = series
            self.window = window
        def dema_indicator(self) -> pd.Series:
            ema1 = _ema(self.series, self.window)
            ema2 = _ema(ema1, self.window)
            return 2 * ema1 - ema2

    class TEMAIndicator:
        def __init__(self, series: pd.Series, window: int = 20):
            self.series = series
            self.window = window
        def tema_indicator(self) -> pd.Series:
            ema1 = _ema(self.series, self.window)
            ema2 = _ema(ema1, self.window)
            ema3 = _ema(ema2, self.window)
            return 3 * ema1 - 3 * ema2 + ema3

    class StochasticOscillator:
        def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, smooth_window: int = 3):
            self.high = high
            self.low = low
            self.close = close
            self.window = window
            self.smooth_window = smooth_window
        def stoch(self) -> pd.Series:
            lowest_low = self.low.rolling(self.window).min()
            highest_high = self.high.rolling(self.window).max()
            stoch = (self.close - lowest_low) / ((highest_high - lowest_low) + 1e-9) * 100
            return stoch
        def stoch_signal(self) -> pd.Series:
            return self.stoch().rolling(self.smooth_window).mean()

    class CCIIndicator:
        def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20):
            self.high = high
            self.low = low
            self.close = close
            self.window = window
        def cci(self) -> pd.Series:
            tp = (self.high + self.low + self.close) / 3
            sma = tp.rolling(self.window).mean()
            mean_dev = (tp - sma).abs().rolling(self.window).mean()
            cci = (tp - sma) / (0.015 * (mean_dev + 1e-9))
            return cci

    class AverageTrueRange:
        def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
            self.high = high
            self.low = low
            self.close = close
            self.window = window
        def average_true_range(self) -> pd.Series:
            tr1 = self.high - self.low
            tr2 = (self.high - self.close.shift()).abs()
            tr3 = (self.low - self.close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(self.window).mean()

    class KeltnerChannel:
        def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20):
            self.high = high
            self.low = low
            self.close = close
            self.window = window
        def keltner_channel_hband(self) -> pd.Series:
            ema = _ema(self.close, self.window)
            atr = AverageTrueRange(self.high, self.low, self.close, self.window).average_true_range()
            return ema + (atr * 2)
        def keltner_channel_lband(self) -> pd.Series:
            ema = _ema(self.close, self.window)
            atr = AverageTrueRange(self.high, self.low, self.close, self.window).average_true_range()
            return ema - (atr * 2)
# Binance client import.  The agent expects a Binance REST client to
# fetch kline and ticker data.  However, some environments may not
# include the ``binance`` library.  To prevent import errors during
# deployment, we attempt the import and provide a stub fallback that
# raises a descriptive error if used.  This allows the module to load
# even when Binance is unavailable.
try:
    from binance.client import Client  # type: ignore
except Exception:
    class Client:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Binance client is not installed. Please install 'python-binance' to fetch price data.")
# Map trading symbols to Binance format.  In some environments the
# ``symbol_mapper`` module may be absent.  Provide a passthrough as
# fallback.
try:
    from symbol_mapper import map_symbol_for_binance  # type: ignore
except Exception:
    def map_symbol_for_binance(symbol: str) -> str:  # type: ignore
        """Fallback symbol mapper: return the symbol unchanged."""
        return symbol
import os
import json
from datetime import datetime
from typing import Optional

# Use zoneinfo from stdlib for timezone conversion (Python ≥3.9)
try:
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:
    ZoneInfo = None  # fallback if not available

# Support/resistance detection.  If the ``price_action`` module is
# unavailable, provide dummy functions that return neutral results.
try:
    from price_action import detect_support_resistance_zones, is_price_near_zone  # type: ignore
except Exception:
    def detect_support_resistance_zones(df):  # type: ignore
        """Fallback: return no support or resistance zones."""
        return {"support": [], "resistance": []}

    def is_price_near_zone(price: float, zones: dict, zone_type: str, tolerance: float) -> bool:  # type: ignore
        """Fallback: always return False indicating price is not near any zone."""
        return False
# Order flow detection and pattern memory.  Provide fallbacks when
# modules are missing.
try:
    from orderflow import detect_aggression  # type: ignore
except Exception:
    def detect_aggression(df):  # type: ignore
        """Fallback order flow: always return neutral."""
        return "neutral"

try:
    from pattern_memory import recall_pattern_confidence  # type: ignore
except Exception:
    def recall_pattern_confidence(symbol: str, pattern_name: str) -> float:  # type: ignore
        """Fallback pattern memory: return zero boost."""
        return 0.0

# Optional imports for pattern detection. In the original repository these
# functions were referenced but not explicitly imported.  Here we attempt
# to import them from plausible modules. If not found, we provide no-op
# fallbacks so that the module does not crash at runtime.  This ensures
# deployments succeed even if pattern detection modules are absent.
try:
    # Try canonical module names first
    from pattern_recognizer import (
        detect_candlestick_patterns,
        detect_triangle_wedge,
        detect_flag_pattern,
    )  # type: ignore
except Exception:
    try:
        # Alternate module structure
        from pattern_recognizer.patterns import (
            detect_candlestick_patterns,
            detect_triangle_wedge,
            detect_flag_pattern,
        )  # type: ignore
    except Exception:
        # Fallback: define dummy functions that return neutral results.
        def detect_candlestick_patterns(df):
            """
            Fallback function when candlestick pattern detectors are unavailable.
            Returns an empty list so that no score is added for patterns.
            """
            return []

        def detect_triangle_wedge(df):
            """
            Fallback function for triangle/wedge detection.
            Returns None, meaning no pattern detected.
            """
            return None

        def detect_flag_pattern(df):
            """
            Fallback function for flag pattern detection.
            Returns False indicating no flag pattern.
            """
            return False

# Path to persistent symbol scores file.
SYMBOL_SCORES_FILE = os.path.join(os.path.dirname(__file__), "symbol_scores.json")

# Initialise a Binance client for price data.  When the Binance
# library is unavailable, ``Client`` is a stub that raises on
# instantiation; in that case, we catch the exception and leave
# ``client`` as ``None``.  Functions that rely on the client must
# handle the ``None`` case gracefully.
try:
    client = Client()
except Exception:
    client = None


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
    # When the Binance client is unavailable, return None immediately.
    if client is None:
        print(f"⚠️ Binance client unavailable; cannot fetch data for {symbol}.")
        return None
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
    if client is None:
        # If the Binance client is unavailable, return an empty list.
        print("⚠️ Binance client unavailable; get_top_symbols returning empty list.")
        return []
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

# ---------------------------------------------------------------------------
# Additional utility functions from the original trade_utils
#
# The original Spot AI agent imported ``simulate_slippage`` and
# ``estimate_commission`` from this module.  These functions are not
# included in the enhanced version by default but are added here as
# simple fallbacks to maintain compatibility with ``agent.py``.  They
# implement basic slippage and commission estimates for spot trading.

def simulate_slippage(price: float, direction: str = "long", slippage_pct: float = 0.0005) -> float:
    """
    Apply a simple slippage adjustment to a price.

    Args:
        price: The fill price before slippage.
        direction: "long" or "short"; long increases the price (worse for buys),
            short decreases the price (worse for sells).
        slippage_pct: The fraction of slippage to apply (e.g., 0.0005 = 0.05%).

    Returns:
        Adjusted price including slippage.
    """
    try:
        slip = float(slippage_pct)
    except Exception:
        slip = 0.0005
    if direction and direction.lower().startswith("s"):
        # For shorts (selling), price decreases due to slippage
        return price * (1 - slip)
    else:
        # For longs (buying), price increases due to slippage
        return price * (1 + slip)


def estimate_commission(symbol: str, quantity: float = 1.0, maker: bool = False) -> float:
    """
    Estimate the commission fee rate for a given trade.

    Args:
        symbol: The trading pair (unused in this simple estimate).
        quantity: Trade quantity (unused; commission is usually percentage based).
        maker: Whether the order is a maker (limit) order (lower fee) or taker (market).

    Returns:
        Commission rate as a percentage of trade value.  Binance fees are roughly
        0.1% for takers and 0.04% for makers, but this can vary by VIP tier.
    """
    # Use typical Binance spot trading fees
    return 0.0004 if maker else 0.001


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
        training_mode = os.getenv("TRAINING_MODE", "false").lower() == "true"

      # NEW: read TRAINING_MODE flag from environment.  When true, skip volume filter.
      training_mode = os.getenv("TRAINING_MODE", "false").lower() == "true"
      # Only enforce the volume threshold when *not* training
      if not training_mode and latest_quote_vol < vol_threshold:
        print(f"⛔ Skipping due to low volume: "
              f"{latest_quote_vol:,.0f} < {vol_threshold:,.0f} "
              f"({vol_factor*100:.0f}% of 20-bar avg)")
        return 0, None, 0, None

      # Only skip on low volume when NOT in training
      if not training_mode and latest_quote_vol < vol_threshold:
         print(f"⛔ Skipping due to low volume: "
               f"{latest_quote_vol:,.0f} < {vol_threshold:,.0f} "
               f"({vol_factor*100:.0f}% of 20-bar avg)")
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
