import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Optional
import traceback
import asyncio
from log_utils import setup_logger

from trade_storage import TRADE_LOG_FILE  # shared trade log path

from volatility_regime import atr_percentile, hurst_exponent  # type: ignore
from multi_timeframe import (
    multi_timeframe_confluence,
    multi_timeframe_indicator_alignment,
)  # type: ignore
from risk_metrics import (
    sharpe_ratio,
    calmar_ratio,
    max_drawdown,
    value_at_risk,
    expected_shortfall,
)  # type: ignore
# Directory containing this module
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the signal log file (formerly trades_log.csv)
SIGNAL_LOG_FILE = os.getenv("SIGNAL_LOG_FILE",
                            os.path.join(_MODULE_DIR, "signal_log.csv"))

logger = setup_logger(__name__)

# Optional TA-Lib imports (with pandas fallbacks if unavailable)
try:
    from ta.trend import (
        EMAIndicator as _TA_EMAIndicator,
        MACD as _TA_MACD,
        ADXIndicator as _TA_ADXIndicator,
        DEMAIndicator as _TA_DEMAIndicator,
        TEMAIndicator as _TA_TEMAIndicator,
    )
    from ta.momentum import (
        RSIIndicator as _TA_RSIIndicator,
        StochasticOscillator as _TA_StochasticOscillator,
        CCIIndicator as _TA_CCIIndicator,
    )
    from ta.volatility import (
        BollingerBands as _TA_BollingerBands,
        AverageTrueRange as _TA_AverageTrueRange,
        KeltnerChannel as _TA_KeltnerChannel,
    )
    from ta.volume import VolumeWeightedAveragePrice as _TA_VolumeWeightedAveragePrice
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
    # Fallback implementations
    def _ema(series: pd.Series, span: int) -> pd.Series:
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
            up_move = self.high.diff()
            down_move = self.low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            tr1 = self.high - self.low
            tr2 = (self.high - self.close.shift()).abs()
            tr3 = (self.low - self.close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(self.window).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(self.window).mean() / (atr + 1e-9))
            minus_di = 100 * (pd.Series(minus_dm).rolling(self.window).mean() / (atr + 1e-9))
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
            return dx.rolling(self.window).mean()
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
            return (self.close - lowest_low) / ((highest_high - lowest_low) + 1e-9) * 100
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

# Binance API client (with fallback if not installed)
try:
    from binance.client import Client  # type: ignore
except Exception:
    class Client:  # dummy fallback
        def __init__(self, *args, **kwargs):
            raise ImportError("Binance client is not installed. Please install 'python-binance' to fetch price data.")

# Symbol mapper fallback (use identity if mapper not available)
try:
    from symbol_mapper import map_symbol_for_binance  # type: ignore
except Exception:
    def map_symbol_for_binance(symbol: str) -> str:
        return symbol

# Timezone support
try:
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:
    ZoneInfo = None

# Support/Resistance detection (fallback returns no zones)
try:
    from price_action import detect_support_resistance_zones, is_price_near_zone  # type: ignore
except Exception:
    def detect_support_resistance_zones(df):
        return {"support": [], "resistance": []}
    def is_price_near_zone(price: float, zones: dict, zone_type: str, tolerance: float) -> bool:
        return False

# Order flow detection fallback
try:
    from orderflow import detect_aggression  # type: ignore
except Exception:
    def detect_aggression(df):
        return "neutral"

# Microstructure metrics fallback
try:
    from microstructure import compute_spread, compute_order_book_imbalance  # type: ignore
except Exception:
    def compute_spread(order_book):
        return float('nan')
    def compute_order_book_imbalance(order_book, depth: int = 10):
        return float('nan')

# Pattern memory fallback
try:
    from pattern_memory import recall_pattern_confidence  # type: ignore
except Exception:
    def recall_pattern_confidence(symbol: str, pattern_name: str) -> float:
        return 0.0

# Pattern detection fallbacks
try:
    from pattern_detection import (
        detect_triangle_wedge,
        detect_flag_pattern,
        detect_head_and_shoulders,
    )  # type: ignore
    from candlestick_patterns import detect_candlestick_patterns  # type: ignore
except Exception:
    try:
        from pattern_recognizer.patterns import (
            detect_candlestick_patterns,
            detect_triangle_wedge,
            detect_flag_pattern,
        )  # type: ignore
        def detect_head_and_shoulders(df):
            return False
    except Exception:
        def detect_candlestick_patterns(df):
            return []
        def detect_triangle_wedge(df):
            return None
        def detect_flag_pattern(df):
            return False
        def detect_head_and_shoulders(df):
            return False

# Path to stored symbol scores
SYMBOL_SCORES_FILE = os.path.join(os.path.dirname(__file__), "symbol_scores.json")

# Initialize Binance client
try:
    client = Client()
except Exception as e:
    logger.warning(
        "Failed to initialize Binance client: %s. Install the 'python-binance' package and ensure network/credentials are configured.",
        e,
        exc_info=True,
    )
    client = None

def get_market_session() -> str:
    """Return the current market session based on local time in Asia/Karachi."""
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
    """Fetch recent OHLCV data for a symbol from Binance."""
    if client is None:
        logger.warning("Binance client unavailable; cannot fetch data for %s.", symbol)
        return None
    try:
        mapped_symbol = map_symbol_for_binance(symbol)
        klines = client.get_klines(symbol=mapped_symbol, interval=Client.KLINE_INTERVAL_5MINUTE, limit=500)
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        df[["open", "high", "low", "close", "volume", "quote_asset_volume"]] = df[
            ["open", "high", "low", "close", "volume", "quote_asset_volume"]
        ].astype(float)
        # Convert timestamp to datetime and set as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df = df.set_index("timestamp")
        df["quote_volume"] = df["quote_asset_volume"]
        return df[["open", "high", "low", "close", "volume", "quote_volume"]]
    except Exception as e:
        logger.warning("Failed to fetch data for %s: %s", symbol, e, exc_info=True)
        return None


async def get_price_data_async(symbol: str) -> Optional[pd.DataFrame]:
    """Asynchronous wrapper around ``get_price_data`` using ``asyncio``."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_price_data, symbol)

def get_order_book(symbol: str, limit: int = 50) -> Optional[dict]:
    """Fetch order book depth from Binance."""
    if client is None:
        return None
    try:
        mapped_symbol = map_symbol_for_binance(symbol)
        book = client.get_order_book(symbol=mapped_symbol, limit=limit)
        return {
            "bids": [(float(p), float(q)) for p, q in book.get("bids", [])],
            "asks": [(float(p), float(q)) for p, q in book.get("asks", [])],
        }
    except Exception:
        return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a suite of technical indicators on a price DataFrame."""
    df = df.copy()
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.debug("[INDICATOR] Missing columns: %s", ', '.join(sorted(missing)))
        return df
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['high', 'low', 'close'])
    try:
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
        vwma_calc = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=20)
        df['vwma'] = vwma_calc.volume_weighted_average_price()
        df['dema_20'] = DEMAIndicator(df['close'], window=20).dema_indicator()
        df['tema_20'] = TEMAIndicator(df['close'], window=20).tema_indicator()
        stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['cci'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        kc = KeltnerChannel(df['high'], df['low'], df['close'], window=20)
        df['kc_upper'] = kc.keltner_channel_hband()
        df['kc_lower'] = kc.keltner_channel_lband()
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr'] = atr.average_true_range()
    except Exception as e:
        logger.warning("[INDICATOR] failed to compute some indicators: %s", e, exc_info=True)
    return df

def get_top_symbols(limit: int = 30) -> list:
    """Return the top quote-volume symbols trading against USDT."""
    if client is None:
        logger.warning("Binance client unavailable; get_top_symbols returning empty list.")
        return []
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
    symbols = [x['symbol'] for x in sorted_tickers if x['symbol'].endswith("USDT") and not x['symbol'].endswith("BUSD")]
    return symbols[:limit]

def compute_performance_metrics(log_file: str = TRADE_LOG_FILE, lookback: int = 100) -> dict:
    """Return risk-adjusted performance metrics from the trade log."""
    if not os.path.exists(log_file):
        return {}
    try:
        cols = [
            "timestamp", "symbol", "direction", "entry", "exit", "outcome",
            "btc_d", "fg", "sent_conf", "sent_bias", "score",
        ]
        df = pd.read_csv(log_file, names=cols, encoding="utf-8")
        df = df.tail(lookback)
        df["entry"] = pd.to_numeric(df["entry"], errors="coerce")
        df["exit"] = pd.to_numeric(df["exit"], errors="coerce")
        df = df.dropna(subset=["entry", "exit"])
        df["ret"] = (df["exit"] - df["entry"]) / df["entry"]
        rets = df["ret"].tolist()
        equity_curve = (1 + df["ret"]).cumprod()
        return {
            "sharpe": sharpe_ratio(rets),
            "calmar": calmar_ratio(rets),
            "max_drawdown": max_drawdown(equity_curve),
            "var": value_at_risk(rets),
            "es": expected_shortfall(rets),
        }
    except Exception:
        return {}


def get_last_trade_outcome(log_file: str = TRADE_LOG_FILE) -> str | None:
    """Return ``'win'`` or ``'loss'`` based on the most recent closed trade.

    The helper is used by the RL position‑sizer to condition its action
    selection on the outcome of the previous trade.  If the trade log is
    missing or empty, ``None`` is returned so that callers can fall back to a
    neutral state.
    """
    if not os.path.exists(log_file):
        return None
    try:
        df = pd.read_csv(log_file, names=[
            "timestamp", "symbol", "direction", "entry", "exit", "outcome",
            "btc_d", "fg", "sent_conf", "sent_bias", "score",
        ], encoding="utf-8")
        if df.empty:
            return None
        last = df.tail(1)
        entry = pd.to_numeric(last["entry"], errors="coerce").iloc[0]
        exit_price = pd.to_numeric(last["exit"], errors="coerce").iloc[0]
        if pd.isna(entry) or pd.isna(exit_price):
            return None
        return "win" if exit_price > entry else "loss"
    except Exception:
        return None

def log_signal(symbol: str, session: str, score: float, direction: Optional[str], weights: dict,
               candle_patterns: list, chart_pattern: Optional[str]) -> None:
    """Append a signal entry to the trades log."""
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
        "chart_pattern": chart_pattern if chart_pattern else "None",
    }
    df_entry = pd.DataFrame([log_entry])
    log_path = SIGNAL_LOG_FILE
    # ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if os.path.exists(log_path):
        df_entry.to_csv(log_path, mode='a', header=False, index=False, encoding="utf-8")
    else:
        df_entry.to_csv(log_path, index=False, encoding="utf-8")

def get_position_size(confidence: float) -> int:
    """Return an integer position size based on the model confidence."""
    if confidence >= 8.5:
        return 100
    elif confidence >= 6.5:
        return 80
    elif confidence >= 5.5:
        return 50
    elif confidence >= 4.5:
        return 20
    else:
        return 0

def simulate_slippage(price: float, direction: str = "long", slippage_pct: float = 0.0005) -> float:
    """Apply a simple slippage adjustment to a price."""
    try:
        slip = float(slippage_pct)
    except Exception:
        slip = 0.0005
    if direction and direction.lower().startswith("s"):
        return price * (1 - slip)
    else:
        return price * (1 + slip)

def estimate_commission(symbol: str, quantity: float = 1.0, maker: bool = False) -> float:
    """Estimate the commission fee rate for a given trade."""
    return 0.0004 if maker else 0.001

def evaluate_signal(price_data: pd.DataFrame, symbol: str = "", sentiment_bias: str = "neutral"):
    """Evaluate a trading signal given a price DataFrame."""
    try:
        if price_data is None or price_data.empty or len(price_data) < 40:
            logger.debug("[DEBUG] Skipping %s: insufficient price data.", symbol)
            return 0, None, 0, None

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(price_data.columns):
            missing = required - set(price_data.columns)
            logger.debug("[DEBUG] Skipping %s: missing columns %s.", symbol, ', '.join(sorted(missing)))
            return 0, None, 0, None

        price_data = price_data.replace([np.inf, -np.inf], np.nan)
        price_data = price_data.dropna(subset=['high', 'low', 'close'])
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
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
        dema_short = DEMAIndicator(close, window=20).dema_indicator()
        dema_long = DEMAIndicator(close, window=50).dema_indicator()
        stoch_obj = StochasticOscillator(high, low, close, window=14, smooth_window=3)
        stoch_k = stoch_obj.stoch()
        stoch_d = stoch_obj.stoch_signal()
        cci = CCIIndicator(high, low, close, window=20).cci()
        atr_p = atr_percentile(high, low, close)
        hurst = hurst_exponent(close)
        def _slope(series: pd.Series) -> float:
            x = np.arange(len(series))
            m, _ = np.polyfit(x, series, 1)
            return float(m)
        confluence = multi_timeframe_confluence(
            price_data[['open', 'high', 'low', 'close', 'volume']],
            ['5T', '15T', '1H'],
            lambda s: _slope(s)
        )
        indicator_alignment = multi_timeframe_indicator_alignment(
            price_data[['open', 'high', 'low', 'close', 'volume']],
            ['1H'],
            {
                'ema_trend': lambda df: EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1]
                - EMAIndicator(df['close'], window=200).ema_indicator().iloc[-1],
                'rsi': lambda df: RSIIndicator(df['close'], window=14).rsi().iloc[-1],
            },
        )
        higher_tf = indicator_alignment.get('1H', {})
        ema_trend_1h = higher_tf.get('ema_trend')
        rsi_1h = higher_tf.get('rsi')
        order_book = get_order_book(symbol)
        if order_book:
            spread = compute_spread(order_book)
            imbalance = compute_order_book_imbalance(order_book)
        else:
            spread = float('nan')
            imbalance = float('nan')
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

        # TRAINING_MODE support: skip volume filter when in training mode
        training_mode = os.getenv("TRAINING_MODE", "false").lower() == "true"
        if not training_mode and latest_quote_vol < vol_threshold:
            logger.info(
                "Skipping due to low volume: %s < %s (%s%% of 20-bar avg)",
                f"{latest_quote_vol:,.0f}",
                f"{vol_threshold:,.0f}",
                f"{vol_factor*100:.0f}",
            )
            return 0, None, 0, None

        base_weights = {
            "ema": 1.4,
            "macd": 1.3,
            "rsi": 1.3,
            "adx": 1.5,
            "vwma": 1.4,
            "bb": 1.3,
            "candle": 1.2,
            "chart": 1.2,
            "flag": 1.0,
            "hs": 1.2,
            "atr": 0.8,
            "hurst": 0.8,
            "confluence": 1.0,
            "flow": 1.5,
            "dema": 1.1,
            "stoch": 1.0,
            "cci": 1.0,
        }
        if session_name == "US":
            base_weights["macd"] += 0.2
            base_weights["stoch"] += 0.2
        w = base_weights
        reinforcement = 1.0
        score = 0.0
        ema_condition = (
            ema_trend_1h is None
            or ema_trend_1h != ema_trend_1h
            or ema_trend_1h > 0
        )
        if ema_short.iloc[-1] > ema_long.iloc[-1] and ema_condition:
            score += w["ema"]
        if macd_line.iloc[-1] > 0:
            score += w["macd"]
        rsi_val = rsi.iloc[-1]
        rsi_condition = (
            rsi_1h is None
            or rsi_1h != rsi_1h
            or rsi_1h > 40
        )
        if rsi_val > 50 and rsi_condition:
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
        if dema_short.iloc[-1] > dema_long.iloc[-1]:
            score += w["dema"]
        if stoch_k.iloc[-1] > stoch_d.iloc[-1] and stoch_k.iloc[-1] < 80:
            score += w["stoch"]
        if cci.iloc[-1] > 0:
            score += w["cci"]
        if atr_p == atr_p:
            if atr_p > 0.75:
                score += w["atr"]
            elif atr_p < 0.25:
                score -= w["atr"]
        if hurst == hurst:
            if hurst > 0.55:
                score += w["hurst"]
            elif hurst < 0.45:
                score -= w["hurst"]
        candle_patterns = detect_candlestick_patterns(price_data)
        # Normalize candlestick pattern output to a list
        if isinstance(candle_patterns, dict):
            triggered_patterns = [p for p, v in candle_patterns.items() if v]
        else:
            triggered_patterns = candle_patterns or []
        chart_pattern = detect_triangle_wedge(price_data)
        flag = detect_flag_pattern(price_data)
        head_shoulders = detect_head_and_shoulders(price_data)
        if triggered_patterns:
            score += w["candle"]
        if chart_pattern:
            score += w["chart"]
        if flag:
            score += w["flag"]
        if head_shoulders:
            score -= w["hs"]
        if all(v > 0 for v in confluence.values() if v == v):
            score += w["confluence"]
        if spread == spread and price_now > 0 and spread / price_now > 0.001:
            logger.warning("Skipping %s: spread %.6f is >0.1%% of price.", symbol, spread)
            return 0, None, 0, None
        if imbalance == imbalance and abs(imbalance) > 0.7:
            logger.warning("Skipping %s: order book imbalance %.2f exceeds threshold.", symbol, imbalance)
            return 0, None, 0, None
        aggression = detect_aggression(price_data)
        if aggression == "buyers in control":
            score += w["flow"]
        elif aggression == "sellers in control":
            score -= w["flow"]
        max_possible = sum(w.values())
        normalized_score = round((score / max_possible) * 10 * reinforcement, 2)
        if sentiment_bias == "bullish" and normalized_score < 5.0:
            normalized_score += 0.8
        elif sentiment_bias == "bearish" and normalized_score > 7.5:
            normalized_score -= 0.8
        normalized_score = round(normalized_score, 2)
        direction = "long" if normalized_score >= 4.5 else None
        position_size = get_position_size(normalized_score)
        zones = detect_support_resistance_zones(price_data)
        zones = zones.to_dict() if isinstance(zones, pd.Series) else (zones or {"support": [], "resistance": []})
        current_price = float(close.iloc[-1])
        if direction == "long":
            near_resistance = is_price_near_zone(current_price, zones, 'resistance', 0.005)
            near_support = is_price_near_zone(current_price, zones, 'support', 0.015 if sentiment_bias == "bullish" else 0.01)
            if near_resistance and normalized_score < 6.5:
                logger.warning("Skipping %s: near resistance zone with score %.2f < 6.5", symbol, normalized_score)
                return 0, None, 0, None
            if not near_support and normalized_score < 5.5:
                logger.warning("Skipping %s: away from support with score %.2f < 5.5", symbol, normalized_score)
                return 0, None, 0, None
        pattern_name = triggered_patterns[0] if triggered_patterns else (chart_pattern if chart_pattern else "None")
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
                "pattern": pattern_name,
            }
            with open(SYMBOL_SCORES_FILE, "w") as f:
                json.dump(scores, f, indent=2)
        except Exception as e:
            logger.warning("[SYMBOL SCORES] Failed to update symbol_scores.json: %s", e, exc_info=True)
        return normalized_score, direction, position_size, pattern_name
    except Exception as e:
        logger.error("Signal evaluation error in %s: %s", symbol, e, exc_info=True)
        traceback.print_exc()
        return 0, None, 0, None
