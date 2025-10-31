import math
import numpy as np
import pandas as pd
import os
import json
import time
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any, Mapping, Callable, Tuple, TypeVar, Dict
import traceback
import asyncio
from log_utils import setup_logger

from weight_optimizer import optimize_indicator_weights

try:  # confidence guard is optional during unit tests
    from confidence_guard import get_adaptive_conf_threshold  # type: ignore
except Exception:
    def get_adaptive_conf_threshold() -> float:  # type: ignore
        """Fallback confidence threshold when adaptive guard is unavailable."""
        return 6.0

from trade_storage import TRADE_HISTORY_FILE, load_trade_history_df  # shared trade log path

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

# Maximum age (seconds) of WebSocket order book data before we consider it stale.
_STREAM_STALENESS_MAX_SECONDS = 5.0
# Cooldown period before re-attempting to use the WebSocket stream after a failure.
_STREAM_BACKOFF_SECONDS = 10.0
# Track symbols that should temporarily bypass the WebSocket stream due to recent issues.
_stream_backoff_until: Dict[str, float] = {}

_T = TypeVar("_T")

_RATE_LIMIT_KEYWORDS = (
    "Too many requests",
    "too many requests",
    "Too Many Requests",
    "-1003",  # Binance rate limit error code
    "IP banned",
    "429",
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if ``exc`` looks like a Binance rate limit error."""

    message = getattr(exc, "message", None)
    if not message:
        message = str(exc)
    if not message:
        return False
    return any(keyword in message for keyword in _RATE_LIMIT_KEYWORDS)


def _call_binance_with_retries(
    action: Callable[[], _T],
    description: str,
    max_attempts: int = 3,
    base_delay: float = 0.5,
) -> Tuple[bool, Optional[_T], Optional[Exception]]:
    """Execute ``action`` with retries and exponential backoff.

    Parameters
    ----------
    action : Callable
        Callable executed with no arguments that performs the Binance API
        request.
    description : str
        Human readable description for logging.
    max_attempts : int, optional
        Number of attempts before giving up, by default 3.
    base_delay : float, optional
        Initial backoff delay in seconds, by default 0.5 seconds.

    Returns
    -------
    Tuple[bool, Optional[_T], Optional[Exception]]
        Tuple of ``(success, result, exception)``.  ``result`` is populated
        only when ``success`` is True.  When ``success`` is False the last
        exception is returned for logging.
    """

    delay = max(base_delay, 0.1)
    last_exception: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = action()
            return True, result, None
        except Exception as exc:  # pragma: no cover - network dependent
            last_exception = exc
            rate_limited = _is_rate_limit_error(exc)
            logger.warning(
                "Attempt %d/%d to %s failed%s: %s",
                attempt,
                max_attempts,
                description,
                " due to rate limit" if rate_limited else "",
                exc,
            )
            if attempt >= max_attempts:
                break
            multiplier = 2.0 if rate_limited else 1.5
            jitter = random.uniform(0.0, base_delay)
            sleep_time = min(delay * multiplier + jitter, 15.0)
            time.sleep(sleep_time)
            delay = min(delay * multiplier, 15.0)
    return False, None, last_exception

# Optional runtime dependency: live market data stream via WebSockets.
try:  # pragma: no cover - import guarded for offline environments
    from market_stream import get_market_stream  # type: ignore
except Exception:  # pragma: no cover - fallback when streaming unavailable
    get_market_stream = None  # type: ignore

# Maximum allowed lag for higher‑timeframe (e.g. 1H) updates.
# The latest completed 1H candle should print shortly after the top of the hour.
# We allow a few extra minutes of slack on top of the expected one hour cadence
# so that slightly delayed exchange updates do not stall the signal pipeline.
HOURLY_BAR_MAX_LAG = timedelta(minutes=5)

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
    from ta.volume import (
        VolumeWeightedAveragePrice as _TA_VolumeWeightedAveragePrice,
        OnBalanceVolumeIndicator as _TA_OnBalanceVolumeIndicator,
    )
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
    OnBalanceVolumeIndicator = _TA_OnBalanceVolumeIndicator
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
        def __init__(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            window: int = 14,
        ):
            self.high = high
            self.low = low
            self.close = close
            self.window = window
            self._plus_di: Optional[pd.Series] = None
            self._minus_di: Optional[pd.Series] = None
            self._adx: Optional[pd.Series] = None

        def _compute(self) -> None:
            if self._adx is not None:
                return
            up_move = self.high.diff()
            down_move = self.low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            tr1 = self.high - self.low
            tr2 = (self.high - self.close.shift()).abs()
            tr3 = (self.low - self.close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(self.window).mean()
            plus_di = 100 * (pd.Series(plus_dm, index=self.high.index).rolling(self.window).mean() / (atr + 1e-9))
            minus_di = 100 * (pd.Series(minus_dm, index=self.high.index).rolling(self.window).mean() / (atr + 1e-9))
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
            self._plus_di = plus_di
            self._minus_di = minus_di
            self._adx = dx.rolling(self.window).mean()

        def adx(self) -> pd.Series:
            self._compute()
            return self._adx if self._adx is not None else pd.Series(dtype=float)

        def adx_pos(self) -> pd.Series:
            self._compute()
            return self._plus_di if self._plus_di is not None else pd.Series(dtype=float)

        def adx_neg(self) -> pd.Series:
            self._compute()
            return self._minus_di if self._minus_di is not None else pd.Series(dtype=float)
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
    class OnBalanceVolumeIndicator:
        def __init__(self, close: pd.Series, volume: pd.Series):
            self.close = close
            self.volume = volume
        def on_balance_volume(self) -> pd.Series:
            obv = [0]
            for i in range(1, len(self.close)):
                if self.close.iloc[i] > self.close.iloc[i - 1]:
                    obv.append(obv[-1] + self.volume.iloc[i])
                elif self.close.iloc[i] < self.close.iloc[i - 1]:
                    obv.append(obv[-1] - self.volume.iloc[i])
                else:
                    obv.append(obv[-1])
            return pd.Series(obv, index=self.close.index)
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

    def is_price_near_zone(
        price: float,
        zones: dict,
        zone_type: str,
        tolerance: float = 0.005,
        atr: Optional[float] = None,
        atr_multiple: Optional[float] = None,
    ) -> bool:
        return False

# Order flow detection fallback
try:
    from orderflow import detect_aggression, OrderFlowAnalysis  # type: ignore
except Exception:
    @dataclass
    class OrderFlowAnalysis:  # type: ignore
        state: str = "neutral"
        features: dict = field(default_factory=dict)

        def __str__(self) -> str:
            return self.state

    def detect_aggression(  # type: ignore
        df,
        order_book=None,
        symbol=None,
        depth: int = 5,
        live_trades=None,
    ):
        return OrderFlowAnalysis()

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
        detect_double_bottom,
        detect_cup_and_handle,
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
        def detect_double_bottom(df):
            return (False, False)
        def detect_cup_and_handle(df):
            return (False, False)
    except Exception:
        def detect_candlestick_patterns(df):
            return []
        def detect_triangle_wedge(df):
            return None
        def detect_flag_pattern(df):
            return False
        def detect_head_and_shoulders(df):
            return False
        def detect_double_bottom(df):
            return (False, False)
        def detect_cup_and_handle(df):
            return (False, False)

# Path to stored symbol scores
SYMBOL_SCORES_FILE = os.path.join(os.path.dirname(__file__), "symbol_scores.json")

_binance_client: Optional[Client] = None
_client_init_error: Optional[str] = None
# Backwards-compatible handle that can be monkeypatched in tests or other
# modules.  When set, ``_get_binance_client`` will return this instance
# without attempting to create a new SDK client.
client: Optional[Client] = None


def _get_binance_client(force_refresh: bool = False) -> Optional[Client]:
    """Return a cached Binance client instance, refreshing on demand."""

    global _binance_client, _client_init_error, client

    if force_refresh:
        # Preserve explicit monkeypatches/overrides.
        if client is not None and client is not _binance_client:
            return client
        _binance_client = None
        client = None

    if client is not None and client is not _binance_client:
        return client

    if _binance_client is not None:
        client = _binance_client
        return _binance_client

    try:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if api_key and api_secret:
            _binance_client = Client(api_key, api_secret)
        else:
            _binance_client = Client()
        _client_init_error = None
        client = _binance_client
        return _binance_client
    except Exception as exc:  # pragma: no cover - depends on environment
        message = str(exc)
        if message != _client_init_error:
            logger.warning(
                "Failed to initialize Binance client: %s. Install the 'python-binance' package and ensure network/credentials are configured.",
                exc,
                exc_info=True,
            )
        _client_init_error = message
        return None

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
    """Fetch recent OHLCV data for a symbol from Binance.

    The bot previously relied on 5‑minute candles which could miss
    intra‑candle spikes.  To improve reactivity this now pulls 1‑minute
    klines so that trade management logic can see the high and low of
    the most recent minute instead of only the 5‑minute close.
    """
    client = _get_binance_client()
    if client is None:
        logger.warning("Binance client unavailable; cannot fetch data for %s.", symbol)
        return None

    mapped_symbol = map_symbol_for_binance(symbol)
    attempts = 2
    last_exception: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            # Use 1‑minute klines for higher‑resolution price checks
            klines = client.get_klines(
                symbol=mapped_symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=500
            )
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "close_time",
                "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore",
            ])
            df[[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_asset_volume",
                "taker_buy_base",
                "taker_buy_quote",
            ]] = df[[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_asset_volume",
                "taker_buy_base",
                "taker_buy_quote",
            ]].astype(float)
            df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce")
            # Convert timestamp to datetime and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
            df = df.sort_values("timestamp").set_index("timestamp")
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            df["quote_volume"] = df["quote_asset_volume"]
            df["taker_sell_base"] = (df["volume"] - df["taker_buy_base"]).clip(lower=0.0)
            df["taker_sell_quote"] = (df["quote_volume"] - df["taker_buy_quote"]).clip(lower=0.0)

            # Fetch the most recent 1‑hour kline and store it for higher timeframe checks
            h_klines = client.get_klines(symbol=mapped_symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=2)
            if h_klines:
                h_df = pd.DataFrame([h_klines[-1]], columns=[
                    "timestamp", "open", "high", "low", "close", "volume", "close_time",
                    "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore",
                ])
                h_df[["open", "high", "low", "close", "volume", "quote_asset_volume"]] = h_df[
                    ["open", "high", "low", "close", "volume", "quote_asset_volume"]
                ].astype(float)
                h_df["timestamp"] = (
                    pd.to_datetime(h_df["timestamp"], unit="ms", errors="coerce", utc=True)
                    .dt.floor("H")
                )
                h_df = h_df.sort_values("timestamp").set_index("timestamp")
                if getattr(h_df.index, "tz", None) is not None:
                    h_df.index = h_df.index.tz_convert("UTC").tz_localize(None)
                h_df["quote_volume"] = h_df["quote_asset_volume"]
                df.attrs["hourly_bar"] = h_df[["open", "high", "low", "close", "volume", "quote_volume"]]

            return df[[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "taker_buy_base",
                "taker_buy_quote",
                "taker_sell_base",
                "taker_sell_quote",
                "number_of_trades",
            ]]
        except Exception as e:
            last_exception = e
            logger.warning(
                "Attempt %d/%d to fetch data for %s failed: %s",
                attempt,
                attempts,
                symbol,
                e,
                exc_info=True,
            )
            # Force a client refresh before retrying in case the connection died.
            client = _get_binance_client(force_refresh=True)
            if client is None:
                break
            # Back off briefly to give the API a chance to recover (e.g. rate limits).
            time.sleep(0.5)

    if last_exception is not None:
        logger.warning("Failed to fetch data for %s after %d attempts: %s", symbol, attempts, last_exception)
    return None


async def get_price_data_async(symbol: str) -> Optional[pd.DataFrame]:
    """Asynchronous wrapper around ``get_price_data`` using ``asyncio``."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_price_data, symbol)

def _get_order_book_rest(symbol: str, limit: int = 50) -> Optional[dict]:
    """Fallback REST request for the order book when streaming is unavailable."""

    client = _get_binance_client()
    if client is None:
        return None
    mapped_symbol = map_symbol_for_binance(symbol)

    def _fetch() -> dict:
        book = client.get_order_book(symbol=mapped_symbol, limit=limit)
        return {
            "bids": [(float(p), float(q)) for p, q in book.get("bids", [])],
            "asks": [(float(p), float(q)) for p, q in book.get("asks", [])],
            "last_update_id": float(book.get("lastUpdateId", 0.0)),
            "last_event_ts": time.time(),
        }

    success, snapshot, error = _call_binance_with_retries(
        _fetch,
        f"fetch order book via REST for {symbol}",
    )
    if success and snapshot is not None:
        return snapshot
    if error is not None:
        logger.debug("REST order book fetch failed for %s: %s", symbol, error)
    return None


def get_order_book(symbol: str, limit: int = 50) -> Optional[dict]:
    """Return a recent order book snapshot, preferring the WebSocket stream."""

    now = time.time()
    backoff_until = _stream_backoff_until.get(symbol, 0.0)
    use_stream = get_market_stream is not None and now >= backoff_until
    if use_stream:
        try:
            stream = get_market_stream()
            snapshot = stream.get_order_book(symbol, depth=limit)
            if snapshot:
                raw_last_update_ts = float(snapshot.get("last_event_ts", 0.0) or 0.0)
                # ``OrderBookState.snapshot`` starts reporting millisecond timestamps once
                # live depth updates flow in. Convert these values back to seconds so the
                # staleness check compares consistent units (``time.time()`` is seconds).
                if raw_last_update_ts > now + 1:
                    last_update_ts = raw_last_update_ts / 1000.0
                elif raw_last_update_ts > 1e11:  # pragma: no cover - defensive cutoff
                    last_update_ts = raw_last_update_ts / 1000.0
                else:
                    last_update_ts = raw_last_update_ts
                is_stale = False
                if last_update_ts:
                    age = now - last_update_ts
                    if age > _STREAM_STALENESS_MAX_SECONDS:
                        logger.debug(
                            "Live order book for %s is stale (%.2fs); falling back to REST.",
                            symbol,
                            age,
                        )
                        is_stale = True
                else:
                    is_stale = True
                if not is_stale:
                    bids = snapshot.get("bids", [])
                    asks = snapshot.get("asks", [])
                    return {
                        "bids": bids[:limit],
                        "asks": asks[:limit],
                        "last_update_id": snapshot.get("last_update_id"),
                        "last_event_ts": snapshot.get("last_event_ts"),
                    }
                _stream_backoff_until[symbol] = now + _STREAM_BACKOFF_SECONDS
        except Exception as exc:
            logger.debug("Live order book unavailable for %s: %s", symbol, exc, exc_info=True)
            _stream_backoff_until[symbol] = now + _STREAM_BACKOFF_SECONDS
    return _get_order_book_rest(symbol, limit=limit)


def update_stop_loss_order(
    symbol: str,
    quantity: float,
    stop_price: float,
    existing_order_id: Optional[str] = None,
    take_profit_price: Optional[float] = None,
) -> Optional[str]:
    """Place or replace a stop-loss or OCO order on Binance.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. ``"BTCUSDT"``.
    quantity : float
        Order size in base units.
    stop_price : float
        Trigger price for the stop-loss order.
    existing_order_id : Optional[str]
        If provided, cancel this order before submitting the new one.
    take_profit_price : Optional[float]
        If provided, submit an OCO order pairing this limit price with the
        stop-loss.  Otherwise a simple stop-loss limit order is used.

    Returns
    -------
    Optional[str]
        The Binance ``orderListId`` (for OCO) or ``orderId`` of the
        newly created order, or ``None`` if the client is unavailable or
        the request fails.
    """
    client = _get_binance_client()
    if client is None:
        logger.warning(
            "Binance client unavailable; cannot update stop-loss for %s.",
            symbol,
        )
        return None
    mapped_symbol = map_symbol_for_binance(symbol)

    if existing_order_id:
        def _cancel_existing() -> bool:
            if take_profit_price is not None and hasattr(client, "cancel_oco_order"):
                client.cancel_oco_order(symbol=mapped_symbol, orderListId=existing_order_id)
            else:
                client.cancel_order(symbol=mapped_symbol, orderId=existing_order_id)
            return True

        success, _, error = _call_binance_with_retries(
            _cancel_existing,
            f"cancel existing stop-loss order for {symbol}",
            max_attempts=2,
        )
        if not success and error is not None:
            logger.warning(
                "Failed to cancel existing stop-loss order for %s: %s",
                symbol,
                error,
            )

    def _place_order() -> dict:
        if take_profit_price is not None and hasattr(client, "create_oco_order"):
            return client.create_oco_order(
                symbol=mapped_symbol,
                side=getattr(Client, "SIDE_SELL", "SELL"),
                quantity=quantity,
                price=float(take_profit_price),
                stopPrice=float(stop_price),
                stopLimitPrice=float(stop_price),
                stopLimitTimeInForce=getattr(Client, "TIME_IN_FORCE_GTC", "GTC"),
            )
        return client.create_order(
            symbol=mapped_symbol,
            side=getattr(Client, "SIDE_SELL", "SELL"),
            type=getattr(Client, "ORDER_TYPE_STOP_LOSS_LIMIT", "STOP_LOSS_LIMIT"),
            quantity=quantity,
            stopPrice=float(stop_price),
            price=float(stop_price),
            timeInForce=getattr(Client, "TIME_IN_FORCE_GTC", "GTC"),
        )

    success, response, error = _call_binance_with_retries(
        _place_order,
        f"place stop-loss order for {symbol}",
    )
    if not success or response is None:
        if error is not None:
            logger.warning("Failed to place stop-loss order for %s: %s", symbol, error)
        return None

    if take_profit_price is not None and hasattr(client, "create_oco_order"):
        order_id = response.get("orderListId")
    else:
        order_id = response.get("orderId")

    status = response.get("status") or response.get("listOrderStatus")
    if status and isinstance(status, str) and status.upper() not in {"NEW", "ACCEPTED", "ACK"}:
        logger.warning(
            "Stop-loss order for %s returned unexpected status %s: %s",
            symbol,
            status,
            response,
        )

    if not order_id:
        logger.warning(
            "Binance response missing order identifier for %s: %s",
            symbol,
            response,
        )
        return None

    return order_id

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
        adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx_indicator.adx().fillna(0)
        try:
            df['di_plus'] = adx_indicator.adx_pos().fillna(0)
            df['di_minus'] = adx_indicator.adx_neg().fillna(0)
        except AttributeError:
            # Older TA versions may not expose DI helpers; fall back to zeros.
            df['di_plus'] = 0.0
            df['di_minus'] = 0.0
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        vwma_calc = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'], window=20)
        df['vwma'] = vwma_calc.volume_weighted_average_price()
        vwap_calc = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'])
        df['vwap'] = vwap_calc.volume_weighted_average_price()
        obv_calc = OnBalanceVolumeIndicator(df['close'], df['volume'])
        df['obv'] = obv_calc.on_balance_volume()
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


def summarise_technical_score(
    indicators: Mapping[str, Any],
    direction: str = "long",
) -> float:
    """Return a bounded 0–10 technical score summarising core indicators.

    The revised version favours smooth, continuous adjustments so that very small
    changes in high-frequency data do not create discontinuous confidence jumps.
    RSI and MACD inputs are normalised via ``tanh`` and ADX is weighted by the
    trend direction inferred from the directional movement index (DI+/DI-).
    """

    def _extract(name: str, default: float) -> float:
        raw = indicators.get(name)
        if raw is None:
            return default
        try:
            if hasattr(raw, "iloc"):
                raw = raw.iloc[-1]
        except Exception:
            pass
        try:
            return float(raw)
        except Exception:
            return default

    def _finite(value: float, fallback: float) -> float:
        return value if math.isfinite(value) else fallback

    rsi = _finite(_extract("rsi", 50.0), 50.0)
    macd = _finite(_extract("macd", 0.0), 0.0)
    macd_signal = _extract("macd_signal", math.nan)
    adx = _finite(_extract("adx", 20.0), 20.0)
    di_plus = _extract("di_plus", math.nan)
    di_minus = _extract("di_minus", math.nan)

    score = 5.0
    bias = (direction or "long").strip().lower()
    if bias not in {"long", "short"}:
        bias = "long"

    # --- RSI: favour oversold readings for longs and overbought for shorts.
    rsi_delta = (50.0 - rsi) / 10.0
    if bias == "short":
        rsi_delta = -rsi_delta
    rsi_component = 2.5 * math.tanh(rsi_delta / 2.0)
    score += rsi_component

    # --- MACD: emphasise histogram slope which responds faster on lower TFs.
    if math.isfinite(macd_signal):
        macd_hist = macd - macd_signal
    else:
        macd_hist = macd
    macd_hist = _finite(macd_hist, 0.0)
    macd_component = 2.0 * math.tanh(macd_hist * 8.0)
    if bias == "short":
        macd_component *= -1.0
    score += macd_component

    # --- ADX + Directional Index: reward strength only when aligned with bias.
    adx_strength = math.tanh((adx - 20.0) / 15.0)
    alignment = 0.0
    if math.isfinite(di_plus) and math.isfinite(di_minus):
        di_diff = di_plus - di_minus
        alignment = math.tanh(di_diff / 20.0)
        if bias == "short":
            alignment = -alignment
    else:
        # Without DI data treat strong trends as a headwind for counter trades.
        alignment = 1.0 if bias == "long" else -0.6

    adx_component = 1.1 * adx_strength * alignment
    if adx_strength > 0 and alignment < 0:
        adx_component -= 0.5 * adx_strength * abs(alignment)
    score += adx_component

    return round(max(0.0, min(score, 10.0)), 2)


def get_top_symbols(limit: int = 30) -> list:
    """Return the top quote-volume symbols trading against USDT."""
    client = _get_binance_client()
    if client is None:
        logger.warning("Binance client unavailable; get_top_symbols returning empty list.")
        return []
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
    symbols = [x['symbol'] for x in sorted_tickers if x['symbol'].endswith("USDT") and not x['symbol'].endswith("BUSD")]
    exclude = {
        token.strip().upper()
        for token in os.getenv("SYMBOL_EXCLUDE", "").split(",")
        if token.strip()
    }
    if exclude:
        symbols = [sym for sym in symbols if sym.upper() not in exclude]
    return symbols[:limit]

def _extract_trade_returns(df: pd.DataFrame) -> pd.Series:
    """Return per-trade returns as decimal values."""
    if df.empty:
        return pd.Series(dtype=float)

    if "pnl_pct" in df.columns:
        returns = pd.to_numeric(df["pnl_pct"], errors="coerce") / 100.0
    elif {"pnl", "notional"}.issubset(df.columns):
        pnl = pd.to_numeric(df["pnl"], errors="coerce")
        notional = pd.to_numeric(df["notional"], errors="coerce").replace(0, np.nan)
        returns = pnl / notional
    else:
        entry = df.get("entry")
        exit_price = df.get("exit")
        if entry is None or exit_price is None:
            return pd.Series(dtype=float)
        entry = pd.to_numeric(entry, errors="coerce")
        exit_price = pd.to_numeric(exit_price, errors="coerce")
        direction = df.get("direction")
        if direction is None:
            direction = pd.Series("long", index=df.index)
        else:
            direction = direction.astype(str).str.lower()
        returns = pd.Series(np.nan, index=df.index, dtype=float)
        mask = entry.notna() & exit_price.notna() & (entry != 0)
        returns.loc[mask] = (exit_price.loc[mask] - entry.loc[mask]) / entry.loc[mask]
        short_mask = mask & direction.eq("short")
        returns.loc[short_mask] = (entry.loc[short_mask] - exit_price.loc[short_mask]) / entry.loc[short_mask]

    returns = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return returns


_DEFAULT_PERFORMANCE_METRICS = {
    "sharpe": float("nan"),
    "calmar": float("nan"),
    "max_drawdown": float("nan"),
    "var": float("nan"),
    "es": float("nan"),
}


def compute_performance_metrics(log_file: str = TRADE_HISTORY_FILE, lookback: int = 100) -> dict:
    """Return risk-adjusted performance metrics from the trade log."""

    use_default_source = log_file == TRADE_HISTORY_FILE
    if not use_default_source and (not log_file or not os.path.exists(log_file)):
        return dict(_DEFAULT_PERFORMANCE_METRICS)

    try:
        if use_default_source:
            df = load_trade_history_df()
        else:
            df = load_trade_history_df(log_file)

        if df.empty:
            return dict(_DEFAULT_PERFORMANCE_METRICS)

        sort_cols = [col for col in ("exit_time", "timestamp") if col in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        if lookback and lookback > 0:
            df = df.tail(lookback)

        returns = _extract_trade_returns(df)
        if returns.empty:
            return dict(_DEFAULT_PERFORMANCE_METRICS)

        equity_curve = (1 + returns).cumprod()
        metrics = {
            "sharpe": sharpe_ratio(returns),
            "calmar": calmar_ratio(returns),
            "max_drawdown": max_drawdown(equity_curve),
            "var": value_at_risk(returns),
            "es": expected_shortfall(returns),
        }
        return {key: float(value) for key, value in metrics.items()}
    except Exception as exc:
        logger.exception(
            "Failed to compute performance metrics from %s: %s", log_file, exc
        )
        return dict(_DEFAULT_PERFORMANCE_METRICS)


def get_last_trade_outcome(log_file: str = TRADE_HISTORY_FILE) -> str | None:
    """Return ``'win'`` or ``'loss'`` based on the most recent closed trade.

    The helper is used by the RL position‑sizer to condition its action
    selection on the outcome of the previous trade.  If the trade log is
    missing or empty, ``None`` is returned so that callers can fall back to a
    neutral state.
    """
    if not os.path.exists(log_file):
        return None
    try:
        df = load_trade_history_df(log_file)
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        try:
            fallback = pd.read_csv(
                log_file,
                names=[
                    "timestamp",
                    "symbol",
                    "direction",
                    "entry",
                    "exit",
                    "outcome",
                    "btc_d",
                    "fg",
                    "sent_conf",
                    "sent_bias",
                    "score",
                ],
                encoding="utf-8",
            )
        except Exception:
            return None
        if fallback.empty:
            return None
        last = fallback.tail(1)
        entry = pd.to_numeric(last.get("entry"), errors="coerce").iloc[0]
        exit_price = pd.to_numeric(last.get("exit"), errors="coerce").iloc[0]
        if pd.isna(entry) or pd.isna(exit_price):
            return None
        direction_series = last.get("direction")
        direction = (
            str(direction_series.iloc[0]).lower()
            if direction_series is not None and not direction_series.empty
            else "long"
        )
        if direction == "short":
            return "win" if exit_price < entry else "loss"
        return "win" if exit_price > entry else "loss"
    try:
        last = df.tail(1)
        entry_series = last.get("entry", last.get("entry_price"))
        exit_series = last.get("exit", last.get("exit_price"))
        if entry_series is None or exit_series is None:
            return None
        entry = pd.to_numeric(entry_series, errors="coerce").iloc[0]
        exit_price = pd.to_numeric(exit_series, errors="coerce").iloc[0]
        if pd.isna(entry) or pd.isna(exit_price):
            return None
        direction_series = last.get("direction")
        direction = (
            str(direction_series.iloc[0]).lower()
            if direction_series is not None and not direction_series.empty
            else "long"
        )
        if direction == "short":
            return "win" if exit_price < entry else "loss"
        return "win" if exit_price > entry else "loss"
    except Exception:
        return None

def get_rl_state(vol_percentile: float | None, log_file: str = TRADE_HISTORY_FILE) -> str:
    """Construct a compound RL state from last outcome and volatility.

    The state combines the result of the most recently closed trade with a
    simple volatility regime derived from the current ATR percentile.  This
    provides additional context for the reinforcement-learning position sizer
    beyond merely "win" or "loss".

    Parameters
    ----------
    vol_percentile : float | None
        Current ATR percentile expressed as a value between 0 and 1.  ``None`` or
        ``NaN`` results are treated as ``unknown``.
    log_file : str, optional
        Path to the trade log inspected for the last trade outcome.

    Returns
    -------
    str
        State label in the form ``"win_high_vol"``, ``"loss_low_vol"`` or
        ``"neutral_mid_vol"`` when no prior trade information exists.
    """
    outcome = get_last_trade_outcome(log_file) or "neutral"
    if vol_percentile is None or np.isnan(vol_percentile):
        vol_bucket = "unknown_vol"
    elif vol_percentile > 0.75:
        vol_bucket = "high_vol"
    elif vol_percentile < 0.25:
        vol_bucket = "low_vol"
    else:
        vol_bucket = "mid_vol"
    return f"{outcome}_{vol_bucket}"

def log_signal(
    symbol: str,
    session: str,
    score: float,
    direction: Optional[str],
    weights: dict,
    candle_patterns: list,
    chart_pattern: Optional[str],
    indicators: Optional[dict] = None,
    feature_values: Optional[dict] = None,
) -> None:
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
    if indicators:
        for name, val in indicators.items():
            log_entry[f"{name}_trigger"] = val
    if feature_values:
        for name, val in feature_values.items():
            log_entry[name] = val
    df_entry = pd.DataFrame([log_entry])
    log_path = SIGNAL_LOG_FILE
    # ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if os.path.exists(log_path):
        df_entry.to_csv(log_path, mode='a', header=False, index=False, encoding="utf-8")
    else:
        df_entry.to_csv(log_path, index=False, encoding="utf-8")

def get_position_size(confidence: float) -> int:
    """Return the target notional size (in USDT) based on confidence."""

    if confidence >= 8.5:
        return 500
    elif confidence >= 6.5:
        return 450
    elif confidence >= 4.5:
        return 400
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

def _select_indicator_params(vol_percentile: float | None) -> tuple[str, dict[str, int | float]]:
    """Return tuned indicator parameters for the current volatility regime."""

    base_params: dict[str, int | float] = {
        "ema_short": 20,
        "ema_long": 50,
        "dema_short": 20,
        "dema_long": 50,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "stoch_window": 14,
        "stoch_smooth": 3,
        "cci_window": 20,
        "bb_window": 20,
        "bb_dev": 2,
        "vwma_window": 20,
    }

    if vol_percentile is None or np.isnan(vol_percentile):
        return "mid", base_params

    low_params = {
        "ema_short": 26,
        "ema_long": 65,
        "dema_short": 26,
        "dema_long": 65,
        "macd_fast": 10,
        "macd_slow": 30,
        "macd_signal": 9,
        "stoch_window": 21,
        "stoch_smooth": 3,
        "cci_window": 30,
        "bb_window": 24,
        "vwma_window": 30,
    }

    high_params = {
        "ema_short": 8,
        "ema_long": 21,
        "dema_short": 8,
        "dema_long": 21,
        "macd_fast": 5,
        "macd_slow": 13,
        "macd_signal": 8,
        "stoch_window": 5,
        "stoch_smooth": 3,
        "cci_window": 14,
        "bb_window": 14,
        "vwma_window": 14,
    }

    if vol_percentile > 0.75:
        params = base_params.copy()
        params.update(high_params)
        return "high", params
    if vol_percentile < 0.25:
        params = base_params.copy()
        params.update(low_params)
        return "low", params

    return "mid", base_params


def evaluate_signal(
    price_data: pd.DataFrame,
    symbol: str = "",
    sentiment_bias: str = "neutral",
    auction_state: Optional[str] = None,
    volume_profile: Optional[Any] = None,
    lvn_entry_level: Optional[float] = None,
    poc_target: Optional[float] = None,
):
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

        def _maybe_float(value: Any) -> Optional[float]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(number):
                return None
            return number

        if auction_state is not None:
            price_data.attrs["auction_state"] = str(auction_state)

        poc_candidate = _maybe_float(poc_target)
        volume_profile_summary: Optional[Dict[str, Any]] = None
        if volume_profile is not None:
            if hasattr(volume_profile, "to_dict"):
                try:
                    volume_profile_summary = volume_profile.to_dict()
                except Exception:
                    volume_profile_summary = None
            elif isinstance(volume_profile, Mapping):
                volume_profile_summary = dict(volume_profile)
            if volume_profile_summary is not None:
                price_data.attrs["volume_profile"] = volume_profile_summary
            poc_from_profile = getattr(volume_profile, "poc", None)
            if poc_candidate is None:
                poc_candidate = _maybe_float(poc_from_profile)
            lvns = getattr(volume_profile, "lvns", None)
            if isinstance(lvns, (list, tuple)) and lvns:
                cleaned_lvns = [
                    float(level)
                    for level in lvns
                    if _maybe_float(level) is not None
                ]
                if cleaned_lvns:
                    price_data.attrs["volume_profile_lvns"] = cleaned_lvns

        lvn_candidate = _maybe_float(lvn_entry_level)
        if lvn_candidate is not None:
            price_data.attrs["lvn_entry_level"] = lvn_candidate
        if poc_candidate is not None:
            price_data.attrs["poc_target"] = poc_candidate

        hourly_bar = price_data.attrs.get("hourly_bar")
        if isinstance(hourly_bar, pd.DataFrame) and not hourly_bar.empty:
            now = datetime.utcnow()
            last_hour = hourly_bar.index[-1]
            # Ensure the last completed 1H candle is not too old. Binance timestamps
            # are aligned to the candle *open* time, so the bar is considered fresh
            # for up to one hour plus a small grace window.
            max_age = timedelta(hours=1) + HOURLY_BAR_MAX_LAG
            if now - last_hour > max_age:
                logger.warning(
                    "Skipping %s: latest 1H bar (%s) is stale (age=%s).",
                    symbol,
                    last_hour,
                    now - last_hour,
                )
                return 0, None, 0, None

        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        volume = price_data['volume']
        atr_p = atr_percentile(high, low, close)
        high_vol_countertrend = bool(np.isfinite(atr_p) and atr_p >= 0.9)
        volatility_regime, indicator_params = _select_indicator_params(atr_p)

        ema_short = EMAIndicator(close, window=indicator_params["ema_short"]).ema_indicator()
        ema_long = EMAIndicator(close, window=indicator_params["ema_long"]).ema_indicator()
        macd_line = MACD(
            close,
            window_slow=indicator_params["macd_slow"],
            window_fast=indicator_params["macd_fast"],
            window_sign=indicator_params["macd_signal"],
        ).macd_diff()
        rsi = RSIIndicator(close, window=14).rsi()
        with np.errstate(invalid='ignore', divide='ignore'):
            adx_indicator = ADXIndicator(high=high, low=low, close=close, window=14)
            adx_series = adx_indicator.adx()
            try:
                di_plus = adx_indicator.adx_pos()
                di_minus = adx_indicator.adx_neg()
            except AttributeError:
                di_plus = pd.Series(0.0, index=adx_series.index)
                di_minus = pd.Series(0.0, index=adx_series.index)
        adx = adx_series.fillna(0)
        di_plus = di_plus.fillna(0)
        di_minus = di_minus.fillna(0)
        bb = BollingerBands(
            close,
            window=int(indicator_params["bb_window"]),
            window_dev=float(indicator_params["bb_dev"]),
        )
        vwma_calc = VolumeWeightedAveragePrice(
            high=high,
            low=low,
            close=close,
            volume=volume,
            window=int(indicator_params["vwma_window"]),
        )
        vwma = vwma_calc.volume_weighted_average_price()
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        dema_short = DEMAIndicator(close, window=indicator_params["dema_short"]).dema_indicator()
        dema_long = DEMAIndicator(close, window=indicator_params["dema_long"]).dema_indicator()
        stoch_obj = StochasticOscillator(
            high,
            low,
            close,
            window=int(indicator_params["stoch_window"]),
            smooth_window=int(indicator_params["stoch_smooth"]),
        )
        stoch_k = stoch_obj.stoch()
        stoch_d = stoch_obj.stoch_signal()
        cci = CCIIndicator(high, low, close, window=int(indicator_params["cci_window"])).cci()
        hurst = hurst_exponent(close)
        def _slope(series: pd.Series) -> float:
            x = np.arange(len(series))
            m, _ = np.polyfit(x, series, 1)
            return float(m)
        hourly_override = None
        if isinstance(hourly_bar, pd.DataFrame) and not hourly_bar.empty:
            hourly_override = hourly_bar[['open', 'high', 'low', 'close', 'volume']]
        confluence = multi_timeframe_confluence(
            price_data[['open', 'high', 'low', 'close', 'volume']],
            ['5T', '15T', '1H', '4H', '1D'],
            lambda s: _slope(s),
            hourly_override=hourly_override,
        )
        # Align key indicators across intraday and higher timeframes so we don't
        # trade against a larger downtrend.  Daily/4H EMA slope acts as a simple
        # higher‑timeframe trend filter.
        indicator_alignment = multi_timeframe_indicator_alignment(
            price_data[['open', 'high', 'low', 'close', 'volume']],
            ['1H', '4H', '1D'],
            {
                'ema_trend': lambda df: EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1]
                - EMAIndicator(df['close'], window=200).ema_indicator().iloc[-1],
                'rsi': lambda df: RSIIndicator(df['close'], window=14).rsi().iloc[-1],
            },
            hourly_override=hourly_override,
        )
        higher_tf_1h = indicator_alignment.get('1H', {})
        ema_trend_1h = higher_tf_1h.get('ema_trend')
        rsi_1h = higher_tf_1h.get('rsi')
        ema_trend_4h = indicator_alignment.get('4H', {}).get('ema_trend')
        ema_trend_1d = indicator_alignment.get('1D', {}).get('ema_trend')
        live_trades = None
        if get_market_stream is not None:
            try:
                live_trades = get_market_stream().get_trade_snapshot(symbol, window_seconds=90)
                if live_trades:
                    price_data.attrs["live_trades"] = live_trades
            except Exception as exc:
                logger.debug("Live trade stream unavailable for %s: %s", symbol, exc, exc_info=True)

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
            "double_bottom": 0.6,
            "cup_handle": 0.6,
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
        w = optimize_indicator_weights(base_weights)
        reinforcement = 1.0
        trend_score = 0.0
        mean_rev_score = 0.0
        structure_score = 0.0
        penalty_score = 0.0
        ema_condition = (
            (ema_trend_1h is None or ema_trend_1h != ema_trend_1h or ema_trend_1h > 0)
            and (ema_trend_4h is None or ema_trend_4h != ema_trend_4h or ema_trend_4h > 0)
            and (ema_trend_1d is None or ema_trend_1d != ema_trend_1d or ema_trend_1d > 0)
        )
        def _linear_score(value: float, lower: float, upper: float) -> float:
            """Return a linear 0-1 score for ``value`` within ``[lower, upper]``."""
            try:
                value = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(value):
                return 0.0
            if upper <= lower:
                return 0.0
            if value <= lower:
                return 0.0
            if value >= upper:
                return 1.0
            return (value - lower) / (upper - lower)

        def _tanh_score(value: float, scale: float) -> float:
            """Smoothly squash ``value``/``scale`` into ``[0, 1]`` using ``tanh``."""
            try:
                value = float(value)
                scale = float(scale)
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(value) or not np.isfinite(scale) or scale <= 0:
                return 0.0
            return float(np.clip(np.tanh(value / scale), 0.0, 1.0))

        ema_latest = float(ema_short.iloc[-1]) if not ema_short.empty else float("nan")
        ema_long_latest = float(ema_long.iloc[-1]) if not ema_long.empty else float("nan")
        ema_diff = ema_latest - ema_long_latest
        rsi_val = float(rsi.iloc[-1]) if not rsi.empty else float("nan")
        bb_lower_val = float(bb_lower.iloc[-1]) if not bb_lower.empty else float("nan")
        counter_trend_mode = False
        ema_flag = 0.0
        if ema_condition and np.isfinite(ema_diff) and ema_diff > 0 and price_now > 0:
            ema_flag = _linear_score(ema_diff / price_now, 0.0, 0.02)
            if ema_flag > 0:
                trend_score += w["ema"] * ema_flag
        else:
            oversold_context = (
                (np.isfinite(rsi_val) and rsi_val < 35.0)
                or (np.isfinite(bb_lower_val) and price_now <= bb_lower_val)
            )
            if (
                high_vol_countertrend
                and oversold_context
                and np.isfinite(ema_diff)
                and ema_diff < 0
                and price_now > 0
            ):
                counter_trend_mode = True
                ema_flag = _linear_score(abs(ema_diff) / price_now, 0.0, 0.02)
                if ema_flag > 0:
                    mean_rev_score += w["ema"] * 0.6 * ema_flag

        macd_latest = float(macd_line.iloc[-1]) if not macd_line.empty else float("nan")
        macd_flag = 0.0
        if np.isfinite(macd_latest) and macd_latest > 0:
            macd_window = macd_line.iloc[-20:].dropna()
            macd_scale = float(macd_window.abs().mean()) if not macd_window.empty else abs(macd_latest)
            if not np.isfinite(macd_scale) or macd_scale <= 1e-8:
                macd_scale = max(abs(macd_latest), 1e-6)
            macd_flag = _tanh_score(macd_latest, macd_scale * 1.5)
            if macd_flag > 0:
                trend_score += w["macd"] * macd_flag

        rsi_condition = (
            rsi_1h is None
            or rsi_1h != rsi_1h
            or rsi_1h > 40
        )
        rsi_flag = 0.0
        if rsi_condition:
            rsi_flag = _linear_score(rsi_val, 50.0, 80.0)
            if rsi_flag > 0:
                trend_score += w["rsi"] * rsi_flag

        adx_val = float(adx.iloc[-1]) if not adx.empty else float("nan")
        adx_flag = _linear_score(adx_val, 20.0, 40.0)
        if adx_flag > 0:
            trend_score += w["adx"] * adx_flag
        vwma_value = vwma.iloc[-1]
        vwma_dev = abs(price_now - vwma_value) / price_now if price_now != 0 else 0.0
        vwma_flag = 0
        if price_now > vwma_value:
            if vwma_dev <= 0.05:
                vwma_flag = 1
                trend_score += w["vwma"]
            elif vwma_dev < 0.10:
                vwma_flag = (0.10 - vwma_dev) / 0.05
                trend_score += w["vwma"] * vwma_flag
        bb_flag = int(bb_lower.iloc[-1] and bb_lower.iloc[-1] < price_now < bb_upper.iloc[-1])
        if bb_flag:
            mean_rev_score += w["bb"]
        dema_flag = int(dema_short.iloc[-1] > dema_long.iloc[-1])
        if dema_flag:
            trend_score += w["dema"]
        stoch_k_val = float(stoch_k.iloc[-1]) if not stoch_k.empty else float("nan")
        stoch_d_val = float(stoch_d.iloc[-1]) if not stoch_d.empty else float("nan")
        stoch_flag = 0.0
        if stoch_k_val == stoch_k_val and stoch_d_val == stoch_d_val and stoch_k_val < 80:
            stoch_flag = _linear_score(stoch_k_val - stoch_d_val, 0.0, 15.0)
            if stoch_flag > 0:
                mean_rev_score += w["stoch"] * stoch_flag
        cci_val = float(cci.iloc[-1]) if not cci.empty else float("nan")
        cci_flag = _linear_score(cci_val, 0.0, 100.0)
        if cci_flag > 0:
            mean_rev_score += w["cci"] * cci_flag
        atr_flag = 0
        if atr_p == atr_p:
            if atr_p > 0.75:
                atr_flag = 1
                trend_score += w["atr"]
            elif atr_p < 0.25:
                atr_flag = -1
                penalty_score += w["atr"]
        hurst_flag = 0
        if hurst == hurst:
            if hurst > 0.55:
                hurst_flag = 1
                trend_score += w["hurst"]
            elif hurst < 0.45:
                hurst_flag = -1
                penalty_score += w["hurst"]
        candle_patterns = detect_candlestick_patterns(price_data)
        # Normalize candlestick pattern output to a list
        if isinstance(candle_patterns, dict):
            triggered_patterns = [p for p, v in candle_patterns.items() if v]
        else:
            triggered_patterns = candle_patterns or []
        candle_flag = int(bool(triggered_patterns))
        chart_pattern = detect_triangle_wedge(price_data)
        chart_flag = int(bool(chart_pattern))
        flag_pattern = detect_flag_pattern(price_data)
        flag_flag = int(bool(flag_pattern))
        head_shoulders = detect_head_and_shoulders(price_data)
        hs_flag = -1 if head_shoulders else 0
        double_bottom, db_vol = detect_double_bottom(price_data)
        double_flag = int(bool(double_bottom))
        cup_handle, cup_vol = detect_cup_and_handle(price_data)
        cup_flag = int(bool(cup_handle))
        if candle_flag:
            structure_score += w["candle"]
        if chart_flag:
            structure_score += w["chart"]
        if flag_flag:
            structure_score += w["flag"]
        if hs_flag:
            penalty_score += w["hs"]
        if double_flag:
            structure_score += w["double_bottom"] * (1.2 if db_vol else 1.0)
        if cup_flag:
            structure_score += w["cup_handle"] * (1.2 if cup_vol else 1.0)
        confluence_flag = int(all(v > 0 for v in confluence.values() if v == v))
        if confluence_flag:
            trend_score += w["confluence"]
        if spread == spread and price_now > 0 and spread / price_now > 0.001:
            logger.warning("Skipping %s: spread %.6f is >0.1%% of price.", symbol, spread)
            return 0, None, 0, None
        imbalance_to_check = imbalance
        aggression = detect_aggression(
            price_data,
            order_book=order_book,
            symbol=symbol,
            live_trades=live_trades,
        )
        flow_features = getattr(aggression, "features", {}) or {}
        feature_imbalance = flow_features.get("order_book_imbalance")
        if feature_imbalance == feature_imbalance:
            imbalance_to_check = feature_imbalance
        if imbalance_to_check == imbalance_to_check and abs(imbalance_to_check) > 0.7:
            logger.warning(
                "Skipping %s: order book imbalance %.2f exceeds threshold.",
                symbol,
                imbalance_to_check,
            )
            return 0, None, 0, None
        flow_score = 0.0
        volume_ratio = None
        price_change_pct = None
        for key, weight in (
            ("trade_imbalance", 0.45),
            ("order_book_imbalance", 0.35),
            ("cvd_change", 0.2),
            ("cvd_divergence", 0.3),
            ("taker_buy_ratio", 0.15),
            ("aggressive_trade_rate", 0.1),
            ("spoofing_intensity", 0.1),
        ):
            value = flow_features.get(key)
            if value == value:
                flow_score += weight * value
        if "volume" in price_data:
            recent = price_data["volume"].tail(5)
            avg_vol = float(recent.mean()) if not recent.empty else float("nan")
            last_vol = float(recent.iloc[-1]) if not recent.empty else float("nan")
            if avg_vol == avg_vol and last_vol == last_vol and avg_vol > 0:
                vol_ratio = (last_vol / avg_vol) - 1.0
                volume_ratio = max(-1.0, min(1.0, vol_ratio))
                flow_score += 0.2 * volume_ratio
        price_change = 0.0
        if len(price_data) >= 5:
            open_ref = float(price_data["open"].iloc[-5])
            close_latest = float(price_data["close"].iloc[-1])
            price_change = close_latest - open_ref
            if open_ref:
                price_change_pct = price_change / open_ref
        if price_change > 0:
            flow_score += 0.1
        elif price_change < 0:
            flow_score -= 0.1
        flow_score = max(-1.0, min(1.0, flow_score))
        flow_flag = 0
        if aggression.state == "buyers in control" or flow_score > 0.2:
            flow_flag = 1
            trend_score += w["flow"] * max(flow_score, 0.0)
        elif aggression.state == "sellers in control" or flow_score < -0.2:
            flow_flag = -1
            penalty_score += w["flow"] * max(-flow_score, 0.0)
        trend_weight_keys = ["ema", "macd", "rsi", "adx", "vwma", "dema", "flow", "confluence", "atr", "hurst"]
        mean_rev_weight_keys = ["bb", "stoch", "cci"]
        structure_weight_keys = ["candle", "chart", "flag", "double_bottom", "cup_handle"]
        trend_max = sum(w[k] for k in trend_weight_keys if k in w)
        mean_rev_max = sum(w[k] for k in mean_rev_weight_keys if k in w)
        structure_max = sum(w[k] for k in structure_weight_keys if k in w)
        trend_total = max(trend_score + structure_score - penalty_score, 0.0)
        mean_rev_total = max(mean_rev_score + structure_score - penalty_score, 0.0)
        trend_norm = round(((trend_total / max(trend_max + structure_max, 1e-6)) * 10 * reinforcement), 2)
        mean_rev_norm = round(((mean_rev_total / max(mean_rev_max + structure_max, 1e-6)) * 10 * reinforcement), 2)

        # Require a higher fraction of the aggregate weight budget before activating.
        base_activation_ratio = 0.6
        base_threshold = round(base_activation_ratio * 10, 2)
        try:
            adaptive_threshold = float(get_adaptive_conf_threshold())
        except Exception:
            adaptive_threshold = base_threshold
        dynamic_threshold = max(base_threshold, adaptive_threshold)
        if atr_p == atr_p:
            if atr_p >= 0.85:
                dynamic_threshold -= 0.4
            elif atr_p <= 0.35:
                dynamic_threshold += 0.4
        dynamic_threshold = float(max(5.5, min(dynamic_threshold, 7.5)))

        setup_type = None
        normalized_score = max(trend_norm, mean_rev_norm)
        TREND_THRESHOLD = max(5.0, dynamic_threshold - 0.5)
        MEAN_REV_THRESHOLD = max(4.5, dynamic_threshold - 1.0)
        COUNTER_TREND_THRESHOLD = max(4.0, dynamic_threshold - 1.5)
        if trend_norm >= TREND_THRESHOLD and trend_norm >= mean_rev_norm:
            normalized_score = trend_norm
            setup_type = "trend"
        elif counter_trend_mode and mean_rev_norm >= COUNTER_TREND_THRESHOLD:
            normalized_score = mean_rev_norm
            setup_type = "counter_trend"
        elif mean_rev_norm >= MEAN_REV_THRESHOLD:
            normalized_score = mean_rev_norm
            setup_type = "mean_reversion"
        if sentiment_bias == "bullish" and normalized_score < 5.0 and setup_type:
            normalized_score += 0.8
        elif sentiment_bias == "bearish" and normalized_score > 7.5 and setup_type:
            normalized_score -= 0.8
        normalized_score = round(normalized_score, 2)
        direction = "long" if setup_type and normalized_score >= dynamic_threshold else None
        position_size = get_position_size(normalized_score)
        zones = detect_support_resistance_zones(price_data)
        zones = zones.to_dict() if isinstance(zones, pd.Series) else (zones or {"support": [], "resistance": []})
        current_price = float(close.iloc[-1])
        atr_value: Optional[float] = None
        if "atr" in price_data:
            try:
                latest_atr = float(price_data["atr"].iloc[-1])
                if np.isfinite(latest_atr) and latest_atr > 0:
                    atr_value = latest_atr
            except Exception:
                atr_value = None
        if direction == "long":
            if atr_value is not None:
                resistance_multiple = 1.0
                support_multiple = 1.5 if sentiment_bias == "bullish" else 1.0
                near_resistance = is_price_near_zone(
                    current_price,
                    zones,
                    'resistance',
                    atr=atr_value,
                    atr_multiple=resistance_multiple,
                )
                near_support = is_price_near_zone(
                    current_price,
                    zones,
                    'support',
                    atr=atr_value,
                    atr_multiple=support_multiple,
                )
            else:
                near_resistance = is_price_near_zone(current_price, zones, 'resistance', proximity=0.005)
                near_support = is_price_near_zone(
                    current_price,
                    zones,
                    'support',
                    proximity=0.015 if sentiment_bias == "bullish" else 0.01,
                )
            min_resistance_score = max(dynamic_threshold + 0.5, 6.5)
            min_support_score = max(dynamic_threshold - 0.5, 5.5)
            if near_resistance and normalized_score < min_resistance_score:
                logger.warning(
                    "Skipping %s: near resistance zone with score %.2f < %.2f",
                    symbol,
                    normalized_score,
                    min_resistance_score,
                )
                return 0, None, 0, None
            if not near_support and normalized_score < min_support_score:
                logger.warning(
                    "Skipping %s: away from support with score %.2f < %.2f",
                    symbol,
                    normalized_score,
                    min_support_score,
                )
                return 0, None, 0, None
        if triggered_patterns:
            pattern_name = triggered_patterns[0]
        elif double_flag:
            pattern_name = "double_bottom"
        elif cup_flag:
            pattern_name = "cup_handle"
        elif chart_flag:
            pattern_name = "triangle_wedge"
        elif flag_flag:
            pattern_name = "flag"
        elif hs_flag:
            pattern_name = "head_and_shoulders"
        else:
            pattern_name = "None"
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
                "setup": setup_type or "none",
                "trend_score": trend_norm,
                "mean_reversion_score": mean_rev_norm,
            }
            with open(SYMBOL_SCORES_FILE, "w") as f:
                json.dump(scores, f, indent=2)
        except Exception as e:
            logger.warning("[SYMBOL SCORES] Failed to update symbol_scores.json: %s", e, exc_info=True)
        spread_bps = None
        if spread == spread and price_now > 0:
            spread_bps = (spread / price_now) * 10_000

        def _safe_float(value: Any) -> Optional[float]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(number):
                return None
            return number

        spoof_val = flow_features.get("spoofing_intensity")
        spoof_alert = 0
        if spoof_val == spoof_val and spoof_val is not None:
            spoof_alert = int(abs(float(spoof_val)) >= 0.5)
        flow_snapshot: dict[str, Any] = {
            "order_flow_score": _safe_float(flow_score),
            "order_flow_flag": float(flow_flag),
            "order_flow_state": getattr(aggression, "state", "neutral") or "neutral",
            "order_book_imbalance": _safe_float(imbalance_to_check),
            "trade_imbalance": _safe_float(flow_features.get("trade_imbalance")),
            "cvd": _safe_float(flow_features.get("cvd")),
            "cvd_change": _safe_float(flow_features.get("cvd_change")),
            "cvd_divergence": _safe_float(flow_features.get("cvd_divergence")),
            "cvd_absorption": _safe_float(flow_features.get("cvd_absorption")),
            "cvd_accumulation": _safe_float(flow_features.get("cvd_accumulation")),
            "taker_buy_ratio": _safe_float(flow_features.get("taker_buy_ratio")),
            "aggressive_trade_rate": _safe_float(flow_features.get("aggressive_trade_rate")),
            "spoofing_intensity": _safe_float(spoof_val),
            "spoofing_alert": spoof_alert,
            "volume_ratio": _safe_float(volume_ratio),
            "price_change_pct": _safe_float(price_change_pct),
            "spread_bps": _safe_float(spread_bps),
        }
        signal_snapshot = {
            "setup_type": setup_type or "none",
            "trend_norm": _safe_float(trend_norm),
            "mean_reversion_norm": _safe_float(mean_rev_norm),
            "activation_threshold": _safe_float(dynamic_threshold),
            "auction_state": str(price_data.attrs.get("auction_state", auction_state or "unknown")),
            "poc_target": _safe_float(price_data.attrs.get("poc_target", poc_candidate)),
            "lvn_entry_level": _safe_float(price_data.attrs.get("lvn_entry_level", lvn_candidate)),
            **flow_snapshot,
        }
        price_data.attrs["signal_features"] = signal_snapshot
        indicator_flags = {
            "ema": ema_flag,
            "macd": macd_flag,
            "rsi": rsi_flag,
            "adx": adx_flag,
            "vwma": vwma_flag,
            "bb": bb_flag,
            "dema": dema_flag,
            "stoch": stoch_flag,
            "cci": cci_flag,
            "atr": atr_flag,
            "hurst": hurst_flag,
            "confluence": confluence_flag,
            "flow": flow_score,
            "flow_flag": flow_flag,
            "order_book_imbalance": flow_snapshot["order_book_imbalance"],
            "cvd_change": flow_snapshot["cvd_change"],
            "cvd_divergence": flow_snapshot["cvd_divergence"],
            "cvd_absorption": flow_snapshot["cvd_absorption"],
            "cvd_accumulation": flow_snapshot["cvd_accumulation"],
            "spoofing_intensity": flow_snapshot["spoofing_intensity"],
            "candle": candle_flag,
            "chart": chart_flag,
            "flag": flag_flag,
            "hs": hs_flag,
            "double_bottom": double_flag,
            "cup_handle": cup_flag,
            "volatility_regime": volatility_regime,
            "trend_norm": trend_norm,
            "mean_reversion_norm": mean_rev_norm,
            "setup_type": setup_type or "none",
            "activation_threshold": dynamic_threshold,
        }
        log_signal(
            symbol,
            session_name,
            normalized_score,
            direction,
            w,
            triggered_patterns,
            chart_pattern,
            indicator_flags,
            signal_snapshot,
        )
        return normalized_score, direction, position_size, pattern_name
    except Exception as e:
        logger.error("Signal evaluation error in %s: %s", symbol, e, exc_info=True)
        traceback.print_exc()
        return 0, None, 0, None
