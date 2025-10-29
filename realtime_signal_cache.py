"""Real-time signal evaluation cache for the trading agent.

This module decouples heavy indicator computation from the synchronous
decision loop.  A background worker continuously refreshes price snapshots,
calls ``evaluate_signal`` and stores the results in-memory so the trading
loop can respond immediately when a trade opportunity appears.  By
precomputing signals we avoid the latency penalty of fetching candles and
deriving indicators right when an order needs to be submitted.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --- RTSC quiet filter ------------------------------------------------
import logging
import time

logger = logging.getLogger("realtime_signal_cache")
_CIRCUIT_BENIGN_SUBSTRINGS = ("insufficient candles",)

# --- RTSC quiet mode (default ON) ---
_RTSQ = os.getenv("RTSC_LOG_QUIET", "1")  # "1" to quiet, "0" to see everything


class _RtscNoiseFilter(logging.Filter):
    """Drop high-volume progress messages but keep start/finish and errors."""

    _DROP_SUBSTRS = (
        # REST chatter
        "REST mirror try",
        "REST mirror OK",
        "REST fetch OK",
        "REST cache update OK",
        "submitted to bg loop",
        "ENTER _refresh_symbol_via_rest",
        "ENTER runner(",
        "Refreshing symbol",
        "ok refresh(",  # benign refresh bookkeeping
        # Kline/trade queue spam
        "market_stream: {\"event\": \"market_queue_drop\"",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True  # always keep errors and above
        msg = record.getMessage()
        # keep just the two lifecycle lines
        if "RTSC warm-up started" in msg or "RTSC warm-up completed" in msg:
            return True
        # drop noisy progress lines
        return not any(s in msg for s in self._DROP_SUBSTRS)


if _RTSQ != "0":
    logger.addFilter(_RtscNoiseFilter())

# Optional: if "market_stream" uses its own logger name, turn it down too.
try:
    logging.getLogger("market_stream").setLevel(logging.ERROR)
except Exception:
    pass
# ----------------------------------------------------------------------

import pandas as pd
import requests  # used for mirror fallback (no proxies)

from rest_prices import rest_backfill_klines, rest_fetch_latest_closed

# python-binance (a.k.a binance-connector legacy) exceptions can vary by version.
# Import defensively so minor packaging changes do not break the cache module.
try:  # pragma: no cover - import guard for optional dependency
    from binance.client import Client as BinanceClient
    from binance.exceptions import BinanceAPIException, BinanceRequestException
except Exception:  # pragma: no cover - allow runtime without python-binance
    BinanceClient = None

    class BinanceAPIException(Exception):
        ...


class BinanceRequestException(Exception):
    ...


LOGGER = logger


# Optional verbose logging for debugging REST refresh flows.
RTSC_DEBUG: bool = os.getenv("RTSC_DEBUG", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Hard switch to force REST refreshes when websockets stall.
RTSC_FORCE_REST: bool = os.getenv("RTSC_FORCE_REST", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Timeout (seconds) for REST kline calls.
RTSC_REST_TIMEOUT: float = float(os.getenv("RTSC_REST_TIMEOUT", "8"))
# Number of klines to request during REST refreshes.
RTSC_REST_LIMIT: int = int(os.getenv("RTSC_REST_LIMIT", "2"))
# Interval for REST klines (should align with trading horizon).
RTSC_REST_INTERVAL: str = os.getenv("RTSC_REST_INTERVAL", "1m")
# Derive an expected bar duration (seconds) from the configured REST interval.


def _interval_to_seconds(value: str) -> Optional[int]:
    """Return ``value`` expressed in seconds if it resembles a kline interval."""

    if not value:
        return None
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw.isdigit():
        try:
            return max(1, int(raw))
        except ValueError:
            return None
    unit = raw[-1]
    amount = raw[:-1]
    factors = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    try:
        base = float(amount)
    except ValueError:
        return None
    factor = factors.get(unit)
    if factor is None:
        return None
    seconds = int(base * factor)
    return seconds if seconds > 0 else None


REST_INTERVAL_SECONDS: Optional[int] = _interval_to_seconds(RTSC_REST_INTERVAL)
# Minimum candles required for the evaluator to produce a trade signal.
RTSC_EVALUATOR_MIN_BARS: int = int(os.getenv("RTSC_MIN_EVAL_BARS", "40"))
# Seconds after which WS is considered stale and REST is triggered.
RTSC_WS_STALE_AFTER_SEC: float = float(os.getenv("RTSC_WS_STALE_AFTER", "12"))

BINANCE_MIRRORS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]

# Optional proxy support for REST client.
HTTP_PROXY = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")


def cache_diagnostics_path() -> Path:
    """Return the path used to persist cache diagnostics snapshots."""

    return Path(os.getenv("RTSC_DIAGNOSTICS_PATH", "rtsc_diagnostics.json")).expanduser()


def cache_state_file() -> str:
    """Backward-compatible alias returning the diagnostics file path as ``str``."""

    return str(cache_diagnostics_path())


_ACTIVE_CACHE: Optional["RealTimeSignalCache"] = None


def set_active_cache(cache: Optional["RealTimeSignalCache"]) -> None:
    """Register ``cache`` as the active :class:`RealTimeSignalCache` instance."""

    global _ACTIVE_CACHE
    _ACTIVE_CACHE = cache


def get_active_cache() -> Optional["RealTimeSignalCache"]:
    """Return the active :class:`RealTimeSignalCache` instance if one is registered."""

    return _ACTIVE_CACHE


def pending_diagnostics(limit: Optional[int] = None) -> Path:
    """Trigger a diagnostics snapshot using the active cache and return its path."""

    cache = get_active_cache()
    if cache is None:
        raise RuntimeError(
            "No active RealTimeSignalCache registered; start the agent to initialise it."
        )

    try:
        cache.pending_diagnostics(limit=limit)
    finally:
        # Even if the underlying call fails to persist, return the expected location so
        # callers can provide actionable feedback.
        pass

    return cache_diagnostics_path()

# --- BEGIN: RTSC warmup/refill core integration --------------------------------

# ===== Config (env tunable; safe defaults) =====
READY_MIN_FRACTION  = float(os.getenv("READY_MIN_FRACTION", "0.80"))   # % of symbols with fresh data
REQUIRE_FULL_WARMUP = os.getenv("RTSC_REQUIRE_FULL_WARMUP", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)
WARMUP_REQUIRED_FRACTION = 1.0 if REQUIRE_FULL_WARMUP else READY_MIN_FRACTION
WARMUP_MAX_SECONDS  = int(os.getenv("WARMUP_MAX_SECONDS", "90"))        # watchdog ceiling
PRIME_LIMIT_MINUTES = int(os.getenv("PRIME_LIMIT_MINUTES", "300"))      # bars for initial prime
FRESHNESS_SECONDS   = int(os.getenv("FRESHNESS_SECONDS", "120"))        # what is considered "fresh"
USE_WS_PRICES       = os.getenv("USE_WS_PRICES", "1").lower() in ("1","true","yes")
ENABLE_PRIME        = os.getenv("ENABLE_RTSCC_PRIME", "1").lower() in ("1","true","yes")
ENABLE_REFRESH      = os.getenv("ENABLE_RTSCC_REFRESH", "1").lower() in ("1","true","yes")
_RAW_REQUIRED_MIN_BARS = int(os.getenv("RTSC_REQUIRED_MIN_BARS", "220"))
REST_REQUIRED_MIN_BARS = max(_RAW_REQUIRED_MIN_BARS, RTSC_EVALUATOR_MIN_BARS)
_RAW_RELAXED_MIN_BARS = int(
    os.getenv("RTSC_RELAXED_MIN_BARS", str(RTSC_EVALUATOR_MIN_BARS))
)
RELAXED_REQUIRED_MIN_BARS = max(
    RTSC_EVALUATOR_MIN_BARS,
    min(REST_REQUIRED_MIN_BARS, _RAW_RELAXED_MIN_BARS),
)
_RAW_WARMUP_BARS = int(os.getenv("RTSC_REST_WARMUP_BARS", "300"))
# Always fetch at least one more bar than required so the most recent candle can
# be discarded when it is still forming without falling below the evaluator
# threshold. This guards against overly aggressive overrides (e.g. setting the
# env var to "3") that previously left the cache with too-few rows.
REST_WARMUP_BARS = max(_RAW_WARMUP_BARS, REST_REQUIRED_MIN_BARS + 1)

# ===== Optional python-binance fallback =====
def _binance_client_from_env():
    try:
        from binance.client import Client
    except Exception:
        return None
    ak = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_KEY")
    sk = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET")
    if not ak or not sk:
        return None
    try:
        return Client(api_key=ak, api_secret=sk)
    except Exception:
        return None


def _binance_fetch_1m(client, symbol: str, limit: int):
    try:
        interval = getattr(client, "KLINE_INTERVAL_1MINUTE", "1m")
        return client.get_klines(symbol=symbol, interval=interval, limit=int(limit)) or []
    except Exception:
        return []


# ===== General helpers =====
def _get_logger(obj):
    lg = getattr(obj, "log", None)
    return lg


def _log(obj, lvl, msg, *args):
    lg = _get_logger(obj)
    try:
        if lg is not None:
            getattr(lg, lvl, lg.info)(msg, *args)
    except Exception:
        # never let logging crash the cache
        pass


def _iter_symbols(obj) -> list[str]:
    syms = getattr(obj, "symbols", [])
    if callable(syms):
        try:
            syms = syms()
        except Exception:
            syms = []
    return list(syms or [])


def _get_rest(obj):
    # try common attribute names used in the project
    for attr in ("rest", "_price_fetcher", "rest_client", "binance", "client"):
        c = getattr(obj, attr, None)
        if c is not None:
            if callable(c) and not any(
                hasattr(c, candidate)
                for candidate in (
                    "get_klines",
                    "fetch_klines",
                    "get_price_data",
                    "get_candles",
                    "get_price_data_async",
                    "get_candles_async",
                )
            ):
                # plain callables (like coroutine fetchers) are unlikely to expose klines APIs
                continue
            return c
    return None


def _fresh_fraction(obj) -> float:
    now = time.time()
    fresh = 0
    syms = _iter_symbols(obj)
    last = getattr(obj, "_last_update_ts", {})
    for s in syms:
        if now - last.get(s, 0) <= FRESHNESS_SECONDS:
            fresh += 1
    return fresh / max(1, len(syms))


def _sufficient_fraction(obj, *, min_bars: int = REST_REQUIRED_MIN_BARS) -> float:
    """Return the fraction of symbols whose cached data meets ``min_bars``."""

    syms = _iter_symbols(obj)
    if not syms:
        return 1.0

    lock = getattr(obj, "_lock", None)
    cache_snapshot: Dict[str, CachedSignal]
    if lock is not None:
        try:
            with lock:  # type: ignore[attr-defined]
                cache_snapshot = dict(getattr(obj, "_cache", {}))
        except Exception:
            cache_snapshot = dict(getattr(obj, "_cache", {}))
    else:
        cache_snapshot = dict(getattr(obj, "_cache", {}))

    key_fn = getattr(obj, "_key", None)

    sufficient = 0
    relaxed_min = getattr(obj, "_relaxed_min_bars", min_bars)
    relaxed_min = max(RTSC_EVALUATOR_MIN_BARS, min(relaxed_min, min_bars))

    for sym in syms:
        key = sym
        if callable(key_fn):
            try:
                key = key_fn(sym)
            except Exception:
                key = str(sym).upper()
        cached = cache_snapshot.get(key)
        rows = 0
        if cached is not None:
            price_df = getattr(cached, "price_data", None)
            if price_df is not None:
                try:
                    rows = len(price_df)
                except Exception:
                    rows = 0
        if rows >= min_bars:
            sufficient += 1
        elif rows >= relaxed_min:
            sufficient += 1

    return sufficient / max(1, len(syms))


def _run_async_allow_nested(coro):
    """
    Run a coroutine even if there is already a running loop.
    We apply nest_asyncio if needed to avoid 'event loop is already running'.
    """

    try:
        import nest_asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except Exception:
        # last resort: create and close a private loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _fetch_candles(rest, symbol: str, limit: int):
    """
    Try the project's fetchers first (sync/async in several spellings).
    Fall back to python-binance REST if needed.
    """

    candidates = [
        ("get_klines", False),
        ("fetch_klines", False),
        ("get_price_data", False),
        ("get_candles", False),
        ("get_price_data_async", True),
        ("get_candles_async", True),
    ]

    now = int(time.time())
    start_s = now - limit * 60
    start_ms, end_ms = start_s * 1000, now * 1000

    for name, is_async in candidates:
        fn = getattr(rest, name, None)
        if not fn:
            continue
        try:
            sig = inspect.signature(fn)
            params = set(sig.parameters.keys())
            kwargs: Dict[str, Any] = {}

            if "symbol" in params:
                kwargs["symbol"] = symbol
            elif "symbols" in params:
                kwargs["symbols"] = [symbol]

            if "interval" in params:
                kwargs["interval"] = "1m"
            elif "timeframe" in params:
                kwargs["timeframe"] = "1m"

            if "limit" in params:
                kwargs["limit"] = int(limit)
            elif "lookback" in params:
                kwargs["lookback"] = int(limit)
            elif "bars" in params:
                kwargs["bars"] = int(limit)

            if "start_ts" in params and "end_ts" in params:
                kwargs["start_ts"], kwargs["end_ts"] = start_ms, end_ms
            elif "start_time" in params and "end_time" in params:
                kwargs["start_time"], kwargs["end_time"] = start_ms, end_ms

            if "timeout" in params:
                kwargs["timeout"] = 15

            if is_async or inspect.iscoroutinefunction(fn):
                candles = _run_async_allow_nested(fn(**kwargs))
            else:
                candles = fn(**kwargs)

            if isinstance(candles, dict):
                candles = candles.get(symbol) or candles.get(symbol.upper()) or candles.get(symbol.lower())

            if candles and len(candles) > 0:
                return candles
        except Exception:
            # try next candidate
            continue

    # hard fallback
    client = _binance_client_from_env()
    if client is not None:
        candles = _binance_fetch_1m(client, symbol, limit)
        if candles:
            return candles

    return []


# --- END: RTSC warmup/refill core integration helpers --------------------------

from observability import log_event, record_metric

PriceFetcher = Callable[[str], Awaitable[pd.DataFrame | None]]
SignalEvaluator = Callable[..., Tuple[float, Optional[str], float, Optional[str]]]

def _klines_to_dataframe(klines: Sequence[Sequence[object]]) -> pd.DataFrame:
    if not klines:
        return pd.DataFrame()

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(list(klines), columns=columns)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
        "number_of_trades",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp").set_index("timestamp")
    tz = getattr(df.index, "tz", None)
    if tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    df["quote_volume"] = df["quote_asset_volume"].fillna(0.0)
    df["taker_sell_base"] = (df["volume"].fillna(0.0) - df["taker_buy_base"].fillna(0.0)).clip(lower=0.0)
    df["taker_sell_quote"] = (
        df["quote_volume"].fillna(0.0) - df["taker_buy_quote"].fillna(0.0)
    ).clip(lower=0.0)

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


@dataclass(slots=True)
class CachedSignal:
    """Container for the latest evaluated signal."""

    symbol: str
    score: float
    direction: Optional[str]
    position_size: float
    pattern: Optional[str]
    price_data: pd.DataFrame
    updated_at: float
    compute_latency: float

    def age(self) -> float:
        """Return the age of the cached signal in seconds."""

        return time.time() - self.updated_at

    def is_fresh(self, max_age: float) -> bool:
        """Whether the cached signal is recent enough for live trading."""

        return self.age() <= max_age


class RealTimeSignalCache:
    """Maintain continuously refreshed trading signals for a universe of symbols."""

    def __init__(
        self,
        price_fetcher: PriceFetcher,
        evaluator: SignalEvaluator,
        refresh_interval: float = 2.0,
        stale_after: Optional[float] = None,
        max_concurrent_fetches: Optional[int] = None,
        *,
        max_concurrency: Optional[int] = None,
        use_streams: bool = False,
    ) -> None:
        self.log = logger
        self._price_fetcher = price_fetcher
        self._evaluator = evaluator
        self._refresh_interval = max(0.5, float(refresh_interval))
        self.use_streams = bool(use_streams)
        if stale_after is None or float(stale_after) <= 0:
            effective_stale_after = self._refresh_interval * 3
        else:
            effective_stale_after = float(stale_after)

        self._stale_after = float(effective_stale_after)

        if max_concurrency is None:
            if max_concurrent_fetches is None:
                max_concurrency = 8
            else:
                max_concurrency = int(max_concurrent_fetches)
        elif max_concurrent_fetches is not None and int(max_concurrency) != int(
            max_concurrent_fetches
        ):
            raise ValueError(
                "Specify only one of max_concurrency or max_concurrent_fetches",
            )

        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")

        self._max_concurrency = int(max_concurrency)
        self._symbols: set[str] = set()
        self._cache: Dict[str, CachedSignal] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._async_wake: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._bg_task: Optional[asyncio.Task] = None
        self._worker_loop: Optional[asyncio.AbstractEventLoop] = None
        self._worker_loop_ready = threading.Event()
        self._context: Dict[str, object] = {}
        self._symbol_added_at: Dict[str, float] = {}
        self._symbol_last_attempt: Dict[str, float] = {}
        self._symbol_last_error: Dict[str, Tuple[float, str]] = {}
        self._primed_symbols: set[str] = set()
        self._priority_symbols: set[str] = set()
        self._rest_refresh_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._default_debounce = 0.8
        self._symbol_debounce: Dict[str, float] = {}
        self._symbol_refresh_overrides: Dict[str, float] = {}
        self._last_priority_request: Dict[str, float] = {}
        self._relaxed_min_bars = RELAXED_REQUIRED_MIN_BARS
        self._cb_threshold = 5
        self._cb_window = 30.0
        self._cb_error_times: "deque[float]" = deque()
        self._cb_open_until = 0.0
        self._cb_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._ws_bridge: Optional[object] = None
        self._ws_callback_registered = False
        self._last_ws_ts: Dict[str, float] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._bg_thread: threading.Thread | None = None
        self._pending_syms: list[str] = []
        self._sem: asyncio.Semaphore | None = None  # initialised when loop starts

        existing_rest = getattr(self, "rest", None)
        if existing_rest is not None:
            self._rest_client = existing_rest
        else:
            self._rest_client = self._init_rest_client()
            if self._rest_client is not None:
                self.rest = self._rest_client
            else:
                self.rest = None

        # --- warmup/refill integration state ---
        self._last_eval: Dict[str, Any] = getattr(self, "_last_eval", {})
        self._last_update_ts: Dict[str, float] = getattr(self, "_last_update_ts", {})
        self._ready = getattr(self, "_ready", False)
        self._warmup_started_at: Optional[float] = getattr(
            self, "_warmup_started_at", None
        )

        evaluator_ref = getattr(self, "evaluator", None)
        if evaluator_ref is not None and not callable(evaluator_ref):
            self._evaluator = evaluator_ref

        self._enable_prime = ENABLE_PRIME
        self._enable_refresh = ENABLE_REFRESH

        _log(
            self,
            "info",
            "RTSC core warmup enabled (prime=%s, refresh=%s, ws=%s | ready>=%.0f%%, bars>=%.0f%%, ceil=%ss, bars=%dm)",
            self._enable_prime,
            self._enable_refresh,
            USE_WS_PRICES,
            READY_MIN_FRACTION * 100,
            WARMUP_REQUIRED_FRACTION * 100,
            WARMUP_MAX_SECONDS,
            PRIME_LIMIT_MINUTES,
        )

        set_active_cache(self)

    @property
    def stale_after(self) -> float:
        """Maximum allowed age (seconds) for cached signals."""

        return self._stale_after

    @stale_after.setter
    def stale_after(self, value: float) -> None:
        """Update the maximum allowed age for cached signals."""

        new_value = max(0.0, float(value))
        if new_value == self._stale_after:
            return
        self._stale_after = new_value
        self._signal_wake()

    @property
    def refresh_interval(self) -> float:
        """Return the background refresh cadence for the cache."""

        return self._refresh_interval

    def _key(self, symbol: str) -> str:
        """Normalize ``symbol`` for internal storage."""

        return str(symbol).upper()

    def _init_rest_client(self) -> Optional["BinanceClient"]:
        """Initialise a python-binance client for REST fallback operations."""

        if BinanceClient is None:
            # logger.warning(
            #     "RTSC: python-binance not available; REST fallback disabled."
            # )
            return None

        api_key = (
            os.getenv("BINANCE_API_KEY")
            or os.getenv("BINANCE_KEY")
            or ""
        )
        api_secret = (
            os.getenv("BINANCE_API_SECRET")
            or os.getenv("BINANCE_SECRET")
            or ""
        )

        try:
            client = BinanceClient(api_key=api_key, api_secret=api_secret)
            # if HTTP_PROXY or HTTPS_PROXY:
            #     logger.info(
            #         "RTSC: REST client initialised (proxies handled by requests layer)."
            #     )
            # else:
            #     logger.info("RTSC: REST client initialised.")
            return client
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("RTSC: Failed to initialise REST client: %s", exc)
            return None

    def _signal_wake(self) -> None:
        """Wake the background worker if it is sleeping."""

        self._wake_event.set()
        loop = self._worker_loop
        async_wake = self._async_wake
        if loop and loop.is_running() and async_wake is not None:
            loop.call_soon_threadsafe(async_wake.set)

    def _on_rest_refresh_done(self, key: str, task: asyncio.Task[Any]) -> None:
        """Cleanup callback for REST refresh tasks to release bookkeeping state."""

        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug("[RTSC] REST refresh task for %s cancelled", key)
        except Exception:
            # logger.warning("[RTSC] REST refresh task for %s raised: %s", key, exc)
            pass
        finally:
            with self._lock:
                existing = self._rest_refresh_tasks.get(key)
                if existing is task:
                    self._rest_refresh_tasks.pop(key, None)

    def _get_ws_bridge(self) -> Optional[object]:
        with self._stream_lock:
            return self._ws_bridge

    def enable_streams(self, ws_bridge: object) -> bool:
        """Attach ``ws_bridge`` to receive streaming updates when enabled."""

        if ws_bridge is None:
            self.disable_streams()
            return False

        if not self.use_streams:
            logger.debug("RTSC: streaming disabled; ignoring enable_streams call")
            return False

        with self._stream_lock:
            already_registered = (
                self._ws_bridge is ws_bridge and self._ws_callback_registered
            )
        if already_registered:
            self._update_stream_symbols()
            return True

        self.disable_streams()
        with self._stream_lock:
            self._ws_bridge = ws_bridge
            self._ws_callback_registered = False

        register_callback = getattr(ws_bridge, "register_callback", None)
        if not callable(register_callback):
            logger.debug("RTSC: WS bridge missing register_callback; cannot enable streams")
            return False

        try:
            register_callback(self.handle_ws_update)
        except Exception:
            logger.debug("RTSC: failed to register WS callback", exc_info=True)
            return False

        with self._stream_lock:
            self._ws_callback_registered = True

        self._update_stream_symbols()
        return True

    def disable_streams(self) -> None:
        """Detach the current WebSocket bridge if one is registered."""

        with self._stream_lock:
            bridge = self._ws_bridge
            registered = self._ws_callback_registered
            self._ws_bridge = None
            self._ws_callback_registered = False

        if bridge is None:
            return

        if registered:
            unregister_callback = getattr(bridge, "unregister_callback", None)
            if callable(unregister_callback):
                try:
                    unregister_callback(self.handle_ws_update)
                except Exception:
                    logger.debug(
                        "RTSC: failed to unregister WS callback", exc_info=True
                    )

    def bars_len(self, symbol: str) -> int:
        key = self._key(symbol)
        with self._lock:
            cached = self._cache.get(key)
        if cached is None or cached.price_data is None:
            return 0
        try:
            return int(len(cached.price_data))
        except Exception:
            return 0

    def force_rest_backfill(self, symbol: str) -> bool:
        """Hard backfill ``symbol`` via REST and commit to cache."""

        key = self._key(symbol)
        client = _get_rest(self)
        if client is None:
            _log(self, "warning", "force_rest_backfill(): no REST client for %s", symbol)
            return False

        try:
            raw = rest_backfill_klines(client, key, RTSC_REST_INTERVAL, REST_WARMUP_BARS)
        except Exception as exc:
            _log(self, "warning", "force_rest_backfill(): REST error for %s: %s", key, exc)
            return False

        df = self._standardize_price_df(raw)
        rows = 0 if df is None else len(df)
        relaxed_min = self._relaxed_min_bars
        if df is None or rows < relaxed_min:
            _log(
                self,
                "warning",
                "force_rest_backfill(): insufficient candles for %s (%d/%d)",
                key,
                rows,
                relaxed_min,
            )
            return False

        if rows < REST_REQUIRED_MIN_BARS:
            _log(
                self,
                "info",
                "force_rest_backfill(): accepting partial snapshot for %s (%d/%d)",
                key,
                rows,
                REST_REQUIRED_MIN_BARS,
            )
            log_event(
                logger,
                "cache_rest_partial",
                symbol=key,
                rows=rows,
                required=REST_REQUIRED_MIN_BARS,
                relaxed=relaxed_min,
                method="force_rest_backfill",
            )

        attempt_ts = time.time()
        success = self._update_cache(key, df, attempt_ts=attempt_ts, prev_age=None)
        if success:
            _log(self, "info", "force_rest_backfill(): %s committed with %d bars", key, rows)
        return success

    def _prime_symbol(self, symbol: str) -> bool:
        if self.bars_len(symbol) >= REST_REQUIRED_MIN_BARS:
            return True

        if _get_rest(self) is not None and self.force_rest_backfill(symbol):
            return True

        fetcher = getattr(self, "_price_fetcher", None)
        if fetcher is None:
            return False

        key = self._key(symbol)
        try:
            maybe_df = fetcher(key)
        except Exception:
            logger.debug("RTSC: price fetcher raised while priming %s", key, exc_info=True)
            return False

        try:
            if inspect.isawaitable(maybe_df):
                df = _run_async_allow_nested(maybe_df)
            else:
                df = maybe_df
        except Exception:
            logger.debug("RTSC: async price fetch failed while priming %s", key, exc_info=True)
            return False

        standardized = self._standardize_price_df(df)
        rows = 0 if standardized is None else len(standardized)
        relaxed_min = self._relaxed_min_bars
        if standardized is None or rows < relaxed_min:
            logger.debug(
                "RTSC: price fetcher returned insufficient candles for %s during prime (%d/%d)",
                key,
                rows,
                relaxed_min,
            )
            return False

        if rows < REST_REQUIRED_MIN_BARS:
            logger.debug(
                "RTSC: accepting partial prime snapshot for %s (%d/%d)",
                key,
                rows,
                REST_REQUIRED_MIN_BARS,
            )
            log_event(
                logger,
                "cache_rest_partial",
                symbol=key,
                rows=rows,
                required=REST_REQUIRED_MIN_BARS,
                relaxed=relaxed_min,
                method="prime_price_fetcher",
            )

        attempt_ts = time.time()
        success = self._update_cache(key, standardized, attempt_ts=attempt_ts, prev_age=None)
        if success:
            _log(
                self,
                "info",
                "prime(): %s primed via price fetcher with %d bars",
                key,
                rows,
            )
        return success

    def _tick_symbol(self, symbol: str) -> bool:
        if self.bars_len(symbol) < REST_REQUIRED_MIN_BARS:
            return self._prime_symbol(symbol)

        client = _get_rest(self)
        if client is None:
            return False

        try:
            latest = rest_fetch_latest_closed(client, self._key(symbol), RTSC_REST_INTERVAL)
        except Exception as exc:
            _log(self, "warning", "tick(): REST error for %s: %s", symbol, exc)
            return False

        latest_df = self._standardize_price_df(latest)
        if latest_df is None or latest_df.empty:
            return False

        key = self._key(symbol)
        with self._lock:
            cached = self._cache.get(key)

        if cached is None or cached.price_data is None or cached.price_data.empty:
            return self._prime_symbol(symbol)

        base_df = cached.price_data
        try:
            last_idx = base_df.index[-1]
        except Exception:
            return self._prime_symbol(symbol)

        add = latest_df[latest_df.index > last_idx]
        if add.empty:
            return True

        combined = pd.concat([base_df, add]).sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="last")]
        attempt_ts = time.time()
        prev_age = None
        try:
            prev_age = attempt_ts - float(cached.updated_at)
        except Exception:
            prev_age = None
        return self._update_cache(key, combined, attempt_ts=attempt_ts, prev_age=prev_age)

    def prime(self):
        """Prime the cache by backfilling symbols via REST."""

        if not self._enable_prime:
            _log(self, "info", "prime(): disabled by config.")
            return

        evaluator = getattr(self, "_evaluator", None)
        if evaluator is None:
            _log(self, "error", "prime(): evaluator missing; skipping prime.")
            return

        symbols = _iter_symbols(self)
        if not symbols:
            return

        for symbol in symbols:
            try:
                self._prime_symbol(symbol)
            except Exception:
                logger.debug("RTSC: error priming %s", symbol, exc_info=True)
            time.sleep(0.02)

        _log(self, "info", "prime(): completed. Fresh=%.0f%%", _fresh_fraction(self) * 100)

    def background_refresh_until_ready(self):
        """Keep fetching short windows until READY_MIN_FRACTION is reached."""

        if not self._enable_refresh:
            _log(self, "info", "refresh(): disabled by config.")
            return

        rest = _get_rest(self)
        if rest is None:
            _log(self, "error", "refresh(): no REST/price_fetcher; disabled.")
            return

        evaluator = getattr(self, "_evaluator", None)
        if evaluator is None:
            _log(self, "error", "refresh(): evaluator missing; disabled.")
            return

        while not self.is_ready():
            for s in _iter_symbols(self):
                try:
                    candles = _fetch_candles(rest, s, 120)
                    if not candles:
                        continue
                    if hasattr(evaluator, "evaluate_from_klines"):
                        ev = evaluator.evaluate_from_klines(s, candles)  # type: ignore[attr-defined]
                        with self._lock:
                            self._last_eval[s] = ev
                            self._last_update_ts[s] = time.time()
                    else:
                        self._evaluate_candles_sync(s, candles)
                except Exception:
                    pass
                time.sleep(0.01)

            fraction = _fresh_fraction(self)
            sufficient_fraction = _sufficient_fraction(self)
            if fraction >= READY_MIN_FRACTION and sufficient_fraction >= WARMUP_REQUIRED_FRACTION:
                self.mark_ready()
                _log(
                    self,
                    "info",
                    "refresh(): threshold met (fresh=%.0f%%, bars=%.0f%%).",
                    fraction * 100,
                    sufficient_fraction * 100,
                )
                break

            time.sleep(0.5)

    def warmup_watchdog(self):
        """Ensure we proceed after a ceiling even if freshness is not met."""

        start = time.time()
        while not self.is_ready():
            frac = _fresh_fraction(self)
            bars_frac = _sufficient_fraction(self)
            elapsed = time.time() - start
            if frac >= READY_MIN_FRACTION and bars_frac >= WARMUP_REQUIRED_FRACTION:
                _log(
                    self,
                    "info",
                    "watchdog: threshold met (fresh=%.0f%%, bars=%.0f%%).",
                    frac * 100,
                    bars_frac * 100,
                )
                self.mark_ready()
                break
            if elapsed >= WARMUP_MAX_SECONDS:
                _log(
                    self,
                    "warning",
                    "watchdog: ceiling reached (%ds). Proceeding with fresh=%.0f%%, bars=%.0f%%.",
                    WARMUP_MAX_SECONDS,
                    frac * 100,
                    bars_frac * 100,
                )
                self.mark_ready()
                break
            time.sleep(1.0)

    # ---------- Small adapters to fit the above ----------
    # If your class already has these, keep your versions.

    def is_ready(self) -> bool:
        return bool(getattr(self, "_ready", False))

    def mark_ready(self, value: bool = True):
        value_bool = bool(value)
        was_ready = bool(getattr(self, "_ready", False))
        self._ready = value_bool
        if value_bool and not was_ready:
            dur = time.time() - getattr(self, "_warmup_started_at", time.time())
            cache_obj = getattr(self, "cache", None)
            if cache_obj is None:
                cache_obj = getattr(self, "_cache", {})
            try:
                entries = len(cache_obj)
            except Exception:
                entries = len(getattr(self, "_cache", {}))
            logger.info("RTSC warm-up completed")

    def _update_stream_symbols(self) -> None:
        if not self.use_streams:
            return
        bridge = self._get_ws_bridge()
        if bridge is None:
            return
        update_symbols = getattr(bridge, "update_symbols", None)
        if not callable(update_symbols):
            return
        symbols_snapshot = sorted(self.symbols())
        try:
            update_symbols(symbols_snapshot)
        except Exception:
            logger.debug("RTSC: failed to update WS bridge symbols", exc_info=True)

    def start(self) -> None:
        """Launch the background refresh worker if it is not already running."""

        def _run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._sem = asyncio.Semaphore(5)
            self._loop_ready.set()
            # LOGGER.warning("[RTSC] bg loop READY")
            try:
                loop.run_forever()
            finally:
                self._loop_ready.clear()
                self._sem = None
                self._loop = None
                # LOGGER.warning("[RTSC] bg loop STOPPED")
                try:
                    loop.close()
                except Exception:
                    logger.debug("[RTSC] error closing bg loop", exc_info=True)

        bg_thread = getattr(self, "_bg_thread", None)
        if bg_thread is None or not bg_thread.is_alive():
            self._loop_ready = getattr(self, "_loop_ready", threading.Event())
            self._loop_ready.clear()
            self._bg_thread = threading.Thread(
                target=_run,
                name="rtsc-bg",
                daemon=True,
            )
            self._bg_thread.start()
        else:
            logger.debug("[RTSC] bg loop already running")

        if not self._loop_ready.wait(timeout=5.0):
            logger.debug("[RTSC] bg loop did not signal readiness within 5s")
        else:
            self.flush_pending()

        if getattr(self, "_bg_task", None) and not self._bg_task.done():
            # logger.info("RTSC: worker already running")
            return
        if getattr(self, "_thread", None) and self._thread.is_alive():
            # logger.info("RTSC: worker already running (thread)")
            return
        self._stop_event.clear()
        self._wake_event.clear()
        self._ready = False
        self._worker_loop_ready.clear()
        try:
            loop = asyncio.get_running_loop()
            # logger.info("RTSC: starting worker on running loop")
            self._bg_task = loop.create_task(self._worker())
        except RuntimeError:
            # logger.info("RTSC: no running loop; starting worker in daemon thread")

            def runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self._worker())
                finally:
                    try:
                        loop.close()
                    except Exception:
                        logger.debug("RTSC: error closing worker loop", exc_info=True)

            self._thread = threading.Thread(target=runner, daemon=True, name="rtsc-worker")
            self._thread.start()
            if not self._worker_loop_ready.wait(timeout=1.0):
                logger.debug("RTSC: worker loop did not signal readiness within 1s")

        t0 = time.time()
        self._warmup_started_at = t0
        try:
            symbols_attr = getattr(self, "symbols", [])
            if callable(symbols_attr):
                symbols_value = symbols_attr()
            else:
                symbols_value = symbols_attr
            n_symbols = len(symbols_value) if symbols_value is not None else 0
            if not n_symbols:
                n_symbols = len(getattr(self, "_symbols", []))
        except Exception:
            n_symbols = -1
        logger.info("RTSC warm-up started")

        if self._enable_prime:
            threading.Thread(target=self.prime, daemon=True, name="rtsc-prime").start()
        if self._enable_refresh:
            threading.Thread(
                target=self.background_refresh_until_ready,
                daemon=True,
                name="rtsc-refresh",
            ).start()
        threading.Thread(
            target=self.warmup_watchdog,
            daemon=True,
            name="rtsc-watchdog",
        ).start()

    def stop(self) -> None:
        """Signal the worker to stop and wait briefly for it to exit."""

        self._stop_event.set()
        self._signal_wake()
        task = self._bg_task
        loop = self._worker_loop
        if task and loop and not task.done():
            loop.call_soon_threadsafe(task.cancel)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._bg_task = None
        self._thread = None
        self._worker_loop = None
        self._worker_loop_ready.clear()
        self._ready = False
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=2.0)
        self._bg_thread = None
        self._loop = None
        self._loop_ready.clear()
        self._sem = None

    def update_universe(self, symbols: Iterable[str]) -> None:
        """Update the symbol universe tracked by the cache."""

        normalized = {self._key(sym) for sym in symbols if sym}
        with self._lock:
            added = normalized - self._symbols
            removed = self._symbols - normalized
            if not added and not removed:
                return
            self._symbols = normalized
            # Remove cache entries for symbols no longer tracked to free memory
            stale_keys = [key for key in self._cache.keys() if key not in normalized]
            for key in stale_keys:
                self._cache.pop(key, None)
            for key in removed:
                self._symbol_added_at.pop(key, None)
                self._symbol_last_attempt.pop(key, None)
                self._symbol_last_error.pop(key, None)
                self._primed_symbols.discard(key)
                self._priority_symbols.discard(key)
                self._last_eval.pop(key, None)
                self._last_update_ts.pop(key, None)
            now = time.time()
            for key in added:
                self._symbol_added_at[key] = now
                self._symbol_last_attempt.pop(key, None)
                self._symbol_last_error.pop(key, None)
                self._primed_symbols.discard(key)
        added_symbols = sorted(added)
        self._signal_wake()
        self._update_stream_symbols()

        if added_symbols:
            missing: list[str] = []
            for sym in added_symbols:
                try:
                    success = self._prime_symbol(sym)
                except Exception:
                    logger.debug(
                        "RTSC: error during prime for %s", sym, exc_info=True
                    )
                    success = False
                if not success:
                    missing.append(sym)
            if missing:
                self.mark_ready(False)
                for sym in missing:
                    try:
                        self.enqueue_refresh(sym)
                    except Exception:
                        logger.debug(
                            "RTSC: failed to enqueue refresh for %s after backfill miss",
                            sym,
                            exc_info=True,
                        )

    def configure_runtime(
        self,
        *,
        default_debounce_ms: int,
        debounce_overrides: Mapping[str, int],
        refresh_overrides: Mapping[str, float],
        circuit_breaker_threshold: int,
        circuit_breaker_window: float,
    ) -> None:
        """Update runtime knobs sourced from configuration."""

        default_seconds = max(0.0, float(default_debounce_ms) / 1000.0)
        with self._lock:
            self._default_debounce = default_seconds
            self._symbol_debounce = {
                self._key(symbol): max(0.0, float(value) / 1000.0)
                for symbol, value in debounce_overrides.items()
                if value is not None
            }
            self._symbol_refresh_overrides = {
                self._key(symbol): max(0.5, float(value))
                for symbol, value in refresh_overrides.items()
                if value is not None
            }
        with self._cb_lock:
            self._cb_threshold = max(1, int(circuit_breaker_threshold))
            self._cb_window = max(5.0, float(circuit_breaker_window))
        self._signal_wake()

    def handle_ws_update(
        self, symbol: str, event_type: str, data: Mapping[str, object]
    ) -> None:
        """Handle WebSocket events and trigger cache refreshes when relevant."""

        if not self.use_streams:
            return
        if not symbol:
            return
        if not isinstance(data, Mapping):
            return
        try:
            event = str(event_type or "").lower()
        except Exception:
            event = ""
        if event != "kline":
            return
        try:
            closed = bool(data.get("x"))
        except Exception:
            closed = False
        if not closed:
            return
        close_ts = data.get("T") or data.get("t") or data.get("closeTime")
        self.on_ws_bar_close(symbol, close_ts)
        loop = self._worker_loop
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.schedule_refresh(symbol), loop)
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            _run_async_allow_nested(self._refresh_symbol_via_rest(symbol))
        else:
            running_loop.create_task(self.schedule_refresh(symbol))

    def on_ws_bar_close(self, symbol: str, close_ts_ms: object | None) -> None:
        """Record a WebSocket bar close for ``symbol``."""

        if not self.use_streams:
            return
        if not symbol:
            return
        key = self._key(symbol)
        timestamp = None
        if close_ts_ms is not None:
            try:
                timestamp = float(close_ts_ms) / 1000.0
            except Exception:
                timestamp = None
        if timestamp is None:
            timestamp = time.time()
        self._last_ws_ts[key] = float(timestamp)
        log_event(
            logger,
            "ws_bar_close",
            symbol=key,
            ws_event="kline",
            close_ts_ms=close_ts_ms,
        )

    def _submit_bg(self, coro: Awaitable[Any]) -> asyncio.Future:
        """Submit a coroutine to the private RTSC background loop."""

        assert (
            self._loop is not None and self._loop_ready.is_set()
        ), "bg loop not ready"
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def enqueue_refresh(self, symbol: str) -> None:
        """Queue a REST refresh for ``symbol`` on the background loop."""

        key = self._key(symbol)
        if not getattr(self, "_loop_ready", None) or not self._loop_ready.is_set() or self._loop is None:
            self._pending_syms = getattr(self, "_pending_syms", [])
            self._pending_syms.append(key)
            # LOGGER.warning(f"[RTSC] queued (loop not ready): {key}")
            return

        async def _runner(sym: str) -> None:
            # LOGGER.warning(f"[RTSC] ENTER runner({sym})")
            sem = self._sem
            if sem is None:
                sem = asyncio.Semaphore(5)
                self._sem = sem
            async with sem:
                # LOGGER.warning(f"[RTSC] ENTER _refresh_symbol_via_rest({sym})")
                try:
                    await asyncio.wait_for(
                        self._refresh_symbol_via_rest(sym), timeout=10
                    )
                    if RTSC_DEBUG:
                        # LOGGER.warning(f"[RTSC] OK refresh({sym})")
                        pass
                except asyncio.TimeoutError:
                    logger.debug("[RTSC] TIMEOUT refresh(%s)", sym)
                except Exception as exc:
                    logger.debug("[RTSC] FAIL refresh(%s): %s", sym, exc)

        fut = self._submit_bg(_runner(key))
        # LOGGER.warning(f"[RTSC] submitted to bg loop: {key} ({fut})")

    async def schedule_refresh(self, symbol: str) -> None:
        """Non-blocking dispatcher: always spawns a refresh task."""

        key = self._key(symbol)

        with self._lock:
            if key not in self._symbols:
                logger.debug("[RTSC] schedule_refresh ignored for unknown symbol %s", key)
                return

        self._signal_wake()
        # logger.info(
        #     f"[RTSC] schedule_refresh({key}) FORCE_REST={os.getenv('RTSC_FORCE_REST','0')}"
        # )
        self.enqueue_refresh(key)

    def flush_pending(self) -> None:
        """Flush any symbols queued before the background loop was ready."""

        syms = getattr(self, "_pending_syms", [])
        if not syms:
            # LOGGER.warning("[RTSC] nothing to flush")
            return
        # LOGGER.warning(f"[RTSC] flushing {len(syms)} queued: {syms}")
        self._pending_syms = []
        for sym in syms:
            self.enqueue_refresh(sym)

    def _ws_is_stale(self, symbol: str) -> bool:
        """Return True when the last WS update for ``symbol`` is stale."""

        last = self._last_ws_ts.get(self._key(symbol))
        if last is None:
            logger.debug("RTSC: No WS timestamp for %s yet → treating as stale", symbol)
            return True
        age = datetime.now(timezone.utc).timestamp() - float(last)
        if age > RTSC_WS_STALE_AFTER_SEC:
            logger.debug(
                "RTSC: WS data for %s is stale (age=%.2fs > %.2fs)",
                symbol,
                age,
                RTSC_WS_STALE_AFTER_SEC,
            )
            return True
        return False

    def get(self, symbol: str) -> Optional[CachedSignal]:
        """Return the cached signal for ``symbol`` if available and fresh."""

        key = self._key(symbol)
        with self._lock:
            entry = self._cache.get(key)
        if entry is None:
            return None
        if not entry.is_fresh(self._stale_after):
            logger.debug("Cached signal for %s is stale (age=%.2fs).", key, entry.age())
            return None
        return entry

    def _evaluate_candles_sync(
        self, symbol: str, candles: Sequence[Sequence[object]]
    ) -> bool:
        """Evaluate ``candles`` synchronously and update internal cache state."""

        if isinstance(candles, pd.DataFrame):
            df = candles.copy()
        else:
            df = _klines_to_dataframe(candles)
        if df.empty:
            return False

        df = self._standardize_price_df(df)
        if df is None or df.empty:
            return False
        key = self._key(symbol)
        with self._lock:
            existing = self._cache.get(key)
        prev_age = existing.age() if existing is not None else None
        attempt_ts = time.time()
        return self._update_cache(
            key,
            df,
            attempt_ts=attempt_ts,
            prev_age=prev_age,
        )

    def symbols(self) -> Sequence[str]:
        """Return the current symbol universe snapshot."""

        with self._lock:
            return tuple(self._symbols)

    def update_context(self, *, sentiment_bias: Optional[str] = None) -> None:
        """Update shared context used during signal evaluation."""

        with self._lock:
            if sentiment_bias is not None:
                self._context["sentiment_bias"] = sentiment_bias

    def pending_diagnostics(self, limit: Optional[int] = None) -> List[Dict[str, object]]:
        """Return diagnostic metadata for symbols missing fresh cache entries."""

        now = time.time()
        with self._lock:
            symbols = tuple(self._symbols)
            cache_snapshot = dict(self._cache)
            added_at = dict(self._symbol_added_at)
            last_attempt = dict(self._symbol_last_attempt)
            last_error = dict(self._symbol_last_error)

        pending: List[Dict[str, object]] = []
        ready: List[str] = []
        max_display_age = max(self._stale_after, self._refresh_interval * 3)

        def _metric(raw: Optional[float]) -> Optional[Dict[str, float]]:
            if raw is None:
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            if value < 0.0:
                value = 0.0
            display = value
            if max_display_age > 0:
                display = min(display, max_display_age)
            return {"raw": value, "display": display}

        def _raw_value(metric: Optional[Dict[str, float]]) -> float:
            if not metric:
                return 0.0
            raw_value = metric.get("raw")
            if isinstance(raw_value, (int, float)):
                return float(raw_value)
            return 0.0
        for symbol in symbols:
            entry = cache_snapshot.get(symbol)
            if entry is not None and entry.is_fresh(self._stale_after):
                ready.append(symbol)
                continue
            waiting_for_metric = _metric(now - added_at[symbol]) if symbol in added_at else None
            stale_age_raw = entry.age() if entry is not None else None
            stale_age_metric = _metric(stale_age_raw)
            request_wait_metric = (
                _metric(now - last_attempt[symbol]) if symbol in last_attempt else None
            )
            error_msg: Optional[str] = None
            error_age_metric: Optional[Dict[str, float]] = None
            if symbol in last_error:
                err_ts, msg = last_error[symbol]
                error_msg = msg
                error_age_metric = _metric(now - err_ts)
            pending.append(
                {
                    "symbol": symbol,
                    "waiting_for": waiting_for_metric,
                    "stale_age": stale_age_metric,
                    "request_wait": request_wait_metric,
                    "last_error": error_msg,
                    "error_age": error_age_metric,
                }
            )

        pending.sort(
            key=lambda item: (
                _raw_value(item["waiting_for"]),
                _raw_value(item["stale_age"]),
            ),
            reverse=True,
        )
        if limit is not None:
            pending = pending[: int(limit)]
        snapshot = {
            "generated_at": float(now),
            "stale_after": float(self._stale_after),
            "refresh_interval": float(self._refresh_interval),
            "universe": list(symbols),
            "ready": ready,
            "pending": pending,
        }
        try:
            path = cache_diagnostics_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
        except Exception:
            logger.debug("Failed to persist cache diagnostics snapshot", exc_info=True)
        return pending

    def force_refresh(self, symbol: str, timeout: float = 15.0) -> bool:
        """
        Backward-compatible synchronous entry that now *runs the async REST refresh
        on the RTSC background loop* and waits for completion up to `timeout`.
        This makes legacy call sites behave correctly without changing their code.
        """

        import traceback

        # logger.warning(
        #     f"[RTSC] force_refresh WRAPPER CALLED for {{symbol}}; self={{id(self)}}\n"
        #     + "".join(traceback.format_stack(limit=6))
        # )

        import asyncio, threading, time

        # Make sure the background loop exists and is running
        if getattr(self, "_loop_ready", None) is None:
            self._loop_ready = threading.Event()

        loop = getattr(self, "_loop", None)
        if loop is None or not self._loop_ready.is_set() or not loop.is_running():
            bg_thread = getattr(self, "_bg_thread", None)
            if bg_thread is None or not bg_thread.is_alive():
                self._loop_ready.clear()

                def _run_loop() -> None:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self._loop = loop
                    self._sem = asyncio.Semaphore(5)
                    self._loop_ready.set()
                    # LOGGER.warning("[RTSC] bg loop READY [force_refresh]")
                    try:
                        loop.run_forever()
                    finally:
                        self._loop_ready.clear()
                        self._sem = None
                        self._loop = None
                        # LOGGER.warning("[RTSC] bg loop STOPPED [force_refresh]")
                        try:
                            loop.close()
                        except Exception:
                            logger.debug("[RTSC] error closing bg loop", exc_info=True)

                try:
                    self._bg_thread = threading.Thread(
                        target=_run_loop,
                        name="rtsc-bg",
                        daemon=True,
                    )
                    self._bg_thread.start()
                except Exception as e:
                    # logger.warning(
                    #     f"[RTSC] force_refresh({symbol}) could not spawn bg loop: {e}"
                    # )
                    return False

            # small wait for READY (avoid a race)
            for _ in range(100):  # up to ~1s in 10ms steps
                loop = getattr(self, "_loop", None)
                if loop is not None and self._loop_ready.is_set() and loop.is_running():
                    break
                time.sleep(0.01)

        if getattr(self, "_loop", None) is None or not self._loop.is_running():
            # logger.warning(f"[RTSC] force_refresh({symbol}) bg loop not running")
            return False

        async def _runner():
            # logger.warning(f"[RTSC] ENTER _refresh_symbol_via_rest({symbol}) [force_refresh]")
            ok = await self._refresh_symbol_via_rest(symbol)
            if ok:
                return True
            # logger.warning(
            #     f"[RTSC] REST force_refresh({symbol}) failed; retrying configured fetcher"
            # )
            return await self._refresh_symbol(symbol, force_rest=False)

        try:
            fut = asyncio.run_coroutine_threadsafe(_runner(), self._loop)
            result = fut.result(timeout=timeout)  # wait for REST path to complete
            if result:
                # logger.warning(f"[RTSC] OK force_refresh({symbol})")
                pass
            else:
                # logger.warning(f"[RTSC] FAIL force_refresh({symbol}): refresh returned False")
                pass
            return bool(result)
        except asyncio.TimeoutError:
            # logger.warning(f"[RTSC] TIMEOUT force_refresh({symbol}) after {timeout:.1f}s")
            return False
        except Exception as e:
            # logger.warning(f"[RTSC] FAIL force_refresh({symbol}): {e}")
            return False

    async def _worker(self) -> None:
        """Background coroutine that refreshes the cache in near real time."""

        loop = asyncio.get_running_loop()
        self._worker_loop = loop
        self._worker_loop_ready.set()
        async_wake = asyncio.Event()
        self._async_wake = async_wake
        # logger.info(
        #     "RTSC: worker started (interval=%.2fs, stale_after=%.2fs, max_concurrency=%d)",
        #     self._refresh_interval,
        #     self._stale_after,
        #     self._max_concurrency,
        # )
        sem = asyncio.Semaphore(self._max_concurrency)
        first = True
        try:
            while not self._stop_event.is_set():
                syms = list(self.symbols())
                with self._lock:
                    active_symbols = set(self._symbols)
                    self._priority_symbols &= active_symbols
                    priority = list(self._priority_symbols)
                    self._priority_symbols.clear()
                baseline_due = self._symbols_due(syms, first_run=first)
                seen: set[str] = set()
                due: List[str] = []
                for sym in priority:
                    if sym in active_symbols and sym not in seen:
                        due.append(sym)
                        seen.add(sym)
                for sym in baseline_due:
                    if sym not in seen:
                        due.append(sym)
                        seen.add(sym)
                # logger.info(
                #     "RTSC: due this cycle = %d / %d (priority=%d)",
                #     len(due),
                #     len(syms),
                #     len(priority),
                # )
                first = False

                async def run_one(sym: str) -> None:
                    async with sem:
                        try:
                            await self._refresh_symbol(sym)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:  # pragma: no cover - defensive logging
                            logger.exception("RTSC: unexpected error refreshing %s", sym)
                            self._record_refresh_error(
                                sym,
                                f"unexpected error: {exc}",
                                attempt_ts=time.time(),
                            )

                if due:
                    await asyncio.gather(
                        *(asyncio.create_task(run_one(sym)) for sym in due)
                    )

                if self._wake_event.is_set():
                    self._wake_event.clear()
                    async_wake.clear()
                    continue

                try:
                    await asyncio.wait_for(async_wake.wait(), timeout=self._refresh_interval)
                except asyncio.TimeoutError:
                    pass
                async_wake.clear()
        except asyncio.CancelledError:
            # logger.info("RTSC worker cancellation received")
            raise
        finally:
            self._async_wake = None
            self._worker_loop = None
            self._bg_task = None
            self._worker_loop_ready.clear()
            # logger.info("RTSC worker stopped")

    def _symbols_due(self, symbols: Sequence[str], *, first_run: bool) -> List[str]:
        """Return the subset of ``symbols`` requiring refresh this cycle."""

        if not symbols:
            return []
        now = time.time()
        with self._lock:
            cache_snapshot = dict(self._cache)
            primed_snapshot = set(self._primed_symbols)
            refresh_overrides = dict(self._symbol_refresh_overrides)

        due: List[str] = []
        for symbol in symbols:
            key = self._key(symbol)
            entry = cache_snapshot.get(key)
            if entry is None:
                due.append(key)
                continue
            if first_run and key not in primed_snapshot:
                due.append(key)
                continue
            interval = refresh_overrides.get(key, self._refresh_interval)
            if now - entry.updated_at >= interval:
                due.append(key)
        return due

    def _prepare_refresh(
        self, symbol: str
    ) -> Optional[Tuple[str, Optional[float], float]]:
        attempt_ts = time.time()
        key = self._key(symbol)
        with self._lock:
            entry = self._cache.get(key)
        prev_age = entry.age() if entry is not None else None
        # logger.info(
        #     "Refreshing symbol %s (age=%s, stale_after=%.1fs)",
        #     key,
        #     f"{prev_age:.1f}s" if prev_age is not None else "None",
        #     self._stale_after,
        # )
        if self._is_circuit_open(attempt_ts):
            log_event(
                logger,
                "cache_circuit_open",
                symbol=key,
                open_until=self._cb_open_until,
                threshold=self._cb_threshold,
                window=self._cb_window,
            )
            return None
        return key, prev_age, attempt_ts

    async def _refresh_symbol(
        self, symbol: str, *, force_rest: Optional[bool] = None
    ) -> bool:
        """Refresh a single symbol and return whether it succeeded."""

        key = self._key(symbol)

        rest_forced = RTSC_FORCE_REST if force_rest is None else bool(force_rest)
        rest_due_to_stale = False
        if not rest_forced and force_rest is not False:
            rest_due_to_stale = self._ws_is_stale(key)

        if rest_forced or rest_due_to_stale:
            rest_task = asyncio.create_task(self._refresh_symbol_via_rest(symbol))
            rest_success = await rest_task
            if rest_success or rest_forced:
                return rest_success
            # logger.info(
            #     "RTSC: REST refresh for %s failed; retrying configured fetcher",
            #     key,
            # )

        prepared = self._prepare_refresh(symbol)
        if prepared is None:
            return False
        key, prev_age, attempt_ts = prepared

        try:
            price_data = await self._price_fetcher(key)
        except Exception as exc:
            self._record_refresh_error(key, f"fetch error: {exc}", attempt_ts=attempt_ts)
            logger.debug("Price fetch failed for %s: %s", key, exc)
            return False

        standardized = self._standardize_price_df(price_data)
        if standardized is None or getattr(standardized, "empty", False):
            logger.debug("Price data for %s could not be standardised; attempting REST backfill.", key)
            primed = await asyncio.to_thread(self._prime_symbol, symbol)
            if primed:
                return True
            self._record_refresh_error(key, "no usable price data returned", attempt_ts=attempt_ts)
            return False

        try:
            rows = int(len(standardized))
        except Exception:
            rows = 0

        if rows < RTSC_EVALUATOR_MIN_BARS:
            logger.debug(
                "Price data for %s has only %d rows (<%d); forcing REST backfill.",
                key,
                rows,
                RTSC_EVALUATOR_MIN_BARS,
            )
            primed = await asyncio.to_thread(self._prime_symbol, symbol)
            if primed:
                return True
            self._record_refresh_error(
                key,
                f"insufficient candles ({rows}/{RTSC_EVALUATOR_MIN_BARS})",
                attempt_ts=attempt_ts,
            )
            return False

        return self._update_cache(
            key,
            standardized,
            attempt_ts=attempt_ts,
            prev_age=prev_age,
        )

    async def _refresh_symbol_via_rest(self, symbol: str) -> bool:
        """Fetch small recent klines via REST mirrors and update the cache."""

        tick_result = await asyncio.to_thread(self._tick_symbol, symbol)
        if tick_result:
            return True

        key = self._key(symbol)
        # LOGGER.warning(f"[RTSC] ENTER _refresh_symbol_via_rest({key})")

        prepared = self._prepare_refresh(symbol)
        if prepared is None:
            # LOGGER.warning(f"[RTSC] REST refresh aborted for {key} (circuit open)")
            return False

        key, _prev_age, attempt_ts = prepared

        existing_rows = 0
        existing_df: Optional[pd.DataFrame] = None
        with self._lock:
            cached_existing = self._cache.get(key)
            if cached_existing is not None:
                price_df = getattr(cached_existing, "price_data", None)
                if price_df is not None and not getattr(price_df, "empty", False):
                    existing_df = price_df.copy()
                    try:
                        existing_rows = len(existing_df)
                    except Exception:
                        existing_rows = 0

        df: Optional[pd.DataFrame] = None
        fetch_exc: Optional[Exception] = None

        rest_client = getattr(self, "rest", None)
        if rest_client is not None:
            try:
                raw_df = await asyncio.to_thread(
                    rest_fetch_latest_closed,
                    rest_client,
                    key,
                    RTSC_REST_INTERVAL,
                )
                df = self._standardize_price_df(raw_df)
            except Exception:
                logger.debug("RTSC: rest_fetch_latest_closed failed for %s", key, exc_info=True)

        if df is None:
            fetch_limit = RTSC_REST_LIMIT
            if existing_rows < REST_REQUIRED_MIN_BARS:
                fetch_limit = max(REST_WARMUP_BARS, RTSC_REST_LIMIT)
            try:
                df = await self._fetch_klines_rest_df(
                    key,
                    interval=RTSC_REST_INTERVAL,
                    limit=fetch_limit,
                    timeout=RTSC_REST_TIMEOUT,
                )
                if df is not None and not df.empty:
                    if RTSC_DEBUG:
                        # LOGGER.warning(f"[RTSC] REST fetch OK for {key}: {len(df)} bars")
                        pass
                else:
                    # LOGGER.warning(f"[RTSC] REST fetch empty for {key}")
                    pass
            except Exception as exc:  # pragma: no cover - debug logging only
                fetch_exc = exc
                # LOGGER.warning(f"[RTSC] FAIL refresh({key}): {exc}")

        if fetch_exc is not None:
            # LOGGER.warning(f"[RTSC] REST fetch exception for {key}: {fetch_exc}")
            self._record_refresh_error(
                key, f"REST fetch exception: {fetch_exc}", attempt_ts=attempt_ts
            )
            return False

        if df is None or df.empty:
            # LOGGER.warning(f"[RTSC] REST fetch produced no data for {key}")
            self._record_refresh_error(
                key, "REST fetch produced no data", attempt_ts=attempt_ts
            )
            return False

        if existing_df is not None and not existing_df.empty:
            try:
                combined = pd.concat([existing_df, df]).sort_index()
                combined = combined.loc[~combined.index.duplicated(keep="last")]
                if len(combined) > REST_WARMUP_BARS:
                    combined = combined.iloc[-REST_WARMUP_BARS:]

                # Guard against scenarios where the REST payload would shrink the
                # cached history (e.g. when only 1-2 candles are returned).  If the
                # existing cache already satisfied our minimum history requirement,
                # prefer that over a truncated snapshot so the evaluator always has
                # adequate context.
                combined_rows = 0
                try:
                    combined_rows = len(combined)
                except Exception:
                    combined_rows = 0
                if combined_rows < max(existing_rows, REST_REQUIRED_MIN_BARS):
                    combined = existing_df

                df = combined
            except Exception:
                logger.debug("RTSC: failed to merge REST snapshot for %s", key, exc_info=True)

        rows = 0
        try:
            rows = len(df)
        except Exception:
            rows = 0

        relaxed_min = self._relaxed_min_bars
        if rows < REST_REQUIRED_MIN_BARS:
            if rows < relaxed_min:
                primed = await asyncio.to_thread(self._prime_symbol, symbol)
                if primed:
                    return True
                self._record_refresh_error(
                    key,
                    f"REST refresh returned insufficient candles ({rows}/{REST_REQUIRED_MIN_BARS})",
                    attempt_ts=attempt_ts,
                    count_toward_breaker=False,
                )
                return False

            log_event(
                logger,
                "cache_rest_partial",
                symbol=key,
                rows=rows,
                required=REST_REQUIRED_MIN_BARS,
                relaxed=relaxed_min,
                method="rest_refresh",
            )

        score = self._quick_score(key, df)
        payload = {
            "df": df,
            "score": score,
            "source": "REST",
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
        }

        cached_score = 0.5 if score is None else float(score)
        direction: Optional[str]
        if cached_score > 0.5:
            direction = "long"
        elif cached_score < 0.5:
            direction = "short"
        else:
            direction = None

        try:
            cached = CachedSignal(
                symbol=key,
                score=cached_score,
                direction=direction,
                position_size=0.0,
                pattern="rest_quick_score",
                price_data=df,
                updated_at=time.time(),
                compute_latency=0.0,
            )
            cached.price_data.attrs["rest_meta"] = {
                "score": payload["score"],
                "source": payload["source"],
                "refreshed_at": payload["refreshed_at"],
            }
            logger.debug(
                "[RTSC] REST payload for %s: rows=%s score=%s",
                key,
                len(df),
                score,
            )

            with self._lock:
                self._cache[key] = cached
                self._last_eval[key] = cached
                self._last_update_ts[key] = cached.updated_at
                self._symbol_last_attempt[key] = cached.updated_at
                self._symbol_last_error.pop(key, None)
                self._primed_symbols.add(key)
        except Exception as exc:
            # LOGGER.warning(f"[RTSC] cache update failed for {key}: {exc}")
            self._record_refresh_error(
                key, f"REST cache update failed: {exc}", attempt_ts=attempt_ts
            )
            # LOGGER.warning(f"[RTSC] REST cache update FAIL for {key}")
            return False

        if RTSC_DEBUG:
            # LOGGER.warning(f"[RTSC] REST cache update OK for {key}")
            pass
        return True

    async def _fetch_klines_rest_df(
        self, symbol: str, interval: str, limit: int, timeout: float
    ) -> Optional[pd.DataFrame]:
        """Compatibility wrapper to keep debug logging focused on REST fetches."""

        return await self._fetch_klines_any(
            symbol, interval=interval, limit=limit, timeout=timeout
        )

    async def _fetch_klines_any(
        self, symbol: str, interval: str, limit: int, timeout: float
    ) -> Optional[pd.DataFrame]:
        """Try multiple Binance endpoints directly; log each attempt."""

        loop = asyncio.get_running_loop()
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        for base in BINANCE_MIRRORS:
            url = f"{base}/api/v3/klines"
            try:
                if RTSC_DEBUG:
                    # logger.info(f"[RTSC] REST mirror try {url} for {symbol}")
                    pass
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: requests.get(
                            url,
                            params=params,
                            timeout=timeout,
                            proxies={},
                        ),
                    ),
                    timeout=timeout + 1,
                )
                response.raise_for_status()
                raw = response.json()
                df = self._shape_klines_df(raw)
                if df is not None and not df.empty:
                    if RTSC_DEBUG:
                        # logger.info(
                        #     f"[RTSC] REST mirror OK {base} for {symbol} (n={len(df)})"
                        # )
                        pass
                    return df
                # logger.warning(f"[RTSC] REST mirror EMPTY {base} for {symbol}")
            except Exception as exc:
                logger.debug(
                    "[RTSC] REST mirror ERROR %s for %s: %s", base, symbol, exc
                )
        return None

    def _shape_klines_df(self, raw) -> Optional[pd.DataFrame]:
        if not raw:
            return None
        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        try:
            df = pd.DataFrame(raw, columns=cols)
        except Exception:
            df = pd.DataFrame(raw)
            df.columns = cols[: len(df.columns)]

        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "taker_buy_base",
            "taker_buy_quote",
            "number_of_trades",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.set_index("close_dt").sort_index()

        df["quote_volume"] = df.get("quote_asset_volume", 0.0).fillna(0.0)
        df["taker_buy_base"] = df.get("taker_buy_base", 0.0).fillna(0.0)
        df["taker_buy_quote"] = df.get("taker_buy_quote", 0.0).fillna(0.0)
        df["taker_sell_base"] = (df["volume"].fillna(0.0) - df["taker_buy_base"]).clip(lower=0.0)
        df["taker_sell_quote"] = (
            df["quote_volume"].fillna(0.0) - df["taker_buy_quote"]
        ).clip(lower=0.0)

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

    def _standardize_price_df(
        self, data: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Ensure DataFrames from helpers match the evaluator schema."""

        if data is None or getattr(data, "empty", True):
            return None

        attrs = dict(getattr(data, "attrs", {}))
        df = data.copy()
        if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_values("timestamp")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.set_index("timestamp")
        elif "close_time" in df.columns:
            df = df.sort_values("close_time")
            df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
            df = df.dropna(subset=["close_time"])
            df = df.set_index("close_time")
        elif "close_dt" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_values("close_dt")
            df["close_dt"] = pd.to_datetime(df["close_dt"], utc=True, errors="coerce")
            df = df.dropna(subset=["close_dt"])
            df = df.set_index("close_dt")

        if not isinstance(df.index, pd.DatetimeIndex):
            return None

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df[~df.index.isna()]
        if df.empty:
            return None

        df = df.sort_index()
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if df.empty:
            return None

        df = df.loc[~df.index.duplicated(keep="last")]

        if REST_INTERVAL_SECONDS:
            deltas = df.index.to_series().diff().dt.total_seconds()
            if not deltas.empty:
                deltas = deltas.fillna(REST_INTERVAL_SECONDS)
                df = df[deltas >= REST_INTERVAL_SECONDS]

        if REST_INTERVAL_SECONDS and not df.empty:
            last_close = df.index[-1]
            now = datetime.now(timezone.utc)
            if (now - last_close).total_seconds() < REST_INTERVAL_SECONDS:
                df = df.iloc[:-1]

        if df.empty:
            return None

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return None

        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

        alias_map = {
            "quote_volume": ["quote_volume", "quote_asset_volume", "qav"],
            "taker_buy_base": ["taker_buy_base", "taker_base_vol"],
            "taker_buy_quote": ["taker_buy_quote", "taker_quote_vol"],
            "number_of_trades": ["number_of_trades", "num_trades"],
        }

        for target, aliases in alias_map.items():
            if target in df.columns:
                df[target] = pd.to_numeric(df[target], errors="coerce")
                continue
            for alias in aliases:
                if alias in df.columns:
                    df[target] = pd.to_numeric(df[alias], errors="coerce")
                    break
            else:
                df[target] = 0.0

        df["quote_volume"] = df["quote_volume"].fillna(0.0)
        df["taker_buy_base"] = df["taker_buy_base"].fillna(0.0)
        df["taker_buy_quote"] = df["taker_buy_quote"].fillna(0.0)
        df["number_of_trades"] = df["number_of_trades"].fillna(0.0)

        if "taker_sell_base" not in df.columns:
            df["taker_sell_base"] = (
                df["volume"].fillna(0.0) - df["taker_buy_base"]
            ).clip(lower=0.0)
        else:
            df["taker_sell_base"] = pd.to_numeric(
                df["taker_sell_base"], errors="coerce"
            ).fillna(0.0)

        if "taker_sell_quote" not in df.columns:
            df["taker_sell_quote"] = (
                df["quote_volume"].fillna(0.0) - df["taker_buy_quote"]
            ).clip(lower=0.0)
        else:
            df["taker_sell_quote"] = pd.to_numeric(
                df["taker_sell_quote"], errors="coerce"
            ).fillna(0.0)

        desired_order = [
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
        ]
        available = [col for col in desired_order if col in df.columns]
        standardized = df[available].sort_index()
        if attrs:
            standardized.attrs.update(attrs)
        return standardized

    def _quick_score(self, symbol: str, df: pd.DataFrame) -> Optional[float]:
        try:
            last = df["close"].iloc[-1]
            prev = df["close"].iloc[-2] if len(df) > 1 else last
            return 0.6 if last > prev else 0.4 if last < prev else 0.5
        except Exception:
            return None

    def _update_cache(
        self,
        key: str,
        price_data: pd.DataFrame,
        *,
        attempt_ts: float,
        prev_age: Optional[float],
    ) -> bool:
        if price_data is None or getattr(price_data, "empty", True):
            _log(self, "warning", "_update_cache(): rejecting empty frame for %s", key)
            return False
        try:
            with self._lock:
                context = dict(self._context)
            sentiment_bias = str(context.get("sentiment_bias", "neutral"))
            eval_start = time.perf_counter()
            score, direction, position_size, pattern = self._evaluator(
                price_data,
                key,
                sentiment_bias=sentiment_bias,
            )
            latency = time.perf_counter() - eval_start
        except Exception as exc:
            self._record_refresh_error(key, "evaluation error", attempt_ts=attempt_ts)
            logger.exception("Signal evaluation failed for %s", key)
            self._register_error(attempt_ts)
            return False

        cached = CachedSignal(
            symbol=key,
            score=float(score),
            direction=direction,
            position_size=float(position_size),
            pattern=pattern,
            price_data=price_data,
            updated_at=time.time(),
            compute_latency=float(latency),
        )
        with self._lock:
            self._symbol_last_attempt[key] = attempt_ts
            self._cache[key] = cached
            self._symbol_last_error.pop(key, None)
            self._primed_symbols.add(key)
            self._last_eval[key] = cached
            self._last_update_ts[key] = cached.updated_at
        # logger.info(
        #     "RTSC: refreshed %s (prev_age=%s)",
        #     key,
        #     f"{prev_age:.1f}s" if prev_age is not None else "None",
        # )
        with self._lock:
            request_time = self._last_priority_request.get(key)
        wake_latency = (attempt_ts - request_time) if request_time else 0.0
        record_metric("signals_fresh", 1.0, labels={"symbol": key})
        record_metric("eval_ms", latency * 1000.0, labels={"symbol": key})
        record_metric("wake_latency_ms", wake_latency * 1000.0, labels={"symbol": key})
        log_event(
            logger,
            "cache_refresh_completed",
            symbol=key,
            compute_latency_ms=latency * 1000.0,
            wake_latency_ms=wake_latency * 1000.0,
            rows=len(price_data) if hasattr(price_data, "__len__") else None,
        )
        with self._cb_lock:
            self._cb_error_times.clear()
            self._cb_open_until = 0.0
        return True

    def _select_symbols_for_refresh(self, symbols: Sequence[str]) -> List[str]:
        """Return the subset of ``symbols`` that require a refresh cycle."""

        now = time.time()
        with self._lock:
            cache_snapshot = dict(self._cache)

        due: List[str] = []
        for symbol in symbols:
            key = self._key(symbol)
            entry = cache_snapshot.get(key)
            if entry is None:
                due.append(key)
                continue
            if now - entry.updated_at >= self._refresh_interval:
                due.append(key)
        return due

    def _record_refresh_error(
        self,
        symbol: str,
        message: str,
        *,
        attempt_ts: Optional[float] = None,
        count_toward_breaker: Optional[bool] = None,
    ) -> None:
        """Record the most recent refresh error for a symbol."""

        ts = attempt_ts or time.time()
        normalized_message = " ".join(str(message).strip().split())
        if len(normalized_message) > 160:
            normalized_message = f"{normalized_message[:157]}..."
        with self._lock:
            key = self._key(symbol)
            self._symbol_last_attempt[key] = ts
            self._symbol_last_error[key] = (ts, normalized_message)
        log_event(
            logger,
            "cache_refresh_error",
            symbol=self._key(symbol),
            message=normalized_message,
        )
        lower_msg = normalized_message.lower()
        benign = any(substr in lower_msg for substr in _CIRCUIT_BENIGN_SUBSTRINGS)
        should_count = count_toward_breaker if count_toward_breaker is not None else not benign
        if should_count:
            self._register_error(ts)

    def _register_error(self, timestamp: float) -> None:
        with self._cb_lock:
            window_start = timestamp - self._cb_window
            self._cb_error_times.append(timestamp)
            while self._cb_error_times and self._cb_error_times[0] < window_start:
                self._cb_error_times.popleft()
            if len(self._cb_error_times) >= self._cb_threshold:
                self._cb_open_until = max(self._cb_open_until, timestamp + self._cb_window)
                log_event(
                    logger,
                    "cache_circuit_tripped",
                    threshold=self._cb_threshold,
                    window=self._cb_window,
                    open_until=self._cb_open_until,
                )

    def _is_circuit_open(self, now: float) -> bool:
        with self._cb_lock:
            return now < self._cb_open_until

    def circuit_breaker_active(self) -> bool:
        """Return whether the evaluator circuit breaker is currently open."""

        return self._is_circuit_open(time.time())

