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
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests  # used for mirror fallback (no proxies)

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)        # make it chatty
logger.propagate = True              # bubble up to root logger

VERSION_TAG = "RTSC-PRIME-UMER-2"
logger.warning("RTSC loaded: %s file=%s", VERSION_TAG, __file__)

print(
    f"RTSC INIT v2025-10-17 file={__file__} loaded at {datetime.now(timezone.utc).isoformat()}"
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

# --- BEGIN: RTSC warmup/refill core integration --------------------------------

# ===== Config (env tunable; safe defaults) =====
READY_MIN_FRACTION  = float(os.getenv("READY_MIN_FRACTION", "0.80"))   # % of symbols with fresh data
WARMUP_MAX_SECONDS  = int(os.getenv("WARMUP_MAX_SECONDS", "90"))        # watchdog ceiling
PRIME_LIMIT_MINUTES = int(os.getenv("PRIME_LIMIT_MINUTES", "300"))      # bars for initial prime
FRESHNESS_SECONDS   = int(os.getenv("FRESHNESS_SECONDS", "120"))        # what is considered "fresh"
USE_WS_PRICES       = os.getenv("USE_WS_PRICES", "1").lower() in ("1","true","yes")
ENABLE_PRIME        = os.getenv("ENABLE_RTSCC_PRIME", "1").lower() in ("1","true","yes")
ENABLE_REFRESH      = os.getenv("ENABLE_RTSCC_REFRESH", "1").lower() in ("1","true","yes")

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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
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
        self._cb_threshold = 5
        self._cb_window = 30.0
        self._cb_error_times: "deque[float]" = deque()
        self._cb_open_until = 0.0
        self._cb_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._ws_bridge: Optional[object] = None
        self._ws_callback_registered = False
        self._last_ws_ts: Dict[str, float] = {}
        self._refresh_sem = asyncio.Semaphore(5)

        existing_rest = getattr(self, "rest", None)
        if existing_rest is not None:
            self._rest_client = existing_rest
        else:
            self._rest_client = self._init_rest_client()
            if self._rest_client is not None:
                self.rest = self._rest_client
            else:
                self.rest = None

        if RTSC_FORCE_REST:
            logger.warning(
                "RealTimeSignalCache: RTSC_FORCE_REST=1 → routing refreshes through REST"
            )

        # --- warmup/refill integration state ---
        self._last_eval: Dict[str, Any] = getattr(self, "_last_eval", {})
        self._last_update_ts: Dict[str, float] = getattr(self, "_last_update_ts", {})
        self._ready = getattr(self, "_ready", False)

        evaluator_ref = getattr(self, "evaluator", None)
        if evaluator_ref is not None and not callable(evaluator_ref):
            self._evaluator = evaluator_ref

        self._enable_prime = ENABLE_PRIME
        self._enable_refresh = ENABLE_REFRESH

        _log(
            self,
            "info",
            "RTSC core warmup enabled (prime=%s, refresh=%s, ws=%s | ready>=%.0f%%, ceil=%ss, bars=%dm)",
            self._enable_prime,
            self._enable_refresh,
            USE_WS_PRICES,
            READY_MIN_FRACTION * 100,
            WARMUP_MAX_SECONDS,
            PRIME_LIMIT_MINUTES,
        )

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
            logger.warning(
                "RTSC: python-binance not available; REST fallback disabled."
            )
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
            if HTTP_PROXY or HTTPS_PROXY:
                logger.info(
                    "RTSC: REST client initialised (proxies handled by requests layer)."
                )
            else:
                logger.info("RTSC: REST client initialised.")
            return client
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("RTSC: Failed to initialise REST client: %s", exc)
            return None

    def _signal_wake(self) -> None:
        """Wake the background worker if it is sleeping."""

        self._wake_event.set()
        loop = self._loop
        async_wake = self._async_wake
        if loop and loop.is_running() and async_wake is not None:
            loop.call_soon_threadsafe(async_wake.set)

    def _on_rest_refresh_done(self, key: str, task: asyncio.Task[Any]) -> None:
        """Cleanup callback for REST refresh tasks to release bookkeeping state."""

        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug("[RTSC] REST refresh task for %s cancelled", key)
        except Exception as exc:
            logger.warning("[RTSC] REST refresh task for %s raised: %s", key, exc)
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

    # ---------- Public API you’re adding ----------

    def prime(self):
        """One-shot initial prime across symbols using REST (with project-first, then fallback)."""

        if not self._enable_prime:
            _log(self, "info", "prime(): disabled by config.")
            return

        rest = _get_rest(self)
        if rest is None:
            _log(self, "error", "prime(): no REST/price_fetcher found; skipping.")
            return

        for _ in range(60):  # up to 30s
            evaluator = getattr(self, "_evaluator", None)
            if evaluator is None:
                time.sleep(0.5)
                continue
            if hasattr(evaluator, "evaluate_from_klines") or callable(evaluator):
                break
            time.sleep(0.5)

        evaluator = getattr(self, "_evaluator", None)
        if evaluator is None:
            _log(self, "error", "prime(): evaluator not ready after 30s; skipping prime.")
            return

        for s in _iter_symbols(self):
            try:
                candles = _fetch_candles(rest, s, PRIME_LIMIT_MINUTES)
                if not candles or len(candles) < 50:
                    _log(
                        self,
                        "warning",
                        "prime(): insufficient candles for %s (got %s).",
                        s,
                        len(candles) if candles else 0,
                    )
                    continue

                if hasattr(evaluator, "evaluate_from_klines"):
                    ev = evaluator.evaluate_from_klines(s, candles)  # type: ignore[attr-defined]
                    with self._lock:
                        self._last_eval[s] = ev
                        self._last_update_ts[s] = time.time()
                else:
                    if not self._evaluate_candles_sync(s, candles):
                        continue
                time.sleep(0.02)
            except Exception as e:  # pragma: no cover - defensive logging
                _log(self, "warning", "prime(): error for %s: %s", s, e)
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
            if fraction >= READY_MIN_FRACTION:
                self.mark_ready()
                _log(
                    self,
                    "info",
                    "refresh(): threshold met (%.0f%% fresh).",
                    fraction * 100,
                )
                break

            time.sleep(0.5)

    def warmup_watchdog(self):
        """Ensure we proceed after a ceiling even if freshness is not met."""

        start = time.time()
        while not self.is_ready():
            frac = _fresh_fraction(self)
            elapsed = time.time() - start
            if frac >= READY_MIN_FRACTION:
                _log(self, "info", "watchdog: threshold met (%.0f%% fresh).", frac * 100)
                self.mark_ready()
                break
            if elapsed >= WARMUP_MAX_SECONDS:
                _log(
                    self,
                    "warning",
                    "watchdog: ceiling reached (%ds). Proceeding with %.0f%% fresh.",
                    WARMUP_MAX_SECONDS,
                    frac * 100,
                )
                self.mark_ready()
                break
            time.sleep(1.0)

    # ---------- Small adapters to fit the above ----------
    # If your class already has these, keep your versions.

    def is_ready(self) -> bool:
        return bool(getattr(self, "_ready", False))

    def mark_ready(self):
        self._ready = True

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

        if getattr(self, "_bg_task", None) and not self._bg_task.done():
            logger.info("RTSC: worker already running")
            return
        if getattr(self, "_thread", None) and self._thread.is_alive():
            logger.info("RTSC: worker already running (thread)")
            return
        self._stop_event.clear()
        self._wake_event.clear()
        self._ready = False
        try:
            loop = asyncio.get_running_loop()
            logger.info("RTSC: starting worker on running loop")
            self._bg_task = loop.create_task(self._worker())
        except RuntimeError:
            logger.info("RTSC: no running loop; starting worker in daemon thread")

            def runner() -> None:
                asyncio.run(self._worker())

            self._thread = threading.Thread(target=runner, daemon=True, name="rtsc-worker")
            self._thread.start()

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
        loop = self._loop
        if task and loop and not task.done():
            loop.call_soon_threadsafe(task.cancel)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._bg_task = None
        self._thread = None
        self._loop = None
        self._ready = False

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
        self._signal_wake()
        self._update_stream_symbols()

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
        loop = self._loop
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

    async def schedule_refresh(self, symbol: str) -> None:
        """Non-blocking dispatcher: always spawns a refresh task."""

        key = self._key(symbol)

        with self._lock:
            if key not in self._symbols:
                logger.debug("[RTSC] schedule_refresh ignored for unknown symbol %s", key)
                return

        self._signal_wake()
        logger.info(f"[RTSC] schedule_refresh({key}) FORCE_REST={RTSC_FORCE_REST}")

        async def _runner() -> None:
            async with self._refresh_sem:
                try:
                    result = await asyncio.wait_for(
                        self._refresh_symbol_via_rest(key),
                        timeout=RTSC_REST_TIMEOUT + 2,
                    )
                    if not result:
                        raise RuntimeError("REST refresh returned no data")
                    logger.info(f"[RTSC] OK refresh({key})")
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[RTSC] TIMEOUT refresh({key}) after {RTSC_REST_TIMEOUT + 2:.1f}s"
                    )
                except Exception as exc:
                    logger.warning(f"[RTSC] FAIL refresh({key}): {exc}")

        asyncio.create_task(_runner())

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
        return pending

    def force_refresh(self, symbol: str, *, timeout: float = 15.0) -> bool:
        """Synchronously refresh ``symbol`` to break through prolonged warm-ups.

        When the asynchronous worker struggles to prime a symbol (for example
        because repeated fetch attempts hit transient API failures), the agent
        can call this helper to perform a blocking refresh.  The call prefers
        to schedule work on the background loop when it is running, falling
        back to executing the refresh in a dedicated event loop otherwise.

        Parameters
        ----------
        symbol:
            Trading pair to refresh.
        timeout:
            Maximum time (seconds) to wait for the background loop to finish
            the refresh when it is already running.  This is ignored when the
            worker is offline and the method spins up a temporary event loop
            instead.
        """

        key = self._key(symbol)
        loop = self._loop
        if loop and loop.is_running():
            logger.info("RTSC: force-refresh scheduling %s on worker loop", key)
            future = asyncio.run_coroutine_threadsafe(self._refresh_symbol(key), loop)
            try:
                return bool(future.result(timeout))
            except FutureTimeout:
                future.cancel()
                logger.warning(
                    "RTSC: force refresh for %s timed out after %.1fs", key, timeout
                )
                return False
            except Exception:
                logger.exception("RTSC: force refresh for %s failed on worker loop", key)
                return False

        logger.info(
            "RTSC: force-refresh running %s on temporary event loop via REST fallback",
            key,
        )
        try:
            rest_success = bool(asyncio.run(self._refresh_symbol_via_rest(key)))
        except Exception:
            logger.exception(
                "RTSC: REST fallback force refresh for %s failed", key
            )
            rest_success = False
        if rest_success:
            return True

        logger.info(
            "RTSC: REST fallback failed for %s; retrying with configured fetcher",
            key,
        )
        try:
            return bool(asyncio.run(self._refresh_symbol(key, force_rest=False)))
        except Exception:
            logger.exception("RTSC: force refresh for %s failed via synchronous path", key)
            return False

    async def _worker(self) -> None:
        """Background coroutine that refreshes the cache in near real time."""

        loop = asyncio.get_running_loop()
        self._loop = loop
        async_wake = asyncio.Event()
        self._async_wake = async_wake
        logger.info(
            "RTSC: worker started (interval=%.2fs, stale_after=%.2fs, max_concurrency=%d)",
            self._refresh_interval,
            self._stale_after,
            self._max_concurrency,
        )
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
                logger.info(
                    "RTSC: due this cycle = %d / %d (priority=%d)",
                    len(due),
                    len(syms),
                    len(priority),
                )
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
            logger.info("RTSC worker cancellation received")
            raise
        finally:
            self._async_wake = None
            self._loop = None
            self._bg_task = None
            logger.info("RTSC worker stopped")

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
        logger.info(
            "Refreshing symbol %s (age=%s, stale_after=%.1fs)",
            key,
            f"{prev_age:.1f}s" if prev_age is not None else "None",
            self._stale_after,
        )
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
            logger.info(
                "RTSC: REST refresh for %s failed; retrying configured fetcher",
                key,
            )

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

        if price_data is None or getattr(price_data, "empty", False):
            self._record_refresh_error(key, "no price data returned", attempt_ts=attempt_ts)
            logger.debug("No price data available for %s", key)
            return False

        return self._update_cache(
            key,
            price_data,
            attempt_ts=attempt_ts,
            prev_age=prev_age,
        )

    async def _refresh_symbol_via_rest(self, symbol: str) -> bool:
        """Fetch small recent klines via REST mirrors and update the cache."""

        key = self._key(symbol)
        logger.info(f"[RTSC] ENTER _refresh_symbol_via_rest({key})")
        try:
            df = await self._fetch_klines_any(
                key,
                interval=RTSC_REST_INTERVAL,
                limit=RTSC_REST_LIMIT,
                timeout=RTSC_REST_TIMEOUT,
            )
        except Exception as exc:
            logger.warning(f"[RTSC] REST fetch exception for {key}: {exc}")
            return False
        if df is None or df.empty:
            logger.warning(f"[RTSC] REST fetch produced no data for {key}")
            return False

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
            logger.warning(f"[RTSC] cache update failed for {key}: {exc}")
            return False

        logger.info(f"[RTSC] cache updated for {key} via REST")
        return True

    async def _fetch_klines_any(
        self, symbol: str, interval: str, limit: int, timeout: float
    ) -> Optional[pd.DataFrame]:
        """Try multiple Binance endpoints directly; log each attempt."""

        loop = asyncio.get_running_loop()
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        for base in BINANCE_MIRRORS:
            url = f"{base}/api/v3/klines"
            try:
                logger.info(f"[RTSC] REST mirror try {url} for {symbol}")
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
                    logger.info(
                        f"[RTSC] REST mirror OK {base} for {symbol} (n={len(df)})"
                    )
                    return df
                logger.warning(f"[RTSC] REST mirror EMPTY {base} for {symbol}")
            except Exception as exc:
                logger.warning(f"[RTSC] REST mirror ERROR {base} for {symbol}: {exc}")
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
            "qav",
            "num_trades",
            "taker_base_vol",
            "taker_quote_vol",
            "ignore",
        ]
        try:
            df = pd.DataFrame(raw, columns=cols)
        except Exception:
            df = pd.DataFrame(raw)
            df.columns = cols[: len(df.columns)]
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df.set_index("close_dt").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

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
        logger.info(
            "RTSC: refreshed %s (prev_age=%s)",
            key,
            f"{prev_age:.1f}s" if prev_age is not None else "None",
        )
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
        self, symbol: str, message: str, *, attempt_ts: Optional[float] = None
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

