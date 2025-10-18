"""Real-time signal evaluation cache for the trading agent.

This module decouples heavy indicator computation from the synchronous
decision loop.  A background worker continuously refreshes price snapshots,
calls ``evaluate_signal`` and stores the results in-memory so the trading
loop can respond immediately when a trade opportunity appears.  By
precomputing signals we avoid the latency penalty of fetching candles and
deriving indicators right when an order needs to be submitted.
"""

from __future__ import annotations

import logging
from concurrent.futures import TimeoutError as FutureTimeout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)        # make it chatty
logger.propagate = True              # bubble up to root logger

VERSION_TAG = "RTSC-PRIME-UMER-2"
logger.warning("RTSC loaded: %s file=%s", VERSION_TAG, __file__)

# --- REST fallback imports / banner ---
import os, asyncio
from datetime import datetime, timezone
import pandas as pd
from binance.client import Client

# One shared public client; no API key needed for public endpoints
REST_CLIENT = Client("", "")

print(
    f"RTSC INIT v2025-10-17 file={__file__} loaded at {datetime.now(timezone.utc).isoformat()}"
)

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from observability import log_event, record_metric

PriceFetcher = Callable[[str], Awaitable[pd.DataFrame | None]]
SignalEvaluator = Callable[..., Tuple[float, Optional[str], float, Optional[str]]]

# --- Thread-pool wrappers: blocking python-binance calls run off the event loop


def _fetch_klines_sync(symbol: str, interval: str = "1m", limit: int = 200):
    """Blocking REST call using python-binance"""

    return REST_CLIENT.get_klines(symbol=symbol, interval=interval, limit=limit)


def _get_ticker_sync(symbol: str):
    return REST_CLIENT.get_symbol_ticker(symbol=symbol)


async def fetch_klines_with_timeout(
    symbol: str,
    interval: str = "1m",
    limit: int = 200,
    timeout: float = 15.0,
):
    """Run the blocking call in a thread pool with timeout protection"""

    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _fetch_klines_sync, symbol, interval, limit),
        timeout=timeout,
    )


async def fetch_ticker_with_timeout(symbol: str, timeout: float = 5.0):
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _get_ticker_sync, symbol),
        timeout=timeout,
    )


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
        self._lock = threading.Lock()
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

    def _signal_wake(self) -> None:
        """Wake the background worker if it is sleeping."""

        self._wake_event.set()
        loop = self._loop
        async_wake = self._async_wake
        if loop and loop.is_running() and async_wake is not None:
            loop.call_soon_threadsafe(async_wake.set)

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
        log_event(logger, "ws_bar_close", symbol=self._key(symbol), event=event)
        self.schedule_refresh(symbol)

    def schedule_refresh(self, symbol: str) -> None:
        """Request an immediate refresh for ``symbol`` when possible."""

        key = self._key(symbol)
        with self._lock:
            if key not in self._symbols:
                return
            now = time.time()
            debounce = self._symbol_debounce.get(key, self._default_debounce)
            last_request = self._last_priority_request.get(key)
            if last_request is not None and now - last_request < debounce:
                return
            self._last_priority_request[key] = now
            if key in self._priority_symbols:
                return
            self._priority_symbols.add(key)
        log_event(
            logger,
            "cache_refresh_requested",
            symbol=key,
            debounce_seconds=debounce,
        )
        self._signal_wake()

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
            rest_success = bool(asyncio.run(self._refresh_symbol_rest(key)))
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

        # Prefer REST while warming up; disable via RTSC_FORCE_REST=0
        if force_rest is not False:
            if force_rest is True or os.getenv("RTSC_FORCE_REST", "1") == "1":
                return await self._refresh_symbol_rest(symbol)

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

    async def _refresh_symbol_rest(self, symbol: str) -> bool:
        """REST fallback refresh using python-binance"""

        prepared = self._prepare_refresh(symbol)
        if prepared is None:
            return False

        key, prev_age, attempt_ts = prepared

        try:
            klines = await fetch_klines_with_timeout(key, "1m", 200, timeout=15.0)
            df = pd.DataFrame(
                klines,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_vol",
                    "trades",
                    "taker_base",
                    "taker_quote",
                    "ignore",
                ],
            )
            for column in ("open", "high", "low", "close", "volume"):
                df[column] = df[column].astype(float)
            for column in ("quote_vol", "taker_base", "taker_quote", "trades"):
                df[column] = pd.to_numeric(df[column], errors="coerce")
            df["ts"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

            self.log.info(
                "[REST] %s: fetched %d bars successfully", key, len(df)
            )

            df = df.dropna(subset=["ts"]).sort_values("ts")
            if df.empty:
                raise ValueError("no price data returned")

            indexed = df.set_index("ts")
            tz = getattr(indexed.index, "tz", None)
            if tz is not None:
                indexed.index = indexed.index.tz_convert("UTC").tz_localize(None)

            indexed["quote_volume"] = indexed["quote_vol"].fillna(0.0)
            indexed["taker_buy_base"] = indexed["taker_base"].fillna(0.0)
            indexed["taker_buy_quote"] = indexed["taker_quote"].fillna(0.0)
            indexed["taker_sell_base"] = (
                indexed["volume"].fillna(0.0) - indexed["taker_buy_base"]
            ).clip(lower=0.0)
            indexed["taker_sell_quote"] = (
                indexed["quote_volume"].fillna(0.0) - indexed["taker_buy_quote"]
            ).clip(lower=0.0)
            indexed["number_of_trades"] = indexed["trades"].fillna(0.0)

            price_data = indexed[
                [
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
            ].copy()
            price_data.attrs["rest_refreshed_at"] = datetime.now(timezone.utc)

            success = self._update_cache(
                key,
                price_data,
                attempt_ts=attempt_ts,
                prev_age=prev_age,
            )
            if success and not df.empty:
                self.log.info(
                    "CachedSignal updated for %s (REST fallback OK, rows=%d, last=%s)",
                    key,
                    len(df),
                    df["ts"].iloc[-1],
                )
            return success
        except Exception as exc:
            self.log.error(
                "[REST] %s: %s: %s",
                key,
                type(exc).__name__,
                exc,
            )
            self._record_refresh_error(
                key, f"REST fetch error: {exc}", attempt_ts=attempt_ts
            )
            logger.debug(
                "REST fallback price fetch failed for %s: %s", key, exc, exc_info=True
            )
            return False

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

