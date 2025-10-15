"""Real-time signal evaluation cache for the trading agent.

This module decouples heavy indicator computation from the synchronous
decision loop.  A background worker continuously refreshes price snapshots,
calls ``evaluate_signal`` and stores the results in-memory so the trading
loop can respond immediately when a trade opportunity appears.  By
precomputing signals we avoid the latency penalty of fetching candles and
deriving indicators right when an order needs to be submitted.
"""

from __future__ import annotations

import logging, os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)        # make it chatty
logger.propagate = True              # bubble up to root logger

VERSION_TAG = "RTSC-PRIME-UMER-2"
logger.warning("RTSC loaded: %s file=%s", VERSION_TAG, __file__)

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PriceFetcher = Callable[[str], Awaitable[pd.DataFrame | None]]
SignalEvaluator = Callable[..., Tuple[float, Optional[str], float, Optional[str]]]


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
    ) -> None:
        self._price_fetcher = price_fetcher
        self._evaluator = evaluator
        self._refresh_interval = max(0.5, float(refresh_interval))
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
        self._thread: Optional[threading.Thread] = None
        self._bg_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._context: Dict[str, object] = {}
        self._symbol_added_at: Dict[str, float] = {}
        self._symbol_last_attempt: Dict[str, float] = {}
        self._symbol_last_error: Dict[str, Tuple[float, str]] = {}
        self._primed_symbols: set[str] = set()

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
        self._wake_event.set()

    @property
    def refresh_interval(self) -> float:
        """Return the background refresh cadence for the cache."""

        return self._refresh_interval

    def _key(self, symbol: str) -> str:
        """Normalize ``symbol`` for internal storage."""

        return str(symbol).upper()

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
        self._wake_event.set()
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
            now = time.time()
            for key in added:
                self._symbol_added_at[key] = now
                self._symbol_last_attempt.pop(key, None)
                self._symbol_last_error.pop(key, None)
                self._primed_symbols.discard(key)
        self._wake_event.set()

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
        for symbol in symbols:
            entry = cache_snapshot.get(symbol)
            if entry is not None and entry.is_fresh(self._stale_after):
                continue
            waiting_for = None
            if symbol in added_at:
                waiting_for = max(0.0, now - added_at[symbol])
                if max_display_age > 0:
                    waiting_for = min(waiting_for, max_display_age)
            stale_age = entry.age() if entry is not None else None
            if stale_age is not None and max_display_age > 0:
                stale_age = min(stale_age, max_display_age)
            request_wait = None
            if symbol in last_attempt:
                request_wait = max(0.0, now - last_attempt[symbol])
                if max_display_age > 0:
                    request_wait = min(request_wait, max_display_age)
            error_msg: Optional[str] = None
            error_age: Optional[float] = None
            if symbol in last_error:
                err_ts, msg = last_error[symbol]
                error_msg = msg
                error_age = max(0.0, now - err_ts)
                if max_display_age > 0:
                    error_age = min(error_age, max_display_age)
            pending.append(
                {
                    "symbol": symbol,
                    "waiting_for": waiting_for,
                    "stale_age": stale_age,
                    "request_wait": request_wait,
                    "last_error": error_msg,
                    "error_age": error_age,
                }
            )

        pending.sort(
            key=lambda item: (
                item["waiting_for"] if item["waiting_for"] is not None else 0.0,
                item["stale_age"] if item["stale_age"] is not None else 0.0,
            ),
            reverse=True,
        )
        if limit is not None:
            pending = pending[: int(limit)]
        return pending

    async def _worker(self) -> None:
        """Background coroutine that refreshes the cache in near real time."""

        loop = asyncio.get_running_loop()
        self._loop = loop
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
                due = self._symbols_due(syms, first_run=first)
                logger.info("RTSC: due this cycle = %d / %d", len(due), len(syms))
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
                    continue

                await asyncio.sleep(self._refresh_interval)
        except asyncio.CancelledError:
            logger.info("RTSC worker cancellation received")
            raise
        finally:
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
            if now - entry.updated_at >= self._refresh_interval:
                due.append(key)
        return due

    async def _refresh_symbol(self, symbol: str) -> bool:
        """Refresh a single symbol and return whether it succeeded."""

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

