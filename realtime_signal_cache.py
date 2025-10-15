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
import logging
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


logger = logging.getLogger(__name__)

VERSION_TAG = "RTSC-2025-10-15-UMER-1"
logger.info("RTSC loaded: %s file=%s", VERSION_TAG, __file__)


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

    def start(self) -> None:
        """Launch the background refresh worker if it is not already running."""

        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="signal-cache", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait briefly for it to exit."""

        self._stop_event.set()
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def update_universe(self, symbols: Iterable[str]) -> None:
        """Update the symbol universe tracked by the cache."""

        normalized = {str(sym).upper() for sym in symbols if sym}
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

        key = str(symbol).upper()
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

    def _run_loop(self) -> None:
        """Background worker that refreshes the cache in near real time."""

        logger.info(
            "Starting real-time signal cache worker (interval=%.2fs, stale_after=%.2fs, max_concurrency=%d)",
            self._refresh_interval,
            self._stale_after,
            self._max_concurrency,
        )
        while not self._stop_event.is_set():
            loop_started = time.perf_counter()
            symbols = self.symbols()
            if symbols:
                with self._lock:
                    primed_snapshot = set(self._primed_symbols)
                symbols_to_prime = [sym for sym in symbols if sym not in primed_snapshot]
                if symbols_to_prime:
                    try:
                        asyncio.run(self._prime_symbols(symbols_to_prime))
                    except Exception:
                        logger.exception("Signal cache prime encountered an error")
                    finally:
                        with self._lock:
                            self._primed_symbols.update(symbols_to_prime)
                symbols_to_refresh = self._select_symbols_for_refresh(symbols)
                if symbols_to_refresh:
                    try:
                        asyncio.run(self._refresh_symbols(symbols_to_refresh))
                    except Exception:
                        logger.exception("Signal cache refresh encountered an error")
            # Wake up sooner if new symbols are pushed; otherwise wait for interval
            elapsed = time.perf_counter() - loop_started
            wait_time = max(0.0, self._refresh_interval - elapsed)
            self._wake_event.wait(timeout=wait_time)
            self._wake_event.clear()
        logger.info("Signal cache worker stopped")

    async def _refresh_symbols(self, symbols: Sequence[str]) -> None:
        """Fetch latest price data and evaluate signals for a batch of symbols."""

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _throttled(symbol: str) -> None:
            async with semaphore:
                try:
                    await self._refresh_symbol(symbol)
                except Exception:
                    logger.exception("Signal cache refresh encountered an unexpected error for %s", symbol)

        tasks = [asyncio.create_task(_throttled(sym)) for sym in symbols]
        await asyncio.gather(*tasks)

    async def _prime_symbols(self, symbols: Sequence[str]) -> None:
        """Run an initial refresh pass for ``symbols`` sequentially."""

        if not symbols:
            return
        logger.info("Signal cache first-run: priming %d symbols", len(symbols))
        for symbol in symbols:
            try:
                await self._refresh_symbol(symbol, propagate_errors=True)
            except Exception as exc:
                logger.warning("Prime failed for %s: %s", symbol, exc, exc_info=True)

    async def _refresh_symbol(self, symbol: str, *, propagate_errors: bool = False) -> bool:
        """Refresh a single symbol and return whether it succeeded."""

        attempt_ts = time.time()
        with self._lock:
            entry = self._cache.get(symbol)
        entry_age = entry.age() if entry is not None else float("nan")
        logger.info(
            "Refreshing symbol %s (age=%.1fs, stale_after=%.1fs)",
            symbol,
            entry_age,
            self._stale_after,
        )
        try:
            price_data = await self._price_fetcher(symbol)
        except Exception as exc:
            self._record_refresh_error(symbol, f"fetch error: {exc}", attempt_ts=attempt_ts)
            logger.debug("Price fetch failed for %s: %s", symbol, exc)
            if propagate_errors:
                raise
            return False

        if price_data is None or getattr(price_data, "empty", False):
            self._record_refresh_error(symbol, "no price data returned", attempt_ts=attempt_ts)
            logger.debug("No price data available for %s", symbol)
            if propagate_errors:
                raise ValueError("no price data returned")
            return False

        try:
            with self._lock:
                context = dict(self._context)
            sentiment_bias = str(context.get("sentiment_bias", "neutral"))
            eval_start = time.perf_counter()
            score, direction, position_size, pattern = self._evaluator(
                price_data,
                symbol,
                sentiment_bias=sentiment_bias,
            )
            latency = time.perf_counter() - eval_start
        except Exception as exc:
            self._record_refresh_error(symbol, "evaluation error", attempt_ts=attempt_ts)
            logger.exception("Signal evaluation failed for %s", symbol)
            if propagate_errors:
                raise
            return False

        cached = CachedSignal(
            symbol=symbol,
            score=float(score),
            direction=direction,
            position_size=float(position_size),
            pattern=pattern,
            price_data=price_data,
            updated_at=time.time(),
            compute_latency=float(latency),
        )
        with self._lock:
            self._symbol_last_attempt[symbol] = attempt_ts
            self._cache[symbol] = cached
            self._symbol_last_error.pop(symbol, None)
        return True

    def _select_symbols_for_refresh(self, symbols: Sequence[str]) -> List[str]:
        """Return the subset of ``symbols`` that require a refresh cycle."""

        now = time.time()
        with self._lock:
            cache_snapshot = dict(self._cache)

        due: List[str] = []
        for symbol in symbols:
            entry = cache_snapshot.get(symbol)
            if entry is None:
                due.append(symbol)
                continue
            if now - entry.updated_at >= self._refresh_interval:
                due.append(symbol)
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
            self._symbol_last_attempt[symbol] = ts
            self._symbol_last_error[symbol] = (ts, normalized_message)

