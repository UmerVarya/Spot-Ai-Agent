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
from typing import Awaitable, Callable, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd


logger = logging.getLogger(__name__)


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
    ) -> None:
        self._price_fetcher = price_fetcher
        self._evaluator = evaluator
        self._refresh_interval = max(0.5, float(refresh_interval))
        self._stale_after = float(stale_after) if stale_after is not None else self._refresh_interval * 3
        self._symbols: set[str] = set()
        self._cache: Dict[str, CachedSignal] = {}
        self._symbol_requested_at: Dict[str, float] = {}
        self._symbol_errors: Dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._context: Dict[str, object] = {}

    @property
    def stale_after(self) -> float:
        """Maximum allowed age (seconds) for cached signals."""

        return self._stale_after

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
        now = time.time()
        with self._lock:
            current = set(self._symbols)
            if normalized == current:
                return
            removed = current - normalized
            added = normalized - current
            self._symbols = normalized
            for key in removed:
                self._cache.pop(key, None)
                self._symbol_requested_at.pop(key, None)
                self._symbol_errors.pop(key, None)
            for key in added:
                self._symbol_requested_at[key] = now
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

    def describe_symbol(self, symbol: str) -> Dict[str, object]:
        """Return diagnostic details for ``symbol``.

        The response includes whether a cached entry exists, its age, how long the
        symbol has been tracked, and the most recent error (if any).  The
        information is intended for logging and troubleshooting and should not be
        used to drive trading decisions directly.
        """

        key = str(symbol).upper()
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            requested_at = self._symbol_requested_at.get(key)
            error = self._symbol_errors.get(key)

        info: Dict[str, object] = {"symbol": key}
        if entry is None:
            info["state"] = "missing"
            info["age"] = None
        else:
            age = now - entry.updated_at
            info["age"] = age
            info["state"] = "fresh" if age <= self._stale_after else "stale"

        if requested_at is not None:
            info["since_requested"] = now - requested_at

        if error is not None:
            message, ts = error
            info["last_error"] = message
            info["last_error_age"] = now - ts

        return info

    def _run_loop(self) -> None:
        """Background worker that refreshes the cache in near real time."""

        logger.info(
            "Starting real-time signal cache worker (interval=%.2fs, stale_after=%.2fs)",
            self._refresh_interval,
            self._stale_after,
        )
        while not self._stop_event.is_set():
            loop_started = time.perf_counter()
            symbols = self.symbols()
            if symbols:
                try:
                    asyncio.run(self._refresh_symbols(symbols))
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

        tasks = [self._price_fetcher(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                message = f"price fetch failed: {result}"
                logger.debug("Price fetch failed for %s: %s", symbol, result)
                self._record_error(symbol, message)
                continue
            if result is None or result.empty:
                message = "price data unavailable"
                logger.debug("No price data available for %s", symbol)
                self._record_error(symbol, message)
                continue
            try:
                with self._lock:
                    context = dict(self._context)
                sentiment_bias = str(context.get("sentiment_bias", "neutral"))
                eval_start = time.perf_counter()
                score, direction, position_size, pattern = self._evaluator(
                    result,
                    symbol,
                    sentiment_bias=sentiment_bias,
                )
                latency = time.perf_counter() - eval_start
            except Exception as exc:
                logger.exception("Signal evaluation failed for %s", symbol)
                self._record_error(symbol, f"evaluation failed: {exc}")
                continue
            cached = CachedSignal(
                symbol=symbol,
                score=float(score),
                direction=direction,
                position_size=float(position_size),
                pattern=pattern,
                price_data=result,
                updated_at=time.time(),
                compute_latency=float(latency),
            )
            with self._lock:
                self._cache[symbol] = cached
                self._symbol_errors.pop(symbol, None)

    def _record_error(self, symbol: str, message: str) -> None:
        """Store the latest error encountered while refreshing ``symbol``."""

        key = str(symbol).upper()
        with self._lock:
            self._symbol_errors[key] = (message, time.time())

