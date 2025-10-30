"""Minimal Binance WebSocket bridge for real-time price updates.

This module exposes :class:`WSPriceBridge`, a light wrapper that runs the
Binance combined stream in a dedicated thread and event loop.  Callers
register callbacks for klines, mini tickers and (optionally) book ticker
updates; the bridge handles reconnections with exponential backoff and
normalises symbols to upper case for downstream consumers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import websockets

import requests

from observability import log_event, record_metric

BINANCE_WS = "wss://stream.binance.com:9443/stream"
MAX_STREAMS_PER_COMBINED = max(1, int(os.getenv("BINANCE_MAX_STREAMS", "200")))
COMBINED_BASE = os.getenv(
    "WS_COMBINED_BASE",
    "wss://stream.binance.com:9443/stream?streams=",
)

logger = logging.getLogger(__name__)

def _stream_names(
    symbols: Iterable[str],
    want_kline_1m: bool = True,
    want_ticker: bool = False,
    want_book: bool = False,
) -> List[str]:
    names = []
    for sym in symbols:
        s = sym.lower()
        if want_kline_1m:
            names.append(f"{s}@kline_1m")
        if want_ticker:
            names.append(f"{s}@ticker")
        if want_book:
            names.append(f"{s}@bookTicker")
    return names


KlineCallback = Callable[[str, str, Dict[str, Any]], None]
TickerCallback = Callable[[str, Dict[str, Any]], None]
BookTickerCallback = Callable[[str, Dict[str, Any]], None]


class WSPriceBridge:
    """Run Binance WebSocket streams and dispatch updates to callbacks."""

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        kline_interval: str = "1m",
        on_kline: Optional[KlineCallback] = None,
        on_ticker: Optional[TickerCallback] = None,
        on_book_ticker: Optional[BookTickerCallback] = None,
        on_stale: Optional[Callable[[float], None]] = None,
        heartbeat_timeout: float = 10.0,
        server_time_sync_interval: float = 120.0,
        max_retries: int = 10,
    ) -> None:
        self._symbols: List[str] = self._normalise_symbols(symbols)
        self._kline_interval = str(kline_interval or "1m").strip()
        if not self._kline_interval:
            self._kline_interval = "1m"
        self._on_kline = on_kline
        self._on_ticker = on_ticker
        self._on_book_ticker = on_book_ticker
        self._on_stale = on_stale
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._resubscribe = threading.Event()
        self._symbols_lock = threading.Lock()
        self._heartbeat_timeout = max(5.0, float(heartbeat_timeout))
        self._last_messages: Dict[int, float] = {}
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._time_sync_interval = max(30.0, float(server_time_sync_interval))
        self._time_sync_stop = threading.Event()
        self._time_sync_thread: Optional[threading.Thread] = None
        self._max_retries = max(1, int(max_retries))
        self._callback_lock = threading.Lock()
        self._extra_callbacks: List[
            Callable[[str, str, Dict[str, Any]], None]
        ] = []
        self._tasks: List[asyncio.Task] = []
        self._urls: List[str] = []
        self._combined_base = COMBINED_BASE
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._conn_lock: Optional[asyncio.Lock] = None
        self.logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background WebSocket thread if it is not running."""

        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="binance-ws-bridge", daemon=True
        )
        self._thread.start()
        if self._time_sync_thread is None or not self._time_sync_thread.is_alive():
            self._time_sync_stop.clear()
            self._time_sync_thread = threading.Thread(
                target=self._time_sync_loop,
                name="binance-time-sync",
                daemon=True,
            )
            self._time_sync_thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Stop the WebSocket bridge and wait for the thread to exit."""

        self._stop.set()
        self._resubscribe.set()
        self._cancel_ws_tasks()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self._loop = None
        self._time_sync_stop.set()
        if self._time_sync_thread and self._time_sync_thread.is_alive():
            self._time_sync_thread.join(timeout=timeout)
        self._time_sync_thread = None

    def update_symbols(self, symbols: Iterable[str]) -> None:
        """Update the subscribed symbols."""

        normalised = self._normalise_symbols(symbols)
        with self._symbols_lock:
            if normalised == self._symbols:
                return
            self._symbols = normalised
        if os.getenv("WS_DEBUG", "0") == "1":
            logger.info("WS bridge subscribing to %d symbols", len(normalised))
        self._resubscribe.set()
        self._cancel_ws_tasks()
        self.start()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(lambda: None)

    def register_callback(
        self, callback: Callable[[str, str, Dict[str, Any]], None]
    ) -> None:
        """Register an external callback for stream events."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._callback_lock:
            if callback in self._extra_callbacks:
                return
            had_callbacks = bool(self._extra_callbacks)
            self._extra_callbacks.append(callback)
        if not had_callbacks and self._on_kline is None:
            self._trigger_resubscribe()

    def unregister_callback(
        self, callback: Callable[[str, str, Dict[str, Any]], None]
    ) -> None:
        with self._callback_lock:
            try:
                self._extra_callbacks.remove(callback)
                has_callbacks = bool(self._extra_callbacks)
            except ValueError:
                return
        if not has_callbacks and self._on_kline is None:
            self._trigger_resubscribe()

    def _emit_callbacks(
        self, symbol: str, event_type: str, payload: Mapping[str, Any]
    ) -> None:
        with self._callback_lock:
            callbacks = list(self._extra_callbacks)
        if not callbacks:
            return
        for callback in callbacks:
            try:
                callback(symbol, event_type, dict(payload))
            except Exception:
                logger.debug("WS bridge external callback failed", exc_info=True)

    def _has_external_callbacks(self) -> bool:
        with self._callback_lock:
            return bool(self._extra_callbacks)

    def _trigger_resubscribe(self) -> None:
        self._resubscribe.set()
        self._cancel_ws_tasks()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(lambda: None)

    def _cancel_ws_tasks(self) -> None:
        if not self._tasks:
            return
        tasks = list(self._tasks)
        loop = self._loop
        if loop and loop.is_running():

            def _cancel() -> None:
                tasks = list(self._tasks)
                for task in tasks:
                    if task is not None and not task.done():
                        task.cancel()
                self._tasks = []

            loop.call_soon_threadsafe(_cancel)
        else:
            for task in tasks:
                task.cancel()
            self._tasks = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        # Ensure the connection lock is bound to the lifetime of this loop.
        self._conn_lock = asyncio.Lock()
        loop.create_task(self._ws_main())
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:  # pragma: no cover - defensive cleanup
                    pass
            loop.close()
            # Avoid holding a lock tied to a closed loop across restarts.
            self._conn_lock = None

    async def _ws_main(self) -> None:
        while not self._stop.is_set():
            self._resubscribe.clear()
            symbols = self._current_symbols()
            if not symbols:
                self._last_messages.clear()
                await asyncio.sleep(1.0)
                continue

            if self._heartbeat_task is None or self._heartbeat_task.done():
                if self._heartbeat_task is not None and self._heartbeat_task.done():
                    try:
                        self._heartbeat_task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.debug(
                            "WS bridge heartbeat watcher finished with error",
                            exc_info=True,
                        )
                self._heartbeat_task = asyncio.create_task(self._heartbeat_watch())
                self._heartbeat_task.add_done_callback(self._on_heartbeat_done)

            has_external_callbacks = self._has_external_callbacks()
            flags = {
                "want_kline_1m": (
                    self._on_kline is not None or has_external_callbacks
                )
                and (self._kline_interval or "1m").strip().lower() == "1m",
                "want_ticker": self._on_ticker is not None,
                "want_book": self._on_book_ticker is not None,
            }

            urls = self._build_combined_urls(symbols, flags)
            if not urls:
                self.logger.info(
                    "WSPriceBridge: no streams to subscribe; skipping connect."
                )
                await asyncio.sleep(1.0)
                continue

            total_batches = len(urls)
            self._initialise_batch_heartbeats(total_batches)
            if self._conn_lock is None:
                self._conn_lock = asyncio.Lock()
            async with self._conn_lock:
                if self._tasks:
                    for task in list(self._tasks):
                        if task and not task.done():
                            task.cancel()
                self._tasks = []
                self._urls = urls

                for index, url in enumerate(urls, start=1):
                    self._tasks.append(
                        asyncio.create_task(
                            self._run_connection(url, index, total_batches)
                        )
                    )

            await asyncio.gather(*self._tasks, return_exceptions=True)

            self._tasks = []
            self._urls = []
            self._last_messages.clear()

            if self._stop.is_set():
                break
            if not self._resubscribe.is_set():
                await asyncio.sleep(1.0)

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except Exception:
                pass
            self._heartbeat_task = None

    def _build_combined_urls(
        self, symbols: Iterable[str], flags: Mapping[str, bool]
    ) -> List[str]:
        interval = (self._kline_interval or "1m").strip().lower() or "1m"
        has_external_callbacks = self._has_external_callbacks()
        want_kline = self._on_kline is not None or has_external_callbacks
        want_ticker = bool(flags.get("want_ticker"))
        want_book = bool(flags.get("want_book"))
        if interval == "1m" and flags.get("want_kline_1m", False):
            streams = _stream_names(
                symbols,
                want_kline_1m=want_kline,
                want_ticker=want_ticker,
                want_book=want_book,
            )
            return list(self._combined_urls(streams, MAX_STREAMS_PER_COMBINED))

        names: List[str] = []
        for symbol in symbols:
            token = str(symbol or "").strip().lower()
            if not token:
                continue
            if want_kline:
                names.append(f"{token}@kline_{interval}")
            if want_ticker:
                names.append(f"{token}@miniTicker")
            if want_book:
                names.append(f"{token}@bookTicker")
        if not names:
            return []
        return list(self._combined_urls(names, MAX_STREAMS_PER_COMBINED))

    def _combined_urls(self, streams: List[str], chunk: int = 200) -> Iterable[str]:
        if not streams:
            return
        base = (
            self._combined_base.rstrip("?") + "?streams="
            if "?" not in self._combined_base
            else self._combined_base + ""
        )
        for i in range(0, len(streams), chunk):
            yield base + "/".join(streams[i : i + chunk])

    async def _run_connection(self, url: str, batch_index: int, total: int) -> None:
        backoff = 1.0
        while not self._stop.is_set() and not self._resubscribe.is_set():
            connect_kwargs = dict(
                ping_interval=None,  # client never sends pings
                ping_timeout=None,  # we won’t wait on client pings
                close_timeout=1,  # fast close
                max_queue=None,  # no backpressure limit
                open_timeout=15,  # keep handshake timeout reasonable
            )
            self.logger.info(
                f"WSPriceBridge: connecting combined stream URL: {url} kwargs={connect_kwargs}"
            )
            try:
                async with websockets.connect(url, **connect_kwargs) as ws:
                    self._ws = ws
                    self.logger.info("WSPriceBridge: connected")
                    backoff = 1.0
                    self._last_messages[batch_index] = time.time()
                    async for raw in ws:
                        self._on_message(raw, batch_index)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.ConnectionClosedError as e:
                code = getattr(e, "code", None)
                self.logger.warning(
                    "WSPriceBridge: connection closed (code=%s) → will reconnect",
                    code,
                )
                await asyncio.sleep(10 if code == 1008 else 3)
            except Exception:
                self.logger.warning(
                    "WSPriceBridge: connection error on batch %d/%d; reconnecting",
                    batch_index,
                    total,
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            else:
                if self._stop.is_set() or self._resubscribe.is_set():
                    break
                self.logger.info(
                    "WSPriceBridge: connection closed for batch %d/%d; reconnecting",
                    batch_index,
                    total,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            finally:
                self._ws = None

    def _initialise_batch_heartbeats(self, total_batches: int) -> None:
        if total_batches <= 0:
            self._last_messages.clear()
            return
        now = time.time()
        self._last_messages = {index: now for index in range(1, total_batches + 1)}

    def _on_message(self, raw: str, batch_index: Optional[int] = None) -> None:
        self._handle_msg(raw, batch_index)

    def _handle_msg(self, raw: str, batch_index: Optional[int] = None) -> None:
        if batch_index is not None:
            self._last_messages[batch_index] = time.time()
        elif self._last_messages:
            # If we lost track of the originating batch, update all to avoid
            # spurious stale triggers while still receiving traffic.
            now = time.time()
            for key in list(self._last_messages.keys()):
                self._last_messages[key] = now
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            logger.debug("WS bridge received non-JSON payload")
            return
        if not isinstance(obj, dict):
            return
        data = obj.get("data", obj)
        if not isinstance(data, dict):
            return
        event = data.get("e")
        if event == "kline":
            kline = data.get("k")
            if isinstance(kline, dict):
                self._dispatch_kline(kline)
            return
        if event in ("24hrMiniTicker", "24hrTicker"):
            self._dispatch_ticker(data)
            return
        if event == "bookTicker":
            self._dispatch_book_ticker(data)
            return
        # Combined stream sometimes nests kline under obj['data']['k'] without 'e'
        if "k" in data:
            kline = data.get("k")
            if isinstance(kline, dict):
                self._dispatch_kline(kline)

    def _dispatch_kline(self, kline: Mapping[str, Any]) -> None:
        symbol = str(kline.get("s") or "").upper()
        interval = str(kline.get("i") or "")
        if not symbol:
            return
        payload = dict(kline)
        if self._on_kline is not None:
            try:
                self._on_kline(symbol, interval, payload)
            except Exception:  # pragma: no cover - callback safety
                logger.exception("WS bridge kline callback failed for %s", symbol)
        self._emit_callbacks(symbol, "kline", payload)

    def _dispatch_ticker(self, payload: Mapping[str, Any]) -> None:
        if self._on_ticker is None:
            return
        symbol = str(payload.get("s") or "").upper()
        if not symbol:
            return
        try:
            self._on_ticker(symbol, dict(payload))
        except Exception:  # pragma: no cover - callback safety
            logger.exception("WS bridge ticker callback failed for %s", symbol)

    def _dispatch_book_ticker(self, payload: Mapping[str, Any]) -> None:
        if self._on_book_ticker is None:
            return
        symbol = str(payload.get("s") or "").upper()
        if not symbol:
            return
        try:
            self._on_book_ticker(symbol, dict(payload))
        except Exception:  # pragma: no cover - callback safety
            logger.exception("WS bridge book ticker callback failed for %s", symbol)

    async def _heartbeat_watch(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self._heartbeat_timeout / 2.0)
                if self._stop.is_set():
                    return
                if not self._last_messages:
                    continue
                now = time.time()
                stale_batches = [
                    (batch_index, now - last)
                    for batch_index, last in self._last_messages.items()
                    if now - last >= self._heartbeat_timeout
                ]
                if not stale_batches:
                    continue
                worst_batch, worst_gap = max(stale_batches, key=lambda item: item[1])
                log_event(
                    logger,
                    "ws_heartbeat_stale",
                    gap_seconds=worst_gap,
                    timeout=self._heartbeat_timeout,
                    stale_batches=[index for index, _ in stale_batches],
                    worst_batch=worst_batch,
                )
                record_metric(
                    "ws_gap_seconds",
                    worst_gap,
                    labels={"stream": "price", "batch": str(worst_batch)},
                )
                self._notify_stale(worst_gap)
                self._resubscribe.set()
                self._cancel_ws_tasks()
                return
        except asyncio.CancelledError:
            raise

    def _on_heartbeat_done(self, task: asyncio.Task) -> None:
        if task is not self._heartbeat_task:
            return
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug(
                "WS bridge heartbeat watcher finished with error", exc_info=True
            )
        if task is self._heartbeat_task:
            self._heartbeat_task = None

    def _notify_stale(self, gap: float) -> None:
        callback = self._on_stale
        if callback is None:
            return
        try:
            callback(gap)
        except Exception:
            logger.debug("WS stale callback failed", exc_info=True)

    def _time_sync_loop(self) -> None:
        while not self._time_sync_stop.is_set():
            if self._stop.is_set():
                break
            try:
                response = requests.get(
                    "https://api.binance.com/api/v3/time", timeout=10
                )
                response.raise_for_status()
                data = response.json()
                server_time = data.get("serverTime")
                if server_time is not None:
                    now_ms = time.time() * 1000.0
                    skew_ms = float(server_time) - float(now_ms)
                    log_event(
                        logger,
                        "server_time_sync",
                        skew_ms=skew_ms,
                    )
                    record_metric(
                        "server_time_skew_ms",
                        skew_ms,
                        labels={"source": "binance"},
                    )
            except Exception:
                logger.debug("Server time sync failed", exc_info=True)
            self._time_sync_stop.wait(self._time_sync_interval)

    def _current_symbols(self) -> List[str]:
        with self._symbols_lock:
            return list(self._symbols)

    @property
    def symbols(self) -> List[str]:
        return self._current_symbols()

    @staticmethod
    def _normalise_symbols(symbols: Iterable[str]) -> List[str]:
        seen = set()
        normalised: List[str] = []
        for sym in symbols:
            token = str(sym or "").strip().lower()
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            normalised.append(token)
        return normalised


__all__ = ["WSPriceBridge", "BINANCE_WS", "COMBINED_BASE", "MAX_STREAMS_PER_COMBINED"]
