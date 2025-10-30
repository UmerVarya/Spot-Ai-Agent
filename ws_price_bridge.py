"""Minimal Binance WebSocket bridge for real-time price updates.

This module exposes :class:`WSPriceBridge`, a light wrapper that runs the
Binance combined stream in a dedicated thread and event loop.  Callers
register callbacks for klines, mini tickers and (optionally) book ticker
updates; the bridge handles reconnections with exponential backoff and
normalises symbols to upper case for downstream consumers.
"""
from __future__ import annotations

import os, asyncio, time, websockets
import json
import logging
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Mapping

import requests

from observability import log_event, record_metric

BINANCE_WS = "wss://stream.binance.com:9443/stream"
WS_BASE = os.getenv(
    "WS_COMBINED_BASE", "wss://stream.binance.com:9443/stream?streams="
)

logger = logging.getLogger(__name__)

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

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
        self._last_message = time.time()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._time_sync_interval = max(30.0, float(server_time_sync_interval))
        self._time_sync_stop = threading.Event()
        self._time_sync_thread: Optional[threading.Thread] = None
        self._max_retries = max(1, int(max_retries))
        self._callback_lock = threading.Lock()
        self._extra_callbacks: List[
            Callable[[str, str, Dict[str, Any]], None]
        ] = []
        self._ws_task: Optional[asyncio.Task] = None
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
        self._cancel_ws_task()
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
        self._cancel_ws_task()
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
            self._extra_callbacks.append(callback)

    def unregister_callback(
        self, callback: Callable[[str, str, Dict[str, Any]], None]
    ) -> None:
        with self._callback_lock:
            try:
                self._extra_callbacks.remove(callback)
            except ValueError:
                return

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

    def _cancel_ws_task(self) -> None:
        task = self._ws_task
        if task is None or task.done():
            return
        loop = self._loop
        if loop and loop.is_running():
            def _cancel() -> None:
                current = self._ws_task
                if current is not None and not current.done():
                    current.cancel()

            loop.call_soon_threadsafe(_cancel)
        else:
            task.cancel()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
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

    async def _ws_main(self) -> None:
        while not self._stop.is_set():
            self._resubscribe.clear()
            symbols = self._current_symbols()
            if not symbols:
                self._last_message = time.time()
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

            try:
                await self._run_connection()
            except asyncio.CancelledError:
                if self._stop.is_set():
                    break
            except Exception:
                logger.warning("WS bridge connection encountered an error", exc_info=True)
                await asyncio.sleep(1.0)
            finally:
                self._ws_task = None

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

# --- inside class WSPriceBridge ---  (replace your existing _run_connection)
    async def _run_connection(self):
        """
        Opens ONE combined-stream websocket to Binance and fans out messages.
        Reconnects with bounded backoff. Reads tuning from environment.
        """
        # Read tuning from env (with sensible defaults)
        BASE   = os.getenv("WS_COMBINED_BASE", "wss://stream.binance.com:9443/stream?streams=")
        BATCH  = int(os.getenv("WS_SUBSCRIBE_BATCH", "200"))   # how many symbols per combined URL
        DELAY  = int(os.getenv("WS_SUBSCRIBE_DELAY_MS", "400")) / 1000.0

        # allow disabling client pings (avoids strict Pong deadlines on noisy hosts)
        raw_pi = os.getenv("WS_PING_INTERVAL", "none").lower()
        raw_pt = os.getenv("WS_PING_TIMEOUT", "none").lower()
        PING_INTERVAL = None if raw_pi == "none" else int(raw_pi)
        PING_TIMEOUT  = None if raw_pt == "none" else int(raw_pt)

        raw_mq = os.getenv("WS_MAX_QUEUE", "0")
        MAX_QUEUE = None if raw_mq == "0" else int(raw_mq)

        min_backoff = int(os.getenv("WS_RECONNECT_MIN_SECONDS", "2"))
        max_backoff = int(os.getenv("WS_RECONNECT_MAX_SECONDS", "60"))

        # Prevent multiple concurrent sockets from this bridge
        if getattr(self, "_ws_task", None) and not self._ws_task.done():
            self.logger.info("WSPriceBridge: connection already running; skip duplicate.")
            return

        # Snapshot the current symbol list (lowercase)
        syms = [s.lower() for s in sorted(set(self.symbols))]
        if not syms:
            self.logger.warning("WSPriceBridge: no symbols to subscribe; delaying 5s.")
            await asyncio.sleep(5)
            return

        # Build combined stream URLs in batches (large lists split across a few sockets)
        urls = []
        interval = (self._kline_interval or "1m").strip().lower() or "1m"
        for chunk in _chunks(syms, BATCH):
            stream_parts: List[str] = []
            for s in chunk:
                stream_parts.append(f"{s}@kline_{interval}")
                if self._on_ticker is not None:
                    stream_parts.append(f"{s}@miniTicker")
                if self._on_book_ticker is not None:
                    stream_parts.append(f"{s}@bookTicker")
            urls.append(BASE + "/".join(stream_parts))

        async def _reader(url):
            backoff = min_backoff
            while True:
                try:
                    self.logger.info(f"WSPriceBridge: connecting combined stream: {url}")
                    ws = await websockets.connect(
                        url,
                        ping_interval=PING_INTERVAL,   # None => no client pings
                        ping_timeout=PING_TIMEOUT,     # None => don’t enforce pong
                        max_queue=MAX_QUEUE,           # None => unbounded
                        close_timeout=1,
                    )
                    self.logger.info("WSPriceBridge: connected.")
                    backoff = min_backoff  # reset backoff after a successful connect
                    async for msg in ws:
                        # hand off immediately; NEVER block this loop with heavy work
                        try:
                            self._on_message(msg)
                        except Exception as e:
                            self.logger.exception(f"WSPriceBridge: handler error: {e}")
                except asyncio.CancelledError:
                    self.logger.info("WSPriceBridge: reader cancelled; closing.")
                    break
                except Exception as e:
                    self.logger.warning(f"WSPriceBridge: connection error: {e}; reconnecting in {backoff}s")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

        # Start one reader per URL (usually this is ONE socket; a few if many symbols)
        tasks = []
        for i, u in enumerate(urls):
            # small stagger to avoid bursty subscribe
            if i:
                await asyncio.sleep(DELAY)
            tasks.append(asyncio.create_task(_reader(u)))

        # Keep a handle so we can avoid duplicate starts later
        self._ws_task = asyncio.create_task(asyncio.gather(*tasks))
        try:
            await self._ws_task
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()

    def _on_message(self, raw: str) -> None:
        self._handle_msg(raw)

    def _handle_msg(self, raw: str) -> None:
        self._last_message = time.time()
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
                gap = time.time() - self._last_message
                if gap >= self._heartbeat_timeout:
                    log_event(
                        logger,
                        "ws_heartbeat_stale",
                        gap_seconds=gap,
                        timeout=self._heartbeat_timeout,
                    )
                    record_metric("ws_gap_seconds", gap, labels={"stream": "price"})
                    self._notify_stale(gap)
                    self._resubscribe.set()
                    self._cancel_ws_task()
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


__all__ = ["WSPriceBridge", "BINANCE_WS", "WS_BASE"]
