"""Minimal Binance WebSocket bridge for real-time price updates.

This module exposes :class:`WSPriceBridge`, a light wrapper that runs the
Binance combined stream in a dedicated thread and event loop.  Callers
register callbacks for klines, mini tickers and (optionally) book ticker
updates; the bridge handles reconnections with exponential backoff and
normalises symbols to upper case for downstream consumers.
"""
from __future__ import annotations

import asyncio
import os
import json
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Mapping

import websockets
import requests

from observability import log_event, record_metric

PING_I = int(os.getenv("WS_PING_INTERVAL", 15))
PING_TO = int(os.getenv("WS_PING_TIMEOUT", 30))
RQ_MIN = int(os.getenv("WS_RECONNECT_MIN_SECONDS", 2))
RQ_MAX = int(os.getenv("WS_RECONNECT_MAX_SECONDS", 60))
_max_queue_env = os.getenv("WS_MAX_QUEUE")
DEFAULT_MAX_QUEUE = 1000
if _max_queue_env is None or not _max_queue_env.strip():
    MAX_Q = DEFAULT_MAX_QUEUE
else:
    try:
        _max_queue_value = int(_max_queue_env)
    except ValueError:
        MAX_Q = DEFAULT_MAX_QUEUE
    else:
        MAX_Q = None if _max_queue_value <= 0 else _max_queue_value
try:
    _batch_env = int(os.getenv("WS_SUBSCRIBE_BATCH", 20))
except ValueError:
    _batch_env = 20
BATCH = max(1, _batch_env)
try:
    _delay_env = int(os.getenv("WS_SUBSCRIBE_DELAY_MS", 400))
except ValueError:
    _delay_env = 400
DELAYMS = max(0, _delay_env)

BINANCE_WS = "wss://stream.binance.com:9443/stream"
WS_BASE = os.getenv(
    "WS_COMBINED_BASE", "wss://stream.binance.com:9443/stream?streams="
)

logger = logging.getLogger(__name__)

_HANDSHAKE_TIMEOUT_MSG = "timed out during opening handshake"


def _ws_connect_kwargs() -> Dict[str, Any]:
    """Return websocket client configuration derived from environment."""

    kwargs: Dict[str, Any] = {
        "ping_interval": float(PING_I),
        "ping_timeout": float(PING_TO),
        "close_timeout": 10,
        "open_timeout": float(os.getenv("WS_OPEN_TIMEOUT", "20")),
    }
    if MAX_Q is not None:
        kwargs["max_queue"] = MAX_Q
    return kwargs


def _compute_backoff_delay(attempt: int) -> float:
    base = max(1.0, float(RQ_MIN))
    cap = max(base, float(RQ_MAX))
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    jitter = 0.5 + random.random() / 2.0
    return max(base, min(cap, delay * jitter))


def _chunks(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    size = max(1, size)
    for index in range(0, len(seq), size):
        yield seq[index : index + size]

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

            chunks = self._chunk_symbols(symbols)
            tasks: List[asyncio.Task] = []
            for index, chunk in enumerate(chunks):
                if index > 0 and DELAYMS:
                    await asyncio.sleep(DELAYMS / 1000.0)
                tasks.append(
                    asyncio.create_task(self._run_connection(chunk, index))
                )
            if not tasks:
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
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                for task in tasks:
                    task.cancel()
                raise
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                if self._heartbeat_task is not None and not self._current_symbols():
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except Exception:
                        pass
                    self._heartbeat_task = None
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except Exception:
                pass
            self._heartbeat_task = None

    async def _run_connection(
        self, chunk: Sequence[str], shard_index: int
    ) -> None:
        attempts = 0
        while not self._stop.is_set() and not self._resubscribe.is_set():
            url = self._build_stream_url(chunk)
            if os.getenv("WS_DEBUG", "0") == "1":
                logger.info(
                    "Connecting to Binance WS shard %d: %s", shard_index, url
                )
            try:
                async with websockets.connect(
                    url,
                    **_ws_connect_kwargs(),
                ) as ws:
                    attempts = 0
                    self._last_message = time.time()
                    await self._listen(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network dependent
                attempts += 1
                delay = _compute_backoff_delay(attempts)
                error_text = str(exc)
                if _HANDSHAKE_TIMEOUT_MSG not in error_text:
                    logger.warning(
                        "WS bridge error on shard %d: %s; retrying in %.1fs",
                        shard_index,
                        exc,
                        delay,
                    )
                if self._stop.is_set() or self._resubscribe.is_set():
                    continue
                await asyncio.sleep(delay)
                record_metric(
                    "ws_reconnects",
                    1.0,
                    labels={"reason": "error", "shard": str(shard_index)},
                )
                if attempts >= self._max_retries:
                    log_event(
                        logger,
                        "ws_reconnect_cap",
                        attempts=attempts,
                        delay=delay,
                        shard=shard_index,
                    )
                    attempts = 0

    async def _listen(self, ws: websockets.WebSocketClientProtocol) -> None:
        while not self._stop.is_set():
            if self._resubscribe.is_set():
                break
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
            except asyncio.TimeoutError:
                try:
                    await ws.ping()
                except Exception:
                    break
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                break
            self._handle_msg(msg)

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

    def _build_stream_url(self, symbols: Sequence[str]) -> str:
        streams: List[str] = []
        for sym in symbols:
            token = sym.lower()
            streams.append(f"{token}@kline_{self._kline_interval}")
            streams.append(f"{token}@miniTicker")
            if self._on_book_ticker is not None:
                streams.append(f"{token}@bookTicker")
        params = "/".join(streams)
        return WS_BASE + params

    def _chunk_symbols(self, symbols: Sequence[str]) -> List[List[str]]:
        return [list(chunk) for chunk in _chunks(symbols, BATCH)]

    def _current_symbols(self) -> List[str]:
        with self._symbols_lock:
            return list(self._symbols)

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
