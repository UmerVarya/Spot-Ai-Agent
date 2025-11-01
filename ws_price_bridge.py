"""Minimal Binance WebSocket bridge for real-time price updates.

This module exposes :class:`WSPriceBridge`, a light wrapper that runs the
Binance combined stream in a dedicated thread and event loop.  Callers
register callbacks for klines, mini tickers and (optionally) book ticker
updates; the bridge handles reconnections with exponential backoff and
normalises symbols to upper case for downstream consumers.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
from contextlib import suppress
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

# Commented out legacy threaded stream import; we now use WSPriceBridge instead.
# from binance import ThreadedWebsocketManager

import aiohttp
import requests
from aiohttp import __version__ as AIOHTTP_VERSION

from observability import log_event, record_metric

BINANCE_WS = "wss://stream.binance.com:9443/stream"
MAX_STREAMS_PER_COMBINED = max(
    1, min(200, int(os.getenv("BINANCE_MAX_STREAMS", "200")))
)
COMBINED_BASE = os.getenv(
    "WS_COMBINED_BASE",
    "wss://stream.binance.com:9443/stream?streams=",
)
MAX_COMBINED_URL_LEN = max(512, int(os.getenv("BINANCE_MAX_URL_LEN", "1900")))
MAX_CONNS = max(1, int(os.getenv("WS_MAX_CONNS", "4")))
IDLE_RECV_TIMEOUT = max(10, int(os.getenv("WS_IDLE_RECV_TIMEOUT", "90")))
WS_BACKEND = (os.getenv("WS_BACKEND", "wsclient") or "wsclient").strip().lower()

WS_MAX_RECONNECTS = int(os.getenv("WS_MAX_RECONNECTS", "20"))
WS_RECONNECT_BASE_MS = int(os.getenv("WS_RECONNECT_BASE_MS", "400"))
WS_RECONNECT_MAX_MS = int(os.getenv("WS_RECONNECT_MAX_MS", "8000"))
WS_SILENCE_SEC = float(os.getenv("WS_SILENCE_SEC", "10.0"))

logger = logging.getLogger(__name__)


def make_streams(
    symbols: Iterable[str],
    *,
    kline_interval: str = "1m",
    include_kline: bool = True,
    include_ticker: bool = True,
    include_book: bool = False,
    ticker_stream: str = "ticker",
    quote_suffix: Optional[str] = None,
) -> List[str]:
    """Return a de-duplicated list of Binance stream names for ``symbols``.

    ``symbols`` are normalised to lower case with leading/trailing whitespace
    removed.  If ``quote_suffix`` is provided, only pairs ending in that suffix
    are kept; by default all symbols are accepted.  Stream names are emitted in
    a stable order with duplicates removed while preserving the first
    occurrence.
    """

    interval = str(kline_interval or "1m").strip().lower() or "1m"
    ticker_name = str(ticker_stream or "ticker").strip() or "ticker"

    ordered: List[str] = []
    seen: set[str] = set()

    def _append(value: str) -> None:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)

    for sym in symbols:
        token = str(sym or "").strip().lower()
        if not token:
            continue
        if quote_suffix and not token.endswith(quote_suffix):
            continue
        if include_kline:
            _append(f"{token}@kline_{interval}")
        if include_ticker:
            _append(f"{token}@{ticker_name}")
        if include_book:
            _append(f"{token}@bookTicker")

    return ordered


def _stream_names(
    symbols: Iterable[str],
    want_kline_1m: bool = True,
    want_ticker: bool = False,
    want_book: bool = False,
) -> List[str]:
    return make_streams(
        symbols,
        kline_interval="1m",
        include_kline=want_kline_1m,
        include_ticker=want_ticker,
        include_book=want_book,
        ticker_stream="ticker",
    )


def _streams_prefix(base: str) -> str:
    base = (base or "").rstrip("?")
    if "streams=" in base:
        if base.endswith("streams"):
            return base + "="
        return base
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}streams="


def _chunk_stream_batches(
    streams: Iterable[str],
    *,
    max_streams: int,
    max_url_len: int,
    prefix: str,
    log: Optional[logging.Logger] = None,
) -> Iterable[List[str]]:
    prefix_len = len(prefix)
    payload_limit = max_url_len - prefix_len
    if payload_limit <= 0:
        if log:
            log.error(
                "WS BRIDGE MARK v3 | prefix too long for url limit | prefix_len=%d | max=%d",
                prefix_len,
                max_url_len,
            )
        return []

    batch: List[str] = []
    batch_len = 0
    max_streams = max(1, min(200, int(max_streams or 1)))
    for raw in streams:
        token_raw = str(raw or "").strip()
        if not token_raw:
            continue

        if "@" in token_raw:
            symbol, sep, suffix = token_raw.partition("@")
            token = f"{symbol.lower()}{sep}{suffix}"
        else:
            token = token_raw.lower()

        token_len = len(token)
        if token_len > payload_limit:
            if log:
                log.error(
                    "WS BRIDGE MARK v3 | stream token exceeds url limit | token=%s | len=%d | limit=%d",
                    token,
                    token_len,
                    payload_limit,
                )
            continue
        projected = batch_len + token_len if not batch else batch_len + token_len + 1
        if len(batch) >= max_streams or projected > payload_limit:
            if batch:
                yield batch
            batch = [token]
            batch_len = token_len
        else:
            batch.append(token)
            batch_len = projected
    if batch:
        yield batch


def _combined_urls(
    symbols: Iterable[str],
    *,
    want_kline_1m: bool = True,
    want_ticker: bool = False,
    want_book: bool = False,
    chunk: Optional[int] = None,
) -> List[str]:
    streams = _stream_names(
        symbols,
        want_kline_1m=want_kline_1m,
        want_ticker=want_ticker,
        want_book=want_book,
    )
    if not streams:
        return []

    limit = MAX_STREAMS_PER_COMBINED if chunk is None else int(chunk)
    limit = max(1, min(200, limit))
    prefix = _streams_prefix(COMBINED_BASE)

    return [
        prefix + "/".join(batch)
        for batch in _chunk_stream_batches(
            streams,
            max_streams=limit,
            max_url_len=MAX_COMBINED_URL_LEN,
            prefix=prefix,
            log=logger,
        )
    ]


KlineCallback = Callable[[str, str, Dict[str, Any]], None]
TickerCallback = Callable[[str, Dict[str, Any]], None]
BookTickerCallback = Callable[[str, Dict[str, Any]], None]


class _WebsocketsPriceBridge:
    """Run Binance WebSocket streams and dispatch updates to callbacks."""

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        kline_interval: str = "1m",
        on_kline: Optional[KlineCallback] = None,
        on_ticker: Optional[TickerCallback] = None,
        on_book_ticker: Optional[BookTickerCallback] = None,
        on_market_event: Optional[Callable[[str, str], Any]] = None,
        on_stale: Optional[Callable[[float], None]] = None,
        heartbeat_timeout: float = 10.0,
        server_time_sync_interval: float = 120.0,
        max_retries: int = 10,
    ) -> None:
        self.logger = logger
        self.logger.warning(
            "WS BRIDGE MARK v3 | backend=aiohttp | file=%s | aiohttp_version=%s",
            __file__,
            AIOHTTP_VERSION,
        )
        self._symbols: List[str] = self._normalise_symbols(symbols)
        self._kline_interval = str(kline_interval or "1m").strip()
        if not self._kline_interval:
            self._kline_interval = "1m"
        self._on_kline = on_kline
        self._on_ticker = on_ticker
        self._on_book_ticker = on_book_ticker
        self._on_market_event = on_market_event
        self._on_stale = on_stale
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._resubscribe = threading.Event()
        self._symbols_lock = threading.Lock()
        self._heartbeat_timeout = max(5.0, float(heartbeat_timeout))
        self._last_messages: Dict[int, float] = {}
        self._last_ticker_fire: Dict[str, float] = {}
        self._ticker_min_interval = 0.75
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
        self._combined_base = _streams_prefix(COMBINED_BASE)
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._conn_lock: Optional[asyncio.Lock] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background WebSocket thread if it is not running."""

        if self._thread and self._thread.is_alive():
            return
        try:
            stream_count = len(self._current_symbols())
        except Exception:
            stream_count = 0
        self.logger.info(
            "WS BRIDGE MARK v3 | WSPriceBridge: starting with %d streams | backend=%s",
            stream_count,
            "aiohttp",
        )
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
            # 🧠 Safety: Don't oversubscribe Binance's combined stream cap
            if len(symbols) > 190:
                logger.warning(
                    "WSPriceBridge: symbol list too large (%d), truncating to 190 for Binance limit",
                    len(symbols),
                )
                truncated = list(symbols[:190])
                replaced_shared = False
                with self._symbols_lock:
                    if self._symbols == symbols:
                        self._symbols = truncated
                        replaced_shared = True
                if replaced_shared:
                    symbols = truncated
                else:
                    # Another thread updated the subscription list while we were
                    # preparing the truncation; respect the newer list instead of
                    # overwriting it with a stale snapshot.
                    symbols = self._current_symbols()
                    if len(symbols) > 190:
                        symbols = symbols[:190]
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

            if len(urls) > MAX_CONNS:
                self.logger.warning(
                    "WS BRIDGE MARK v3 | too many url batches=%d | max_conns=%d | truncating",
                    len(urls),
                    MAX_CONNS,
                )
                urls = urls[:MAX_CONNS]

            total_batches = len(urls)
            self._initialise_batch_heartbeats(total_batches)
            if self._conn_lock is None:
                self._conn_lock = asyncio.Lock()
            timeout_cfg = aiohttp.ClientTimeout(
                total=None,
                sock_connect=15,
                sock_read=IDLE_RECV_TIMEOUT,
            )
            session = aiohttp.ClientSession(timeout=timeout_cfg)
            try:
                async with self._conn_lock:
                    if self._tasks:
                        for task in list(self._tasks):
                            if task and not task.done():
                                task.cancel()
                    self._tasks = []
                    self._urls = urls
                    self._session = session

                    for index, url in enumerate(urls, start=1):
                        self._tasks.append(
                            asyncio.create_task(
                                self._run_connection(session, url, index, total_batches)
                            )
                        )

                await asyncio.gather(*self._tasks, return_exceptions=True)
            finally:
                if self._session is session:
                    self._session = None
                self._tasks = []
                self._urls = []
                self._last_messages.clear()
                with suppress(Exception):
                    await session.close()

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
            streams = make_streams(
                symbols,
                kline_interval="1m",
                include_kline=want_kline,
                include_ticker=want_ticker,
                include_book=want_book,
                ticker_stream="ticker",
            )
            self.logger.warning(
                "WS BRIDGE MARK v3 | streams=%d | chunk=200", len(streams)
            )
            urls = list(self._combined_urls(streams, chunk=200))
            self.logger.warning(
                "WS BRIDGE MARK v3 | combined urls=%d | mode=kline_1m",
                len(urls),
            )
            return urls

        names = make_streams(
            symbols,
            kline_interval=interval,
            include_kline=want_kline,
            include_ticker=want_ticker,
            include_book=want_book,
            ticker_stream="miniTicker",
        )
        if not names:
            return []
        self.logger.warning(
            "WS BRIDGE MARK v3 | streams=%d | chunk=200", len(names)
        )
        urls = list(self._combined_urls(names, chunk=200))
        self.logger.warning(
            "WS BRIDGE MARK v3 | combined urls=%d | mode=mix",
            len(urls),
        )
        return urls

    def _combined_urls(self, streams: List[str], chunk: int = 200) -> Iterable[str]:
        if not streams:
            return []
        try:
            chunk = int(chunk)
        except (TypeError, ValueError):
            chunk = 200
        chunk = max(1, min(200, chunk))
        prefix = _streams_prefix(self._combined_base)
        for batch in _chunk_stream_batches(
            streams,
            max_streams=chunk,
            max_url_len=MAX_COMBINED_URL_LEN,
            prefix=prefix,
            log=self.logger,
        ):
            url = prefix + "/".join(batch)
            self.logger.warning(
                "WS BRIDGE MARK v3 | combined url built | batch_size=%d | url_len=%d",
                len(batch),
                len(url),
            )
            yield url

    async def _run_connection(
        self,
        session: aiohttp.ClientSession,
        url: str,
        batch_index: int,
        total: int,
    ) -> None:
        def _reconnect_delay(attempt: int) -> float:
            step = max(1, attempt)
            base = min(30.0, 1.5 * (2 ** (step - 1)))
            jitter = 0.2 * (step % 3)
            return base + jitter

        payload = ""
        if "streams=" in url:
            payload = url.split("streams=", 1)[1]
        stream_count = len([token for token in payload.split("/") if token]) or 1
        attempt = 0

        while not self._stop.is_set() and not self._resubscribe.is_set():
            try:
                self.logger.warning(
                    "WS BRIDGE MARK v3 | aiohttp=%s | url=%s | n_streams=%d | url_len=%d",
                    AIOHTTP_VERSION,
                    url,
                    stream_count,
                    len(url),
                )
                async with session.ws_connect(
                    url,
                    heartbeat=30,
                    max_msg_size=0,
                    receive_timeout=IDLE_RECV_TIMEOUT,
                ) as ws:
                    self._ws = ws
                    attempt = 0
                    self._last_messages[batch_index] = time.monotonic()
                    self.logger.warning(
                        "WS BRIDGE MARK v3 | connected | batch=%d/%d | streams=%d | url_len=%d",
                        batch_index,
                        total,
                        stream_count,
                        len(url),
                    )
                    async for msg in ws:
                        if self._stop.is_set() or self._resubscribe.is_set():
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            self._last_messages[batch_index] = time.monotonic()
                            self._on_message(msg.data, batch_index)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            exc = ws.exception()
                            raise exc if exc is not None else RuntimeError("WS error")
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            break
                        else:
                            continue

                if self._stop.is_set() or self._resubscribe.is_set():
                    break
                attempt += 1
                delay = _reconnect_delay(attempt)
                self.logger.warning(
                    "WS BRIDGE MARK v3 | socket ended | batch=%d/%d | reconnect in %.1fs",
                    batch_index,
                    total,
                    delay,
                )
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                if self._stop.is_set() or self._resubscribe.is_set():
                    break
                attempt += 1
                delay = _reconnect_delay(attempt)
                self.logger.warning(
                    "WS BRIDGE MARK v3 | idle timeout | batch=%d/%d | reconnect in %.1fs",
                    batch_index,
                    total,
                    delay,
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                if self._stop.is_set() or self._resubscribe.is_set():
                    break
                attempt += 1
                delay = _reconnect_delay(attempt)
                try:
                    logger.warning(
                        "WS reconnect (%s) in %.1fs | url_len=%d",
                        type(exc).__name__,
                        delay,
                        len(url),
                    )
                except Exception:
                    pass
                await asyncio.sleep(delay)
            finally:
                self._ws = None

    def _initialise_batch_heartbeats(self, total_batches: int) -> None:
        if total_batches <= 0:
            self._last_messages.clear()
            return
        now = time.monotonic()
        self._last_messages = {index: now for index in range(1, total_batches + 1)}

    def _on_message(self, raw: str, batch_index: Optional[int] = None) -> None:
        self._handle_msg(raw, batch_index)

    def _handle_msg(self, raw: str, batch_index: Optional[int] = None) -> None:
        if batch_index is not None:
            self._last_messages[batch_index] = time.monotonic()
        elif self._last_messages:
            # If we lost track of the originating batch, update all to avoid
            # spurious stale triggers while still receiving traffic.
            now = time.monotonic()
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
        if payload.get("x"):
            self._schedule_market_event(symbol, "kline_close")

    def _dispatch_ticker(self, payload: Mapping[str, Any]) -> None:
        symbol = str(payload.get("s") or "").upper()
        if not symbol:
            return
        if self._on_ticker is not None:
            try:
                self._on_ticker(symbol, dict(payload))
            except Exception:  # pragma: no cover - callback safety
                logger.exception("WS bridge ticker callback failed for %s", symbol)
        now = time.monotonic()
        last = self._last_ticker_fire.get(symbol, 0.0)
        if now - last >= self._ticker_min_interval:
            self._last_ticker_fire[symbol] = now
            self._schedule_market_event(symbol, "ticker")

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
                now = time.monotonic()
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

    async def _invoke_market_event(self, symbol: str, kind: str) -> None:
        callback = self._on_market_event
        if callback is None or not symbol:
            return
        try:
            result = callback(symbol, kind)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug(
                "WS bridge market event callback failed for %s", symbol, exc_info=True
            )

    def _schedule_market_event(self, symbol: str, kind: str) -> None:
        callback = self._on_market_event
        if callback is None or not symbol:
            return
        loop = self._loop
        if loop is None or not loop.is_running():
            try:
                result = callback(symbol, kind)
                if inspect.isawaitable(result):
                    asyncio.run(result)
            except Exception:
                logger.debug(
                    "WS bridge market event callback failed for %s", symbol, exc_info=True
                )
            return
        coro = self._invoke_market_event(symbol, kind)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, loop)

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


if WS_BACKEND == "wsclient":
    try:
        from ws_backend_client import WSClientBridge as _WSClientBridge
    except Exception:  # pragma: no cover - optional dependency
        logger.warning(
            "WS BRIDGE MARK v3 | wsclient backend unavailable; falling back to websockets",
            exc_info=True,
        )
        WSPriceBridge = _WebsocketsPriceBridge
    else:

        class _WSClientPriceBridge:
            """Threaded price bridge backed by websocket-client."""

            def __init__(
                self,
                symbols: Iterable[str],
                *,
                kline_interval: str = "1m",
                on_kline: Optional[KlineCallback] = None,
                on_ticker: Optional[TickerCallback] = None,
                on_book_ticker: Optional[BookTickerCallback] = None,
                on_market_event: Optional[Callable[[str, str], Any]] = None,
                on_stale: Optional[Callable[[float], None]] = None,
                heartbeat_timeout: float = 10.0,
                server_time_sync_interval: float = 120.0,
                max_retries: int = 10,
            ) -> None:
                self.logger = logger
                self._symbols_lock = threading.RLock()
                self._symbols: List[str] = _WebsocketsPriceBridge._normalise_symbols(symbols)
                self._kline_interval = str(kline_interval or "1m").strip() or "1m"
                self._on_kline = on_kline
                self._on_ticker = on_ticker
                self._on_book_ticker = on_book_ticker
                self._on_market_event = on_market_event
                self._on_stale = on_stale
                self._heartbeat_timeout = max(5.0, float(heartbeat_timeout))
                self._lock = threading.RLock()
                self._stop = threading.Event()
                self._clients: List[_WSClientBridge] = []
                self._last_messages: Dict[int, float] = {}
                self._heartbeat_thread: Optional[threading.Thread] = None
                self._callback_lock = threading.Lock()
                self._extra_callbacks: List[
                    Callable[[str, str, Dict[str, Any]], None]
                ] = []
                self._running = False
                self._combined_base = _streams_prefix(COMBINED_BASE)
                self._last_ticker_fire: Dict[str, float] = {}
                self._ticker_min_interval = 0.75
                self._last_msg_mono = time.monotonic()
                self._reconnects = 0
                self._max_reconnects = max(1, int(WS_MAX_RECONNECTS))
                self._reconnect_backoff_ms = max(1, int(WS_RECONNECT_BASE_MS))
                self._silence_threshold = max(
                    float(self._heartbeat_timeout), float(WS_SILENCE_SEC), 1.0
                )

            # ------------------------------------------------------------------
            # Public API
            # ------------------------------------------------------------------
            def start(self) -> None:
                with self._lock:
                    if self._running:
                        return
                    symbols = self._current_symbols()
                batches = self._build_batches(symbols)
                stream_count = sum(len(batch) for batch in batches)
                self.logger.info(
                    "WS BRIDGE MARK v3 | WSPriceBridge: starting with %d streams | backend=%s",
                    stream_count,
                    "wsclient",
                )
                if not batches:
                    self.logger.info(
                        "WS BRIDGE MARK v3 | WSPriceBridge: no streams to subscribe; skipping connect."
                    )
                    return
                clients: List[_WSClientBridge] = []
                now = time.monotonic()
                for index, batch in enumerate(batches, start=1):
                    if not batch:
                        continue
                    clients.append(
                        _WSClientBridge(
                            batch,
                            self._make_handler(index),
                            base_url=self._combined_base,
                        )
                    )
                with self._lock:
                    self._stop.clear()
                    self._clients = clients
                    self._last_messages = {
                        index: now for index in range(1, len(clients) + 1)
                    }
                    self._last_msg_mono = now
                    self._running = bool(clients)
                for client in clients:
                    try:
                        client.start()
                    except Exception:
                        self.logger.warning(
                            "WS BRIDGE MARK v3 | wsclient start failed", exc_info=True
                        )
                with self._lock:
                    if self._running and self._heartbeat_thread is None:
                        self._heartbeat_thread = threading.Thread(
                            target=self._heartbeat_loop,
                            name="wsclient-heartbeat",
                            daemon=True,
                        )
                        self._heartbeat_thread.start()

            def stop(self, timeout: float = 3.0) -> None:
                self._shutdown_clients(timeout=timeout, log=True)

            def update_symbols(self, symbols: Iterable[str]) -> None:
                normalised = self._normalise_symbols(symbols)
                with self._symbols_lock:
                    if normalised == self._symbols:
                        return
                    self._symbols = normalised
                self._restart_clients()

            def register_callback(
                self, callback: Callable[[str, str, Dict[str, Any]], None]
            ) -> None:
                if not callable(callback):
                    raise TypeError("callback must be callable")
                with self._callback_lock:
                    if callback in self._extra_callbacks:
                        return
                    had_callbacks = bool(self._extra_callbacks)
                    self._extra_callbacks.append(callback)
                if not had_callbacks and self._on_kline is None:
                    self._restart_clients()

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
                    self._restart_clients()

            @property
            def symbols(self) -> List[str]:
                return self._current_symbols()

            # ------------------------------------------------------------------
            # Internal helpers
            # ------------------------------------------------------------------
            def _restart_clients(self) -> None:
                with self._lock:
                    running = self._running
                if not running:
                    return
                self._shutdown_clients(timeout=3.0, log=False)
                self.start()

            def _shutdown_clients(self, timeout: float, log: bool) -> None:
                current_thread = threading.current_thread()
                with self._lock:
                    clients = list(self._clients)
                    thread = self._heartbeat_thread
                    self._clients = []
                    self._heartbeat_thread = None
                    self._running = False
                    self._last_messages.clear()
                    self._stop.set()
                for client in clients:
                    try:
                        client.stop()
                    except Exception:
                        self.logger.debug(
                            "WS BRIDGE MARK v3 | wsclient stop failed", exc_info=True
                        )
                if thread and thread is not current_thread:
                    thread.join(timeout=timeout)
                if log:
                    self.logger.info("WS BRIDGE MARK v3 | WSPriceBridge: stopped")

            def _current_symbols(self) -> List[str]:
                with self._symbols_lock:
                    return list(self._symbols)

            def _normalise_symbols(self, symbols: Iterable[str]) -> List[str]:
                return _WebsocketsPriceBridge._normalise_symbols(symbols)

            def _build_batches(self, symbols: Iterable[str]) -> List[List[str]]:
                interval = (self._kline_interval or "1m").strip().lower() or "1m"
                has_external_callbacks = self._has_external_callbacks()
                want_kline = self._on_kline is not None or has_external_callbacks
                want_ticker = self._on_ticker is not None
                want_book = self._on_book_ticker is not None
                streams = make_streams(
                    symbols,
                    kline_interval=interval,
                    include_kline=want_kline,
                    include_ticker=want_ticker,
                    include_book=want_book,
                    ticker_stream="ticker",
                )
                if not streams:
                    return []
                prefix = _streams_prefix(self._combined_base)
                batches = list(
                    _chunk_stream_batches(
                        streams,
                        max_streams=MAX_STREAMS_PER_COMBINED,
                        max_url_len=MAX_COMBINED_URL_LEN,
                        prefix=prefix,
                        log=self.logger,
                    )
                )
                if len(batches) > MAX_CONNS:
                    self.logger.warning(
                        "WS BRIDGE MARK v3 | too many url batches=%d | max_conns=%d | truncating",
                        len(batches),
                        MAX_CONNS,
                    )
                    batches = batches[:MAX_CONNS]
                return [batch for batch in batches if batch]

            def _make_handler(self, batch_index: int) -> Callable[[str], None]:
                def _handler(msg: str, _idx: int = batch_index) -> None:
                    self._handle_msg(msg, _idx)

                return _handler

            def _handle_msg(
                self, raw: str, batch_index: Optional[int] = None
            ) -> None:
                now = time.monotonic()
                if batch_index is not None:
                    with self._lock:
                        self._last_messages[batch_index] = now
                        self._last_msg_mono = now
                else:
                    with self._lock:
                        if self._last_messages:
                            for key in list(self._last_messages.keys()):
                                self._last_messages[key] = now
                        self._last_msg_mono = now
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
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
                    except Exception:
                        logger.exception("WS bridge kline callback failed for %s", symbol)
                self._emit_callbacks(symbol, "kline", payload)
                if payload.get("x"):
                    self._fire_market_event(symbol, "kline_close")

            def _dispatch_ticker(self, payload: Mapping[str, Any]) -> None:
                symbol = str(payload.get("s") or "").upper()
                if not symbol:
                    return
                if self._on_ticker is not None:
                    try:
                        self._on_ticker(symbol, dict(payload))
                    except Exception:
                        logger.exception("WS bridge ticker callback failed for %s", symbol)
                now = time.monotonic()
                last = self._last_ticker_fire.get(symbol, 0.0)
                if now - last >= self._ticker_min_interval:
                    self._last_ticker_fire[symbol] = now
                    self._fire_market_event(symbol, "ticker")

            def _dispatch_book_ticker(self, payload: Mapping[str, Any]) -> None:
                if self._on_book_ticker is None:
                    return
                symbol = str(payload.get("s") or "").upper()
                if not symbol:
                    return
                try:
                    self._on_book_ticker(symbol, dict(payload))
                except Exception:
                    logger.exception(
                        "WS bridge book ticker callback failed for %s", symbol
                    )

            def _heartbeat_loop(self) -> None:
                try:
                    interval = max(0.5, self._heartbeat_timeout / 2.0)
                    while not self._stop.wait(interval):
                        if self._watchdog_tick():
                            return
                        with self._lock:
                            if not self._running:
                                continue
                            last = dict(self._last_messages)
                        if not last:
                            continue
                        now = time.monotonic()
                        gap = max(now - ts for ts in last.values())
                        if gap >= self._heartbeat_timeout:
                            self._notify_stale(gap)
                            self._schedule_reconnect()
                            return
                finally:
                    self.logger.debug(
                        "WS BRIDGE MARK v3 | wsclient heartbeat loop exiting"
                    )

            def _watchdog_tick(self) -> bool:
                if self._stop.is_set():
                    return False
                now = time.monotonic()
                gap = now - self._last_msg_mono
                if gap > self._silence_threshold:
                    self._notify_stale(gap)
                    self.logger.info(
                        {"event": "ws_gap_fallback", "gap_seconds": round(gap, 3)}
                    )
                    self._schedule_reconnect()
                    return True
                return False

            def _schedule_reconnect(self) -> None:
                if self._stop.is_set():
                    return
                if self._reconnects >= self._max_reconnects:
                    self.logger.error(
                        "Max reconnections %s reached; holding stream.",
                        self._max_reconnects,
                    )
                    return

                self._reconnects += 1
                delay_ms = min(self._reconnect_backoff_ms, WS_RECONNECT_MAX_MS)
                self.logger.warning(
                    "Reconnecting WS (attempt %s/%s) in %sms ...",
                    self._reconnects,
                    self._max_reconnects,
                    delay_ms,
                )
                time.sleep(delay_ms / 1000.0)

                try:
                    self._restart_clients()
                except Exception as exc:
                    self.logger.exception("WS reopen failed: %s", exc)
                else:
                    with self._lock:
                        running = self._running
                    if not running:
                        self._reconnects = 0
                        self._reconnect_backoff_ms = max(
                            1, int(WS_RECONNECT_BASE_MS)
                        )
                        return
                    self._reconnects = 0
                    self._reconnect_backoff_ms = max(
                        1, int(WS_RECONNECT_BASE_MS)
                    )
                    self._last_msg_mono = time.monotonic()
                    return

                self._reconnect_backoff_ms = min(
                    int(math.ceil(self._reconnect_backoff_ms * 1.7)),
                    WS_RECONNECT_MAX_MS,
                )

            def _notify_stale(self, gap: float) -> None:
                callback = self._on_stale
                if callback is None:
                    return
                try:
                    callback(gap)
                except Exception:
                    logger.debug("WS stale callback failed", exc_info=True)

            def _fire_market_event(self, symbol: str, kind: str) -> None:
                callback = self._on_market_event
                if callback is None or not symbol:
                    return
                try:
                    result = callback(symbol, kind)
                    if inspect.isawaitable(result):
                        asyncio.run(result)
                except Exception:
                    logger.debug(
                        "WS bridge market event callback failed for %s", symbol, exc_info=True
                    )

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
                        logger.debug(
                            "WS bridge external callback failed", exc_info=True
                        )

            def _has_external_callbacks(self) -> bool:
                with self._callback_lock:
                    return bool(self._extra_callbacks)

        WSPriceBridge = _WSClientPriceBridge
else:
    WSPriceBridge = _WebsocketsPriceBridge

__all__ = [
    "WSPriceBridge",
    "BINANCE_WS",
    "COMBINED_BASE",
    "MAX_STREAMS_PER_COMBINED",
    "make_streams",
]
