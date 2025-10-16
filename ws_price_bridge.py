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
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Mapping

import websockets

BINANCE_WS = "wss://stream.binance.com:9443/stream"

logger = logging.getLogger(__name__)

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
    ) -> None:
        self._symbols: List[str] = self._normalise_symbols(symbols)
        self._kline_interval = str(kline_interval or "1m").strip()
        if not self._kline_interval:
            self._kline_interval = "1m"
        self._on_kline = on_kline
        self._on_ticker = on_ticker
        self._on_book_ticker = on_book_ticker
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._resubscribe = threading.Event()
        self._symbols_lock = threading.Lock()

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

    def update_symbols(self, symbols: Iterable[str]) -> None:
        """Update the subscribed symbols."""

        normalised = self._normalise_symbols(symbols)
        with self._symbols_lock:
            if normalised == self._symbols:
                return
            self._symbols = normalised
        logger.info("WS bridge subscribing to %d symbols", len(normalised))
        self._resubscribe.set()
        self.start()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(lambda: None)

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
        backoff = 1.0
        while not self._stop.is_set():
            symbols = self._current_symbols()
            if not symbols:
                await asyncio.sleep(1.0)
                continue
            url = self._build_stream_url(symbols)
            logger.info("Connecting to Binance WS: %s", url)
            try:
                async with websockets.connect(
                    url, ping_interval=15, ping_timeout=15
                ) as ws:
                    backoff = 1.0
                    await self._listen(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("WS bridge error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _listen(self, ws: websockets.WebSocketClientProtocol) -> None:
        while not self._stop.is_set():
            if self._resubscribe.is_set():
                self._resubscribe.clear()
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
        if not symbol or self._on_kline is None:
            return
        try:
            self._on_kline(symbol, interval, dict(kline))
        except Exception:  # pragma: no cover - callback safety
            logger.exception("WS bridge kline callback failed for %s", symbol)

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

    def _build_stream_url(self, symbols: Sequence[str]) -> str:
        streams: List[str] = []
        for sym in symbols:
            streams.append(f"{sym}@kline_{self._kline_interval}")
            streams.append(f"{sym}@miniTicker")
            if self._on_book_ticker is not None:
                streams.append(f"{sym}@bookTicker")
        params = "/".join(streams)
        return f"{BINANCE_WS}?streams={params}"

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


__all__ = ["WSPriceBridge", "BINANCE_WS"]
