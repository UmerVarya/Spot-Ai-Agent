"""Threaded Binance user data stream bridge."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Callable, Dict, Optional

import requests
import websockets

logger = logging.getLogger(__name__)

BINANCE_API_BASE = os.getenv("BINANCE_API_BASE", "https://api.binance.com")
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"

UserStreamCallback = Callable[[Dict[str, Any]], None]


class UserDataStreamBridge:
    """Bridge Binance user data stream into callback-based handlers."""

    def __init__(
        self,
        *,
        on_event: Optional[UserStreamCallback] = None,
        keepalive_interval: float = 30 * 60,
    ) -> None:
        self._on_event = on_event
        self._keepalive_interval = max(60.0, float(keepalive_interval))
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop = threading.Event()
        self._listen_key_lock = threading.Lock()
        self._listen_key: Optional[str] = None
        self._api_key = os.getenv("BINANCE_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._api_key:
            logger.debug("BINANCE_API_KEY missing; skipping user data stream bridge")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="binance-user-stream", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self._loop = None
        listen_key = self._listen_key
        if listen_key:
            try:
                requests.delete(
                    f"{BINANCE_API_BASE}/api/v3/userDataStream",
                    headers={"X-MBX-APIKEY": self._api_key or ""},
                    params={"listenKey": listen_key},
                    timeout=10,
                )
            except Exception:
                logger.debug("Failed to delete listen key", exc_info=True)
        with self._listen_key_lock:
            self._listen_key = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.create_task(self._stream_main())
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                try:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
            loop.close()

    async def _stream_main(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            listen_key = await asyncio.to_thread(self._ensure_listen_key)
            if not listen_key:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
                continue
            url = f"{BINANCE_WS_BASE}/{listen_key}"
            logger.info("Connecting to Binance user stream")
            keepalive_task = self._loop.create_task(self._keepalive_loop(listen_key)) if self._loop else None
            try:
                async with websockets.connect(url, ping_interval=15, ping_timeout=15) as ws:
                    backoff = 1.0
                    await self._listen(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("User data stream error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
            finally:
                if keepalive_task:
                    keepalive_task.cancel()
                    try:
                        await keepalive_task
                    except Exception:
                        pass
                await asyncio.to_thread(self._refresh_listen_key)

    async def _listen(self, ws: websockets.WebSocketClientProtocol) -> None:
        while not self._stop.is_set():
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
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("User stream received non-JSON payload")
            return
        if not isinstance(payload, dict):
            return
        if callable(self._on_event):
            try:
                self._on_event(dict(payload))
            except Exception:
                logger.exception("User stream callback failed")

    async def _keepalive_loop(self, listen_key: str) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self._keepalive_interval)
            await asyncio.to_thread(self._keepalive, listen_key)

    def _keepalive(self, listen_key: str) -> None:
        try:
            requests.put(
                f"{BINANCE_API_BASE}/api/v3/userDataStream",
                headers={"X-MBX-APIKEY": self._api_key or ""},
                params={"listenKey": listen_key},
                timeout=10,
            )
        except Exception:
            logger.debug("Failed to keep listen key alive", exc_info=True)

    def _ensure_listen_key(self) -> Optional[str]:
        with self._listen_key_lock:
            if self._listen_key:
                return self._listen_key
        try:
            response = requests.post(
                f"{BINANCE_API_BASE}/api/v3/userDataStream",
                headers={"X-MBX-APIKEY": self._api_key or ""},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            listen_key = data.get("listenKey")
            if listen_key:
                with self._listen_key_lock:
                    self._listen_key = listen_key
                return listen_key
        except Exception:
            logger.warning("Failed to obtain Binance listen key", exc_info=True)
        return None

    def _refresh_listen_key(self) -> None:
        with self._listen_key_lock:
            self._listen_key = None


__all__ = ["UserDataStreamBridge"]
