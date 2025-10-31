import json
import logging
import threading
import time
from typing import Callable, List, Optional

import websocket  # websocket-client

LOG = logging.getLogger("WSClientBridge")

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="


class WSClientBridge:
    def __init__(self, streams: List[str], on_message: Callable[[str], None]):
        self.streams = [s.strip().lower() for s in streams if s]
        self.on_message = on_message
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self._app: Optional[websocket.WebSocketApp] = None
        self._lock = threading.RLock()

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._stop = True
            try:
                if self._app is not None:
                    self._app.keep_running = False
                    try:
                        self._app.close()
                    except Exception:
                        pass
            finally:
                thread = self._thread
                self._thread = None
                self._app = None
        if thread:
            thread.join(timeout=5.0)

    def _run(self):
        backoff = 1.0
        url = BINANCE_WS_BASE + "/".join(self.streams)
        while not self._stop:
            ws = None
            try:
                LOG.info(
                    "WS BRIDGE MARK v3 | wsclient connecting url=%s | n_streams=%d",
                    url,
                    len(self.streams),
                )
                ws = websocket.WebSocketApp(
                    url,
                    on_message=lambda _ws, msg: self._on_msg(msg),
                    on_error=lambda _ws, err: LOG.warning("wsclient error: %s", err),
                    on_close=lambda _ws, code, msg: self._on_close(code, msg),
                )
                with self._lock:
                    self._app = ws
                ws.run_forever(
                    ping_interval=None,
                    ping_timeout=None,
                    origin=None,
                    http_proxy_host=None,
                    http_proxy_port=None,
                )
            except Exception as e:
                LOG.warning("wsclient exception: %s", e)
            finally:
                with self._lock:
                    if self._app is ws:
                        self._app = None
            if self._stop:
                break
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 30.0)
            LOG.info(
                "WS BRIDGE MARK v3 | wsclient reconnecting backoff=%.1fs", backoff
            )
        with self._lock:
            if self._thread is threading.current_thread():
                self._thread = None

    def _on_msg(self, msg: str):
        try:
            if self.on_message:
                self.on_message(msg)
        except Exception:
            LOG.exception("wsclient on_message error")

    def _on_close(self, code, msg):
        LOG.info("wsclient closed code=%s msg=%s", code, msg)
        with self._lock:
            self._app = None
