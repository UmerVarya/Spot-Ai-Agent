"""Utilities for streaming high-frequency market data from Binance.

The scalping agent previously relied on REST endpoints that expose
minute-level klines and static depth snapshots.  To react to
microstructure changes the agent now maintains lightweight in-memory
views of the live order book and recent trades using Binance WebSocket
streams.  The :class:`LiveMarketDataManager` class encapsulates the
socket lifecycle and exposes thread-safe accessors for the rest of the
code base.

The manager is intentionally resilient: when the ``python-binance``
library is missing or a network error occurs, all public methods return
``None`` so that callers can gracefully fall back to REST polling.  This
keeps the module safe to import in offline test environments.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional, Tuple

from log_utils import setup_logger
from symbol_mapper import map_symbol_for_binance

try:  # pragma: no cover - depends on optional runtime dependency
    from binance import Client
    from binance import ThreadedWebsocketManager
except Exception:  # pragma: no cover - gracefully handle missing package
    Client = None  # type: ignore
    ThreadedWebsocketManager = None  # type: ignore


logger = setup_logger(__name__)


class LiveMarketDataManager:
    """Maintain lightweight live market data caches via WebSockets.

    Parameters
    ----------
    depth_levels : int, optional
        Number of depth levels to retain for each side of the book.
    trade_window : int, optional
        Rolling window (in seconds) used for aggregating trade flow
        statistics such as cumulative volume delta (CVD).
    """

    def __init__(self, depth_levels: int = 50, trade_window: int = 300) -> None:
        self._depth_levels = depth_levels
        self._trade_window = trade_window
        self._twm: Optional[ThreadedWebsocketManager] = None
        self._rest_client: Optional[Client] = None
        self._streams: Dict[str, Dict[str, Optional[str]]] = {}
        self._order_books: Dict[str, Dict[str, object]] = {}
        self._trade_windows: Dict[str, Deque[Dict[str, float]]] = defaultdict(deque)
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._manager_lock = threading.Lock()
        self._available = Client is not None and ThreadedWebsocketManager is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_manager(self) -> bool:
        if not self._available:
            return False
        if self._twm is not None:
            return True
        with self._manager_lock:
            if self._twm is not None:
                return True
            try:
                api_key = os.getenv("BINANCE_API_KEY")
                api_secret = os.getenv("BINANCE_API_SECRET")
                self._twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
                self._twm.start()
                logger.info("Started Binance ThreadedWebsocketManager")
                return True
            except Exception as exc:  # pragma: no cover - depends on network
                logger.warning("Failed to start Binance websocket manager: %s", exc, exc_info=True)
                self._twm = None
                return False

    def _get_rest_client(self) -> Optional[Client]:
        if not self._available:
            return None
        if self._rest_client is not None:
            return self._rest_client
        try:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            self._rest_client = Client(api_key, api_secret)
            return self._rest_client
        except Exception as exc:  # pragma: no cover - depends on network
            logger.warning("Failed to initialise REST client for websocket bootstrap: %s", exc)
            self._rest_client = None
            return None

    def _bootstrap_order_book(self, symbol: str) -> None:
        rest_client = self._get_rest_client()
        if rest_client is None:
            return
        try:
            snapshot = rest_client.get_order_book(symbol=symbol, limit=self._depth_levels)
        except Exception as exc:  # pragma: no cover - depends on network
            logger.warning("Failed to fetch initial order book snapshot for %s: %s", symbol, exc)
            return
        bids = {float(price): float(qty) for price, qty in snapshot.get("bids", [])}
        asks = {float(price): float(qty) for price, qty in snapshot.get("asks", [])}
        self._order_books[symbol] = {
            "bids": bids,
            "asks": asks,
            "last_update_id": snapshot.get("lastUpdateId", 0),
            "last_update_ts": time.time(),
        }

    def _handle_depth_message(self, symbol: str, message: Dict[str, object]) -> None:
        if not message or "e" not in message:
            return
        if message.get("e") == "error":
            logger.warning("Depth stream for %s reported error: %s", symbol, message)
            return
        try:
            bids: Iterable[Tuple[str, str]] = message.get("b", [])  # type: ignore[assignment]
            asks: Iterable[Tuple[str, str]] = message.get("a", [])  # type: ignore[assignment]
            update_id = int(message.get("u", 0))
        except Exception:
            return
        lock = self._locks[symbol]
        with lock:
            book = self._order_books.setdefault(symbol, {
                "bids": {},
                "asks": {},
                "last_update_id": 0,
                "last_update_ts": 0.0,
            })
            last_update_id = int(book.get("last_update_id", 0))
            if update_id <= last_update_id:
                return
            book_bids: Dict[float, float] = book.setdefault("bids", {})  # type: ignore[assignment]
            book_asks: Dict[float, float] = book.setdefault("asks", {})  # type: ignore[assignment]
            for price_str, qty_str in bids:
                price = float(price_str)
                qty = float(qty_str)
                if qty <= 0:
                    book_bids.pop(price, None)
                else:
                    book_bids[price] = qty
            for price_str, qty_str in asks:
                price = float(price_str)
                qty = float(qty_str)
                if qty <= 0:
                    book_asks.pop(price, None)
                else:
                    book_asks[price] = qty
            book["last_update_id"] = update_id
            book["last_update_ts"] = time.time()

    def _handle_trade_message(self, symbol: str, message: Dict[str, object]) -> None:
        if not message or "e" not in message:
            return
        if message.get("e") == "error":
            logger.warning("Trade stream for %s reported error: %s", symbol, message)
            return
        try:
            price = float(message.get("p", 0.0))
            quantity = float(message.get("q", 0.0))
            is_buyer_maker = bool(message.get("m", False))
            trade_ts_ms = int(message.get("T", int(time.time() * 1000)))
        except Exception:
            return
        trade_ts = trade_ts_ms / 1000.0
        buy_qty = 0.0 if is_buyer_maker else quantity
        sell_qty = quantity if is_buyer_maker else 0.0
        entry = {
            "ts": trade_ts,
            "buy_qty": buy_qty,
            "sell_qty": sell_qty,
            "qty": quantity,
            "price": price,
            "quote_buy": buy_qty * price,
            "quote_sell": sell_qty * price,
        }
        lock = self._locks[symbol]
        with lock:
            window = self._trade_windows[symbol]
            window.append(entry)
            cutoff = trade_ts - self._trade_window
            while window and window[0]["ts"] < cutoff:
                window.popleft()

    def _start_symbol_streams(self, mapped_symbol: str) -> bool:
        if not self._ensure_manager():
            return False
        if mapped_symbol in self._streams:
            return True
        self._bootstrap_order_book(mapped_symbol)
        if self._twm is None:
            return False
        try:
            depth_key = self._twm.start_depth_socket(
                callback=lambda msg, sym=mapped_symbol: self._handle_depth_message(sym, msg),
                symbol=mapped_symbol,
                depth=self._depth_levels,
            )
            trade_key = self._twm.start_aggtrade_socket(
                callback=lambda msg, sym=mapped_symbol: self._handle_trade_message(sym, msg),
                symbol=mapped_symbol,
            )
            self._streams[mapped_symbol] = {"depth": depth_key, "trade": trade_key}
            logger.info("Subscribed to depth/trade streams for %s", mapped_symbol)
            return True
        except Exception as exc:  # pragma: no cover - depends on network
            logger.warning("Failed to subscribe to websocket streams for %s: %s", mapped_symbol, exc)
            self._streams.pop(mapped_symbol, None)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_symbol(self, symbol: str) -> bool:
        """Ensure that live streams for ``symbol`` are running."""

        if not self._available:
            return False
        mapped_symbol = map_symbol_for_binance(symbol)
        return self._start_symbol_streams(mapped_symbol)

    def get_order_book(self, symbol: str, depth: int = 50) -> Optional[Dict[str, object]]:
        """Return the most recent order book snapshot for ``symbol``."""

        if not self.ensure_symbol(symbol):
            return None
        mapped_symbol = map_symbol_for_binance(symbol)
        lock = self._locks[mapped_symbol]
        with lock:
            book = self._order_books.get(mapped_symbol)
            if not book:
                return None
            bids_dict: Dict[float, float] = book.get("bids", {})  # type: ignore[assignment]
            asks_dict: Dict[float, float] = book.get("asks", {})  # type: ignore[assignment]
            if not bids_dict or not asks_dict:
                return None
            bids_sorted = sorted(bids_dict.items(), key=lambda item: item[0], reverse=True)[:depth]
            asks_sorted = sorted(asks_dict.items(), key=lambda item: item[0])[:depth]
            return {
                "bids": bids_sorted,
                "asks": asks_sorted,
                "last_update_id": book.get("last_update_id"),
                "last_update_ts": book.get("last_update_ts"),
            }

    def get_trade_flow(self, symbol: str, window_seconds: Optional[int] = None) -> Optional[Dict[str, float]]:
        """Return aggregated trade flow metrics for ``symbol``.

        Parameters
        ----------
        symbol : str
            Trading pair in the agent's notation (e.g. ``"BTCUSDT"``).
        window_seconds : int, optional
            Override the aggregation horizon. Defaults to the manager's
            ``trade_window`` value.
        """

        if not self.ensure_symbol(symbol):
            return None
        mapped_symbol = map_symbol_for_binance(symbol)
        horizon = window_seconds or self._trade_window
        lock = self._locks[mapped_symbol]
        with lock:
            window = self._trade_windows.get(mapped_symbol)
            if not window:
                return None
            now = time.time()
            cutoff = now - horizon
            while window and window[0]["ts"] < cutoff:
                window.popleft()
            if not window:
                return None
            buy_volume = sum(entry["buy_qty"] for entry in window)
            sell_volume = sum(entry["sell_qty"] for entry in window)
            total_volume = sum(entry["qty"] for entry in window)
            quote_buy = sum(entry["quote_buy"] for entry in window)
            quote_sell = sum(entry["quote_sell"] for entry in window)
            trade_count = float(len(window))
            first_ts = window[0]["ts"]
            last_ts = window[-1]["ts"]
            duration = max(last_ts - first_ts, 1e-6)
            trades_per_second = trade_count / duration
            return {
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "net_volume": buy_volume - sell_volume,
                "total_volume": total_volume,
                "buy_quote_volume": quote_buy,
                "sell_quote_volume": quote_sell,
                "net_quote_volume": quote_buy - quote_sell,
                "trade_count": trade_count,
                "trades_per_second": trades_per_second,
                "window_seconds": duration,
                "last_update_ts": last_ts,
            }


# Global singleton used by the trading agent.
market_data_stream = LiveMarketDataManager()

