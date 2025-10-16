"""Utilities for streaming Binance market data via WebSockets.

This module exposes a lightweight manager that keeps a rolling in-memory
order book and taker-flow statistics for each subscribed symbol.  The trading
agent can query these structures to obtain second-by-second microstructure
signals such as order book imbalance, cumulative volume delta (CVD), and trade
rates without waiting for REST snapshots.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

from observability import log_event, record_metric

try:  # pragma: no cover - optional import during docs build
    from binance import ThreadedWebsocketManager  # type: ignore
    from binance.client import Client  # type: ignore
except Exception:  # pragma: no cover - handled gracefully in runtime
    ThreadedWebsocketManager = None  # type: ignore
    Client = None  # type: ignore

try:
    from symbol_mapper import map_symbol_for_binance  # type: ignore
except Exception:  # pragma: no cover - fallback mapping
    def map_symbol_for_binance(symbol: str) -> str:
        return symbol.upper()

logger = logging.getLogger(__name__)


@dataclass
class TradeEntry:
    """Representation of a single trade for rolling statistics."""

    timestamp: float
    base_qty: float
    quote_qty: float
    is_taker_buy: bool


class RollingTradeStats:
    """Maintain rolling taker flow statistics for a symbol."""

    def __init__(self, window_seconds: int = 120) -> None:
        self.window_seconds = window_seconds
        self._entries: Deque[TradeEntry] = deque()
        self._lock = threading.Lock()
        self._buy_base = 0.0
        self._sell_base = 0.0
        self._buy_quote = 0.0
        self._sell_quote = 0.0
        self._buy_count = 0
        self._sell_count = 0
        self._last_event_ts = 0.0
        # Cumulative figures since stream started for coarse CVD estimates
        self._cumulative_net_base = 0.0
        self._cumulative_total_base = 0.0
        self._cumulative_net_quote = 0.0
        self._cumulative_total_quote = 0.0

    def add_trade(self, base_qty: float, price: float, is_taker_buy: bool, timestamp: Optional[float]) -> None:
        if not (base_qty > 0 and price > 0):
            return
        event_ts = float(timestamp) if timestamp is not None else time.time()
        quote_qty = base_qty * price
        entry = TradeEntry(timestamp=event_ts, base_qty=base_qty, quote_qty=quote_qty, is_taker_buy=is_taker_buy)
        with self._lock:
            self._entries.append(entry)
            if is_taker_buy:
                self._buy_base += base_qty
                self._buy_quote += quote_qty
                self._buy_count += 1
                self._cumulative_net_base += base_qty
                self._cumulative_net_quote += quote_qty
            else:
                self._sell_base += base_qty
                self._sell_quote += quote_qty
                self._sell_count += 1
                self._cumulative_net_base -= base_qty
                self._cumulative_net_quote -= quote_qty
            self._cumulative_total_base += base_qty
            self._cumulative_total_quote += quote_qty
            self._last_event_ts = event_ts
            self._trim_locked(event_ts)

    def _trim_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._entries and self._entries[0].timestamp < cutoff:
            stale = self._entries.popleft()
            if stale.is_taker_buy:
                self._buy_base -= stale.base_qty
                self._buy_quote -= stale.quote_qty
                self._buy_count -= 1
            else:
                self._sell_base -= stale.base_qty
                self._sell_quote -= stale.quote_qty
                self._sell_count -= 1

    def _build_price_footprint(
        self,
        entries: Iterable[TradeEntry],
        bin_pct: float = 0.0001,
    ) -> Tuple[List[Dict[str, float]], float]:
        """Aggregate executed trades into price-level bins."""

        entries_list = [entry for entry in entries if entry.base_qty > 0 and entry.quote_qty > 0]
        if not entries_list:
            return [], 0.0

        last_price = None
        for entry in reversed(entries_list):
            price = entry.quote_qty / entry.base_qty
            if math.isfinite(price) and price > 0:
                last_price = price
                break

        if not last_price or last_price <= 0:
            return [], 0.0

        bin_size = max(last_price * bin_pct, 1e-9)
        bins: Dict[float, List[float]] = defaultdict(lambda: [0.0, 0.0])

        for entry in entries_list:
            price = entry.quote_qty / entry.base_qty
            if not math.isfinite(price) or price <= 0:
                continue
            bin_index = int(round(price / bin_size)) if bin_size > 0 else 0
            price_level = round(bin_index * bin_size, 8)
            bucket = bins[price_level]
            if entry.is_taker_buy:
                bucket[0] += entry.base_qty
            else:
                bucket[1] += entry.base_qty

        footprint = [
            {
                "price": float(level),
                "buy_volume": float(values[0]),
                "sell_volume": float(values[1]),
            }
            for level, values in sorted(bins.items())
            if values[0] > 0 or values[1] > 0
        ]

        return footprint, float(bin_size)

    def snapshot(self, window_seconds: Optional[int] = None) -> Dict[str, float]:
        window = int(window_seconds) if window_seconds is not None else self.window_seconds
        now = time.time()
        with self._lock:
            entries_list: List[TradeEntry]
            if window != self.window_seconds:
                cutoff = now - window
                buy_base = 0.0
                sell_base = 0.0
                buy_quote = 0.0
                sell_quote = 0.0
                buy_count = 0
                sell_count = 0
                entries_list = []
                for entry in self._entries:
                    if entry.timestamp < cutoff:
                        continue
                    entries_list.append(entry)
                    if entry.is_taker_buy:
                        buy_base += entry.base_qty
                        buy_quote += entry.quote_qty
                        buy_count += 1
                    else:
                        sell_base += entry.base_qty
                        sell_quote += entry.quote_qty
                        sell_count += 1
            else:
                self._trim_locked(now)
                entries_list = list(self._entries)
                buy_base = max(self._buy_base, 0.0)
                sell_base = max(self._sell_base, 0.0)
                buy_quote = max(self._buy_quote, 0.0)
                sell_quote = max(self._sell_quote, 0.0)
                buy_count = max(self._buy_count, 0)
                sell_count = max(self._sell_count, 0)

            total_base = buy_base + sell_base
            total_quote = buy_quote + sell_quote
            total_trades = buy_count + sell_count
            trade_rate = total_trades / max(window, 1)
            footprint_bins, bin_size = self._build_price_footprint(entries_list)
            return {
                "window_seconds": float(window),
                "last_event_ts": float(self._last_event_ts),
                "buy_base_volume": float(buy_base),
                "sell_base_volume": float(sell_base),
                "buy_quote_volume": float(buy_quote),
                "sell_quote_volume": float(sell_quote),
                "net_base_volume": float(buy_base - sell_base),
                "net_quote_volume": float(buy_quote - sell_quote),
                "total_base_volume": float(total_base),
                "total_quote_volume": float(total_quote),
                "buy_trades": float(buy_count),
                "sell_trades": float(sell_count),
                "total_trades": float(total_trades),
                "trade_rate_per_sec": float(trade_rate),
                "cumulative_net_base_volume": float(self._cumulative_net_base),
                "cumulative_total_base_volume": float(self._cumulative_total_base),
                "cumulative_net_quote_volume": float(self._cumulative_net_quote),
                "cumulative_total_quote_volume": float(self._cumulative_total_quote),
                "price_footprint_bins": footprint_bins,
                "price_footprint_bin_size": float(bin_size),
            }


class OrderBookState:
    """Maintain an incremental view of the order book."""

    def __init__(self, depth: int = 50) -> None:
        self.depth = depth
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_update_id: int = 0
        self.last_event_ts: float = 0.0
        self._lock = threading.Lock()

    def apply_snapshot(self, bids: Iterable[Tuple[str, str]], asks: Iterable[Tuple[str, str]], last_update_id: int) -> None:
        with self._lock:
            self.bids = {float(price): float(qty) for price, qty in bids if float(qty) > 0}
            self.asks = {float(price): float(qty) for price, qty in asks if float(qty) > 0}
            self.last_update_id = int(last_update_id)
            self.last_event_ts = time.time()
            self._trim_locked()

    def apply_diff(self, update: dict) -> None:
        bids = update.get("b", [])
        asks = update.get("a", [])
        final_update_id = int(update.get("u", 0))
        first_update_id = int(update.get("U", final_update_id))
        event_time = float(update.get("E", time.time()))
        with self._lock:
            if self.last_update_id and final_update_id <= self.last_update_id:
                return
            if self.last_update_id and first_update_id > self.last_update_id + 1:
                # Gap detected; discard to force refresh
                return
            for price_str, qty_str in bids:
                price = float(price_str)
                qty = float(qty_str)
                if qty <= 0:
                    self.bids.pop(price, None)
                else:
                    self.bids[price] = qty
            for price_str, qty_str in asks:
                price = float(price_str)
                qty = float(qty_str)
                if qty <= 0:
                    self.asks.pop(price, None)
                else:
                    self.asks[price] = qty
            self.last_update_id = final_update_id
            self.last_event_ts = max(self.last_event_ts, event_time)
            self._trim_locked()

    def mark_stale(self) -> None:
        with self._lock:
            self.last_update_id = 0

    def _trim_locked(self) -> None:
        if len(self.bids) > self.depth * 4:
            top_bids = sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)[: self.depth * 2]
            self.bids = dict(top_bids)
        if len(self.asks) > self.depth * 4:
            top_asks = sorted(self.asks.items(), key=lambda kv: kv[0])[: self.depth * 2]
            self.asks = dict(top_asks)

    def snapshot(self, depth: Optional[int] = None) -> Dict[str, List[Tuple[float, float]]]:
        book_depth = depth or self.depth
        with self._lock:
            bids = sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)[:book_depth]
            asks = sorted(self.asks.items(), key=lambda kv: kv[0])[:book_depth]
            return {
                "bids": [(float(price), float(qty)) for price, qty in bids],
                "asks": [(float(price), float(qty)) for price, qty in asks],
                "last_update_id": float(self.last_update_id),
                "last_event_ts": float(self.last_event_ts),
            }


class BinanceMarketStream:
    """Manage Binance WebSocket subscriptions for depth and trade data."""

    _SUPPORTED_SOCKET_DEPTHS: Tuple[int, ...] = (5, 10, 20)

    def __init__(self, depth: int = 50, trade_window_seconds: int = 120) -> None:
        self.depth = depth
        self.trade_window_seconds = trade_window_seconds
        self._socket_depth = self._resolve_socket_depth(depth)
        if self._socket_depth != depth:
            logger.debug(
                "Requested depth %s is not supported for Binance sockets; using %s.",
                depth,
                self._socket_depth,
            )
        self._twm: Optional[ThreadedWebsocketManager] = None
        self._client: Optional[Client] = None
        self._manager_lock = threading.Lock()
        self._symbol_streams: Dict[str, Dict[str, int]] = {}
        self._order_books: Dict[str, OrderBookState] = {}
        self._trades: Dict[str, RollingTradeStats] = {}
        self._symbol_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._disabled = False

    def _ensure_manager(self) -> None:
        if self._disabled:
            return
        if ThreadedWebsocketManager is None or Client is None:
            logger.debug("Binance WebSocket dependencies unavailable; disabling stream.")
            self._disabled = True
            return
        with self._manager_lock:
            if self._twm is None:
                try:
                    api_key = os.getenv("BINANCE_API_KEY")
                    api_secret = os.getenv("BINANCE_API_SECRET")
                    self._twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
                    self._twm.start()
                except Exception as exc:  # pragma: no cover - network dependent
                    logger.warning("Failed to start Binance WebSocket manager: %s", exc)
                    self._twm = None
                    self._disabled = True
                    return
            if self._client is None:
                try:
                    api_key = os.getenv("BINANCE_API_KEY")
                    api_secret = os.getenv("BINANCE_API_SECRET")
                    if api_key and api_secret:
                        self._client = Client(api_key, api_secret)
                    else:
                        self._client = Client()
                except Exception as exc:  # pragma: no cover - network dependent
                    logger.warning("Failed to create Binance REST client for stream: %s", exc)
                    self._client = None

    def _reset_symbol(self, symbol: str) -> None:
        self._order_books.pop(symbol, None)
        self._trades.pop(symbol, None)
        streams = self._symbol_streams.pop(symbol, {})
        if self._twm:
            for stream_id in streams.values():
                try:
                    self._twm.stop_socket(stream_id)
                except Exception:  # pragma: no cover - best effort
                    continue

    def _resolve_socket_depth(self, requested_depth: int) -> int:
        """Return the closest supported depth level for Binance sockets."""

        if requested_depth <= 0:
            return self._SUPPORTED_SOCKET_DEPTHS[0]
        for level in self._SUPPORTED_SOCKET_DEPTHS:
            if requested_depth <= level:
                return level
        return self._SUPPORTED_SOCKET_DEPTHS[-1]

    def _process_depth_message(
        self, state: OrderBookState, msg: dict, mapped_symbol: str
    ) -> None:
        """Apply a depth message to ``state`` handling diff and partial formats."""

        if not isinstance(msg, dict):
            return

        if msg.get("e") == "depthUpdate":
            state.apply_diff(msg)
            if state.last_update_id == 0:
                # Force resubscribe on gap
                logger.debug(
                    "Depth stream gap detected for %s; requesting resubscribe.", mapped_symbol
                )
                state.mark_stale()
                self._reset_symbol(mapped_symbol)
            return

        if {"lastUpdateId", "bids", "asks"}.issubset(msg.keys()):
            state.apply_snapshot(
                msg.get("bids", []),
                msg.get("asks", []),
                msg.get("lastUpdateId", 0),
            )

    def _ensure_symbol(self, symbol: str) -> None:
        mapped_symbol = map_symbol_for_binance(symbol)
        if self._disabled:
            return
        self._ensure_manager()
        if self._disabled or self._twm is None:
            return
        symbol_lock = self._symbol_locks[mapped_symbol]
        with symbol_lock:
            if mapped_symbol in self._symbol_streams:
                return
            state = OrderBookState(depth=min(self.depth, self._socket_depth))
            trades = RollingTradeStats(window_seconds=self.trade_window_seconds)
            snapshot_ok = False
            if self._client is not None:
                try:
                    snapshot = self._client.get_order_book(symbol=mapped_symbol, limit=max(self.depth, 100))
                    if snapshot:
                        state.apply_snapshot(snapshot.get("bids", []), snapshot.get("asks", []), snapshot.get("lastUpdateId", 0))
                        snapshot_ok = True
                except Exception as exc:  # pragma: no cover - network dependent
                    logger.debug("Failed to fetch initial order book snapshot for %s: %s", mapped_symbol, exc)
            if not snapshot_ok:
                logger.debug("Order book snapshot unavailable for %s; live updates may be delayed.", mapped_symbol)
            self._order_books[mapped_symbol] = state
            self._trades[mapped_symbol] = trades

            def _handle_depth(msg: dict) -> None:
                self._process_depth_message(state, msg, mapped_symbol)

            def _handle_trade(msg: dict) -> None:
                if not isinstance(msg, dict):
                    return
                qty = float(msg.get("q", 0.0))
                price = float(msg.get("p", 0.0))
                is_buyer_maker = bool(msg.get("m", False))
                # In Binance streams ``m`` indicates whether the buyer is the market maker.
                # When the buyer is the maker, the taker was a sell order.
                is_taker_buy = not is_buyer_maker
                event_ts = float(msg.get("E")) / 1000.0 if msg.get("E") else None
                trades.add_trade(qty, price, is_taker_buy, event_ts)

            try:
                depth_socket = self._twm.start_depth_socket(
                    callback=_handle_depth,
                    symbol=mapped_symbol.lower(),
                )
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Failed to subscribe to depth stream for %s: %s", mapped_symbol, exc)
                self._disabled = True
                return
            trade_socket = None
            try:
                trade_socket = self._twm.start_aggtrade_socket(callback=_handle_trade, symbol=mapped_symbol.lower())
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("Failed to subscribe to trade stream for %s: %s", mapped_symbol, exc)
            self._symbol_streams[mapped_symbol] = {
                "depth": depth_socket,
                "trade": trade_socket or 0,
            }

    def get_order_book(self, symbol: str, depth: int = 50) -> Optional[Dict[str, List[Tuple[float, float]]]]:
        mapped_symbol = map_symbol_for_binance(symbol)
        if self._disabled:
            return None
        self._ensure_symbol(symbol)
        state = self._order_books.get(mapped_symbol)
        if state is None:
            return None
        snapshot = state.snapshot(depth=depth)
        if not snapshot.get("bids") and not snapshot.get("asks"):
            return None
        return snapshot

    def get_trade_snapshot(self, symbol: str, window_seconds: int = 60) -> Optional[Dict[str, float]]:
        mapped_symbol = map_symbol_for_binance(symbol)
        if self._disabled:
            return None
        self._ensure_symbol(symbol)
        stats = self._trades.get(mapped_symbol)
        if stats is None:
            return None
        return stats.snapshot(window_seconds=window_seconds)

    def stop(self) -> None:  # pragma: no cover - only used in live runs
        if self._twm is not None:
            try:
                self._twm.stop()
            except Exception:
                pass
            self._twm = None
        self._symbol_streams.clear()


_global_stream: Optional[BinanceMarketStream] = None
_global_lock = threading.Lock()


def get_market_stream() -> BinanceMarketStream:
    """Return a process-wide singleton ``BinanceMarketStream`` instance."""

    global _global_stream
    with _global_lock:
        if _global_stream is None:
            _global_stream = BinanceMarketStream()
        return _global_stream


class ExponentialBackoff:
    """Simple exponential backoff helper."""

    def __init__(
        self,
        *,
        base: float = 1.0,
        factor: float = 2.0,
        max_interval: float = 60.0,
        jitter: float = 0.25,
    ) -> None:
        self.base = float(base)
        self.factor = float(factor)
        self.max_interval = float(max_interval)
        self.jitter = float(jitter)
        self._attempts = 0
        self._lock = threading.Lock()

    def next_interval(self) -> float:
        with self._lock:
            interval = min(self.max_interval, self.base * (self.factor ** self._attempts))
            if self.jitter:
                jitter = interval * random.uniform(-self.jitter, self.jitter)
                interval = max(self.base, interval + jitter)
            self._attempts += 1
            return interval

    def reset(self) -> None:
        with self._lock:
            self._attempts = 0


class BinanceEventStream:
    """Event-driven wrapper around Binance WebSocket feeds.

    The class exposes a thread-safe queue of market events.  Each event is a
    dictionary with at least ``type`` and ``symbol`` keys.  Connection health is
    continuously monitored via heartbeat timestamps.  If the stream becomes
    stale the sockets are restarted with exponential backoff.  When sockets
    cannot be established the class falls back to REST polling to keep prices
    reasonably fresh until WebSocket connectivity resumes.
    """

    HEARTBEAT_SECONDS = 10.0
    STALE_AFTER_SECONDS = 30.0

    def __init__(
        self,
        symbols: Optional[Iterable[str]] = None,
        *,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_queue: int = 100,
    ) -> None:
        self._symbols = {map_symbol_for_binance(symbol) for symbol in (symbols or [])}
        self._on_event = on_event
        self._queue_maxsize = max(1, int(max_queue))
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self._queue_maxsize)
        self._manager_lock = threading.Lock()
        self._twm: Optional[ThreadedWebsocketManager] = None
        self._client: Optional[Client] = None
        self._stream_ids: Dict[str, Dict[str, int]] = {}
        self._user_socket_id: Optional[int] = None
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_message = 0.0
        self._connected = threading.Event()
        self._backoff = ExponentialBackoff(base=1.0, factor=2.0, max_interval=90.0)
        self._disabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def event_queue(self) -> "queue.Queue[Dict[str, Any]]":
        return self._queue

    def set_symbols(self, symbols: Iterable[str]) -> None:
        mapped = {map_symbol_for_binance(symbol) for symbol in symbols}
        with self._manager_lock:
            if mapped == self._symbols:
                return
            removed = self._symbols - mapped
            added = mapped - self._symbols
            self._symbols = mapped
        for symbol in removed:
            self._stop_symbol(symbol)
        for symbol in added:
            self._start_symbol(symbol)

    def start(self) -> None:
        if self._disabled:
            return
        self._ensure_manager()
        if self._disabled:
            return
        self._stop_event.clear()
        with self._manager_lock:
            for symbol in list(self._symbols):
                self._start_symbol(symbol)
            self._start_user_socket()
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
        self._backoff.reset()

    def stop(self) -> None:
        self._stop_event.set()
        with self._manager_lock:
            for symbol in list(self._stream_ids):
                self._stop_symbol(symbol)
            if self._twm is not None:
                try:
                    if self._user_socket_id is not None:
                        self._twm.stop_socket(self._user_socket_id)
                except Exception:
                    pass
            self._user_socket_id = None
        if self._twm is not None:
            try:
                self._twm.stop()
            except Exception:
                pass
            self._twm = None
        self._connected.clear()

    def ensure_alive(self) -> None:
        if self._stop_event.is_set() or self._disabled:
            return
        now = time.time()
        if now - self._last_message > self.STALE_AFTER_SECONDS:
            logger.warning("Binance event stream heartbeat stale; reconnecting sockets")
            self._restart_streams()

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def poll_rest(self, symbols: Optional[Iterable[str]] = None) -> None:
        if self._client is None:
            return
        for symbol in symbols or list(self._symbols):
            try:
                ticker = self._client.get_symbol_ticker(symbol=symbol)
                price = float(ticker.get("price", 0.0))
            except Exception as exc:  # pragma: no cover - network dependent
                logger.debug("REST price poll failed for %s: %s", symbol, exc)
                continue
            event = {
                "type": "rest_price",
                "symbol": symbol,
                "price": price,
                "timestamp": time.time(),
            }
            self._publish_event(event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_manager(self) -> None:
        if ThreadedWebsocketManager is None or Client is None:
            logger.debug("Binance WebSocket dependencies unavailable; disabling event stream.")
            self._disabled = True
            return
        with self._manager_lock:
            if self._twm is None:
                try:
                    api_key = os.getenv("BINANCE_API_KEY")
                    api_secret = os.getenv("BINANCE_API_SECRET")
                    self._twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
                    self._twm.start()
                except Exception as exc:  # pragma: no cover - network dependent
                    logger.warning("Failed to start Binance event stream manager: %s", exc)
                    self._twm = None
                    self._disabled = True
                    return
            if self._client is None:
                try:
                    api_key = os.getenv("BINANCE_API_KEY")
                    api_secret = os.getenv("BINANCE_API_SECRET")
                    self._client = Client(api_key, api_secret) if api_key and api_secret else Client()
                except Exception as exc:  # pragma: no cover - network dependent
                    logger.warning("Failed to create Binance REST client: %s", exc)
                    self._client = None

    def _start_symbol(self, symbol: str) -> None:
        if self._disabled or self._twm is None:
            return

        def handle_trade(msg: dict) -> None:
            if not isinstance(msg, dict):
                return
            price = float(msg.get("p") or msg.get("c") or 0.0)
            quantity = float(msg.get("q") or msg.get("Q") or 0.0)
            event_time = float(msg.get("E") or msg.get("T") or time.time() * 1000) / 1000.0
            event = {
                "type": "trade",
                "symbol": symbol,
                "price": price,
                "quantity": quantity,
                "payload": msg,
                "timestamp": event_time,
            }
            self._publish_event(event)

        def handle_kline(msg: dict) -> None:
            if not isinstance(msg, dict):
                return
            payload = msg.get("k") or msg
            event_time = float(payload.get("T") or payload.get("t") or time.time() * 1000) / 1000.0
            close_price = float(payload.get("c") or payload.get("C") or 0.0)
            event = {
                "type": "kline",
                "symbol": symbol,
                "price": close_price,
                "payload": payload,
                "timestamp": event_time,
            }
            self._publish_event(event)

        try:
            trade_socket = self._twm.start_aggtrade_socket(callback=handle_trade, symbol=symbol.lower())
            kline_socket = self._twm.start_kline_socket(callback=handle_kline, symbol=symbol.lower())
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Failed to subscribe to Binance streams for %s: %s", symbol, exc)
            self._disabled = True
            return
        self._stream_ids[symbol] = {"trade": trade_socket, "kline": kline_socket}

    def _stop_symbol(self, symbol: str) -> None:
        streams = self._stream_ids.pop(symbol, None)
        if self._twm is None or not streams:
            return
        for stream_id in streams.values():
            if not stream_id:
                continue
            try:
                self._twm.stop_socket(stream_id)
            except Exception:  # pragma: no cover - best effort
                continue

    def _start_user_socket(self) -> None:
        if self._client is None or self._twm is None:
            return

        def handle_user_data(msg: dict) -> None:
            if not isinstance(msg, dict):
                return
            event_type = msg.get("e")
            if event_type not in {"executionReport", "outboundAccountPosition"}:
                return
            event = {
                "type": "order_update",
                "symbol": msg.get("s"),
                "payload": msg,
                "timestamp": float(msg.get("E", time.time() * 1000)) / 1000.0,
            }
            self._publish_event(event)

        try:
            listen_key = None
            try:
                listen_key = self._client.stream_get_listen_key()  # type: ignore[attr-defined]
            except Exception:
                listen_key = None
            if not listen_key:
                listen_key = self._client.new_listen_key()  # type: ignore[attr-defined]
            self._user_socket_id = self._twm.start_user_socket(callback=handle_user_data, listen_key=listen_key)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.debug("User data stream unavailable: %s", exc)
            self._user_socket_id = None

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.HEARTBEAT_SECONDS)
            self.ensure_alive()

    def _restart_streams(self) -> None:
        if self._disabled:
            return
        interval = self._backoff.next_interval()
        logger.info("Restarting Binance streams in %.2fs", interval)
        time.sleep(interval)
        with self._manager_lock:
            for symbol in list(self._stream_ids):
                self._stop_symbol(symbol)
            if self._twm is not None and self._user_socket_id is not None:
                try:
                    self._twm.stop_socket(self._user_socket_id)
                except Exception:
                    pass
                self._user_socket_id = None
        self._ensure_manager()
        if self._disabled:
            return
        with self._manager_lock:
            for symbol in list(self._symbols):
                self._start_symbol(symbol)
            self._start_user_socket()

    def _publish_event(self, event: Dict[str, Any]) -> None:
        self._last_message = max(self._last_message, float(event.get("timestamp", time.time())))
        self._connected.set()
        try:
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    dropped_type = str(event.get("type"))
                    log_event(
                        logger,
                        "market_queue_drop",
                        queue_size=self._queue.qsize(),
                        max_size=self._queue_maxsize,
                        event_type=dropped_type,
                    )
                    record_metric(
                        "market_queue_drop", 1.0, labels={"type": dropped_type}
                    )
            self._queue.put_nowait(event)
            record_metric(
                "market_queue_len", float(self._queue.qsize()), labels={"queue": "market"}
            )
        except queue.Full:
            logger.debug("Dropping event due to full queue: %s", event.get("type"))
        if callable(self._on_event):
            try:
                self._on_event(event)
            except Exception:
                logger.exception("Event handler raised an exception")


__all__ = [
    "RollingTradeStats",
    "OrderBookState",
    "BinanceMarketStream",
    "get_market_stream",
    "BinanceEventStream",
    "ExponentialBackoff",
]
