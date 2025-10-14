"""Central state registry shared across event-driven subsystems."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional


class CentralState:
    """Maintain shared state for real-time and background subsystems.

    The state container stores the most recent context that the decision engine
    depends on (macro regime, news filters, market microstructure snapshots,
    etc.).  All updates are protected by a re-entrant lock so that multiple
    worker threads may publish results without racing with the consumer.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {
            "macro": {"data": None, "timestamp": 0.0},
            "news": {"data": None, "timestamp": 0.0},
            "narratives": {},
            "prices": {},
            "klines": {},
            "orders": [],
            "connection": {"status": "disconnected", "last_event": 0.0},
        }

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def update_section(self, section: str, value: Any, *, timestamp: Optional[float] = None) -> None:
        """Replace an entire section atomically."""

        ts = float(timestamp if timestamp is not None else time.time())
        with self._lock:
            self._state.setdefault(section, {})
            self._state[section] = {"data": value, "timestamp": ts}

    def merge_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Merge ``updates`` into ``section`` keeping timestamps consistent."""

        now = time.time()
        with self._lock:
            existing = self._state.setdefault(section, {"data": {}, "timestamp": now})
            data = existing.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            data.update(updates)
            existing["data"] = data
            existing["timestamp"] = now

    def append_to_section(self, section: str, value: Any, *, max_items: Optional[int] = None) -> None:
        """Append ``value`` to a list section while keeping only ``max_items``."""

        now = time.time()
        with self._lock:
            existing = self._state.setdefault(section, {"data": [], "timestamp": now})
            data = existing.get("data")
            if not isinstance(data, list):
                data = []
            data.append(value)
            if max_items is not None and max_items > 0:
                data[:] = data[-max_items:]
            existing["data"] = data
            existing["timestamp"] = now

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy snapshot of the internal state."""

        with self._lock:
            return deepcopy(self._state)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return a shallow copy of a section's metadata and payload."""

        with self._lock:
            value = self._state.get(section, {"data": None, "timestamp": 0.0})
            return {"data": deepcopy(value.get("data")), "timestamp": float(value.get("timestamp", 0.0))}

    # ------------------------------------------------------------------
    # Specialised helpers for streaming events
    # ------------------------------------------------------------------
    def update_price(self, symbol: str, price: float, *, timestamp: Optional[float] = None) -> None:
        now = float(timestamp if timestamp is not None else time.time())
        with self._lock:
            prices = self._state.setdefault("prices", {})
            if not isinstance(prices, dict):
                prices = {}
                self._state["prices"] = prices
            prices[symbol] = {"price": float(price), "timestamp": now}
            connection = self._state.setdefault("connection", {})
            if isinstance(connection, dict):
                connection["last_event"] = now

    def update_kline(self, symbol: str, kline: Dict[str, Any], *, timestamp: Optional[float] = None) -> None:
        now = float(timestamp if timestamp is not None else time.time())
        with self._lock:
            klines = self._state.setdefault("klines", {})
            if not isinstance(klines, dict):
                klines = {}
                self._state["klines"] = klines
            klines[symbol] = {"payload": deepcopy(kline), "timestamp": now}
            connection = self._state.setdefault("connection", {})
            if isinstance(connection, dict):
                connection["last_event"] = now

    def append_order_update(self, update: Dict[str, Any]) -> None:
        self.append_to_section("orders", deepcopy(update), max_items=200)

    def set_connection_status(self, status: str) -> None:
        with self._lock:
            connection = self._state.setdefault("connection", {})
            if not isinstance(connection, dict):
                connection = {}
                self._state["connection"] = connection
            connection["status"] = status
            connection["last_event"] = time.time()

    def merge_narrative(self, key: str, values: Dict[str, Any]) -> None:
        with self._lock:
            narratives = self._state.setdefault("narratives", {})
            if not isinstance(narratives, dict):
                narratives = {}
                self._state["narratives"] = narratives
            entry = narratives.setdefault(key, {})
            entry.update(values)
            entry["timestamp"] = time.time()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def latest_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            prices = self._state.get("prices")
            if isinstance(prices, dict):
                entry = prices.get(symbol)
                if isinstance(entry, dict):
                    price = entry.get("price")
                    return float(price) if price is not None else None
        return None

    def tracked_symbols(self) -> Iterable[str]:
        with self._lock:
            prices = self._state.get("prices")
            if isinstance(prices, dict):
                return list(prices.keys())
        return []


__all__ = ["CentralState"]
