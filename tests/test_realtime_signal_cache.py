import asyncio
import time
from typing import Awaitable, Callable, Dict, Iterable, Optional, Tuple

import pandas as pd
import pytest

from realtime_signal_cache import CachedSignal, RealTimeSignalCache


async def _dummy_fetcher(symbol: str) -> Optional[pd.DataFrame]:
    return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp.utcnow()])


def _dummy_evaluator(*args, **kwargs) -> Tuple[float, Optional[str], float, Optional[str]]:
    return 0.0, "long", 1.0, None


class _DummyBridge:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[str, str, Dict[str, object]], None]] = []
        self.symbols: list[str] = []

    def register_callback(
        self, callback: Callable[[str, str, Dict[str, object]], None]
    ) -> None:
        self.callbacks.append(callback)

    def unregister_callback(
        self, callback: Callable[[str, str, Dict[str, object]], None]
    ) -> None:
        try:
            self.callbacks.remove(callback)
        except ValueError:
            pass

    def update_symbols(self, symbols: Iterable[str]) -> None:
        self.symbols = list(symbols)


def _build_cache(
    fetcher: Callable[[str], Awaitable[Optional[pd.DataFrame]]] = _dummy_fetcher,
    evaluator: Callable[..., Tuple[float, Optional[str], float, Optional[str]]] = _dummy_evaluator,
    *,
    refresh_interval: float = 1.0,
    stale_after: Optional[float] = 3.0,
    use_streams: bool = False,
) -> RealTimeSignalCache:
    cache = RealTimeSignalCache(
        fetcher,
        evaluator,
        refresh_interval=refresh_interval,
        stale_after=stale_after,
        use_streams=use_streams,
    )
    cache.configure_runtime(
        default_debounce_ms=800,
        debounce_overrides={},
        refresh_overrides={},
        circuit_breaker_threshold=5,
        circuit_breaker_window=30.0,
    )
    return cache


def test_cache_uses_default_stale_multiplier() -> None:
    cache = _build_cache(stale_after=None, refresh_interval=2.5)
    assert cache.stale_after == pytest.approx(2.5 * 3)


def test_cache_accepts_max_concurrency_alias() -> None:
    cache = RealTimeSignalCache(
        _dummy_fetcher,
        _dummy_evaluator,
        refresh_interval=1.0,
        stale_after=3.0,
        max_concurrent_fetches=4,
    )
    assert cache._max_concurrency == 4


def test_select_symbols_for_refresh_marks_old_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIGNAL_REFRESH_INTERVAL", raising=False)
    monkeypatch.delenv("SIGNAL_STALE_AFTER", raising=False)
    cache = _build_cache()
    now = time.time()
    df = pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp.utcnow()])
    cache.update_universe(["BTCUSDT", "ETHUSDT"])
    cache._cache["BTCUSDT"] = CachedSignal(
        symbol="BTCUSDT",
        score=1.0,
        direction="long",
        position_size=1.0,
        pattern=None,
        price_data=df,
        updated_at=now - 5.0,
        compute_latency=0.01,
    )
    cache._cache["ETHUSDT"] = CachedSignal(
        symbol="ETHUSDT",
        score=1.0,
        direction="long",
        position_size=1.0,
        pattern=None,
        price_data=df,
        updated_at=now,
        compute_latency=0.01,
    )

    due = cache._select_symbols_for_refresh(["BTCUSDT", "ETHUSDT"])
    assert "BTCUSDT" in due
    assert "ETHUSDT" not in due


def test_pending_diagnostics_clamps_large_ages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIGNAL_REFRESH_INTERVAL", raising=False)
    monkeypatch.delenv("SIGNAL_STALE_AFTER", raising=False)
    cache = _build_cache()
    cache.update_universe(["XRPUSDT"])
    df = pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp.utcnow()])
    cache._cache["XRPUSDT"] = CachedSignal(
        symbol="XRPUSDT",
        score=1.0,
        direction="long",
        position_size=1.0,
        pattern=None,
        price_data=df,
        updated_at=time.time() - 120.0,
        compute_latency=0.01,
    )
    cache._symbol_added_at["XRPUSDT"] = time.time() - 500.0
    cache._symbol_last_attempt["XRPUSDT"] = time.time() - 400.0
    cache._symbol_last_error["XRPUSDT"] = (time.time() - 300.0, "fetch error")

    diagnostics = cache.pending_diagnostics()
    assert diagnostics
    entry = diagnostics[0]

    waiting_for = entry["waiting_for"]
    assert isinstance(waiting_for, dict)
    assert waiting_for["display"] <= cache.stale_after
    assert waiting_for["raw"] >= cache.stale_after

    stale_age = entry["stale_age"]
    assert isinstance(stale_age, dict)
    assert stale_age["display"] <= cache.stale_after
    assert stale_age["raw"] >= cache.stale_after

    request_wait = entry["request_wait"]
    assert isinstance(request_wait, dict)
    assert request_wait["display"] <= cache.stale_after
    assert request_wait["raw"] >= cache.stale_after

    error_age = entry["error_age"]
    assert isinstance(error_age, dict)
    assert error_age["display"] <= cache.stale_after
    assert error_age["raw"] >= cache.stale_after


def test_force_refresh_without_worker_primes_cache() -> None:
    calls: list[str] = []

    async def fetcher(symbol: str) -> Optional[pd.DataFrame]:
        calls.append(symbol)
        await asyncio.sleep(0)
        return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp.utcnow()])

    cache = _build_cache(fetcher=fetcher)
    cache.update_universe(["BTCUSDT"])
    assert cache.get("BTCUSDT") is None

    success = cache.force_refresh("BTCUSDT", timeout=2.0)
    assert success
    cached = cache.get("BTCUSDT")
    assert cached is not None
    assert calls == ["BTCUSDT"]


def test_force_refresh_while_worker_running() -> None:
    calls: list[str] = []

    async def fetcher(symbol: str) -> Optional[pd.DataFrame]:
        calls.append(symbol)
        await asyncio.sleep(0.01)
        return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp.utcnow()])

    cache = _build_cache(fetcher=fetcher)
    cache.update_universe(["ETHUSDT"])
    cache.start()
    try:
        success = cache.force_refresh("ETHUSDT", timeout=2.0)
        assert success
        assert cache.get("ETHUSDT") is not None
        assert calls  # fetcher invoked either by worker or manual refresh
    finally:
        cache.stop()


def test_schedule_refresh_respects_debounce() -> None:
    cache = _build_cache()
    cache.update_universe(["ETHUSDT"])
    cache.schedule_refresh("ETHUSDT")
    first_request = cache._last_priority_request.get("ETHUSDT")
    assert first_request is not None
    cache.schedule_refresh("ETHUSDT")
    assert len(cache._priority_symbols) == 1


def test_circuit_breaker_trips_after_errors() -> None:
    cache = _build_cache()
    cache.configure_runtime(
        default_debounce_ms=500,
        debounce_overrides={},
        refresh_overrides={},
        circuit_breaker_threshold=2,
        circuit_breaker_window=60.0,
    )
    cache.update_universe(["BTCUSDT"])
    now = time.time()
    cache._record_refresh_error("BTCUSDT", "boom", attempt_ts=now)
    cache._record_refresh_error("BTCUSDT", "boom", attempt_ts=now + 1)
    assert cache.circuit_breaker_active()


def test_stream_enable_requires_flag() -> None:
    cache = _build_cache(use_streams=False)
    bridge = _DummyBridge()
    assert cache.enable_streams(bridge) is False
    assert bridge.callbacks == []


def test_stream_enable_registers_and_updates_symbols() -> None:
    cache = _build_cache(use_streams=True)
    cache.update_universe(["ETHUSDT", "BTCUSDT"])
    bridge = _DummyBridge()
    assert cache.enable_streams(bridge) is True
    assert bridge.callbacks == [cache.handle_ws_update]
    assert bridge.symbols == ["BTCUSDT", "ETHUSDT"]
    cache.disable_streams()
    assert bridge.callbacks == []


def test_on_ws_bar_close_updates_last_timestamp() -> None:
    cache = _build_cache(use_streams=True)
    cache.update_universe(["BTCUSDT"])
    close_ts_ms = 1_600_000_000_000
    cache.on_ws_bar_close("BTCUSDT", close_ts_ms)
    stored = cache._last_update_ts.get("BTCUSDT")
    assert stored is not None
    assert stored == pytest.approx(close_ts_ms / 1000.0)


def test_handle_ws_update_triggers_bar_close_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _build_cache(use_streams=True)
    cache.update_universe(["ETHUSDT"])

    observed: Dict[str, Tuple[str, Optional[int]]] = {}

    def record_bar_close(symbol: str, close_ts_ms: Optional[int]) -> None:
        observed["bar_close"] = (symbol, close_ts_ms)

    scheduled: list[str] = []

    def record_refresh(symbol: str) -> None:
        scheduled.append(symbol)

    monkeypatch.setattr(cache, "on_ws_bar_close", record_bar_close)
    monkeypatch.setattr(cache, "schedule_refresh", record_refresh)

    def fake_create_task(coro):
        asyncio.run(coro)
        return object()

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    cache.handle_ws_update("ETHUSDT", "kline", {"x": True, "T": 1_700_000_000_000})

    assert observed["bar_close"] == ("ETHUSDT", 1_700_000_000_000)
    assert scheduled == ["ETHUSDT"]
