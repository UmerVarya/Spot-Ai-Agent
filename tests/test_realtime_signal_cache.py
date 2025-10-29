import asyncio
import time
from types import MethodType
from typing import Awaitable, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

import pytest

import realtime_signal_cache as rtsc
from realtime_signal_cache import CachedSignal, RealTimeSignalCache


def _make_price_df(rows: int = 60) -> pd.DataFrame:
    index = pd.date_range(
        end=pd.Timestamp.utcnow().floor("T"), periods=rows, freq="T", tz="UTC"
    )
    base = np.linspace(1.0, 1.0 + rows * 0.01, rows)
    data = {
        "open": base,
        "high": base + 0.05,
        "low": base - 0.05,
        "close": base + 0.01,
        "volume": np.full(rows, 10.0),
    }
    df = pd.DataFrame(data, index=index)
    df["quote_volume"] = df["volume"]
    df["taker_buy_base"] = df["volume"] * 0.5
    df["taker_buy_quote"] = df["quote_volume"] * 0.5
    df["taker_sell_base"] = df["volume"] - df["taker_buy_base"]
    df["taker_sell_quote"] = df["quote_volume"] - df["taker_buy_quote"]
    df["number_of_trades"] = np.full(rows, 12.0)
    return df


async def _dummy_fetcher(symbol: str) -> Optional[pd.DataFrame]:
    return _make_price_df()


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
        return _make_price_df()

    cache = _build_cache(fetcher=fetcher)
    cache.update_universe(["BTCUSDT"])
    assert cache.get("BTCUSDT") is None

    success = cache.force_refresh("BTCUSDT", timeout=2.0)
    assert success
    cached = cache.get("BTCUSDT")
    assert cached is not None
    assert calls and set(calls) == {"BTCUSDT"}


def test_prime_symbol_falls_back_to_price_fetcher(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _build_cache(fetcher=_dummy_fetcher)
    cache.rest = None
    cache._rest_client = None
    cache.update_universe(["BTCUSDT"])
    monkeypatch.setattr(rtsc, "REST_REQUIRED_MIN_BARS", 10)
    monkeypatch.setattr(cache, "force_rest_backfill", lambda symbol: False)

    assert cache.bars_len("BTCUSDT") == 0

    success = cache._prime_symbol("BTCUSDT")
    assert success
    assert cache.bars_len("BTCUSDT") >= rtsc.REST_REQUIRED_MIN_BARS


def test_refresh_symbol_force_rest_false_bypasses_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("realtime_signal_cache.RTSC_FORCE_REST", True)

    rest_calls: list[str] = []
    fetch_calls: list[str] = []

    async def fetcher(symbol: str) -> Optional[pd.DataFrame]:
        fetch_calls.append(symbol)
        await asyncio.sleep(0)
        return _make_price_df()

    async def rest_fetch(self: RealTimeSignalCache, symbol: str) -> bool:
        rest_calls.append(symbol)
        return False

    cache = _build_cache(fetcher=fetcher)
    cache.update_universe(["BTCUSDT"])
    cache._refresh_symbol_via_rest = MethodType(rest_fetch, cache)

    success = asyncio.run(cache._refresh_symbol("BTCUSDT", force_rest=False))

    assert success is True
    assert fetch_calls and fetch_calls[-1] == "BTCUSDT"
    assert rest_calls == []


def test_force_refresh_while_worker_running() -> None:
    calls: list[str] = []

    async def fetcher(symbol: str) -> Optional[pd.DataFrame]:
        calls.append(symbol)
        await asyncio.sleep(0.01)
        return _make_price_df()

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


def test_schedule_refresh_triggers_rest_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    cache = _build_cache()
    cache.update_universe(["ETHUSDT"])
    cache.start()
    cache.flush_pending()

    async def fake_rest(symbol: str) -> bool:
        calls.append(symbol)
        return True

    monkeypatch.setattr(cache, "_refresh_symbol_via_rest", fake_rest)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cache.schedule_refresh("ETHUSDT"))
        loop.run_until_complete(asyncio.sleep(0.05))
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    cache.flush_pending()
    cache.stop()

    assert calls == ["ETHUSDT"]


def test_on_ws_bar_close_records_timestamp() -> None:
    cache = _build_cache(use_streams=True)
    cache.update_universe(["BTCUSDT"])
    close_ts_ms = 1_700_000_000_000
    cache.on_ws_bar_close("BTCUSDT", close_ts_ms)
    recorded = cache._last_ws_ts.get("BTCUSDT")
    assert recorded is not None
    assert recorded == pytest.approx(close_ts_ms / 1000.0)


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
