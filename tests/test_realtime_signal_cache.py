import asyncio
import time
from typing import Awaitable, Callable, Optional, Tuple

import pandas as pd
import pytest

from realtime_signal_cache import CachedSignal, RealTimeSignalCache


async def _dummy_fetcher(symbol: str) -> Optional[pd.DataFrame]:
    return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp.utcnow()])


def _dummy_evaluator(*args, **kwargs) -> Tuple[float, Optional[str], float, Optional[str]]:
    return 0.0, "long", 1.0, None


def _build_cache(
    fetcher: Callable[[str], Awaitable[Optional[pd.DataFrame]]] = _dummy_fetcher,
    evaluator: Callable[..., Tuple[float, Optional[str], float, Optional[str]]] = _dummy_evaluator,
    *,
    refresh_interval: float = 1.0,
    stale_after: Optional[float] = 3.0,
) -> RealTimeSignalCache:
    return RealTimeSignalCache(
        fetcher,
        evaluator,
        refresh_interval=refresh_interval,
        stale_after=stale_after,
    )


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
    assert entry["waiting_for"] <= cache.stale_after
    assert entry["stale_age"] <= cache.stale_after
    assert entry["request_wait"] <= cache.stale_after
    assert entry["error_age"] <= cache.stale_after


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
