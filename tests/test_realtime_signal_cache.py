import time
from collections.abc import Awaitable, Callable
from typing import Optional, Tuple

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


def test_cache_uses_environment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNAL_REFRESH_INTERVAL", "5.5")
    monkeypatch.setenv("SIGNAL_STALE_AFTER", "42")
    cache = _build_cache(stale_after=None)
    assert cache.refresh_interval == pytest.approx(5.5)
    assert cache.stale_after == pytest.approx(42)


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
