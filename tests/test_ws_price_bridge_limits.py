from __future__ import annotations

import importlib

import ws_price_bridge as bridge


def _parse_streams(url: str) -> list[str]:
    prefix = "streams="
    _, _, tail = url.partition(prefix)
    return tail.split("/") if tail else []


def test_combined_urls_single_batch_under_limit(monkeypatch):
    monkeypatch.setattr(bridge, "MAX_STREAMS_PER_COMBINED", 5)
    symbols = [f"SYM{i}USDT" for i in range(5)]

    urls = bridge._combined_urls(symbols, want_kline_1m=True)

    assert len(urls) == 1
    streams = _parse_streams(urls[0])
    assert len(streams) == len(symbols)
    assert len(streams) <= bridge.MAX_STREAMS_PER_COMBINED


def test_combined_urls_multiple_batches_over_limit(monkeypatch):
    monkeypatch.setattr(bridge, "MAX_STREAMS_PER_COMBINED", 3)
    symbols = [f"SYM{i}USDT" for i in range(5)]

    urls = bridge._combined_urls(symbols, want_kline_1m=True)

    assert len(urls) == 2
    batch_sizes = [len(_parse_streams(url)) for url in urls]
    assert all(size <= bridge.MAX_STREAMS_PER_COMBINED for size in batch_sizes)
    assert sum(batch_sizes) == len(symbols)


def test_max_streams_per_combined_is_clamped(monkeypatch):
    monkeypatch.setenv("BINANCE_MAX_STREAMS", "500")
    reloaded = importlib.reload(bridge)

    try:
        assert reloaded.MAX_STREAMS_PER_COMBINED == 200
    finally:
        monkeypatch.delenv("BINANCE_MAX_STREAMS", raising=False)
        importlib.reload(bridge)


def test_make_streams_filters_and_deduplicates():
    streams = bridge.make_streams(
        [" BTCUSDT ", "ETHBTC", "BTCUSDT", "SOLUSDT"],
        include_kline=True,
        include_ticker=True,
        include_book=True,
    )

    assert streams == [
        "btcusdt@kline_1m",
        "btcusdt@ticker",
        "btcusdt@bookTicker",
        "solusdt@kline_1m",
        "solusdt@ticker",
        "solusdt@bookTicker",
    ]
