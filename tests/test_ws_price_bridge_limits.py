from __future__ import annotations

import importlib

import pytest

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


def test_make_streams_accepts_all_pairs_by_default():
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
        "ethbtc@kline_1m",
        "ethbtc@ticker",
        "ethbtc@bookTicker",
        "solusdt@kline_1m",
        "solusdt@ticker",
        "solusdt@bookTicker",
    ]


def test_make_streams_can_filter_by_suffix():
    streams = bridge.make_streams(
        ["BTCUSDT", "ETHBTC", "SOLBUSD"],
        include_kline=False,
        include_ticker=True,
        include_book=False,
        quote_suffix="busd",
    )

    assert streams == ["solbusd@ticker"]


def test_ws_bridge_normalises_combined_base(monkeypatch):
    monkeypatch.setenv("WS_BACKEND", "wsclient")
    monkeypatch.setenv("WS_COMBINED_BASE", "wss://example.com/stream")
    reloaded = importlib.reload(bridge)

    try:
        instance = reloaded.WSPriceBridge(["BTCUSDT"])

        assert instance._combined_base == "wss://example.com/stream?streams="
    finally:
        monkeypatch.delenv("WS_BACKEND", raising=False)
        monkeypatch.delenv("WS_COMBINED_BASE", raising=False)
        importlib.reload(bridge)


def test_wsclient_bridge_base_url_is_normalised_directly():
    pytest.importorskip("websocket")
    import ws_backend_client as backend
    reloaded_client = importlib.reload(backend)

    bridge_client = reloaded_client.WSClientBridge(
        ["btcusdt@kline_1m"],
        lambda _msg: None,
        base_url="wss://example.com/stream",
    )

    assert bridge_client._base_url == "wss://example.com/stream?streams="
