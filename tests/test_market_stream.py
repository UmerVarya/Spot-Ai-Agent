import importlib
import sys
import types

import market_stream


def test_process_depth_message_handles_partial_snapshot() -> None:
    stream = market_stream.BinanceMarketStream()
    state = market_stream.OrderBookState(depth=5)
    state.apply_snapshot([("100", "1")], [("101", "2")], 10)

    partial_message = {
        "lastUpdateId": 20,
        "bids": [["100.5", "3"]],
        "asks": [["101.5", "4"]],
    }

    stream._process_depth_message(state, partial_message, "btcusdt")

    snapshot = state.snapshot()
    assert snapshot["last_update_id"] == 20.0
    assert snapshot["bids"] == [(100.5, 3.0)]
    assert snapshot["asks"] == [(101.5, 4.0)]


def test_process_depth_message_handles_diff_update() -> None:
    stream = market_stream.BinanceMarketStream()
    state = market_stream.OrderBookState(depth=5)
    state.apply_snapshot([("100", "1")], [("101", "2")], 10)

    diff_message = {
        "e": "depthUpdate",
        "E": 1700000000,
        "U": 11,
        "u": 11,
        "b": [["100", "0.5"]],
        "a": [["101", "0"]],
    }

    stream._process_depth_message(state, diff_message, "btcusdt")

    snapshot = state.snapshot()
    assert snapshot["last_update_id"] == 11.0
    assert snapshot["bids"] == [(100.0, 0.5)]
    assert snapshot["asks"] == []


def test_event_queue_is_bounded() -> None:
    stream = market_stream.BinanceEventStream(max_queue=2)
    for idx in range(5):
        stream._publish_event({"type": "test", "symbol": "BTCUSDT", "timestamp": idx})
    assert stream.event_queue.qsize() <= 2


def test_threaded_websocket_manager_import_prefers_streams() -> None:
    global market_stream

    original_modules = {
        name: sys.modules.get(name)
        for name in ("binance", "binance.streams", "binance.client")
    }
    for name in ["binance", "binance.streams", "binance.client"]:
        sys.modules.pop(name, None)

    fake_binance = types.ModuleType("binance")
    fake_streams = types.ModuleType("binance.streams")

    class DummyManager:  # pragma: no cover - simple sentinel
        pass

    fake_streams.ThreadedWebsocketManager = DummyManager  # type: ignore[attr-defined]

    fake_client = types.ModuleType("binance.client")

    class DummyClient:  # pragma: no cover - simple sentinel
        pass

    fake_client.Client = DummyClient  # type: ignore[attr-defined]
    fake_binance.client = fake_client  # type: ignore[attr-defined]
    fake_binance.streams = fake_streams  # type: ignore[attr-defined]
    fake_binance.ThreadedWebsocketManager = object()  # type: ignore[attr-defined]

    sys.modules["binance"] = fake_binance
    sys.modules["binance.streams"] = fake_streams
    sys.modules["binance.client"] = fake_client

    try:
        market_stream = importlib.reload(market_stream)
        assert market_stream.ThreadedWebsocketManager is DummyManager
        assert getattr(sys.modules["binance"], "ThreadedWebsocketManager") is DummyManager
    finally:
        for name in ["binance", "binance.streams", "binance.client"]:
            sys.modules.pop(name, None)
        for name, module in original_modules.items():
            if module is not None:
                sys.modules[name] = module
        market_stream = importlib.reload(market_stream)
