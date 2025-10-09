from market_stream import BinanceMarketStream, OrderBookState


def test_process_depth_message_handles_partial_snapshot() -> None:
    stream = BinanceMarketStream()
    state = OrderBookState(depth=5)
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
    stream = BinanceMarketStream()
    state = OrderBookState(depth=5)
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
