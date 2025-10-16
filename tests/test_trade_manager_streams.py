from __future__ import annotations

import trade_manager as tm


def test_process_book_ticker_triggers_manage(monkeypatch):
    tm._LIVE_MARKET.clear()
    tm._LAST_MANAGE_TRIGGER.clear()

    # Provide a single active trade and force an exit signal when price exceeds threshold.
    monkeypatch.setattr(tm, "load_active_trades", lambda: [{"symbol": "BTCUSDT"}])
    monkeypatch.setattr(
        tm,
        "should_exit_position",
        lambda trade, current_price, recent_high, recent_low: [{"type": "tp1"}]
        if current_price and current_price >= 105
        else [],
    )

    calls: list[str] = []

    def _mock_manage() -> None:
        calls.append("managed")

    monkeypatch.setattr(tm, "manage_trades", _mock_manage)

    tm.process_book_ticker("BTCUSDT", {"b": "105", "a": "105.5", "E": 1_000})

    snapshot = tm._LIVE_MARKET.get("BTCUSDT")
    assert snapshot is not None
    assert snapshot["bid"] == 105.0
    assert snapshot["ask"] == 105.5
    assert calls == ["managed"]


def test_process_user_stream_event_updates_price(monkeypatch):
    tm._LIVE_MARKET.clear()
    tm._LAST_MANAGE_TRIGGER.clear()

    calls: list[str] = []

    def _mock_manage() -> None:
        calls.append("managed")

    monkeypatch.setattr(tm, "manage_trades", _mock_manage)

    tm.process_user_stream_event(
        {"e": "executionReport", "s": "ETHUSDT", "X": "FILLED", "L": "2010.5", "E": 2_000}
    )

    snapshot = tm._LIVE_MARKET.get("ETHUSDT")
    assert snapshot is not None
    assert snapshot["price"] == 2010.5
    assert calls == ["managed"]


def test_process_live_kline_deduplicates_close(monkeypatch):
    tm._KLINE_CLOSE_IDS.clear()
    calls: list[tuple] = []

    def _noop_update(*args, **kwargs):
        return None

    monkeypatch.setattr(tm, "_update_live_market", _noop_update)

    def _recorder(symbol, price, high, low):
        calls.append((symbol, price, high, low))

    monkeypatch.setattr(tm, "_check_live_triggers", _recorder)

    payload = {"c": "100", "h": "101", "l": "99", "T": 1234, "x": True}
    tm.process_live_kline("BTCUSDT", "1m", payload)
    tm.process_live_kline("BTCUSDT", "1m", payload)

    assert len(calls) == 1
