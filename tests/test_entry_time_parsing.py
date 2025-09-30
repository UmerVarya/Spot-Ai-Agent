import pandas as pd
from datetime import datetime, timedelta

import trade_manager


def test_manage_trades_parses_iso_entry_time(monkeypatch):
    """Ensure manage_trades handles ISO8601 entry_time with timezone."""
    now = datetime.utcnow()
    iso_entry = (now - trade_manager.MAX_HOLDING_TIME - timedelta(minutes=1)).replace(microsecond=0).isoformat() + "Z"
    trade = {
        "symbol": "BTCUSDT",
        "direction": "long",
        "entry": 100.0,
        "position_size": 1,
        "sl": 90.0,
        "tp1": 110.0,
        "tp2": 120.0,
        "tp3": 130.0,
        "status": {"tp1": False, "tp2": False, "tp3": False},
        "entry_time": iso_entry,
    }

    def fake_load_active_trades():
        return [trade]

    saved = {}

    def fake_save_active_trades(trades):
        saved["trades"] = trades

    def fake_price_data(symbol):
        return pd.DataFrame({"close": [100.0], "high": [100.0], "low": [100.0]})

    def fake_commission(symbol, quantity, maker):
        return 0.0

    def fake_slippage(price, direction):
        return price

    called = {}

    def fake_log_trade_result(trade, **kwargs):
        called.update(kwargs)

    monkeypatch.setattr(trade_manager, "load_active_trades", fake_load_active_trades)
    monkeypatch.setattr(trade_manager, "save_active_trades", fake_save_active_trades)
    monkeypatch.setattr(trade_manager, "get_price_data", fake_price_data)
    monkeypatch.setattr(trade_manager, "estimate_commission", fake_commission)
    monkeypatch.setattr(trade_manager, "simulate_slippage", fake_slippage)
    monkeypatch.setattr(trade_manager, "log_trade_result", fake_log_trade_result)
    monkeypatch.setattr(trade_manager, "_update_rl", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "send_email", lambda *args, **kwargs: None)

    trade_manager.manage_trades()

    assert called.get("outcome") == "time_exit"
    assert saved["trades"] == []


def test_manage_trades_time_exit_without_price_data(monkeypatch):
    """Verify time-based exits still occur when price data is missing."""
    now = datetime.utcnow()
    stale_entry = (now - trade_manager.MAX_HOLDING_TIME - timedelta(minutes=1)).replace(microsecond=0)
    trade = {
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 200.0,
        "last_price": 205.0,
        "position_size": 2,
        "sl": 180.0,
        "tp1": 220.0,
        "tp2": 230.0,
        "tp3": 240.0,
        "status": {"tp1": False, "tp2": False, "tp3": False},
        "entry_time": stale_entry.strftime("%Y-%m-%d %H:%M:%S"),
    }

    def fake_load_active_trades():
        return [trade]

    saved = {}

    def fake_save_active_trades(trades):
        saved["trades"] = trades

    def fake_commission(symbol, quantity, maker):
        return 0.0

    def fake_slippage(price, direction):
        return price

    captured = {}

    def fake_log_trade_result(trade_record, **kwargs):
        captured["trade"] = trade_record.copy()
        captured.update(kwargs)

    monkeypatch.setattr(trade_manager, "load_active_trades", fake_load_active_trades)
    monkeypatch.setattr(trade_manager, "save_active_trades", fake_save_active_trades)
    monkeypatch.setattr(trade_manager, "get_price_data", lambda symbol: None)
    monkeypatch.setattr(trade_manager, "estimate_commission", fake_commission)
    monkeypatch.setattr(trade_manager, "simulate_slippage", fake_slippage)
    monkeypatch.setattr(trade_manager, "log_trade_result", fake_log_trade_result)
    monkeypatch.setattr(trade_manager, "_update_rl", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "send_email", lambda *args, **kwargs: None)

    trade_manager.manage_trades()

    assert captured.get("outcome") == "time_exit"
    assert captured.get("exit_price") == 205.0
    assert captured["trade"].get("exit_price") == 205.0
    assert saved["trades"] == []

