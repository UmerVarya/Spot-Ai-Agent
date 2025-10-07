from datetime import datetime, timedelta

import pandas as pd
import pytest

import trade_manager
from orderflow import OrderFlowAnalysis


def _basic_indicators():
    return {
        "atr": pd.Series([1.0, 1.0]),
        "rsi": pd.Series([55, 55]),
        "macd": pd.Series([0.1, 0.1]),
        "macd_signal": pd.Series([0.05, 0.05]),
        "ema_20": pd.Series([104.0, 104.0]),
        "ema_50": pd.Series([103.5, 103.5]),
        "vwap": None,
        "adx": pd.Series([20.0, 20.5]),
        "kc_lower": pd.Series([98.0, 98.0]),
    }


def test_break_even_sl_moves_on_strong_orderflow(monkeypatch):
    now = datetime.utcnow().replace(microsecond=0)
    entry_time = (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    index = pd.to_datetime([
        now - timedelta(minutes=2),
        now - timedelta(minutes=1),
    ])
    price_data = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.5],
            "close": [101.0, 102.5],
            "volume": [1500.0, 1800.0],
            "quote_volume": [150000.0, 185000.0],
            "taker_buy_base": [900.0, 1100.0],
            "taker_buy_quote": [91000.0, 113000.0],
            "taker_sell_base": [600.0, 700.0],
            "taker_sell_quote": [59000.0, 72000.0],
            "number_of_trades": [120.0, 140.0],
        },
        index=index,
    )

    trade = {
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 100.0,
        "position_size": 1.0,
        "sl": 95.0,
        "tp1": 110.0,
        "tp2": 120.0,
        "tp3": 130.0,
        "status": {"tp1": False, "tp2": False, "tp3": False, "sl": False},
        "entry_time": entry_time,
    }

    monkeypatch.setattr(trade_manager, "load_active_trades", lambda: [trade])

    saved_trades = {}

    def fake_save_active_trades(trades):
        saved_trades["trades"] = trades

    monkeypatch.setattr(trade_manager, "save_active_trades", fake_save_active_trades)
    monkeypatch.setattr(trade_manager, "get_price_data", lambda symbol: price_data)
    monkeypatch.setattr(trade_manager, "calculate_indicators", lambda df: _basic_indicators())
    monkeypatch.setattr(trade_manager, "estimate_commission", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(trade_manager, "simulate_slippage", lambda price, direction=None: price)
    monkeypatch.setattr(trade_manager, "_update_rl", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "send_email", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "log_trade_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "plan_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "detect_sell_pressure", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "get_order_book", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "should_exit_early", lambda *args, **kwargs: (False, None))
    monkeypatch.setattr(trade_manager, "should_exit_position", lambda *args, **kwargs: [])
    monkeypatch.setattr(trade_manager, "execute_exit_trade", lambda *args, **kwargs: (0.0, 0.0, 0.0))

    sl_updates = {}

    def fake_update_sl(trade_obj, new_sl):
        sl_updates["value"] = new_sl
        trade_obj["sl"] = new_sl

    monkeypatch.setattr(trade_manager, "_update_stop_loss", fake_update_sl)

    strong_flow = OrderFlowAnalysis(
        state="buyers in control",
        features={"cvd_change": 0.35, "trade_imbalance": 0.3},
    )
    monkeypatch.setattr(trade_manager, "detect_aggression", lambda *args, **kwargs: strong_flow)

    trade_manager.manage_trades()

    assert pytest.approx(trade["sl"]) == trade["entry"]
    assert trade["status"].get("flow_break_even") is True
    assert sl_updates["value"] == trade["entry"]
    assert saved_trades["trades"][0]["sl"] == trade["entry"]
