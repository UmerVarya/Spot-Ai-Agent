from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

import trade_manager


def test_manage_trades_waits_for_post_entry_candle(monkeypatch):
    """Ensure drawdown exits wait for at least one post-entry candle."""

    now = datetime.utcnow().replace(microsecond=0)
    entry_time = (now - timedelta(seconds=30)).isoformat()
    index = pd.to_datetime([now - timedelta(minutes=2), now - timedelta(minutes=1)])
    price_data = pd.DataFrame(
        {
            "open": [105.0, 104.5],
            "high": [106.0, 105.0],
            "low": [100.0, 100.5],
            "close": [104.0, 104.5],
            "volume": [1000, 1200],
        },
        index=index,
    )

    trade = {
        "symbol": "BTCUSDT",
        "direction": "long",
        "entry": 105.0,
        "position_size": 1.0,
        "sl": 95.0,
        "tp1": 115.0,
        "tp2": 125.0,
        "tp3": 135.0,
        "status": {"tp1": False, "tp2": False, "tp3": False},
        "entry_time": entry_time,
    }

    monkeypatch.setattr(trade_manager, "load_active_trades", lambda: [trade])

    saved = {}

    def fake_save_active_trades(trades):
        saved["trades"] = trades

    monkeypatch.setattr(trade_manager, "save_active_trades", fake_save_active_trades)
    monkeypatch.setattr(trade_manager, "get_price_data", lambda symbol: price_data)
    monkeypatch.setattr(
        trade_manager,
        "calculate_indicators",
        lambda df: {
            "atr": pd.Series([1.0, 1.0]),
            "rsi": pd.Series([55, 55]),
            "macd": pd.Series([0.1, 0.1]),
            "macd_signal": pd.Series([0.05, 0.05]),
            "ema_20": pd.Series([104.0, 104.0]),
            "ema_50": pd.Series([103.5, 103.5]),
            "vwap": None,
        },
    )
    monkeypatch.setattr(trade_manager, "estimate_commission", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(trade_manager, "simulate_slippage", lambda price, direction=None: price)
    monkeypatch.setattr(trade_manager, "_update_rl", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "send_email", lambda *args, **kwargs: None)
    monkeypatch.setattr(trade_manager, "log_trade_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        trade_manager,
        "detect_aggression",
        lambda *args, **kwargs: SimpleNamespace(state="neutral", features={}),
    )
    monkeypatch.setattr(
        trade_manager,
        "analyze_macro_sentiment",
        lambda: {"bias": "neutral", "confidence": 0},
    )

    trade_manager.manage_trades()

    assert saved["trades"], "Active trades should be preserved"
    managed_trade = saved["trades"][0]
    assert managed_trade.get("outcome") is None
    assert managed_trade.get("symbol") == "BTCUSDT"
