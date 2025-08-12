import json
from pathlib import Path
import trade_storage


def test_save_and_load_active_trades(tmp_path, monkeypatch):
    path = tmp_path / "active.json"
    monkeypatch.setattr(trade_storage, "ACTIVE_TRADES_FILE", str(path))
    trades = [{"symbol": "BTCUSDT"}]
    trade_storage.save_active_trades(trades)
    loaded = trade_storage.load_active_trades()
    assert loaded == trades
