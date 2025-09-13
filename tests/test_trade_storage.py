import json
from pathlib import Path
import trade_storage
import trade_manager
import csv


def test_save_and_load_active_trades(tmp_path, monkeypatch):
    path = tmp_path / "active.json"
    monkeypatch.setattr(trade_storage, "ACTIVE_TRADES_FILE", str(path))
    trades = [{"symbol": "BTCUSDT"}]
    trade_storage.save_active_trades(trades)
    loaded = trade_storage.load_active_trades()
    assert loaded == trades


def test_log_trade_result_extended_fields(tmp_path, monkeypatch):
    csv_path = tmp_path / "log.csv"
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))
    trade = {
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 1000,
        "size": 1000,
        "position_size": 1,
        "strategy": "test",
        "session": "Asia",
        "sentiment_bias": "bullish",
        "sentiment_confidence": 8.0,
        "volatility": 50.0,
        "htf_trend": 10.0,
        "order_imbalance": 5.0,
        "macro_indicator": 20.0,
    }
    trade_storage.log_trade_result(trade, outcome="tp1", exit_price=1100)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["sentiment_bias"] == "bullish"
    assert rows[0]["sentiment_confidence"] == "8.0"
    assert "volatility" in rows[0]
    assert "macro_indicator" in rows[0]
    assert "llm_error" in rows[0]
    assert rows[0]["timestamp"].endswith("Z")
    assert float(rows[0]["pnl"]) == 100.0
    assert float(rows[0]["pnl_pct"]) == 10.0
    assert float(rows[0]["pnl_tp1"]) == 0.0
    assert float(rows[0]["pnl_tp2"]) == 0.0
    assert float(rows[0]["size_tp1"]) == 0.0
    assert float(rows[0]["size_tp2"]) == 0.0
    assert float(rows[0]["notional_tp1"]) == 0.0
    assert float(rows[0]["notional_tp2"]) == 0.0
    assert rows[0]["win"] == "True"


def test_log_trade_result_partial_tp_fields(tmp_path, monkeypatch):
    csv_path = tmp_path / "log.csv"
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))
    trade = {
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 1000,
        "size": 500,
        "position_size": 0.5,
        "strategy": "test",
        "session": "Asia",
    }
    trade_storage.log_trade_result(trade, outcome="tp1_partial", exit_price=1100)
    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["pnl"]) == 50.0
    assert float(row["pnl_tp1"]) == 50.0
    assert float(row["size_tp1"]) == 0.5
    assert float(row["notional_tp1"]) == 500.0
    assert float(row["pnl_tp2"]) == 0.0
    assert row["win"] == "True"


def test_win_flag_negative_pnl(tmp_path, monkeypatch):
    csv_path = tmp_path / "log.csv"
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))
    trade = {
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 1000,
        "size": 1000,
        "position_size": 1,
        "strategy": "test",
        "session": "Asia",
    }
    trade_storage.log_trade_result(trade, outcome="sl", exit_price=900)
    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["win"] == "False"


def test_log_trade_result_writes_header_if_file_empty(tmp_path, monkeypatch):
    """Regression test for missing header when history file pre-exists."""
    csv_path = tmp_path / "log.csv"
    csv_path.touch()  # create zero-byte file
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))
    trade = {"symbol": "BTCUSDT", "direction": "long", "entry": 1000, "size": 1000, "position_size": 1}
    trade_storage.log_trade_result(trade, outcome="tp1", exit_price=1100)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Without the header fix, ``rows`` would be empty because the first trade
    # becomes the header.  We expect one row with a "symbol" column.
    assert rows and rows[0]["symbol"] == "BTCUSDT"


def test_log_trade_result_rewrites_missing_header(tmp_path, monkeypatch):
    """If the existing log lacks a header, it should be inserted."""
    csv_path = tmp_path / "log.csv"
    csv_path.write_text("BTCUSDT,1000,1100\n")  # simulate data without header
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))
    trade = {
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 2000,
        "size": 1000,
        "position_size": 1,
        "strategy": "test",
        "session": "Asia",
    }
    trade_storage.log_trade_result(trade, outcome="tp1", exit_price=2100)
    with open(csv_path) as f:
        first = f.readline().strip().lower()
    assert first.startswith("trade_id,")


def test_size_as_notional_without_position_size(tmp_path, monkeypatch):
    csv_path = tmp_path / "log.csv"
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))
    trade = {
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 2000,
        "size": 100,  # treated as notional
        "strategy": "test",
        "session": "Asia",
    }
    trade_storage.log_trade_result(trade, outcome="tp1", exit_price=2200)
    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["notional"]) == 100.0
    assert float(row["pnl"]) == 10.0
    assert float(row["pnl_pct"]) == 10.0


def test_quantity_size_when_notional_disabled(tmp_path, monkeypatch):
    csv_path = tmp_path / "log.csv"
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))
    monkeypatch.setattr(trade_storage, "SIZE_AS_NOTIONAL", False)
    trade = {
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 2000,
        "size": 0.05,  # quantity when SIZE_AS_NOTIONAL is False
        "strategy": "test",
        "session": "Asia",
    }
    trade_storage.log_trade_result(trade, outcome="tp1", exit_price=2200)
    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["notional"]) == 100.0
    assert float(row["pnl"]) == 10.0
    assert float(row["pnl_pct"]) == 10.0


def test_duplicate_trade_guard(tmp_path, monkeypatch):
    path = tmp_path / "active.json"
    monkeypatch.setattr(trade_storage, "ACTIVE_TRADES_FILE", str(path))
    trade = {"symbol": "BTCUSDT", "entry": 100, "direction": "long"}
    assert trade_manager.create_new_trade(trade) is True
    assert trade_manager.create_new_trade(trade) is False
    trades = trade_storage.load_active_trades()
    assert len(trades) == 1


def test_store_trade_skips_duplicates(tmp_path, monkeypatch):
    path = tmp_path / "active.json"
    monkeypatch.setattr(trade_storage, "ACTIVE_TRADES_FILE", str(path))
    first = {"symbol": "ETHUSDT", "entry": 100}
    second = {"symbol": "ETHUSDT", "entry": 200}
    assert trade_storage.store_trade(first) is True
    assert trade_storage.store_trade(second) is False
    trades = trade_storage.load_active_trades()
    assert len(trades) == 1
    assert trades[0]["entry"] == 100
