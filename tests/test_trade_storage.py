import csv
import pandas as pd

import trade_storage
import trade_manager


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
        "pattern": "double_bottom",
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
    assert rows[0]["tp1_partial"] == "False"
    assert rows[0]["tp2_partial"] == "False"
    assert rows[0]["pattern"] == "double_bottom"


def test_log_trade_result_reassigns_misplaced_strategy_and_session(tmp_path, monkeypatch):
    csv_path = tmp_path / "log.csv"
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))
    trade = {
        "symbol": "BTCUSDT",
        "direction": "long",
        "entry": 1000,
        "size": 1000,
        "position_size": 1,
        "strategy": "2025-09-27T13:46:34Z",
        "session": "2025-09-27T18:46:41Z",
        "pattern": "mean_reversion",
    }

    trade_storage.log_trade_result(
        trade,
        outcome="tp1",
        exit_price=1100,
        exit_time="2025-09-27T18:46:41.123Z",
    )

    df = pd.read_csv(csv_path)
    assert df.loc[0, "entry_time"] == "2025-09-27T13:46:34Z"
    assert df.loc[0, "strategy"] == "mean_reversion"
    session_value = df.loc[0, "session"]
    assert pd.isna(session_value) or session_value == "N/A"
    assert str(df.loc[0, "exit_time"]).startswith("2025-09-27T18:46:41.123")


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
    assert row["tp1_partial"] == "True"
    assert row["tp2_partial"] == "False"


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
    assert float(row["pnl"]) < 0


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


def test_log_trade_result_archives_missing_header(tmp_path, monkeypatch):
    """Legacy files without headers are archived and replaced."""

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

    backups = list(csv_path.parent.glob("log.csv.legacy-*"))
    assert backups, "Expected legacy log to be archived"
    assert "BTCUSDT" in backups[0].read_text()

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["symbol"] == "ETHUSDT"
    assert "BTCUSDT" not in rows[0].values()


def test_log_trade_result_consolidates_duplicate_rows(tmp_path, monkeypatch):
    csv_path = tmp_path / "log.csv"
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))

    trade = {
        "trade_id": "abc123",
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 1000,
        "size": 500,
        "position_size": 0.5,
        "strategy": "test",
        "session": "Asia",
    }

    trade_storage.log_trade_result(trade, outcome="tp1_partial", exit_price=1100)
    # A repeated log for the same trade should not produce duplicate rows in the
    # consolidated history file.
    trade_storage.log_trade_result(trade, outcome="tp1_partial", exit_price=1100)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["trade_id"] == "abc123"
    assert rows[0]["tp1_partial"] == "True"


def test_load_trade_history_df_skips_headerless_csv(tmp_path):
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("BTCUSDT,1000,1100\n")

    df = trade_storage.load_trade_history_df(path=str(legacy))

    assert df.empty


def test_log_trade_result_writes_file_when_db_cursor_present(tmp_path, monkeypatch):
    """Trades should always append to CSV even if a DB cursor is configured."""
    csv_path = tmp_path / "log.csv"

    class DummyCursor:
        def __init__(self):
            self.calls = []

        def execute(self, query, params=None):
            self.calls.append((query, params))

    dummy_cursor = DummyCursor()
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(csv_path))
    monkeypatch.setattr(trade_storage, "DB_CURSOR", dummy_cursor)
    monkeypatch.setattr(trade_storage, "Json", lambda x: x)

    trade = {
        "symbol": "ETHUSDT",
        "direction": "long",
        "entry": 1000,
        "size": 1000,
        "position_size": 1,
        "strategy": "test",
        "session": "Asia",
    }

    trade_storage.log_trade_result(trade, outcome="tp1", exit_price=1100)

    # Should write a row to CSV
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and rows[0]["symbol"] == "ETHUSDT"

    # And the DB cursor should have been used
    assert dummy_cursor.calls


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


def test_load_trade_history_df_alias_normalisation(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        "time,symbol,direction,outcome,entry_price,exit_price,sent_conf\n"
        "2024-01-01T00:00:00Z,BTCUSDT,long,tp1,100,110,7.5\n"
    )
    df = trade_storage.load_trade_history_df(str(csv_path))
    assert "timestamp" in df.columns
    assert "sentiment_confidence" in df.columns
    assert float(df["sentiment_confidence"].iloc[0]) == 7.5


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
