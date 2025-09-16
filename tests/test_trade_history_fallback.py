import importlib

import trade_storage


def test_load_history_uses_legacy_file_when_primary_empty(monkeypatch, tmp_path):
    primary = tmp_path / "primary.csv"
    legacy = tmp_path / "legacy.csv"
    primary.write_text("")
    legacy.write_text(
        "trade_id,timestamp,symbol,entry,exit,size,direction,outcome,pnl\n"
        "1,2024-01-01T00:00:00Z,BTCUSDT,100,110,1,long,tp1,10\n"
    )

    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(primary))
    monkeypatch.setattr(trade_storage, "_LEGACY_HISTORY_FILES", [str(legacy)])
    monkeypatch.setattr(trade_storage, "_HISTORY_ENV_OVERRIDE", False)

    df = trade_storage.load_trade_history_df()
    assert not df.empty
    assert "BTCUSDT" in df["symbol"].astype(str).tolist()


def test_completed_trades_env_alias(monkeypatch, tmp_path):
    alias_file = tmp_path / "alias.csv"
    alias_file.write_text("")

    with monkeypatch.context() as m:
        m.delenv("TRADE_HISTORY_FILE", raising=False)
        m.setenv("COMPLETED_TRADES_FILE", str(alias_file))
        importlib.reload(trade_storage)
        assert trade_storage.TRADE_HISTORY_FILE == str(alias_file)

    importlib.reload(trade_storage)
