import pandas as pd
import pytest
import trade_storage


def test_deduplicate_history(tmp_path, monkeypatch):
    data = [
        {
            "trade_id": "1",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-01-01T00:00:00Z",
            "exit_time": "2024-01-01T01:00:00Z",
            "entry": 100.0,
            "exit": 110.0,
            "size": 50.0,
            "position_size": 0.5,
            "strategy": "s1",
            "outcome": "tp1_partial",
            "fees": 0,
            "slippage": 0,
        },
        {
            "trade_id": "1",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-01-01T00:00:00Z",
            "exit_time": "2024-01-01T02:00:00Z",
            "entry": 100.0,
            "exit": 120.0,
            "size": 50.0,
            "position_size": 0.5,
            "strategy": "s1",
            "outcome": "tp2_partial",
            "fees": 0,
            "slippage": 0,
        },
        {
            "trade_id": "2",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-01-01T00:00:00Z",
            "exit_time": "2024-01-01T03:00:00Z",
            "entry": 200.0,
            "exit": 210.0,
            "size": 200.0,
            "position_size": 1.0,
            "strategy": "s1",
            "outcome": "tp2",
            "fees": 0,
            "slippage": 0,
        },
    ]
    df = pd.DataFrame(data)
    hist_file = tmp_path / "completed.csv"
    df.to_csv(hist_file, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(hist_file))
    result = trade_storage.load_trade_history_df().sort_values("trade_id").reset_index(drop=True)

    # trade 1 collapsed into a single row
    t1 = result[result["trade_id"].astype(str) == "1"].iloc[0]
    assert bool(t1["tp1_partial"])
    assert bool(t1["tp2_partial"])
    assert t1["pnl"] == 15.0
    assert t1["pnl_tp1"] == 5.0
    assert t1["pnl_tp2"] == 10.0
    assert t1["notional"] == 100.0
    assert t1["notional_tp1"] == 50.0
    assert t1["notional_tp2"] == 50.0
    assert t1["notional_tp3"] == 0.0
    assert t1["size"] == 100.0
    assert t1["size_tp1"] == 0.5
    assert t1["size_tp2"] == 0.5
    assert t1["size_tp3"] == 0.0
    assert t1["pnl_tp3"] == 0.0
    assert t1["position_size"] == 1.0

    # second trade remains separate
    assert len(result) == 2
    t2 = result[result["trade_id"].astype(str) == "2"].iloc[0]
    assert t2["pnl"] == 10.0
    assert not bool(t2["tp1_partial"])
    assert not bool(t2["tp2_partial"])
    assert t2["pnl_tp3"] == 0.0


def test_deduplicate_history_uses_final_exit_size_for_remaining_leg(tmp_path, monkeypatch):
    data = [
        {
            "trade_id": "10",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-05-01T00:00:00Z",
            "exit_time": "2024-05-01T01:00:00Z",
            "entry": 100.0,
            "exit": 105.0,
            "size": 400.0,
            "position_size": 4.0,
            "strategy": "s5",
            "outcome": "tp1_partial",
        },
        {
            "trade_id": "10",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-05-01T00:00:00Z",
            "exit_time": "2024-05-01T03:00:00Z",
            "entry": 100.0,
            "exit": 110.0,
            "size": 600.0,
            "position_size": 0.0,
            "initial_size": 10.0,
            "final_exit_size": 6.0,
            "strategy": "s5",
            "outcome": "tp3",
        },
    ]

    df = pd.DataFrame(data)
    hist_file = tmp_path / "final_exit.csv"
    df.to_csv(hist_file, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(hist_file))

    result = trade_storage.load_trade_history_df()
    trade = result.iloc[0]

    # Final leg should contribute only the remaining 6 units instead of the
    # original 10-unit initial position.
    assert trade["position_size"] == pytest.approx(10.0)
    assert trade["size_tp1"] == pytest.approx(4.0)
    assert trade["size_tp3"] == pytest.approx(6.0)
    assert trade["pnl"] == pytest.approx(80.0)


def test_deduplicate_history_tp3_allocation(tmp_path, monkeypatch):
    data = [
        {
            "trade_id": "5",
            "symbol": "ETHUSDT",
            "direction": "long",
            "entry_time": "2024-04-01T00:00:00Z",
            "exit_time": "2024-04-01T01:00:00Z",
            "entry": 100.0,
            "exit": 110.0,
            "size": 50.0,
            "position_size": 0.5,
            "strategy": "s4",
            "outcome": "tp1_partial",
            "fees": 0.1,
            "slippage": 0.0,
        },
        {
            "trade_id": "5",
            "symbol": "ETHUSDT",
            "direction": "long",
            "entry_time": "2024-04-01T00:00:00Z",
            "exit_time": "2024-04-01T02:00:00Z",
            "entry": 100.0,
            "exit": 130.0,
            "size": 50.0,
            "position_size": 0.5,
            "strategy": "s4",
            "outcome": "tp3",
            "fees": 0.4,
            "slippage": 0.0,
        },
    ]

    df = pd.DataFrame(data)
    hist_file = tmp_path / "completed.csv"
    df.to_csv(hist_file, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(hist_file))

    result = trade_storage.load_trade_history_df()
    trade = result.iloc[0]

    assert trade["pnl"] == pytest.approx(19.6)
    assert trade["pnl_tp1"] == pytest.approx(4.9)
    assert trade["pnl_tp3"] == pytest.approx(14.7)
    assert trade["size_tp3"] == pytest.approx(0.5)
    assert trade["notional_tp3"] == pytest.approx(50.0)


def test_deduplicate_history_cumulative_fees(tmp_path, monkeypatch):
    data = [
        {
            "trade_id": "3",
            "symbol": "ETHUSDT",
            "direction": "long",
            "entry_time": "2024-02-01T00:00:00Z",
            "exit_time": "2024-02-01T01:00:00Z",
            "entry": 100.0,
            "exit": 110.0,
            "size": 50.0,
            "position_size": 0.5,
            "strategy": "s2",
            "outcome": "tp1_partial",
            "fees": 0.2,
            "slippage": 0.1,
        },
        {
            "trade_id": "3",
            "symbol": "ETHUSDT",
            "direction": "long",
            "entry_time": "2024-02-01T00:00:00Z",
            "exit_time": "2024-02-01T02:00:00Z",
            "entry": 100.0,
            "exit": 120.0,
            "size": 50.0,
            "position_size": 0.5,
            "strategy": "s2",
            "outcome": "tp2",
            "fees": 1.1,  # cumulative fees from both legs
            "slippage": 0.25,  # cumulative slippage from both legs
        },
    ]
    df = pd.DataFrame(data)
    hist_file = tmp_path / "completed.csv"
    df.to_csv(hist_file, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(hist_file))

    result = trade_storage.load_trade_history_df()
    trade = result.iloc[0]

    # Total net PnL subtracts cumulative costs only once
    assert trade["pnl"] == pytest.approx(13.65)
    assert trade["fees"] == pytest.approx(1.1)
    assert trade["slippage"] == pytest.approx(0.25)

    # Stage allocations use net contributions per leg
    assert trade["tp1_partial"]
    assert trade["pnl_tp1"] == pytest.approx(4.7)
    # Remaining leg contribution equals total minus TP1 allocation
    assert (trade["pnl"] - trade["pnl_tp1"]) == pytest.approx(8.95)


def test_deduplicate_history_partial_fees_without_final_exit(tmp_path, monkeypatch):
    data = [
        {
            "trade_id": "4",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-03-01T00:00:00Z",
            "exit_time": "2024-03-01T01:00:00Z",
            "entry": 100.0,
            "exit": 110.0,
            "size": 50.0,
            "position_size": 0.5,
            "strategy": "s3",
            "outcome": "tp1_partial",
            "fees": 0.2,
            "slippage": 0.0,
        },
        {
            "trade_id": "4",
            "symbol": "BTCUSDT",
            "direction": "long",
            "entry_time": "2024-03-01T00:00:00Z",
            "exit_time": "2024-03-01T02:00:00Z",
            "entry": 100.0,
            "exit": 115.0,
            "size": 50.0,
            "position_size": 0.5,
            "strategy": "s3",
            "outcome": "tp2_partial",
            "fees": 0.9,
            "slippage": 0.0,
        },
    ]

    df = pd.DataFrame(data)
    hist_file = tmp_path / "completed.csv"
    df.to_csv(hist_file, index=False)
    monkeypatch.setattr(trade_storage, "TRADE_HISTORY_FILE", str(hist_file))

    result = trade_storage.load_trade_history_df()
    trade = result.iloc[0]

    assert trade["fees"] == pytest.approx(1.1)
    assert trade["pnl"] == pytest.approx(11.4)
