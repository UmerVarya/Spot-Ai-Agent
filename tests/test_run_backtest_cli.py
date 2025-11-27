import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from backtest.engine import BacktestConfig, BacktestResult
from backtest.types import BacktestProgress
from run_backtest_cli import run_cli


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    timestamps = pd.date_range("2024-01-01", periods=20, freq="1min", tz="UTC")
    prices = pd.Series(range(1, 21), index=timestamps, dtype=float)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices + 0.25,
            "volume": 1000.0,
            "quote_volume": 1000.0,
        }
    )
    csv_path = tmp_path / "TEST_1m.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def stub_backtest_result() -> BacktestResult:
    cfg = BacktestConfig(
        start_ts=pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc)),
        end_ts=pd.Timestamp(datetime(2024, 1, 2, tzinfo=timezone.utc)),
        initial_capital=10_000.0,
    )
    trades = pd.DataFrame(
        [
            {
                "symbol": "TEST",
                "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2024-01-01T00:10:00Z"),
                "net_return": 0.01,
                "net_pnl_quote": 100.0,
                "r_multiple": 1.0,
            }
        ]
    )
    equity = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T00:10:00Z")],
            "equity": [10_100.0],
            "drawdown_pct": [0.0],
            "drawdown": [0.0],
        }
    )
    metrics = {"win_rate": 100.0}
    return BacktestResult(cfg, trades, equity, metrics, pd.DataFrame(), {}, None, {})


def test_cli_writes_outputs_with_meta(tmp_path: Path, sample_csv: Path, stub_backtest_result: BacktestResult, monkeypatch):
    out_dir = tmp_path / "out"

    def _fake_run(csv_paths, cfg, symbols=None, progress_callback=None):
        if progress_callback:
            progress_callback(BacktestProgress(phase="simulating", current=10, total=10))
        return stub_backtest_result

    monkeypatch.chdir(Path.cwd())
    monkeypatch.setattr("run_backtest_cli.run_backtest_from_csv_paths", _fake_run)

    args = [
        "--symbols",
        "TEST",
        "--timeframe",
        "1m",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-01",
        "--csv-paths",
        str(sample_csv),
        "--out-dir",
        str(out_dir),
        "--score-threshold",
        "0.0",
        "--min-prob",
        "0.0",
    ]

    exit_code = run_cli(args)
    assert exit_code == 0

    meta_files = list(out_dir.glob("*_meta.json"))
    assert meta_files, "meta.json not written"
    metrics_files = list(out_dir.glob("*_metrics.json"))
    trades_files = list(out_dir.glob("*_trades.csv"))
    equity_files = list(out_dir.glob("*_equity.csv"))

    assert metrics_files and trades_files and equity_files

    meta = json.loads(meta_files[0].read_text())
    assert meta["status"] == "completed"
    assert meta["progress"] == 1.0
    assert meta["total_bars"] >= meta["current_bar"]

    metrics = json.loads(metrics_files[0].read_text())
    assert metrics.get("total_trades") == 1.0
    equity = pd.read_csv(equity_files[0])
    assert not equity.empty


def test_cli_handles_no_trades(tmp_path: Path, sample_csv: Path, monkeypatch):
    out_dir = tmp_path / "out"
    cfg = BacktestConfig(
        start_ts=pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc)),
        end_ts=pd.Timestamp(datetime(2024, 1, 2, tzinfo=timezone.utc)),
        initial_capital=10_000.0,
    )
    empty_result = BacktestResult(cfg, pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), {}, None, {})

    def _fake_run(csv_paths, cfg, symbols=None, progress_callback=None):
        if progress_callback:
            progress_callback(BacktestProgress(phase="simulating", current=0, total=0))
        return empty_result

    monkeypatch.setattr("run_backtest_cli.run_backtest_from_csv_paths", _fake_run)

    args = [
        "--symbols",
        "TEST",
        "--timeframe",
        "1m",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-01",
        "--csv-paths",
        str(sample_csv),
        "--out-dir",
        str(out_dir),
    ]

    exit_code = run_cli(args)
    assert exit_code == 0

    trades_files = list(out_dir.glob("*_trades.csv"))
    equity_files = list(out_dir.glob("*_equity.csv"))
    metrics_files = list(out_dir.glob("*_metrics.json"))
    meta_files = list(out_dir.glob("*_meta.json"))

    assert trades_files and equity_files and metrics_files and meta_files

    trades = pd.read_csv(trades_files[0])
    assert trades.empty
    assert list(trades.columns)

    equity = pd.read_csv(equity_files[0])
    assert not equity.empty
    metrics = json.loads(metrics_files[0].read_text())
    assert metrics.get("total_trades") == 0.0
    meta = json.loads(meta_files[0].read_text())
    assert meta["status"] == "completed"
    assert meta["progress"] == 1.0
