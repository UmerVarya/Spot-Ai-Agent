import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.engine import BacktestConfig, BacktestResult
from backtest.types import BacktestProgress
import backtest_cli


def _sample_csv(tmp_path: Path) -> Path:
    timestamps = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 100.0,
            "quote_volume": 100.0,
        }
    )
    path = tmp_path / "TEST_1m.csv"
    df.to_csv(path, index=False)
    return path


def _stub_result() -> BacktestResult:
    cfg = BacktestConfig(
        start_ts=pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc)),
        end_ts=pd.Timestamp(datetime(2024, 1, 3, tzinfo=timezone.utc)),
        initial_capital=10_000.0,
    )
    trades = pd.DataFrame(
        [
            {
                "symbol": "TEST",
                "entry_time": pd.Timestamp("2024-01-01T00:00:00Z"),
                "exit_time": pd.Timestamp("2024-01-01T00:01:00Z"),
                "net_pnl_quote": 10.0,
                "r_multiple": 1.0,
            }
        ]
    )
    equity = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T00:01:00Z")],
            "equity": [10_010.0],
            "drawdown_pct": [0.0],
            "drawdown": [0.0],
        }
    )
    return BacktestResult(cfg, trades, equity, {}, pd.DataFrame(), {}, None, {})


def test_cli_writes_artifacts_and_meta(tmp_path, monkeypatch):
    csv_path = _sample_csv(tmp_path)
    result = _stub_result()

    def _fake_ensure(symbols, timeframe, start, end, data_dir):
        return [csv_path]

    def _fake_run(csv_paths, cfg, symbols=None, progress_callback=None):
        if progress_callback:
            progress_callback(BacktestProgress(phase="simulating", current=5, total=5))
        return result

    monkeypatch.setattr("backtest.run.ensure_ohlcv_csvs", _fake_ensure)
    monkeypatch.setattr("backtest.run.run_backtest_from_csv_paths", _fake_run)

    out_dir = tmp_path / "out"
    args = [
        "--symbols",
        "TEST",
        "--timeframe",
        "1m",
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-02",
        "--score-threshold",
        "0.3",
        "--min-prob",
        "0.6",
        "--exit-mode",
        "atr_trailing",
        "--trade-size-usd",
        "750",
        "--fee-bps",
        "5",
        "--slippage-bps",
        "1",
        "--random-seed",
        "42",
        "--output-dir",
        str(out_dir),
    ]

    exit_code = backtest_cli.main(args)
    assert exit_code == 0

    meta_files = list(out_dir.glob("*_meta.json"))
    metrics_files = list(out_dir.glob("*_metrics.json"))
    trades_files = list(out_dir.glob("*_trades.csv"))
    equity_files = list(out_dir.glob("*_equity.csv"))

    assert meta_files and metrics_files and trades_files and equity_files

    meta = json.loads(meta_files[0].read_text())
    assert meta["symbols"] == ["TEST"]
    assert meta["timeframe"] == "1m"
    assert meta["start_date"] == "2024-01-01"
    assert meta["end_date"] == "2024-01-02"
    assert meta["params"]["score_threshold"] == 0.3
    assert meta["params"]["min_prob"] == 0.6
    assert meta["params"]["exit_mode"] == "atr_trailing"
    assert meta["params"]["trade_size_usd"] == 750.0
    assert meta["params"]["fee_bps"] == 5.0
    assert meta["params"]["slippage_bps"] == 1.0
    assert meta["params"]["random_seed"] == 42
    assert meta["status"] == "completed"

    trades = pd.read_csv(trades_files[0])
    assert list(trades.columns)
    equity = pd.read_csv(equity_files[0])
    assert not equity.empty


def test_cli_legacy_symbol_flag(tmp_path, monkeypatch):
    csv_path = _sample_csv(tmp_path)
    result = _stub_result()

    monkeypatch.setattr("backtest.run.ensure_ohlcv_csvs", lambda *_, **__: [csv_path])
    monkeypatch.setattr("backtest.run.run_backtest_from_csv_paths", lambda *_, **__: result)

    out_dir = tmp_path / "out"
    args = [
        "--symbol",
        "TEST",
        "--timeframe",
        "1m",
        "--months",
        "1",
        "--output-dir",
        str(out_dir),
    ]

    exit_code = backtest_cli.main(args)
    assert exit_code == 0

    assert list(out_dir.glob("*_meta.json")), "meta.json not written for legacy symbol"
