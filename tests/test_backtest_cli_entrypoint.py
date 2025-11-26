import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import backtest_cli
from backtest.engine import BacktestConfig, BacktestResult


@pytest.fixture
def sample_result():
    cfg = BacktestConfig(
        start_ts=pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc)),
        end_ts=pd.Timestamp(datetime(2024, 1, 2, tzinfo=timezone.utc)),
    )
    trades = pd.DataFrame(
        [
            {"symbol": "BTCUSDT", "entry_time": "2024-01-01", "exit_time": "2024-01-01", "return": 0.01},
        ]
    )
    equity = pd.DataFrame({"timestamp": ["2024-01-01"], "equity": [10_100]})
    metrics = {"win_rate": 0.5, "profit_factor": 1.2}
    return BacktestResult(cfg, trades, equity, metrics, pd.DataFrame(), {}, None, {})


def test_cli_writes_expected_outputs(tmp_path: Path, sample_result, monkeypatch):
    called_args = {}

    def _fake_run_backtest(**kwargs):
        called_args.update(kwargs)
        return sample_result

    monkeypatch.setattr(backtest_cli, "run_backtest", _fake_run_backtest)

    start = "2024-01-01"
    end = "2024-01-02"
    exit_code = backtest_cli.main(
        [
            "--symbol",
            "BTCUSDT",
            "--timeframe",
            "1m",
            "--start",
            start,
            "--end",
            end,
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert called_args["symbol"] == "BTCUSDT"
    assert called_args["timeframe"] == "1m"

    base = tmp_path / "BTCUSDT_1m_2024-01-01_2024-01-02"
    trades_path = base.with_name(base.name + "_trades.csv")
    equity_path = base.with_name(base.name + "_equity.csv")
    metrics_path = base.with_name(base.name + "_metrics.csv")

    assert trades_path.exists()
    assert equity_path.exists()
    assert metrics_path.exists()

    metrics_df = pd.read_csv(metrics_path)
    assert metrics_df.loc[0, "win_rate"] == 0.5


def test_month_resolution_requires_range(monkeypatch):
    # Ensure users must pass range arguments
    with pytest.raises(ValueError):
        backtest_cli._resolve_date_range(argparse.Namespace(start=None, end=None, months=None))
