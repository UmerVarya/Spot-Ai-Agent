import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import backtest_cli
from backtest.filesystem import discover_backtest_files, build_backtest_output_paths
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

    def _fake_launch_backtest(**kwargs):
        called_args.update(kwargs)
        paths = build_backtest_output_paths(kwargs["backtest_id"], kwargs["out_dir"])
        paths["trades"].parent.mkdir(parents=True, exist_ok=True)
        sample_result.trades.to_csv(paths["trades"], index=False)
        sample_result.equity_curve.to_csv(paths["equity"], index=False)
        metrics_path = paths["metrics"]
        pd.DataFrame([sample_result.metrics]).to_json(metrics_path, orient="records")
        paths["meta"].write_text(json.dumps({"status": "completed"}))
        return {"paths": paths}

    monkeypatch.setattr(backtest_cli, "launch_backtest", _fake_launch_backtest)

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
    assert called_args["symbols"] == ["BTCUSDT"]
    assert called_args["timeframe"] == "1m"

    base = tmp_path / "BTCUSDT_1m_2024-01-01_2024-01-02"
    trades_path = base.with_name(base.name + "_trades.csv")
    equity_path = base.with_name(base.name + "_equity.csv")
    metrics_path = base.with_name(base.name + "_metrics.json")

    assert trades_path.exists()
    assert equity_path.exists()
    assert metrics_path.exists()

    metrics_df = pd.read_json(metrics_path)
    assert metrics_df.loc[0, "win_rate"] == 0.5

    discovered = discover_backtest_files(tmp_path)
    kinds = {f.kind for f in discovered}
    assert kinds == {"trades", "equity", "metrics"}


def test_month_resolution_requires_range(monkeypatch):
    # Ensure users must pass range arguments
    with pytest.raises(ValueError):
        backtest_cli._resolve_date_range(argparse.Namespace(start=None, end=None, months=None))
