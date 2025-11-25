import json
from pathlib import Path

import pandas as pd
import pytest

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


def test_cli_writes_outputs(tmp_path: Path, sample_csv: Path, monkeypatch):
    out_dir = tmp_path / "out"

    # Ensure tests do not call external data sources
    monkeypatch.chdir(Path.cwd())

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

    expected_summary = out_dir / "summary_1m_20240101_20240101.json"
    assert expected_summary.exists()

    summary = json.loads(expected_summary.read_text())
    assert "overall" in summary

    trades_files = list(out_dir.glob("*_trades.csv"))
    equity_files = list(out_dir.glob("*_equity.csv"))
    assert trades_files, "Trades CSV not written"
    assert equity_files, "Equity CSV not written"
