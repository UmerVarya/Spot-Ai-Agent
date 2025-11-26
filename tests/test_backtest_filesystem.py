from pathlib import Path

from backtest.filesystem import discover_backtest_files, get_backtest_dir


def test_discover_backtest_files_parses_pattern(tmp_path: Path):
    valid = tmp_path / "BTCUSDT_1m_2024-01-01_2024-02-01_trades.csv"
    valid.touch()
    newer = tmp_path / "ETHUSDT_1m_2024-03-01_2024-03-15_metrics.csv"
    newer.touch()
    invalid = tmp_path / "random.txt"
    invalid.touch()

    files = discover_backtest_files(tmp_path)

    assert len(files) == 2
    # Sorted by end date descending
    assert files[0].path == newer
    assert files[0].symbol == "ETHUSDT"
    assert files[0].timeframe == "1m"
    assert files[0].start == "2024-03-01"
    assert files[0].end == "2024-03-15"
    assert files[0].kind == "metrics"


def test_get_backtest_dir_creates_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTESTS_DIR", str(tmp_path / "nested"))
    path = get_backtest_dir()
    assert path.exists()
    assert path.is_dir()
