from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest import data as data_loader
from backtest.data_manager import ensure_ohlcv_csvs


class DummyBinanceClient:
    def __init__(self, interval_ms: int = 60_000) -> None:
        self.calls = []
        self.interval_ms = interval_ms

    def get_historical_klines(self, symbol, interval, start_str=None, end_str=None):  # pragma: no cover - thin shim
        self.calls.append((symbol, interval, start_str, end_str))
        start_ms = int(start_str)
        end_ms = int(end_str)
        rows = []
        current = start_ms
        while current <= end_ms:
            rows.append(
                [
                    current,
                    "1",
                    "2",
                    "0.5",
                    "1.5",
                    "100",
                    current + self.interval_ms - 1,
                    "200",
                    "1",
                    "50",
                    "100",
                    "0",
                ]
            )
            current += self.interval_ms
        return rows


def test_ensure_ohlcv_creates_and_appends(monkeypatch, tmp_path: Path) -> None:
    client = DummyBinanceClient()
    monkeypatch.setattr("backtest.data_manager._get_binance_client", lambda: client)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    mid_end = start + timedelta(minutes=2)

    csv_paths = ensure_ohlcv_csvs(["BTCUSDT"], "1m", start, mid_end, data_dir=tmp_path)
    assert csv_paths and csv_paths[0].exists()

    df = pd.read_csv(csv_paths[0])
    assert len(df) == 3
    assert df["timestamp"].is_unique

    final_end = start + timedelta(minutes=4)
    csv_paths = ensure_ohlcv_csvs(["BTCUSDT"], "1m", start, final_end, data_dir=tmp_path)

    df_updated = pd.read_csv(csv_paths[0])
    assert len(df_updated) == 5  # two additional rows appended
    assert df_updated["timestamp"].is_unique
    assert client.calls[-1][2] > client.calls[0][2]  # tail fetch should start after initial window


def test_fetches_missing_leading_candle(monkeypatch, tmp_path: Path) -> None:
    client = DummyBinanceClient()
    monkeypatch.setattr("backtest.data_manager._get_binance_client", lambda: client)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    interval = "1m"

    existing_start = start + timedelta(minutes=1)
    existing_df = pd.DataFrame(
        {
            "timestamp": [existing_start, existing_start + timedelta(minutes=1)],
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [100, 100],
            "quote_volume": [200, 200],
        }
    )
    csv_path = tmp_path / "BTCUSDT_1m.csv"
    existing_df.to_csv(csv_path, index=False)

    end = existing_start + timedelta(minutes=1)
    csv_paths = ensure_ohlcv_csvs(["BTCUSDT"], interval, start, end, data_dir=tmp_path)

    assert csv_paths and csv_paths[0].exists()
    df = pd.read_csv(csv_paths[0])
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    assert len(df) == 3
    assert timestamps.min() == start
    assert timestamps.max() == end
    assert len(client.calls) == 1
    assert client.calls[0][2] == int(start.timestamp() * 1000)
    assert client.calls[0][3] == int(start.timestamp() * 1000)


def test_uses_cached_range_without_download(monkeypatch, tmp_path: Path) -> None:
    client = DummyBinanceClient()
    monkeypatch.setattr("backtest.data_manager._get_binance_client", lambda: client)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(minutes=4)
    index = pd.date_range(start=start, periods=5, freq="T", tz=timezone.utc)
    existing_df = pd.DataFrame(
        {
            "timestamp": index,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 100.0,
            "quote_volume": 200.0,
        }
    )
    csv_path = tmp_path / "BTCUSDT_1m.csv"
    existing_df.to_csv(csv_path, index=False)

    csv_paths = ensure_ohlcv_csvs(["BTCUSDT"], "1m", start, end, data_dir=tmp_path)

    assert csv_paths and csv_paths[0].exists()
    assert client.calls == []


def test_load_csv_paths_reuses_cache(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "BTCUSDT_1m.csv"
    index = pd.date_range(start=datetime(2024, 1, 1, tzinfo=timezone.utc), periods=3, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": index,
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [10, 20, 30],
            "quote_volume": [20, 40, 60],
        }
    )
    df.to_csv(csv_path, index=False)

    real_read_csv = data_loader.pd.read_csv
    read_count = {"count": 0}

    def _counting_read_csv(*args, **kwargs):
        read_count["count"] += 1
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(data_loader.pd, "read_csv", _counting_read_csv)

    start = index.min()
    end = index.max()
    first_load = data_loader.load_csv_paths([csv_path], start=start, end=end)
    second_load = data_loader.load_csv_paths([csv_path], start=start, end=end)

    assert read_count["count"] == 1
    assert first_load.keys() == second_load.keys()

    data_loader.invalidate_cache_for_paths([csv_path])
    data_loader.load_csv_paths([csv_path], start=start, end=end)
    assert read_count["count"] == 2


def test_load_csv_paths_end_is_exclusive(tmp_path: Path) -> None:
    csv_path = tmp_path / "BTCUSDT_1h.csv"
    index = pd.date_range(start=datetime(2024, 8, 1, tzinfo=timezone.utc), periods=26, freq="H")
    df = pd.DataFrame(
        {
            "timestamp": index,
            "open": range(len(index)),
            "high": range(len(index)),
            "low": range(len(index)),
            "close": range(len(index)),
            "volume": [100] * len(index),
            "quote_volume": [200] * len(index),
        }
    )
    df.to_csv(csv_path, index=False)

    end = datetime(2024, 8, 2, tzinfo=timezone.utc)
    windowed = data_loader.load_csv_paths([csv_path], start=index.min(), end=end)

    assert list(windowed.keys()) == ["BTCUSDT"]
    df_windowed = windowed["BTCUSDT"]
    assert df_windowed.index.max() == pd.Timestamp("2024-08-01 23:00", tz="UTC")
    assert pd.Timestamp("2024-08-02", tz="UTC") not in df_windowed.index


def test_load_csv_paths_skips_missing_files(tmp_path: Path) -> None:
    missing_path = tmp_path / "DOES_NOT_EXIST.csv"

    result = data_loader.load_csv_paths([missing_path], use_cache=True)

    assert result == {}
