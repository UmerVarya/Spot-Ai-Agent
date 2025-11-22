from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

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
