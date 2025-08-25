import pandas as pd
from datetime import datetime

import multi_timeframe

def _build_df(start, periods, freq="1T"):
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    data = {
        "open": range(periods),
        "high": range(periods),
        "low": range(periods),
        "close": range(periods),
        "volume": range(periods),
    }
    return pd.DataFrame(data, index=idx)

def test_resample_ohlcv_labels_right():
    df = _build_df("2024-01-01 00:00:00", 5)  # 5 one-minute bars
    resampled = multi_timeframe.resample_ohlcv(df, "5T")
    assert resampled.index[-1] == pd.Timestamp("2024-01-01 00:05:00")

def test_multi_timeframe_confluence_drops_open_bar(monkeypatch):
    df = _build_df("2024-01-01 09:00:00", 90)  # 1.5 hours of data

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2024, 1, 1, 10, 30, tzinfo=tz)

    monkeypatch.setattr(multi_timeframe, "datetime", FixedDateTime)

    def length(series):
        return len(series)

    res = multi_timeframe.multi_timeframe_confluence(df, ["1H"], length)
    assert res["1H"] == 2.0  # only two closed hourly bars
