import pandas as pd
from datetime import datetime, timezone, timedelta

import multi_timeframe as mt
from multi_timeframe import resample_ohlcv, multi_timeframe_confluence


def _make_df(index):
    return pd.DataFrame({
        'open': range(len(index)),
        'high': range(len(index)),
        'low': range(len(index)),
        'close': range(len(index)),
        'volume': 1,
    }, index=index)


def test_resample_ohlcv_labels_right():
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    index = pd.date_range(end=now - timedelta(minutes=1), periods=10, freq='T', tz=timezone.utc)
    df = _make_df(index)

    resampled = resample_ohlcv(df, '5T')

    # Bars should be labelled by their closing time (right edge).
    # First bar should close five minutes after the start of the
    # initial bin (i.e., at the right edge of the 5‑minute window).
    expected_first = index[0].floor('5T') + timedelta(minutes=5)
    assert resampled.index[0] == expected_first
    assert all(ts.minute % 5 == 0 for ts in resampled.index)

    # The latest resampled bar should not extend beyond the current time.
    assert resampled.index[-1] <= pd.Timestamp(datetime.now(timezone.utc))


def test_multi_timeframe_confluence_skips_future_bars(monkeypatch):
    fixed_now = datetime(2024, 1, 1, 10, 6, tzinfo=timezone.utc)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is None else fixed_now.astimezone(tz)

    # Patch datetime used inside module to ensure deterministic behaviour.
    monkeypatch.setattr(mt.dt, "datetime", FixedDateTime)

    start = fixed_now - timedelta(minutes=9)
    end = fixed_now + timedelta(minutes=6)
    index = pd.date_range(start=start, end=end, freq='T', tz=timezone.utc)
    df = _make_df(index)

    resampled_all = resample_ohlcv(df, '5T')
    assert resampled_all.index[-1] > pd.Timestamp(fixed_now)

    results = multi_timeframe_confluence(df, ['5T'], lambda s: float(len(s)))
    expected = resample_ohlcv(df, '5T')
    expected = expected[expected.index <= pd.Timestamp(fixed_now)]
    assert results['5T'] == float(len(expected))
    assert expected.index[-1] <= pd.Timestamp(fixed_now)
