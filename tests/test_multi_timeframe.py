import pandas as pd
from datetime import datetime
import multi_timeframe


def _sample_df():
    idx = pd.date_range('2020-01-01', periods=10, freq='1T')
    data = {
        'open': range(10),
        'high': range(10),
        'low': range(10),
        'close': range(10),
        'volume': range(10),
    }
    return pd.DataFrame(data, index=idx)


def test_resample_ohlcv_labels_right():
    df = _sample_df()
    res = multi_timeframe.resample_ohlcv(df, '5T')
    assert res.index[-1] == pd.Timestamp('2020-01-01 00:10')


def test_multi_timeframe_confluence_skips_incomplete_bar(monkeypatch):
    df = _sample_df()
    fixed_now = datetime(2020, 1, 1, 0, 9)

    class DummyDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now if tz is None else fixed_now.replace(tzinfo=tz)

    monkeypatch.setattr(multi_timeframe, 'datetime', DummyDateTime)
    result = multi_timeframe.multi_timeframe_confluence(df, ['5T'], lambda s: s.iloc[-1])
    assert result == {'5T': 5}
