import numpy as np
import pandas as pd

from backtest.legacy import Backtester
from trade_utils import precompute_backtest_indicators


def _build_price_df(periods: int = 120, freq: str = "T") -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq=freq)
    base = np.linspace(100, 110, len(index))
    return pd.DataFrame(
        {
            "open": base * 0.999,
            "high": base * 1.001,
            "low": base * 0.998,
            "close": base,
            "volume": np.linspace(50, 60, len(index)),
        },
        index=index,
    )


def test_precompute_trend_states_fill_dataframe():
    df = _build_price_df(periods=360, freq="T")

    precompute_backtest_indicators(df)

    assert "trend_1h_state" in df.columns
    assert "trend_4h_state" in df.columns
    assert df["trend_1h_state"].isna().sum() == 0
    assert df["trend_4h_state"].isna().sum() == 0


def test_backtester_progress_counts_all_bars():
    df = _build_price_df(periods=30, freq="T")
    historical_data = {"BTCUSDT": df}
    progress_events = []

    def progress_callback(progress):
        progress_events.append((progress.phase, progress.current, progress.total))

    bt = Backtester(
        historical_data,
        evaluate_signal=lambda *_args, **_kwargs: {"score": 1.0, "direction": "long", "confidence": 1.0},
        predict_prob=lambda *_args, **_kwargs: 0.6,
        macro_filter=lambda: True,
        position_size_func=lambda _confidence: 0.01,
    )

    bt.run(
        {
            "min_score": 0.0,
            "min_prob": 0.0,
            "atr_mult_sl": 1.5,
            "tp_rungs": (1.0, 2.0),
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "latency_bars": 0,
            "max_concurrent": 1,
            "is_backtest": True,
        },
        progress_callback=progress_callback,
    )

    assert progress_events, "Progress callback should be invoked"
    phases = [event[0] for event in progress_events]
    assert "simulating" in phases
    last_event = progress_events[-1]
    assert last_event[1] == last_event[2] == len(df)
