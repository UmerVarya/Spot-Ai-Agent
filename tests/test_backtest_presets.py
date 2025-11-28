import numpy as np
import pandas as pd

from backtest.presets import (
    BacktestPresetConfig,
    PRESET_FULL_AUDIT,
    PRESET_QUICK_SMOKE,
    PRESET_STANDARD_RESEARCH,
    resolve_preset,
)
from backtest.legacy import Backtester


def _build_df(periods: int = 40) -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.utcnow(), periods=periods, freq="T")
    base = np.linspace(100, 120, len(index))
    return pd.DataFrame(
        {
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base * 1.005,
            "volume": np.linspace(10, 20, len(index)),
        },
        index=index,
    )


def test_resolve_preset_defaults():
    assert resolve_preset(None).name == PRESET_STANDARD_RESEARCH.name
    assert resolve_preset(PRESET_QUICK_SMOKE).name == PRESET_QUICK_SMOKE.name
    assert resolve_preset(PRESET_FULL_AUDIT.name).name == PRESET_FULL_AUDIT.name


def test_custom_preset_streams_progress_and_disables_slippage():
    df = _build_df()
    historical_data = {"BTCUSDT": df}
    progress_events = []

    preset = BacktestPresetConfig(
        name="UnitTest",
        enable_intrabar_simulation=False,
        collect_microstructure=False,
        enable_per_bar_debug_logging=False,
        enable_slippage_model=False,
        enable_assertions=False,
        enable_rich_metrics=False,
        progress_log_interval_bars=1,
        processing_chunk_size=5,
    )

    bt = Backtester(
        historical_data,
        evaluate_signal=lambda *_args, **_kwargs: {"score": 1.0, "direction": "long", "confidence": 1.0},
        predict_prob=lambda *_args, **_kwargs: 0.6,
        macro_filter=lambda: True,
        position_size_func=lambda _confidence: 0.01,
    )

    result = bt.run(
        {
            "min_score": 0.0,
            "min_prob": 0.0,
            "atr_mult_sl": 1.5,
            "tp_rungs": (0.5,),
            "fee_bps": 0.0,
            "slippage_bps": 250.0,
            "latency_bars": 0,
            "max_concurrent": 1,
            "is_backtest": True,
        },
        progress_callback=lambda p: progress_events.append(p),
        preset=preset,
    )

    assert progress_events and any(evt.phase == "simulating" for evt in progress_events)

    trades = result.get("trades") or []
    assert trades, "Expected at least one trade"
    first_trade = trades[0]
    entry_time = first_trade["entry_time"]
    expected_open = float(df.loc[entry_time]["open"])
    assert abs(float(first_trade["entry_price"]) - expected_open) < 1e-6
