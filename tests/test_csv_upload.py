import pandas as pd
from backtest import generate_trades_from_ohlcv

def test_generate_trades_from_ohlcv_labels_tp_sl():
    df = pd.DataFrame(
        {
            "open": [100, 100, 100],
            "high": [100, 103, 99],
            "low": [100, 98, 90],
            "close": [100, 102, 95],
        },
        index=pd.date_range("2023-01-01", periods=3, freq="D"),
    )
    trades = generate_trades_from_ohlcv(df, symbol="TEST", take_profit=0.02, stop_loss=0.03)
    assert len(trades) == 2
    assert trades[0]["outcome"] == "tp1"
    assert trades[1]["outcome"] == "sl"
